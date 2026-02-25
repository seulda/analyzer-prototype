"""
SAM + OpenCV 지붕 세그멘테이션 (경사 지붕 다면 검출)
=====================================================
MobileSAM 포인트 프롬프트 → 건물 전체 마스크 → K-means 밝기 클러스터링
→ 경사면별 폴리곤 분리 → 각도 스냅

출력 포맷:
    [
        {"class": "building_outline", ...},   # 건물 전체 윤곽 (줌 판정용)
        {"class": "roof_face", ...},           # 경사면 1
        {"class": "roof_face", ...},           # 경사면 2
        ...
    ]
"""

import os
import io
import math

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import silhouette_score
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon

# MobileSAM imports
from mobile_sam import sam_model_registry, SamPredictor

# ---------------------------------------------------------------------------
# 모델 로드 (모듈 임포트 시 1회)
# ---------------------------------------------------------------------------

_MODEL_TYPE = "vit_t"
_CHECKPOINT = os.environ.get(
    "SAM_CHECKPOINT",
    os.path.join(os.path.dirname(__file__), "models", "mobile_sam.pt"),
)

_device = "cpu"
if torch.cuda.is_available():
    _device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    _device = "mps"

print(f"[SAM] Loading MobileSAM from {_CHECKPOINT} ...")

_sam = sam_model_registry[_MODEL_TYPE](checkpoint=_CHECKPOINT)
_sam.to(device=_device)
_sam.eval()

_predictor = SamPredictor(_sam)

print(f"[SAM] MobileSAM loaded on {_device}")

# 워밍업 추론 (MPS 셰이더 컴파일 대기)
_warmup_img = np.zeros((64, 64, 3), dtype=np.uint8)
_predictor.set_image(_warmup_img)
_predictor.predict(
    point_coords=np.array([[32, 32]]),
    point_labels=np.array([1]),
    multimask_output=True,
)
print("[SAM] Warmup complete")


# ---------------------------------------------------------------------------
# 각도 스냅 유틸리티
# ---------------------------------------------------------------------------

def _snap_angle(dx: float, dy: float, snap_deg: float = 15.0) -> tuple[float, float]:
    """
    벡터 (dx, dy)를 snap_deg 단위로 가장 가까운 각도에 스냅.
    건물 윤곽의 직각/평행 모서리를 깔끔하게 정리합니다.
    """
    angle = math.atan2(dy, dx)
    snap_rad = math.radians(snap_deg)
    snapped = round(angle / snap_rad) * snap_rad
    length = math.hypot(dx, dy)
    return length * math.cos(snapped), length * math.sin(snapped)


def _snap_polygon(points: list[tuple[int, int]], snap_deg: float = 15.0) -> list[tuple[int, int]]:
    """
    폴리곤 꼭짓점을 각도 스냅하여 직선화.
    첫 점은 고정하고 이후 점들을 스냅된 벡터로 재계산.
    """
    if len(points) < 3:
        return points

    snapped = [points[0]]
    for i in range(1, len(points)):
        prev = snapped[-1]
        dx = float(points[i][0] - prev[0])
        dy = float(points[i][1] - prev[1])
        sdx, sdy = _snap_angle(dx, dy, snap_deg)
        snapped.append((int(round(prev[0] + sdx)), int(round(prev[1] + sdy))))

    return snapped


# ---------------------------------------------------------------------------
# Shapely 유틸리티
# ---------------------------------------------------------------------------

def _largest_polygon(geom) -> ShapelyPolygon | None:
    """Geometry에서 가장 큰 Polygon을 추출. 유효하지 않으면 None."""
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "Polygon":
        return geom if geom.area > 0 else None
    if geom.geom_type == "MultiPolygon":
        polys = [g for g in geom.geoms if g.area > 0]
        return max(polys, key=lambda g: g.area) if polys else None
    # GeometryCollection 등
    polys = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon" and g.area > 0]
    return max(polys, key=lambda g: g.area) if polys else None


def _extract_polygons(geom) -> list[ShapelyPolygon]:
    """Geometry에서 모든 Polygon을 리스트로 추출."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    return [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]


# ---------------------------------------------------------------------------
# 내부 함수: 마스크 → 폴리곤
# ---------------------------------------------------------------------------

def _mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.015,
    snap_deg: float = 15.0,
    min_area: float = 0,
) -> list[dict] | None:
    """
    바이너리 마스크 → OpenCV 윤곽 → Douglas-Peucker → 각도 스냅 → points 리스트.
    반환: {"x": int, "y": int} 리스트 또는 None (유효하지 않으면)
    """
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < min_area:
        return None

    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_ratio * peri
    approx = cv2.approxPolyDP(contour, epsilon, True)

    vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

    if len(vertices) < 3:
        return None

    if snap_deg > 0:
        vertices = _snap_polygon(vertices, snap_deg)

    return [{"x": v[0], "y": v[1]} for v in vertices]


# ---------------------------------------------------------------------------
# Step 1: 건물 전체 마스크 추출
# ---------------------------------------------------------------------------

def _get_building_mask(
    np_image: np.ndarray,
    click_x: int,
    click_y: int,
) -> tuple[np.ndarray, float]:
    """
    SAM argmax(scores) (최고 점수 마스크)를 건물 전체 마스크로 사용.
    마스크가 너무 작으면 해당 bbox를 box prompt로 SAM 재호출.

    Returns:
        (building_mask, confidence)
    """
    _predictor.set_image(np_image)
    h, w = np_image.shape[:2]

    input_point = np.array([[click_x, click_y]])
    input_label = np.array([1])

    masks, scores, _ = _predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # argmax(scores) = 샘플 노트북과 동일 방식 (최고 점수 마스크)
    best_idx = int(np.argmax(scores))
    building_mask = masks[best_idx]
    confidence = float(scores[best_idx])

    # mask[2]의 면적이 이미지의 0.5% 미만이면 부족하다고 판단 → box prompt
    mask_area = building_mask.sum()
    image_area = h * w
    if mask_area < image_area * 0.005:
        # mask[2]의 bbox를 box prompt로 SAM 재호출
        ys, xs = np.where(building_mask)
        if len(xs) > 0:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            # 박스를 10% 여유 확장
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            box[0] = max(0, box[0] - int(bw * 0.1))
            box[1] = max(0, box[1] - int(bh * 0.1))
            box[2] = min(w, box[2] + int(bw * 0.1))
            box[3] = min(h, box[3] + int(bh * 0.1))

            masks2, scores2, _ = _predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                box=box[None, :],
                multimask_output=True,
            )
            best_idx = int(np.argmax(scores2))
            building_mask = masks2[best_idx]
            confidence = float(scores2[best_idx])

    return building_mask, confidence


# ---------------------------------------------------------------------------
# Step 2: 건물 내 경사면 분리 (K-means 밝기 클러스터링)
# ---------------------------------------------------------------------------

def _split_roof_faces(
    np_image: np.ndarray,
    building_mask: np.ndarray,
    min_face_ratio: float = 0.02,
) -> list[np.ndarray]:
    """
    건물 마스크 영역 내에서 V채널 K-means로 경사면 분리.
    k 선택: valid_ccs × silhouette 스코어 최대 (품질과 면 수의 균형).

    Returns:
        face_masks: 각 경사면의 바이너리 마스크 리스트
    """
    h, w = np_image.shape[:2]

    # V채널 (밝기) 추출
    hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
    mask_y, mask_x = np.where(building_mask)
    if len(mask_y) == 0:
        return []

    v_pixels = hsv[mask_y, mask_x, 2].astype(np.float32).reshape(-1, 1)

    building_area = building_mask.sum()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)

    if len(v_pixels) < 10:
        return [building_mask]

    # --- 전처리: V채널 범위가 좁으면 k=2 강제 ---
    # 이상치(깊은 그림자/반사) 제외를 위해 10-90 백분위 사용
    v_p10 = float(np.percentile(v_pixels, 10))
    v_p90 = float(np.percentile(v_pixels, 90))
    v_range = v_p90 - v_p10
    max_k = 4 if v_range >= 30 else 2
    print(f"[FACE] V p10={v_p10:.0f}, p90={v_p90:.0f}, range={v_range:.0f} → max_k={max_k}")

    # --- 각 k 후보: K-means → silhouette × valid_ccs 스코어 ---
    best_k = 1
    best_score = 0.0
    best_labels = None
    best_centers = np.array([0.0])

    for k in range(2, max_k + 1):
        if len(v_pixels) < k:
            break

        _, labels, centers = cv2.kmeans(
            v_pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        labels_flat = labels.flatten()

        unique_labels = np.unique(labels_flat)
        if len(unique_labels) < 2:
            continue

        # silhouette
        sample_size = min(5000, len(v_pixels))
        if sample_size < len(v_pixels):
            indices = np.random.choice(len(v_pixels), sample_size, replace=False)
            sil = silhouette_score(v_pixels[indices], labels_flat[indices])
        else:
            sil = silhouette_score(v_pixels, labels_flat)

        if sil < 0.2:
            print(f"[FACE] k={k} sil={sil:.4f} (below 0.2, skip)")
            continue

        # 유효 CC 수 계산 (adaptive_ratio를 k에 맞춰 적용)
        k_ratio = max(min_face_ratio, 0.15 / k)
        tmp_label_map = np.zeros((h, w), dtype=np.int32)
        tmp_label_map[mask_y, mask_x] = labels_flat + 1
        valid_ccs = 0
        for cid in range(k):
            cluster_mask = (tmp_label_map == (cid + 1)).astype(np.uint8)
            num_labels, cc_labels = cv2.connectedComponents(cluster_mask)
            for cc_id in range(1, num_labels):
                if (cc_labels == cc_id).sum() >= building_area * k_ratio:
                    valid_ccs += 1

        # 균형 스코어: 면 수 × 클러스터 품질
        combined = valid_ccs * sil
        print(f"[FACE] k={k} sil={sil:.4f} ccs={valid_ccs} ratio={k_ratio:.3f} score={combined:.3f}")

        if combined > best_score:
            best_score = combined
            best_k = k
            best_labels = labels_flat
            best_centers = centers.flatten()  # V값 클러스터 중심

    print(f"[FACE] selected k={best_k}, score={best_score:.3f}")

    if best_k <= 1 or best_labels is None:
        return [building_mask]

    # --- 선택된 k의 labels로 face_id_map 구축 ---
    # min_face_ratio를 k에 반비례로 동적 조정
    # k=2 → 7.5%, k=3 → 5%, k=4 → 3.75%
    adaptive_ratio = max(min_face_ratio, 0.15 / best_k)
    print(f"[FACE] adaptive min_face_ratio={adaptive_ratio:.3f} ({adaptive_ratio*100:.1f}%)")

    label_map = np.zeros((h, w), dtype=np.int32)
    label_map[mask_y, mask_x] = best_labels + 1

    next_face_id = 1
    face_id_map = np.zeros((h, w), dtype=np.int32)

    for cluster_id in range(best_k):
        cluster_mask = (label_map == (cluster_id + 1)).astype(np.uint8)
        num_labels, cc_labels = cv2.connectedComponents(cluster_mask)

        for cc_id in range(1, num_labels):
            cc_mask = cc_labels == cc_id
            cc_area = cc_mask.sum()

            if cc_area < building_area * adaptive_ratio:
                continue

            face_id_map[cc_mask] = next_face_id
            next_face_id += 1

    if next_face_id == 1:
        return [building_mask]

    # --- 디버그: face_id_map 시각화 저장 ---
    try:
        colors = [
            (0,0,0), (255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255), (0,255,255), (128,0,255),
            (255,128,0), (0,128,255), (128,255,0), (255,0,128),
        ]
        debug_img = np_image.copy()
        overlay = np.zeros_like(debug_img)
        for fid in range(1, next_face_id):
            c = colors[fid % len(colors)]
            overlay[face_id_map == fid] = c
        # 건물 영역만 반투명 오버레이
        mask_bool = building_mask.astype(bool)
        debug_img[mask_bool] = (debug_img[mask_bool] * 0.4 + overlay[mask_bool] * 0.6).astype(np.uint8)
        # 윤곽선
        for fid in range(1, next_face_id):
            fm = (face_id_map == fid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, contours, -1, (255,255,255), 1)
        Image.fromarray(debug_img).save("/tmp/debug_faces_before_merge.png")
        print(f"[DEBUG] saved /tmp/debug_faces_before_merge.png ({next_face_id-1} faces)")
    except Exception as e:
        print(f"[DEBUG] save failed: {e}")

    # 미할당 픽셀 재배분: 가장 가까운 유효 면에 편입
    unassigned = building_mask.astype(bool) & (face_id_map == 0)

    if np.any(unassigned):
        distances = []
        for fid in range(1, next_face_id):
            face_bin = (face_id_map == fid).astype(np.uint8)
            dist = cv2.distanceTransform(
                1 - face_bin, cv2.DIST_L2, 5,
            )
            distances.append(dist)
        dist_stack = np.stack(distances, axis=0)
        nearest = np.argmin(dist_stack, axis=0) + 1

        face_id_map[unassigned] = nearest[unassigned]

    # --- 후처리: 인접 유사 밝기 면 병합 ---
    # threshold = 클러스터 중심간 최소 거리의 절반
    # → 같은 클러스터 파편끼리만 병합, 다른 클러스터는 보존
    v_channel = hsv[:, :, 2]
    kernel = np.ones((3, 3), np.uint8)
    sorted_centers = np.sort(best_centers)
    min_gap = float(np.min(np.diff(sorted_centers))) if len(sorted_centers) > 1 else 255
    merge_threshold = min_gap * 0.5
    print(f"[FACE] cluster centers={sorted_centers.astype(int)}, min_gap={min_gap:.1f}, merge_thr={merge_threshold:.1f}")

    face_ids = set(int(x) for x in np.unique(face_id_map) if x > 0)
    # 원본 밝기 고정 (cascade 방지 — 병합해도 비교 기준은 원래 값)
    orig_mean_v = {fid: float(v_channel[face_id_map == fid].mean()) for fid in face_ids}
    alive = set(face_ids)

    merged = True
    while merged:
        merged = False
        for fid_a in sorted(alive):
            dilated = cv2.dilate(
                (face_id_map == fid_a).astype(np.uint8), kernel,
            )
            neighbors = set(int(x) for x in np.unique(face_id_map[dilated > 0])) & alive - {fid_a}

            for fid_b in sorted(neighbors):
                if abs(orig_mean_v[fid_a] - orig_mean_v[fid_b]) <= merge_threshold:
                    face_id_map[face_id_map == fid_b] = fid_a
                    alive.discard(fid_b)
                    merged = True
                    break
            if merged:
                break

    print(f"[FACE] after merge: {len(alive)} faces (threshold={merge_threshold:.1f})")

    # --- 디버그: merge 후 시각화 저장 ---
    try:
        colors = [
            (0,0,0), (255,0,0), (0,255,0), (0,0,255),
            (255,255,0), (255,0,255), (0,255,255), (128,0,255),
            (255,128,0), (0,128,255), (128,255,0), (255,0,128),
        ]
        debug_img2 = np_image.copy()
        overlay2 = np.zeros_like(debug_img2)
        for i, fid in enumerate(sorted(alive)):
            c = colors[(i + 1) % len(colors)]
            overlay2[face_id_map == fid] = c
        mask_bool2 = building_mask.astype(bool)
        debug_img2[mask_bool2] = (debug_img2[mask_bool2] * 0.4 + overlay2[mask_bool2] * 0.6).astype(np.uint8)
        for fid in sorted(alive):
            fm = (face_id_map == fid).astype(np.uint8) * 255
            contours, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img2, contours, -1, (255,255,255), 1)
        Image.fromarray(debug_img2).save("/tmp/debug_faces_after_merge.png")
        print(f"[DEBUG] saved /tmp/debug_faces_after_merge.png ({len(alive)} faces)")
    except Exception as e:
        print(f"[DEBUG] save failed: {e}")

    # face_id_map → 개별 마스크 리스트
    face_masks = []
    for fid in sorted(alive):
        fm = face_id_map == fid
        if fm.any():
            face_masks.append(fm)

    print(f"[FACE] final face count: {len(face_masks)}")
    return face_masks


# ---------------------------------------------------------------------------
# 공개 함수: Step 1 — 건물 윤곽만 추출
# ---------------------------------------------------------------------------

def segment_outline(
    image_bytes: bytes,
    click_x: int | None = None,
    click_y: int | None = None,
    epsilon_ratio: float = 0.015,
    snap_deg: float = 15.0,
    min_area_ratio: float = 0.005,
) -> tuple[dict | None, np.ndarray | None]:
    """
    Step 1만: SAM → building_mask → outline polygon.

    Returns:
        (outline_pred, building_mask)
        outline_pred: {"class": "building_outline", ...} 또는 None
        building_mask: numpy 바이너리 마스크 또는 None
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)
    h, w = np_image.shape[:2]
    image_area = h * w

    if click_x is None:
        click_x = w // 2
    if click_y is None:
        click_y = h // 2

    building_mask, confidence = _get_building_mask(np_image, click_x, click_y)

    outline_points = _mask_to_polygon(
        building_mask,
        epsilon_ratio=epsilon_ratio,
        snap_deg=snap_deg,
        min_area=image_area * min_area_ratio,
    )

    if outline_points is None:
        return None, None

    building_pixel_area = int(building_mask.sum())

    outline_pred = {
        "class": "building_outline",
        "confidence": round(confidence, 4),
        "points": outline_points,
        "pixel_area": building_pixel_area,
    }

    return outline_pred, building_mask


# ---------------------------------------------------------------------------
# 공개 함수: Step 2 — 건물 마스크로 면 분리
# ---------------------------------------------------------------------------

def segment_faces(
    image_bytes: bytes,
    building_mask: np.ndarray,
    outline_confidence: float,
    epsilon_ratio: float = 0.015,
) -> list[dict]:
    """
    Step 2: building_mask로 K-means 면 분리. face predictions 반환.

    Args:
        image_bytes: 위성 이미지 바이트
        building_mask: 건물 바이너리 마스크
        outline_confidence: 건물 윤곽 신뢰도 (face 신뢰도 계산용)

    Returns:
        face predictions 리스트 (roof_face들)
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)

    building_pixel_area = int(building_mask.sum())
    face_masks = _split_roof_faces(np_image, building_mask)

    # 클리핑용 건물 윤곽 폴리곤 생성 (outline과 동일 파라미터)
    clip_outline = _mask_to_polygon(
        building_mask, epsilon_ratio=epsilon_ratio, snap_deg=15.0, min_area=0,
    )
    clip_poly = None
    if clip_outline:
        clip_poly = ShapelyPolygon([(p["x"], p["y"]) for p in clip_outline])
        if not clip_poly.is_valid:
            clip_poly = clip_poly.buffer(0)

    # 1) 각 face 마스크 → 폴리곤 → 건물 윤곽으로 클리핑
    face_entries = []  # (shapely_poly, pixel_area)
    for face_mask in face_masks:
        face_points = _mask_to_polygon(
            face_mask,
            epsilon_ratio=epsilon_ratio * 0.5,
            snap_deg=30,
            min_area=0,
        )
        if face_points is None:
            continue

        face_poly = ShapelyPolygon([(p["x"], p["y"]) for p in face_points])
        if not face_poly.is_valid:
            face_poly = face_poly.buffer(0)

        if clip_poly is not None:
            face_poly = face_poly.intersection(clip_poly)

        face_poly = _largest_polygon(face_poly)
        if face_poly is None:
            continue

        face_entries.append((face_poly, int(face_mask.sum())))

    # 2) 겹침 제거: 면적 큰 순으로 우선권
    face_entries.sort(key=lambda e: e[0].area, reverse=True)
    assigned = ShapelyPolygon()  # 빈 폴리곤
    non_overlapping = []
    for poly, px_area in face_entries:
        remaining = poly.difference(assigned)
        remaining = _largest_polygon(remaining)
        if remaining is None:
            continue
        non_overlapping.append((remaining, px_area))
        assigned = assigned.union(remaining)

    # 3) 빈틈 채우기: 건물 윤곽 중 어떤 face에도 속하지 않은 영역 → 가장 가까운 face에 병합
    if clip_poly is not None and non_overlapping:
        gap = clip_poly.difference(assigned)
        gap_parts = _extract_polygons(gap)
        for gp in gap_parts:
            if gp.area < 0.5:
                continue
            nearest_idx = min(
                range(len(non_overlapping)),
                key=lambda i: non_overlapping[i][0].distance(gp),
            )
            merged = non_overlapping[nearest_idx][0].union(gp)
            merged = _largest_polygon(merged)
            if merged is not None:
                non_overlapping[nearest_idx] = (merged, non_overlapping[nearest_idx][1])

    # 4) Shapely → points dict 변환
    face_predictions = []
    for poly, px_area in non_overlapping:
        coords = list(poly.exterior.coords[:-1])
        face_points = [
            {"x": round(x, 2), "y": round(y, 2)}
            for x, y in coords
        ]
        if len(face_points) < 3:
            continue

        face_confidence = outline_confidence * min(
            1.0, px_area / max(building_pixel_area, 1),
        )
        face_predictions.append({
            "class": "roof_face",
            "confidence": round(face_confidence, 4),
            "points": face_points,
            "pixel_area": px_area,
        })

    # 면 분리 결과가 없으면 건물 전체를 단일 roof_face로
    if not face_predictions:
        outline_points = _mask_to_polygon(building_mask, epsilon_ratio=epsilon_ratio, snap_deg=15.0, min_area=0)
        if outline_points:
            face_predictions.append({
                "class": "roof_face",
                "confidence": round(outline_confidence, 4),
                "points": outline_points,
                "pixel_area": building_pixel_area,
            })

    return face_predictions


# ---------------------------------------------------------------------------
# 메인 세그멘테이션 함수 (호환성 유지)
# ---------------------------------------------------------------------------

def segment_building(
    image_bytes: bytes,
    click_x: int | None = None,
    click_y: int | None = None,
    epsilon_ratio: float = 0.015,
    snap_deg: float = 15.0,
    min_area_ratio: float = 0.005,
) -> list[dict]:
    """
    MobileSAM + OpenCV로 건물 지붕을 세그멘테이션합니다.
    내부적으로 segment_outline + segment_faces를 순차 호출합니다.

    Returns:
        predictions:
            [0] = {"class": "building_outline", ...}
            [1:] = {"class": "roof_face", ...}
    """
    outline_pred, building_mask = segment_outline(
        image_bytes, click_x, click_y, epsilon_ratio, snap_deg, min_area_ratio,
    )

    if outline_pred is None:
        return []

    predictions = [outline_pred]

    face_predictions = segment_faces(
        image_bytes, building_mask, outline_pred["confidence"], epsilon_ratio,
    )
    predictions.extend(face_predictions)

    return predictions
