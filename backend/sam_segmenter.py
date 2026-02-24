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
from shapely.geometry import Polygon as ShapelyPolygon, LineString
from shapely.ops import split as shapely_split, unary_union
from sklearn.metrics import silhouette_score

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

def _boundary_linearity(label_map: np.ndarray, mask_pixels_y: np.ndarray, mask_pixels_x: np.ndarray) -> float:
    """
    label map에서 클러스터 간 경계의 직선성을 측정.
    각 경계를 직선 피팅 후 평균 잔차(residual)를 계산.
    반환값이 낮을수록 경계가 직선에 가까움 (좋은 분할).
    """
    h, w = label_map.shape
    boundary_groups: dict[tuple[int, int], list[tuple[float, float]]] = {}

    # 수평 경계
    diff_h = label_map[:, :-1] != label_map[:, 1:]
    valid_h = (label_map[:, :-1] > 0) & (label_map[:, 1:] > 0) & diff_h
    ys, xs = np.where(valid_h)
    for y, x in zip(ys, xs):
        a, b = int(label_map[y, x]), int(label_map[y, x + 1])
        key = (min(a, b), max(a, b))
        boundary_groups.setdefault(key, []).append((float(x) + 0.5, float(y)))

    # 수직 경계
    diff_v = label_map[:-1, :] != label_map[1:, :]
    valid_v = (label_map[:-1, :] > 0) & (label_map[1:, :] > 0) & diff_v
    ys, xs = np.where(valid_v)
    for y, x in zip(ys, xs):
        a, b = int(label_map[y, x]), int(label_map[y + 1, x])
        key = (min(a, b), max(a, b))
        boundary_groups.setdefault(key, []).append((float(x), float(y) + 0.5))

    if not boundary_groups:
        return float("inf")

    # 각 경계 그룹의 직선 피팅 잔차 계산
    residuals = []
    for pair, pixels in boundary_groups.items():
        if len(pixels) < 5:
            continue
        pts = np.array(pixels, dtype=np.float32)
        pts_cv = pts.reshape(-1, 1, 2)
        line_params = cv2.fitLine(pts_cv, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()

        # 각 점에서 직선까지의 거리
        dx = pts[:, 0] - x0
        dy = pts[:, 1] - y0
        # 직선에 수직인 거리 = |dx * vy - dy * vx|
        dists = np.abs(dx * vy - dy * vx)
        residuals.append(float(np.mean(dists)))

    if not residuals:
        return float("inf")

    return float(np.mean(residuals))


def _find_optimal_k(
    pixels: np.ndarray,
    mask_pixels_y: np.ndarray,
    mask_pixels_x: np.ndarray,
    h: int,
    w: int,
    building_area: int,
    max_k: int = 16,
    max_residual_ratio: float = 0.25,
) -> int:
    """
    k를 2부터 올려가며, silhouette score + 경계 직선성으로 최적 k 선택.

    멈추는 조건:
    - silhouette score < 0.15 (클러스터 분리가 약함)
    - 새로 생긴 경계의 직선 잔차 > building_size * max_residual_ratio (경계가 구불구불)
    - 두 조건 모두 충족하는 마지막 k를 채택

    max_k: 탐색 상한 (기본 16)
    max_residual_ratio: 건물 크기 대비 허용 잔차 비율 (기본 0.25)
    """
    if len(pixels) < 10:
        return 1

    # 건물 크기 기준 상대적 잔차 한계
    building_size = math.sqrt(building_area)
    max_residual = building_size * max_residual_ratio
    print(f"[K-means] building_size={building_size:.0f}px, max_residual={max_residual:.1f}px ({max_residual_ratio*100:.0f}%)")

    best_k = 1
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)

    for k in range(2, max_k + 1):
        if len(pixels) < k * 20:  # 클러스터당 최소 20개 픽셀
            break

        _, labels, _ = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )
        labels_flat = labels.flatten()

        unique_labels = np.unique(labels_flat)
        if len(unique_labels) < 2:
            break

        # silhouette score (샘플링)
        sample_size = min(5000, len(pixels))
        if sample_size < len(pixels):
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            sil_score = silhouette_score(pixels[indices], labels_flat[indices])
        else:
            sil_score = silhouette_score(pixels, labels_flat)

        # 경계 직선성 측정
        label_map = np.zeros((h, w), dtype=np.int32)
        label_map[mask_pixels_y, mask_pixels_x] = labels_flat + 1
        residual = _boundary_linearity(label_map, mask_pixels_y, mask_pixels_x)

        print(f"[K-means] k={k}: silhouette={sil_score:.3f}, residual={residual:.2f}px ({residual/building_size*100:.1f}%)")

        if sil_score < 0.15:
            print(f"[K-means] k={k} 탈락: silhouette 부족")
            break

        if residual > max_residual:
            print(f"[K-means] k={k} 탈락: 경계 비직선 (>{max_residual:.1f}px)")
            break

        best_k = k

    print(f"[K-means] 최적 k={best_k}")
    return best_k


def _split_roof_faces(
    np_image: np.ndarray,
    building_mask: np.ndarray,
    min_face_ratio: float = 0.05,
) -> list[np.ndarray]:
    """
    건물 마스크 영역 내에서 밝기 기반 K-means로 경사면 분리.
    Label map 방식: 각 픽셀은 정확히 하나의 면에만 속함 (overlap 없음).

    Returns:
        face_masks: 각 경사면의 바이너리 마스크 리스트
    """
    h, w = np_image.shape[:2]

    # 건물 마스크 영역의 픽셀 좌표 추출
    mask_pixels_y, mask_pixels_x = np.where(building_mask)
    if len(mask_pixels_y) == 0:
        return []

    # Lab 색공간 (밝기+색상, 인지적으로 균일)
    lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[mask_pixels_y, mask_pixels_x, 0]  # 0~255 (OpenCV scale)
    a = lab[mask_pixels_y, mask_pixels_x, 1]  # 0~255 (centered at 128)
    b = lab[mask_pixels_y, mask_pixels_x, 2]  # 0~255 (centered at 128)

    # 정규화된 좌표 (0~1 범위)
    x_norm = mask_pixels_x.astype(np.float32) / w
    y_norm = mask_pixels_y.astype(np.float32) / h

    # 5D 피처: [L, a, b, x*spatial_weight, y*spatial_weight]
    spatial_weight = 0.3  # 색상 대비 공간의 상대적 가중치
    pixels = np.column_stack([
        L / 255.0,            # 0~1 정규화
        a / 255.0,            # 0~1 정규화
        b / 255.0,            # 0~1 정규화
        x_norm * spatial_weight,
        y_norm * spatial_weight,
    ])

    building_area = building_mask.sum()

    # 최적 K 선택 (silhouette + 경계 직선성)
    optimal_k = _find_optimal_k(pixels, mask_pixels_y, mask_pixels_x, h, w, building_area)

    if optimal_k <= 1:
        return [building_mask]

    # K-means 클러스터링
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
    _, labels, _ = cv2.kmeans(
        pixels, optimal_k, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
    )
    labels_flat = labels.flatten()

    # Label map 생성 (각 건물 픽셀에 클러스터 ID 배정, 0 = 미할당)
    label_map = np.zeros((h, w), dtype=np.int32)
    label_map[mask_pixels_y, mask_pixels_x] = labels_flat + 1  # 1-indexed (vectorized)

    # 각 클러스터 → connected components → 소면적 CC를 미할당(0)으로 전환
    next_face_id = 1
    face_id_map = np.zeros((h, w), dtype=np.int32)  # 최종 면 ID 맵

    for cluster_id in range(optimal_k):
        cluster_mask = (label_map == (cluster_id + 1)).astype(np.uint8)
        num_labels, cc_labels = cv2.connectedComponents(cluster_mask)

        for cc_id in range(1, num_labels):
            cc_mask = cc_labels == cc_id
            cc_area = cc_mask.sum()

            if cc_area < building_area * min_face_ratio:
                # 소면적 → 미할당으로 전환 (나중에 재배분)
                continue

            face_id_map[cc_mask] = next_face_id
            next_face_id += 1

    # 유효한 면이 없으면 건물 마스크 통째로 반환
    if next_face_id == 1:
        return [building_mask]

    # 미할당 픽셀 재배분: 가장 가까운 유효 면에 편입
    unassigned = building_mask.astype(bool) & (face_id_map == 0)

    if np.any(unassigned):
        # 각 면까지의 거리 → 미할당 픽셀을 최근접 면에 배정
        num_faces = next_face_id - 1
        distances = []
        for fid in range(1, next_face_id):
            face_bin = (face_id_map == fid).astype(np.uint8)
            dist = cv2.distanceTransform(
                1 - face_bin, cv2.DIST_L2, 5,
            )
            distances.append(dist)
        dist_stack = np.stack(distances, axis=0)
        nearest = np.argmin(dist_stack, axis=0) + 1  # 1-indexed face ID

        face_id_map[unassigned] = nearest[unassigned]

    # face_id_map → 개별 마스크 리스트로 변환
    face_masks = []
    for fid in range(1, next_face_id):
        fm = face_id_map == fid
        if fm.any():
            face_masks.append(fm)

    return face_masks


def _find_split_lines(face_id_map: np.ndarray) -> list[LineString]:
    """
    face_id_map에서 인접 면 간 경계 픽셀을 추출하고 직선으로 피팅.
    각 능선(ridgeline)을 하나의 직선으로 반환.
    """
    h, w = face_id_map.shape
    boundary_groups: dict[tuple[int, int], list[tuple[float, float]]] = {}

    # 수평 경계 (좌우 인접 비교)
    diff_h = face_id_map[:, :-1] != face_id_map[:, 1:]
    valid_h = (face_id_map[:, :-1] > 0) & (face_id_map[:, 1:] > 0) & diff_h
    ys, xs = np.where(valid_h)
    for y, x in zip(ys, xs):
        a, b = int(face_id_map[y, x]), int(face_id_map[y, x + 1])
        key = (min(a, b), max(a, b))
        boundary_groups.setdefault(key, []).append((float(x) + 0.5, float(y)))

    # 수직 경계 (상하 인접 비교)
    diff_v = face_id_map[:-1, :] != face_id_map[1:, :]
    valid_v = (face_id_map[:-1, :] > 0) & (face_id_map[1:, :] > 0) & diff_v
    ys, xs = np.where(valid_v)
    for y, x in zip(ys, xs):
        a, b = int(face_id_map[y, x]), int(face_id_map[y + 1, x])
        key = (min(a, b), max(a, b))
        boundary_groups.setdefault(key, []).append((float(x), float(y) + 0.5))

    # 각 경계 그룹에 직선 피팅
    lines = []
    for pair, pixels in boundary_groups.items():
        if len(pixels) < 10:
            continue
        pts = np.array(pixels, dtype=np.float32).reshape(-1, 1, 2)
        line_params = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()

        # 이미지를 충분히 넘는 길이로 연장
        t = float(max(h, w)) * 3
        line = LineString([
            (x0 - vx * t, y0 - vy * t),
            (x0 + vx * t, y0 + vy * t),
        ])
        lines.append(line)

    return lines


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
    outline_pred: dict | None = None,
    epsilon_ratio: float = 0.015,
) -> list[dict]:
    """
    Step 2: building_mask → K-means 면 분리 → 능선(ridgeline) 추출 →
    outline 폴리곤을 직선으로 split → 퍼즐 조각 face predictions.

    핵심: 각 면의 윤곽을 독립 추출하지 않고, outline을 능선 직선으로 "칼로 잘라서"
    겹침/빈공간 없는 퍼즐 조각을 생성.
    """
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)
    h, w = np_image.shape[:2]

    building_pixel_area = int(building_mask.sum())
    face_masks = _split_roof_faces(np_image, building_mask)

    # building outline 폴리곤 생성
    outline_shape = None
    if outline_pred and "points" in outline_pred:
        outline_pts = [(p["x"], p["y"]) for p in outline_pred["points"]]
        if len(outline_pts) >= 3:
            try:
                outline_shape = ShapelyPolygon(outline_pts)
                if not outline_shape.is_valid:
                    outline_shape = outline_shape.buffer(0)
            except Exception:
                outline_shape = None

    face_predictions = []

    if outline_shape is not None and len(face_masks) > 1:
        # --- 능선 split 방식: label map → ridgeline → outline split ---

        # face_id_map 재구성 (label map)
        face_id_map = np.zeros((h, w), dtype=np.int32)
        for i, mask in enumerate(face_masks):
            face_id_map[mask] = i + 1

        # 능선 직선 추출
        split_lines = _find_split_lines(face_id_map)

        if split_lines:
            # outline을 능선 직선으로 순차 split (칼로 자르기)
            pieces = [outline_shape]
            for line in split_lines:
                new_pieces = []
                for piece in pieces:
                    try:
                        result = shapely_split(piece, line)
                        for geom in result.geoms:
                            if geom.geom_type == "Polygon" and not geom.is_empty:
                                new_pieces.append(geom)
                    except Exception:
                        new_pieces.append(piece)
                pieces = new_pieces

            # 각 조각을 face에 배정 (label map의 대표점으로 판별)
            face_polygon_groups: dict[int, list] = {}
            for piece in pieces:
                if piece.is_empty or piece.geom_type != "Polygon":
                    continue
                pt = piece.representative_point()
                px = max(0, min(w - 1, int(round(pt.x))))
                py = max(0, min(h - 1, int(round(pt.y))))
                fid = int(face_id_map[py, px])
                if fid == 0:
                    continue
                face_polygon_groups.setdefault(fid, []).append(piece)

            # 같은 face의 조각 병합 → prediction 생성
            for fid in sorted(face_polygon_groups.keys()):
                polys = face_polygon_groups[fid]
                merged = unary_union(polys)

                if merged.geom_type == "MultiPolygon":
                    merged = max(merged.geoms, key=lambda g: g.area)
                if merged.geom_type != "Polygon" or merged.is_empty:
                    continue

                simplified = merged.simplify(1.5, preserve_topology=True)
                if simplified.is_empty or simplified.geom_type != "Polygon":
                    simplified = merged

                final_points = [
                    {"x": int(round(x)), "y": int(round(y))}
                    for x, y in simplified.exterior.coords[:-1]
                ]
                if len(final_points) < 3:
                    continue

                face_pixel_area = int(face_masks[fid - 1].sum())
                face_confidence = outline_confidence * min(
                    1.0, face_pixel_area / max(building_pixel_area, 1),
                )
                face_predictions.append({
                    "class": "roof_face",
                    "confidence": round(face_confidence, 4),
                    "points": final_points,
                    "pixel_area": face_pixel_area,
                })

            face_predictions.sort(key=lambda p: p["pixel_area"], reverse=True)

    # 단일 면이거나 split 결과 없으면 outline 전체를 단일 roof_face로
    if not face_predictions:
        outline_points = _mask_to_polygon(
            building_mask, epsilon_ratio=epsilon_ratio, snap_deg=15.0, min_area=0,
        )
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
        image_bytes, building_mask, outline_pred["confidence"],
        outline_pred=outline_pred, epsilon_ratio=epsilon_ratio,
    )
    predictions.extend(face_predictions)

    return predictions
