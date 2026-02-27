"""
SAM + Distance Transform 지붕 면 분리
======================================
MobileSAM → 건물 마스크 + 윤곽 폴리곤
  → 폴리곤 각 변(edge)까지 distance transform
  → 건물 마스크 내 픽셀을 가장 가까운 변에 할당
  → 변 당 1개 face (소면적 제거)
  → 변의 outward normal → azimuth
"""

import os
import io
import math
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
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
# 유틸리티
# ---------------------------------------------------------------------------

def _snap_angle(dx: float, dy: float, snap_deg: float = 15.0) -> tuple[float, float]:
    angle = math.atan2(dy, dx)
    snap_rad = math.radians(snap_deg)
    snapped = round(angle / snap_rad) * snap_rad
    length = math.hypot(dx, dy)
    return length * math.cos(snapped), length * math.sin(snapped)


def _snap_polygon(points: list[tuple[int, int]], snap_deg: float = 15.0) -> list[tuple[int, int]]:
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


def _mask_to_polygon(
    mask: np.ndarray,
    epsilon_ratio: float = 0.015,
    snap_deg: float = 15.0,
    min_area: float = 0,
) -> list[dict] | None:
    mask_uint8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    vertices = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
    if len(vertices) < 3:
        return None
    if snap_deg > 0:
        vertices = _snap_polygon(vertices, snap_deg)
    return [{"x": v[0], "y": v[1]} for v in vertices]


def _simplify_polygon(pts: list[tuple[float, float]], min_edge_ratio: float = 0.08) -> list[tuple[float, float]]:
    """짧은 변 제거하여 폴리곤 단순화."""
    if len(pts) <= 3:
        return pts

    n = len(pts)
    perimeter = sum(
        math.hypot(pts[(i + 1) % n][0] - pts[i][0], pts[(i + 1) % n][1] - pts[i][1])
        for i in range(n)
    )
    min_edge_len = perimeter * min_edge_ratio

    simplified = list(pts)
    changed = True
    while changed and len(simplified) > 3:
        changed = False
        new_pts = []
        skip_next = False
        m = len(simplified)
        for i in range(m):
            if skip_next:
                skip_next = False
                continue
            edge_len = math.hypot(
                simplified[(i + 1) % m][0] - simplified[i][0],
                simplified[(i + 1) % m][1] - simplified[i][1],
            )
            if edge_len < min_edge_len and len(simplified) - (1 if changed else 0) > 3:
                mx = (simplified[i][0] + simplified[(i + 1) % m][0]) / 2
                my = (simplified[i][1] + simplified[(i + 1) % m][1]) / 2
                new_pts.append((mx, my))
                skip_next = True
                changed = True
            else:
                new_pts.append(simplified[i])
        if changed:
            simplified = new_pts

    return simplified


# ---------------------------------------------------------------------------
# Step 1: 건물 전체 마스크 추출
# ---------------------------------------------------------------------------

def _get_building_mask(
    np_image: np.ndarray,
    click_x: int,
    click_y: int,
) -> tuple[np.ndarray, float]:
    _predictor.set_image(np_image)
    h, w = np_image.shape[:2]
    input_point = np.array([[click_x, click_y]])
    input_label = np.array([1])

    masks, scores, _ = _predictor.predict(
        point_coords=input_point, point_labels=input_label, multimask_output=True,
    )
    best_idx = int(np.argmax(scores))
    building_mask = masks[best_idx]
    confidence = float(scores[best_idx])

    mask_area = building_mask.sum()
    image_area = h * w
    if mask_area < image_area * 0.005:
        ys, xs = np.where(building_mask)
        if len(xs) > 0:
            box = np.array([xs.min(), ys.min(), xs.max(), ys.max()])
            bw, bh = box[2] - box[0], box[3] - box[1]
            box[0] = max(0, box[0] - int(bw * 0.1))
            box[1] = max(0, box[1] - int(bh * 0.1))
            box[2] = min(w, box[2] + int(bw * 0.1))
            box[3] = min(h, box[3] + int(bh * 0.1))
            masks2, scores2, _ = _predictor.predict(
                point_coords=input_point, point_labels=input_label,
                box=box[None, :], multimask_output=True,
            )
            best_idx = int(np.argmax(scores2))
            building_mask = masks2[best_idx]
            confidence = float(scores2[best_idx])

    return building_mask, confidence


# ===========================================================================
# Step 2: K-means + Distance Transform 병렬 대조 면 분리
# ===========================================================================


def _kmeans_gate(
    image_bytes: bytes, building_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    """K-means 판별 + face_id_map 반환."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)
    h, w = np_image.shape[:2]
    building_bool = building_mask.astype(bool)
    ys, xs = np.where(building_bool)

    if len(xs) < 100:
        return np.zeros((h, w), dtype=np.int32), 1

    # Lab + 정규화 좌표 5D
    lab = cv2.cvtColor(np_image, cv2.COLOR_RGB2Lab).astype(np.float32)
    pixels = lab[ys, xs]
    coords = np.stack([xs / w, ys / h], axis=1).astype(np.float32) * 50
    features = np.hstack([pixels, coords])

    # 최적 k 탐색 (silhouette)
    max_k = min(6, len(features) // 50)
    if max_k < 2:
        return np.zeros((h, w), dtype=np.int32), 1

    best_k, best_score, best_labels = 1, -1.0, None
    sample_n = min(5000, len(features))

    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, n_init=3, max_iter=100, random_state=42)
        labels = km.fit_predict(features)
        score = silhouette_score(features, labels, sample_size=sample_n)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_score < 0.15:
        return np.zeros((h, w), dtype=np.int32), 1

    face_id_map = np.full((h, w), -1, dtype=np.int32)
    face_id_map[ys, xs] = best_labels
    print(f"[KMeans] k={best_k}, silhouette={best_score:.3f}")
    return face_id_map, best_k


def _dt_faces(
    outline_pred: dict, building_mask: np.ndarray,
) -> tuple[np.ndarray, list, int, bool]:
    """Distance Transform face 맵 생성."""
    points = outline_pred["points"]
    h, w = building_mask.shape

    pts = [(p["x"], p["y"]) for p in points]
    orig_n = len(pts)
    pts = _simplify_polygon(pts)
    n = len(pts)
    if n != orig_n:
        print(f"[DT] polygon simplified: {orig_n} → {n} vertices")

    signed_area = sum(
        pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
        for i in range(n)
    ) / 2
    clockwise = signed_area > 0

    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    edge_distances = []
    for (x1, y1), (x2, y2) in edges:
        edge_img = np.ones((h, w), dtype=np.uint8)
        cv2.line(edge_img, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), 0, 1)
        dist = cv2.distanceTransform(edge_img, cv2.DIST_L2, 5)
        edge_distances.append(dist)

    dist_stack = np.stack(edge_distances, axis=0)
    nearest_edge = np.argmin(dist_stack, axis=0)

    return nearest_edge, edges, n, clockwise


def _cross_validate(
    nearest_edge: np.ndarray,
    face_id_map: np.ndarray,
    building_mask: np.ndarray,
    n: int,
    dominance_threshold: float = 0.55,
) -> list[tuple[int, int]]:
    """DT 경계 vs K-means 경계 대조 → 합칠 쌍 목록.

    면 전체의 dominant K-means cluster를 비교하여 판정.
    경계 픽셀이 아닌 면 전체를 보는 이유: DT 경계(대각선)가
    K-means 경계를 가로지르면 경계 픽셀의 mode가 같게 나올 수 있음.
    """
    building_bool = building_mask.astype(bool)
    merge_pairs = []
    kernel = np.ones((3, 3), np.uint8)

    # 각 DT face의 dominant K-means cluster 계산
    face_dominant = {}
    for i in range(n):
        mask_i = building_bool & (nearest_edge == i)
        km_vals = face_id_map[mask_i]
        valid = km_vals[km_vals >= 0]
        if len(valid) == 0:
            face_dominant[i] = (-1, 0.0)
        else:
            counts = np.bincount(valid)
            mode = int(counts.argmax())
            frac = counts[mode] / len(valid)
            face_dominant[i] = (mode, frac)

    for i in range(n):
        mask_i = building_bool & (nearest_edge == i)
        if mask_i.sum() == 0:
            continue
        dilated_i = cv2.dilate(mask_i.astype(np.uint8), kernel) > 0

        for j in range(i + 1, n):
            mask_j = building_bool & (nearest_edge == j)
            if mask_j.sum() == 0:
                continue

            # 인접 확인
            if (dilated_i & mask_j).sum() < 5:
                continue

            mode_i, frac_i = face_dominant[i]
            mode_j, frac_j = face_dominant[j]

            if mode_i < 0 or mode_j < 0:
                merge_pairs.append((i, j))
            elif mode_i == mode_j:
                merge_pairs.append((i, j))
            elif frac_i < dominance_threshold and frac_j < dominance_threshold:
                # 양쪽 다 애매하면 합침
                merge_pairs.append((i, j))
            # else: 다른 dominant cluster → 진짜 경계 → 유지

    print(f"[XV] face dominants: {face_dominant}")
    return merge_pairs


def segment_faces(
    outline_pred: dict,
    building_mask: np.ndarray,
    image_bytes: bytes,
    epsilon_ratio: float = 0.015,
) -> list[dict]:
    """
    K-means + Distance Transform 병렬 대조 면 분리.

    Path A: K-means → face_id_map (색상/위치 클러스터)
    Path B: DT → nearest_edge (기하학 면)
    대조 → 일치하는 DT 경계만 유지, 나머지 합침
    """
    confidence = outline_pred["confidence"]
    building_bool = building_mask.astype(bool)
    building_area = int(building_bool.sum())
    h, w = building_mask.shape

    # Path A: K-means
    face_id_map, k = _kmeans_gate(image_bytes, building_mask)

    # k=1 → 단일면 즉시 반환 (평탄 지붕)
    if k <= 1:
        print("[Faces] k=1 → single face (flat roof)")
        pts = _mask_to_polygon(building_mask, epsilon_ratio, snap_deg=15.0)
        if pts is None:
            return [], []
        return [{
            "class": "roof_face",
            "confidence": round(confidence, 4),
            "points": pts,
            "pixel_area": building_area,
        }], []

    # Path B: Distance Transform
    nearest_edge, edges, n, clockwise = _dt_faces(outline_pred, building_mask)

    # 대조
    merge_pairs = _cross_validate(nearest_edge, face_id_map, building_mask, n)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in merge_pairs:
        union(i, j)

    # 그룹별 마스크 합침
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    min_area = building_area * 0.05
    face_predictions = []

    for _root, members in groups.items():
        group_mask = np.zeros((h, w), dtype=bool)
        for i in members:
            group_mask |= building_bool & (nearest_edge == i)

        px_area = int(group_mask.sum())
        if px_area < min_area:
            continue

        # Azimuth: 그룹 내 가장 긴 변의 outward normal
        longest_len, azimuth = 0, None
        for i in members:
            (x1, y1), (x2, y2) = edges[i]
            edge_len = math.hypot(x2 - x1, y2 - y1)
            if edge_len > longest_len:
                longest_len = edge_len
                dx, dy = x2 - x1, y2 - y1
                if clockwise:
                    azimuth = math.degrees(math.atan2(dy, dx)) % 360
                else:
                    azimuth = math.degrees(math.atan2(-dy, -dx)) % 360

        face_points = _mask_to_polygon(group_mask, epsilon_ratio, snap_deg=0, min_area=0)
        if face_points is None:
            continue

        face_conf = confidence * min(1.0, px_area / max(building_area, 1))
        pred = {
            "class": "roof_face",
            "confidence": round(face_conf, 4),
            "points": face_points,
            "pixel_area": px_area,
        }
        if azimuth is not None:
            pred["azimuth_deg"] = round(azimuth, 1)
        face_predictions.append(pred)

    # 결과 없으면 단일 face 폴백
    if not face_predictions:
        outline_points = _mask_to_polygon(building_mask, epsilon_ratio, snap_deg=15.0)
        if outline_points:
            face_predictions.append({
                "class": "roof_face",
                "confidence": round(confidence, 4),
                "points": outline_points,
                "pixel_area": building_area,
            })

    # --- Debug layers: K-means clusters + DT faces (merge 전) ---
    debug_preds = []
    unique_clusters = np.unique(face_id_map[building_bool])
    for cid in unique_clusters[unique_clusters >= 0]:
        cmask = (face_id_map == int(cid)) & building_bool
        cpts = _mask_to_polygon(cmask, epsilon_ratio, snap_deg=0, min_area=0)
        if cpts:
            debug_preds.append({
                "class": "debug_kmeans", "confidence": 1.0,
                "points": cpts, "pixel_area": int(cmask.sum()),
            })
    for i in range(n):
        dmask = building_bool & (nearest_edge == i)
        if dmask.sum() < min_area:
            continue
        dpts = _mask_to_polygon(dmask, epsilon_ratio, snap_deg=0, min_area=0)
        if dpts:
            debug_preds.append({
                "class": "debug_dt", "confidence": 1.0,
                "points": dpts, "pixel_area": int(dmask.sum()),
            })

    print(f"[Faces] k={k}, {n} edges, {len(merge_pairs)} merges → {len(face_predictions)} faces")
    return face_predictions, debug_preds


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
        building_mask, epsilon_ratio=epsilon_ratio, snap_deg=snap_deg,
        min_area=image_area * min_area_ratio,
    )
    if outline_points is None:
        return None, None

    outline_pred = {
        "class": "building_outline",
        "confidence": round(confidence, 4),
        "points": outline_points,
        "pixel_area": int(building_mask.sum()),
    }
    return outline_pred, building_mask


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
    outline_pred, building_mask = segment_outline(
        image_bytes, click_x, click_y, epsilon_ratio, snap_deg, min_area_ratio,
    )
    if outline_pred is None:
        return []

    predictions = [outline_pred]
    face_predictions, _debug = segment_faces(outline_pred, building_mask, image_bytes, epsilon_ratio)
    predictions.extend(face_predictions)
    return predictions
