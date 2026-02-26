"""
SAM + Straight Skeleton 지붕 세그멘테이션
==========================================
MobileSAM 포인트 프롬프트 → 건물 전체 마스크 → 폴리곤 각 변(edge) 기반 면 분리
→ distance transform으로 building_mask 내 픽셀을 가장 가까운 변에 할당
→ 변의 outward normal → azimuth 계산

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
from shapely.geometry import Polygon as ShapelyPolygon, MultiPolygon, LineString
from shapely.ops import split as shapely_split, unary_union

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
# Step 2: Straight Skeleton 기반 면 분리
# ---------------------------------------------------------------------------

def segment_faces_skeleton(
    outline_pred: dict,
    building_mask: np.ndarray,
    epsilon_ratio: float = 0.015,
) -> list[dict]:
    """
    건물 윤곽 폴리곤을 angle bisector로 기하학적 분할.
    윤곽 폴리곤 자체를 쪼개므로 빈틈/삐져나감 없이 완벽한 타일링.
    """
    points = outline_pred["points"]  # [{"x": int, "y": int}, ...]
    confidence = outline_pred["confidence"]
    h, w = building_mask.shape
    building_bool = building_mask.astype(bool)
    building_area = int(building_bool.sum())
    n = len(points)

    pts = [(p["x"], p["y"]) for p in points]

    # 1. 폴리곤 winding 판별 (signed area)
    signed_area = sum(
        pts[i][0] * pts[(i+1)%n][1] - pts[(i+1)%n][0] * pts[i][1]
        for i in range(n)
    ) / 2
    clockwise = signed_area > 0  # 이미지 좌표(y↓)에서 양수 = 시계방향

    # 2. 윤곽 폴리곤
    outline_poly = ShapelyPolygon(pts)
    if not outline_poly.is_valid:
        outline_poly = outline_poly.buffer(0)

    # 3. 각 꼭짓점의 angle bisector 계산 → split line 생성
    extent = max(h, w) * 2.0
    bisector_lines = []

    for j in range(n):
        prev_j = (j - 1) % n
        next_j = (j + 1) % n

        # 인접 변 방향
        d_in = (pts[j][0] - pts[prev_j][0], pts[j][1] - pts[prev_j][1])
        d_out = (pts[next_j][0] - pts[j][0], pts[next_j][1] - pts[j][1])

        len_in = math.hypot(*d_in)
        len_out = math.hypot(*d_out)
        if len_in < 1e-10 or len_out < 1e-10:
            continue

        d_in = (d_in[0] / len_in, d_in[1] / len_in)
        d_out = (d_out[0] / len_out, d_out[1] / len_out)

        # 내향 법선 → bisector
        if clockwise:
            n_in = (-d_in[1], d_in[0])
            n_out = (-d_out[1], d_out[0])
        else:
            n_in = (d_in[1], -d_in[0])
            n_out = (d_out[1], -d_out[0])

        bx = n_in[0] + n_out[0]
        by = n_in[1] + n_out[1]
        blen = math.hypot(bx, by)
        if blen < 1e-10:
            continue
        bx, by = bx / blen, by / blen

        # 꼭짓점에서 살짝 바깥 → 안쪽 끝까지 직선
        line = LineString([
            (pts[j][0] - bx * 0.5, pts[j][1] - by * 0.5),
            (pts[j][0] + bx * extent, pts[j][1] + by * extent),
        ])
        bisector_lines.append(line)

    # 4. bisector로 윤곽 폴리곤 분할
    pieces = [outline_poly]
    for bline in bisector_lines:
        new_pieces = []
        for piece in pieces:
            try:
                result = shapely_split(piece, bline)
                for geom in result.geoms:
                    for poly in _extract_polygons(geom):
                        if poly.area > 0:
                            new_pieces.append(poly)
            except Exception:
                new_pieces.append(piece)
        if new_pieces:
            pieces = new_pieces

    # 5. 각 조각 → 가장 가까운 변 할당 (distance transform)
    edges = []
    for i in range(n):
        edges.append((pts[i], pts[(i + 1) % n]))

    edge_distances = []
    for (x1, y1), (x2, y2) in edges:
        edge_img = np.ones((h, w), dtype=np.uint8)
        cv2.line(edge_img, (int(x1), int(y1)), (int(x2), int(y2)), 0, 1)
        dist = cv2.distanceTransform(edge_img, cv2.DIST_L2, 5)
        edge_distances.append(dist)
    dist_stack = np.stack(edge_distances, axis=0)
    nearest_edge = np.argmin(dist_stack, axis=0)

    # 6. 같은 변에 속하는 조각들을 합침 (union)
    edge_piece_groups: dict[int, list] = {}
    for piece in pieces:
        if piece.area < 1:
            continue
        cx, cy = piece.centroid.x, piece.centroid.y
        ci = max(0, min(h - 1, int(round(cy))))
        cj = max(0, min(w - 1, int(round(cx))))
        edge_idx = int(nearest_edge[ci, cj])
        edge_piece_groups.setdefault(edge_idx, []).append(piece)

    min_face_area = outline_poly.area * 0.05
    face_predictions = []

    for edge_idx, group in edge_piece_groups.items():
        merged = unary_union(group)
        merged = _largest_polygon(merged)
        if merged is None or merged.area < min_face_area:
            continue

        # azimuth 계산
        (x1, y1), (x2, y2) = edges[edge_idx]
        dx, dy = x2 - x1, y2 - y1
        if clockwise:
            azimuth = math.degrees(math.atan2(dy, dx)) % 360
        else:
            azimuth = math.degrees(math.atan2(-dy, -dx)) % 360

        # 픽셀 면적 (building_mask 기준)
        piece_coords = np.array(list(merged.exterior.coords), dtype=np.int32)
        piece_raster = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(piece_raster, [piece_coords], 1)
        px_area = int((building_bool & (piece_raster > 0)).sum())

        coords = list(merged.exterior.coords[:-1])
        face_points = [{"x": round(x, 2), "y": round(y, 2)} for x, y in coords]
        if len(face_points) < 3:
            continue

        face_confidence = confidence * min(1.0, px_area / max(building_area, 1))
        face_predictions.append({
            "class": "roof_face",
            "confidence": round(face_confidence, 4),
            "points": face_points,
            "pixel_area": px_area,
            "azimuth_deg": round(azimuth, 1),
        })

    # 결과 없으면 단일 face 폴백
    if not face_predictions:
        outline_points = _mask_to_polygon(building_mask, epsilon_ratio, snap_deg=15.0, min_area=0)
        if outline_points:
            face_predictions.append({
                "class": "roof_face",
                "confidence": round(confidence, 4),
                "points": outline_points,
                "pixel_area": building_area,
            })

    print(f"[SKELETON] final face count: {len(face_predictions)}")
    return face_predictions


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
    내부적으로 segment_outline + segment_faces_skeleton을 순차 호출합니다.

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

    face_predictions = segment_faces_skeleton(outline_pred, building_mask, epsilon_ratio)
    predictions.extend(face_predictions)

    return predictions
