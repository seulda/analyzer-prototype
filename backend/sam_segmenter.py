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
    SAM mask[2] (가장 큰 영역 = whole object)를 건물 전체 마스크로 사용.
    mask[2]가 너무 작으면 해당 bbox를 box prompt로 SAM 재호출.

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

    # mask[2] = 가장 큰 영역 (whole object 계층)
    building_mask = masks[2]
    confidence = float(scores[2])

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

def _find_optimal_k(pixels: np.ndarray, max_k: int = 4) -> int:
    """
    실루엣 점수를 사용하여 최적 K 선택.
    k=2,3,4 중 가장 높은 실루엣 점수를 가진 k 반환.
    모든 k에서 실루엣 점수가 낮으면(< 0.3) k=1 반환 (단일면).
    """
    if len(pixels) < 10:
        return 1

    best_k = 1
    best_score = 0.3  # 최소 임계값

    for k in range(2, max_k + 1):
        if len(pixels) < k:
            break

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1.0)
        _, labels, _ = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )

        labels_flat = labels.flatten()
        # 각 클러스터에 최소 포인트가 있어야 실루엣 점수 계산 가능
        unique_labels = np.unique(labels_flat)
        if len(unique_labels) < 2:
            continue

        # 실루엣 점수 계산 (샘플링으로 속도 향상)
        sample_size = min(5000, len(pixels))
        if sample_size < len(pixels):
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            score = silhouette_score(pixels[indices], labels_flat[indices])
        else:
            score = silhouette_score(pixels, labels_flat)

        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def _split_roof_faces(
    np_image: np.ndarray,
    building_mask: np.ndarray,
    min_face_ratio: float = 0.02,
) -> list[np.ndarray]:
    """
    건물 마스크 영역 내에서 밝기 기반 K-means로 경사면 분리.
    Label map 방식: 각 픽셀은 정확히 하나의 면에만 속함 (overlap 없음).

    Returns:
        face_masks: 각 경사면의 바이너리 마스크 리스트
    """
    h, w = np_image.shape[:2]

    # HSV 변환 → V채널 (밝기)
    hsv = cv2.cvtColor(np_image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]

    # 건물 마스크 영역의 밝기 값 추출
    mask_pixels_y, mask_pixels_x = np.where(building_mask)
    if len(mask_pixels_y) == 0:
        return []

    brightness_values = v_channel[mask_pixels_y, mask_pixels_x].astype(np.float32)
    pixels = brightness_values.reshape(-1, 1)

    building_area = building_mask.sum()

    # 최적 K 선택
    optimal_k = _find_optimal_k(pixels)

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


# ---------------------------------------------------------------------------
# 메인 세그멘테이션 함수
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
    경사 지붕의 경우 각 면을 별도 폴리곤으로 분리하여 반환합니다.

    Args:
        image_bytes: 위성 이미지 바이트
        click_x, click_y: 포인트 프롬프트 좌표 (None이면 이미지 중앙)
        epsilon_ratio: Douglas-Peucker 단순화 비율 (둘레 대비)
        snap_deg: 각도 스냅 단위 (도)
        min_area_ratio: 최소 윤곽 면적 비율 (이미지 대비)

    Returns:
        predictions:
            [0] = {"class": "building_outline", ...}  # 건물 전체 (줌 판정용)
            [1:] = {"class": "roof_face", ...}         # 경사면별 폴리곤
    """
    # 이미지 로드
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    np_image = np.array(pil_image)
    h, w = np_image.shape[:2]
    image_area = h * w

    # 클릭 좌표 기본값 = 이미지 중앙
    if click_x is None:
        click_x = w // 2
    if click_y is None:
        click_y = h // 2

    # Step 1: 건물 전체 마스크 추출
    building_mask, confidence = _get_building_mask(np_image, click_x, click_y)

    # 건물 전체 윤곽 폴리곤 생성 (줌 판정용)
    outline_points = _mask_to_polygon(
        building_mask,
        epsilon_ratio=epsilon_ratio,
        snap_deg=snap_deg,
        min_area=image_area * min_area_ratio,
    )

    if outline_points is None:
        return []

    building_pixel_area = int(building_mask.sum())

    predictions = [{
        "class": "building_outline",
        "confidence": round(confidence, 4),
        "points": outline_points,
        "pixel_area": building_pixel_area,
    }]

    # Step 2: 건물 내 경사면 분리
    face_masks = _split_roof_faces(np_image, building_mask)

    for face_mask in face_masks:
        face_points = _mask_to_polygon(
            face_mask,
            epsilon_ratio=epsilon_ratio * 0.5,  # 면은 약한 단순화 (경계 충실도 vs 직선 균형)
            snap_deg=30,  # 면은 약한 snap (15°→30°, 직각/평행만 스냅)
            min_area=0,  # _split_roof_faces()에서 이미 2% 필터링 완료
        )

        if face_points is None:
            continue

        face_pixel_area = int(face_mask.sum())
        face_confidence = confidence * min(1.0, face_pixel_area / max(building_pixel_area, 1))

        predictions.append({
            "class": "roof_face",
            "confidence": round(face_confidence, 4),
            "points": face_points,
            "pixel_area": face_pixel_area,
        })

    # 면 분리 결과가 없으면 건물 전체를 단일 roof_face로 추가
    if len(predictions) == 1:
        predictions.append({
            "class": "roof_face",
            "confidence": round(confidence, 4),
            "points": outline_points,
            "pixel_area": building_pixel_area,
        })

    return predictions
