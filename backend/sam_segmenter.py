"""
SAM + OpenCV 지붕 세그멘테이션 (경사 지붕 다면 검출)
=====================================================
MobileSAM 포인트 프롬프트 → 건물 전체 마스크 → Edge 기반 용마루(棟) 검출
→ Canny + HoughLinesP → 직선 병합 → 스코어링 → Shapely split → 경사면 분리

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
from shapely.ops import split as shapely_split

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
# Step 2: 건물 내 경사면 분리 (Edge 기반 용마루 검출)
# ---------------------------------------------------------------------------

def _preprocess_roof_region(
    np_image: np.ndarray,
    building_mask: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], float] | None:
    """
    A. 전처리: crop → grayscale → blur → Canny → 마스킹.

    Returns:
        (edges, crop_bbox, diag) 또는 None (마스크 너무 작음)
        crop_bbox = (y1, y2, x1, x2) — 원본 좌표계
    """
    ys, xs = np.where(building_mask)
    if len(ys) == 0:
        return None

    y1, y2 = int(ys.min()), int(ys.max())
    x1, x2 = int(xs.min()), int(xs.max())
    bh, bw = y2 - y1, x2 - x1

    # 건물 bbox가 100px 미만이면 단일면 폴백
    if bh < 100 or bw < 100:
        print(f"[RIDGE] bbox too small ({bw}x{bh}), fallback to single face")
        return None

    # 패딩 (10px or 10% 중 큰 것)
    pad = max(10, int(max(bh, bw) * 0.1))
    h_img, w_img = np_image.shape[:2]
    y1p = max(0, y1 - pad)
    y2p = min(h_img, y2 + pad)
    x1p = max(0, x1 - pad)
    x2p = min(w_img, x2 + pad)

    crop = np_image[y1p:y2p, x1p:x2p]
    crop_mask = building_mask[y1p:y2p, x1p:x2p].astype(bool)

    diag = math.hypot(y2p - y1p, x2p - x1p)

    print(f"[RIDGE] crop size={crop.shape}, mask_pixels={crop_mask.sum()}, diag={diag:.0f}")

    # Grayscale → Gaussian blur (완화된 sigma)
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    sigma = max(0.8, diag / 150.0)
    ksize = int(sigma * 6) | 1  # 홀수 보장
    ksize = max(3, ksize)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)

    # Adaptive Canny — 건물 내부 픽셀 기준 median
    median_val = float(np.median(blurred[crop_mask]))
    low = max(10, int(0.4 * median_val))
    high = max(low + 10, int(0.9 * median_val))
    edges = cv2.Canny(blurred, low, high)

    print(f"[RIDGE] blur sigma={sigma:.2f} ksize={ksize}, median={median_val:.0f}, "
          f"canny low={low} high={high}")

    edges_before_mask = int(edges.sum() // 255)

    # 건물 마스크를 erosion → 윤곽 경계 edge 제거 (내부 용마루만 남김)
    erode_px = max(3, int(diag * 0.02))
    erode_kernel = np.ones((erode_px, erode_px), np.uint8)
    inner_mask = cv2.erode(crop_mask.astype(np.uint8), erode_kernel).astype(bool)
    edges[~inner_mask] = 0
    print(f"[RIDGE] erode {erode_px}px: mask {crop_mask.sum()} → inner {inner_mask.sum()}")

    edges_after_mask = int(edges.sum() // 255)
    print(f"[RIDGE] edges before mask={edges_before_mask}, after mask={edges_after_mask}")

    # 디버그: Canny edge 오버레이 (마스킹 후에도 저장)
    try:
        dbg = crop.copy()
        dbg[edges > 0] = (0, 255, 0)
        # 건물 마스크 윤곽도 표시
        mask_u8 = crop_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(dbg, contours, -1, (255, 255, 0), 1)
        Image.fromarray(dbg).save("/tmp/debug_ridge_edges.png")
        print("[DEBUG] saved /tmp/debug_ridge_edges.png")
    except Exception as e:
        print(f"[DEBUG] edge save failed: {e}")

    if edges_after_mask == 0:
        print("[RIDGE] no Canny edges inside building mask, fallback to single face")
        return None

    print(f"[RIDGE] preprocess: bbox=({x1p},{y1p})-({x2p},{y2p}), diag={diag:.0f}, "
          f"edge_pixels={edges_after_mask}")

    # 디버그: Canny edge 오버레이
    try:
        dbg = crop.copy()
        dbg[edges > 0] = (0, 255, 0)
        Image.fromarray(dbg).save("/tmp/debug_ridge_edges.png")
        print("[DEBUG] saved /tmp/debug_ridge_edges.png")
    except Exception as e:
        print(f"[DEBUG] edge save failed: {e}")

    return edges, (y1p, y2p, x1p, x2p), diag


def _detect_candidate_lines(
    edges: np.ndarray,
    diag: float,
) -> list[tuple[int, int, int, int]]:
    """
    B. HoughLinesP로 직선 검출.
    50개 초과 시 threshold 20%씩 증가 (최대 3회).

    Returns:
        list of (x1, y1, x2, y2) in crop 좌표
    """
    threshold = max(10, int(diag * 0.10))
    min_line_len = max(10, int(diag * 0.12))
    max_line_gap = max(5, int(diag * 0.10))

    lines = None
    for attempt in range(4):
        raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=threshold,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap,
        )
        if raw is None:
            break
        lines = [tuple(l[0]) for l in raw]
        if len(lines) <= 50:
            break
        # threshold 20% 증가
        threshold = int(threshold * 1.2)
        print(f"[RIDGE] too many lines ({len(lines)}), raising threshold to {threshold}")

    if lines is None:
        print("[RIDGE] HoughLinesP returned no lines")
        return []

    print(f"[RIDGE] detected {len(lines)} candidate lines (threshold={threshold})")
    return lines


def _merge_parallel_lines(
    lines: list[tuple[int, int, int, int]],
    diag: float,
    angle_tol: float = 15.0,
    dist_ratio: float = 0.08,
) -> list[dict]:
    """
    C. 유사 직선 병합.

    Returns:
        list of {"angle": float, "endpoints": (x1,y1,x2,y2),
                 "length": float, "members": int}
    """
    if not lines:
        return []

    # 각 직선의 각도, 중점, 길이 계산
    line_info = []
    for x1, y1, x2, y2 in lines:
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180  # [0, 180)
        length = math.hypot(x2 - x1, y2 - y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        line_info.append({
            "endpoints": (x1, y1, x2, y2),
            "angle": angle,
            "length": length,
            "cx": cx, "cy": cy,
        })

    # 각도 기준 그룹화
    line_info.sort(key=lambda li: li["angle"])
    groups = []
    used = [False] * len(line_info)

    for i, li in enumerate(line_info):
        if used[i]:
            continue
        group = [i]
        used[i] = True
        for j in range(i + 1, len(line_info)):
            if used[j]:
                continue
            # 각도 차이 (wrap-around 고려)
            da = abs(li["angle"] - line_info[j]["angle"])
            da = min(da, 180 - da)
            if da < angle_tol:
                group.append(j)
                used[j] = True
        groups.append(group)

    # 그룹 내 수직 거리 기준 서브그룹화 → 병합
    dist_threshold = diag * dist_ratio
    merged = []

    for group in groups:
        if len(group) == 1:
            idx = group[0]
            li = line_info[idx]
            merged.append({
                "angle": li["angle"],
                "endpoints": li["endpoints"],
                "length": li["length"],
                "members": 1,
            })
            continue

        # 그룹의 대표 각도 (길이가중평균)
        total_len = sum(line_info[i]["length"] for i in group)
        avg_angle = sum(line_info[i]["angle"] * line_info[i]["length"] for i in group) / total_len
        ref_rad = math.radians(avg_angle)

        # 수직 거리 기준 서브그룹화
        # 수직방향 벡터: (-sin(angle), cos(angle))
        perp_nx = -math.sin(ref_rad)
        perp_ny = math.cos(ref_rad)

        # 각 직선의 수직 투영 거리
        proj_dists = []
        for i in group:
            li = line_info[i]
            d = li["cx"] * perp_nx + li["cy"] * perp_ny
            proj_dists.append((d, i))
        proj_dists.sort(key=lambda x: x[0])

        # 거리 기준 서브그룹
        subgroups = [[proj_dists[0]]]
        for k in range(1, len(proj_dists)):
            if abs(proj_dists[k][0] - proj_dists[k - 1][0]) < dist_threshold:
                subgroups[-1].append(proj_dists[k])
            else:
                subgroups.append([proj_dists[k]])

        # 각 서브그룹 → 대표 직선 (길이가중평균 + 투영 범위)
        dir_nx = math.cos(ref_rad)
        dir_ny = math.sin(ref_rad)

        for sg in subgroups:
            indices = [s[1] for s in sg]
            sg_total_len = sum(line_info[i]["length"] for i in indices)
            sg_angle = sum(line_info[i]["angle"] * line_info[i]["length"] for i in indices) / sg_total_len

            # 투영 범위로 대표 직선 endpoints 계산
            all_pts = []
            for i in indices:
                x1, y1, x2, y2 = line_info[i]["endpoints"]
                all_pts.extend([(x1, y1), (x2, y2)])

            # 직선 방향 투영
            projs = [px * dir_nx + py * dir_ny for px, py in all_pts]
            min_proj = min(projs)
            max_proj = max(projs)

            # 수직 방향 평균 (가중)
            avg_perp = sum(
                (line_info[i]["cx"] * perp_nx + line_info[i]["cy"] * perp_ny) * line_info[i]["length"]
                for i in indices
            ) / sg_total_len

            # 대표 직선 endpoints
            rx1 = min_proj * dir_nx + avg_perp * perp_nx
            ry1 = min_proj * dir_ny + avg_perp * perp_ny
            rx2 = max_proj * dir_nx + avg_perp * perp_nx
            ry2 = max_proj * dir_ny + avg_perp * perp_ny

            merged.append({
                "angle": sg_angle,
                "endpoints": (int(round(rx1)), int(round(ry1)), int(round(rx2)), int(round(ry2))),
                "length": math.hypot(rx2 - rx1, ry2 - ry1),
                "members": len(indices),
            })

    print(f"[RIDGE] merged {len(lines)} lines → {len(merged)} representative lines")
    return merged


def _select_ridge_lines(
    merged_lines: list[dict],
    edges: np.ndarray,
    building_mask_crop: np.ndarray,
    diag: float,
    min_face_ratio: float,
) -> list[dict]:
    """
    D. 용마루 선별 — 스코어링.

    스코어 = 0.30×길이 + 0.25×엣지강도 + 0.30×분할균형 + 0.15×멤버수보너스

    Returns:
        선별된 ridge lines (최대 3개)
    """
    if not merged_lines:
        return []

    h, w = edges.shape[:2]
    mask_area = building_mask_crop.astype(bool).sum()

    scored = []
    for ml in merged_lines:
        x1, y1, x2, y2 = ml["endpoints"]
        line_len = ml["length"]

        # 길이 스코어 (0-1), 최소 0.15 (완화)
        s_length = min(1.0, line_len / diag)
        if s_length < 0.10:
            print(f"[RIDGE]   skip ({x1},{y1})-({x2},{y2}): length={s_length:.2f} < 0.10")
            continue

        # 엣지강도: 직선 위 Canny edge 비율, 최소 0.10 (완화)
        line_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
        line_pixels = line_mask > 0
        total_on_line = line_pixels.sum()
        if total_on_line == 0:
            continue
        edge_on_line = (edges[line_pixels] > 0).sum()
        s_edge = edge_on_line / total_on_line
        if s_edge < 0.05:
            print(f"[RIDGE]   skip ({x1},{y1})-({x2},{y2}): edge={s_edge:.2f} < 0.05")
            continue

        # 분할균형: 직선으로 나뉜 건물 마스크 양쪽 비율
        # 간단한 half-plane 분류
        angle_rad = math.radians(ml["angle"])
        nx = -math.sin(angle_rad)  # 법선벡터
        ny = math.cos(angle_rad)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2

        mask_ys, mask_xs = np.where(building_mask_crop)
        if len(mask_ys) == 0:
            continue
        dots = (mask_xs - mx) * nx + (mask_ys - my) * ny
        side_a = (dots > 0).sum()
        side_b = (dots <= 0).sum()
        total_sides = side_a + side_b
        if total_sides == 0:
            continue
        s_balance = min(side_a, side_b) / total_sides * 2
        if s_balance < 0.10:
            print(f"[RIDGE]   skip ({x1},{y1})-({x2},{y2}): balance={s_balance:.2f} < 0.10")
            continue

        # 멤버수 보너스
        s_members = min(1.0, ml["members"] / 3.0)

        score = 0.30 * s_length + 0.25 * s_edge + 0.30 * s_balance + 0.15 * s_members

        print(f"[RIDGE] line ({x1},{y1})-({x2},{y2}): len={s_length:.2f} edge={s_edge:.2f} "
              f"bal={s_balance:.2f} mem={s_members:.2f} → score={score:.3f}")

        scored.append({**ml, "score": score,
                       "s_length": s_length, "s_edge": s_edge,
                       "s_balance": s_balance})

    # 스코어 < 0.20 → 전부 버림
    scored = [s for s in scored if s["score"] >= 0.20]
    if not scored:
        print("[RIDGE] no lines passed score threshold (0.20)")
        return []

    # 스코어 내림차순 정렬
    scored.sort(key=lambda s: s["score"], reverse=True)

    # 최대 5개 선택, 유사 직선 중복 제거
    selected = []
    for candidate in scored:
        if len(selected) >= 5:
            break
        # 이미 선택된 직선과 각도 25° 이내 + 수직거리 diag*0.15 이내면 중복
        is_dup = False
        for sel in selected:
            da = abs(candidate["angle"] - sel["angle"])
            da = min(da, 180 - da)
            if da < 25:
                # 수직 거리 체크
                ref_rad = math.radians(sel["angle"])
                perp_nx = -math.sin(ref_rad)
                perp_ny = math.cos(ref_rad)
                cx1 = (candidate["endpoints"][0] + candidate["endpoints"][2]) / 2
                cy1 = (candidate["endpoints"][1] + candidate["endpoints"][3]) / 2
                cx2 = (sel["endpoints"][0] + sel["endpoints"][2]) / 2
                cy2 = (sel["endpoints"][1] + sel["endpoints"][3]) / 2
                vdist = abs((cx1 - cx2) * perp_nx + (cy1 - cy2) * perp_ny)
                if vdist < diag * 0.15:
                    is_dup = True
                    print(f"[RIDGE]   dedup: angle_diff={da:.1f}° vdist={vdist:.0f} < {diag*0.15:.0f}")
                    break
        if not is_dup:
            selected.append(candidate)

    print(f"[RIDGE] selected {len(selected)} ridge lines from {len(scored)} candidates")
    return selected


def _split_mask_by_lines(
    building_mask: np.ndarray,
    ridge_lines: list[dict],
    crop_bbox: tuple[int, int, int, int],
    diag: float,
    min_face_ratio: float,
) -> list[np.ndarray]:
    """
    E. 마스크 분할 — Shapely split + 래스터화.

    Args:
        building_mask: 원본 좌표계 bool mask
        ridge_lines: crop 좌표계 직선
        crop_bbox: (y1, y2, x1, x2) — crop→원본 변환용
        diag: 대각선 길이
        min_face_ratio: 최소 면적 비율

    Returns:
        list of bool face masks (원본 좌표계)
    """
    h, w = building_mask.shape
    y1p, y2p, x1p, x2p = crop_bbox
    building_area = building_mask.astype(bool).sum()

    # building_mask → Shapely Polygon (가장 큰 외곽 윤곽)
    mask_uint8 = building_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [building_mask]

    largest_contour = max(contours, key=cv2.contourArea)
    if len(largest_contour) < 3:
        return [building_mask]

    coords = [(int(pt[0][0]), int(pt[0][1])) for pt in largest_contour]
    if len(coords) < 3:
        return [building_mask]

    building_poly = ShapelyPolygon(coords)
    if not building_poly.is_valid:
        building_poly = building_poly.buffer(0)
    if building_poly.is_empty:
        return [building_mask]

    # 각 용마루 → 원본 좌표로 변환 → 양방향 diag*3 연장 → split
    current_pieces = [building_poly]
    extend_len = diag * 3

    for rl in ridge_lines:
        cx1, cy1, cx2, cy2 = rl["endpoints"]
        # crop 좌표 → 원본 좌표
        ox1, oy1 = cx1 + x1p, cy1 + y1p
        ox2, oy2 = cx2 + x1p, cy2 + y1p

        # 양방향 연장
        dx = ox2 - ox1
        dy = oy2 - oy1
        length = math.hypot(dx, dy)
        if length < 1e-6:
            continue
        ux, uy = dx / length, dy / length
        ext_x1 = ox1 - ux * extend_len
        ext_y1 = oy1 - uy * extend_len
        ext_x2 = ox2 + ux * extend_len
        ext_y2 = oy2 + uy * extend_len

        split_line = LineString([(ext_x1, ext_y1), (ext_x2, ext_y2)])

        new_pieces = []
        for piece in current_pieces:
            try:
                result = shapely_split(piece, split_line)
                for geom in result.geoms:
                    if geom.geom_type == "Polygon" and geom.area > 0:
                        new_pieces.append(geom)
                    elif geom.geom_type == "MultiPolygon":
                        for g in geom.geoms:
                            if g.area > 0:
                                new_pieces.append(g)
            except Exception:
                new_pieces.append(piece)

        if new_pieces:
            current_pieces = new_pieces

    # --- face_id_map 방식: 모든 건물 픽셀을 정확히 1개 면에 할당 ---
    min_area = building_area * min_face_ratio
    building_bool = building_mask.astype(bool)

    # 1) 각 Shapely polygon → face_id_map에 래스터화 (우선순위: 큰 면 먼저)
    current_pieces.sort(key=lambda p: p.area, reverse=True)
    face_id_map = np.zeros((h, w), dtype=np.int32)
    piece_list = []  # (face_id, polygon) 유효 면만
    next_id = 1

    for poly in current_pieces:
        if poly.is_empty or poly.area < 1:
            continue
        exterior_coords = np.array(poly.exterior.coords, dtype=np.int32)
        tmp = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(tmp, [exterior_coords], 255)
        # 건물 마스크 내 + 아직 미할당인 픽셀만
        assignable = (tmp > 0) & building_bool & (face_id_map == 0)
        if assignable.sum() > 0:
            face_id_map[assignable] = next_id
            piece_list.append(next_id)
            next_id += 1

    # 2) 미할당 건물 픽셀 → 가장 가까운 면에 할당 (distance transform)
    unassigned = building_bool & (face_id_map == 0)
    if np.any(unassigned) and piece_list:
        distances = []
        for fid in piece_list:
            face_bin = (face_id_map == fid).astype(np.uint8)
            dist = cv2.distanceTransform(1 - face_bin, cv2.DIST_L2, 5)
            distances.append(dist)
        dist_stack = np.stack(distances, axis=0)
        nearest = np.argmin(dist_stack, axis=0)  # index into piece_list
        for i, fid in enumerate(piece_list):
            assign_here = unassigned & (nearest == i)
            face_id_map[assign_here] = fid

    # 3) 소면적 면 → 이웃 면에 병합
    # a) min_face_ratio 미달 면 무조건 병합
    # b) 그 후에도 면 수가 직선수+2를 초과하면, 가장 작은 면부터 병합
    alive = set(piece_list)
    kernel = np.ones((5, 5), np.uint8)

    # a) 절대 소면적 병합
    changed = True
    while changed:
        changed = False
        for fid in sorted(alive, key=lambda f: (face_id_map == f).sum()):
            fmask = face_id_map == fid
            if fmask.sum() >= min_area:
                continue
            dilated = cv2.dilate(fmask.astype(np.uint8), kernel)
            neighbors = set(int(x) for x in np.unique(face_id_map[dilated > 0])) - {0, fid}
            neighbors = neighbors & alive
            if neighbors:
                merge_to = max(neighbors, key=lambda n: (face_id_map == n).sum())
                print(f"[RIDGE]   merge tiny face {fid} ({fmask.sum()}px) → {merge_to}")
                face_id_map[fmask] = merge_to
                alive.discard(fid)
                changed = True

    # 4) face_id_map → 개별 마스크 리스트
    face_masks = []
    for fid in sorted(alive):
        fm = face_id_map == fid
        if fm.sum() >= min_area:
            face_masks.append(fm)

    if not face_masks:
        return [building_mask]

    print(f"[RIDGE] split into {len(face_masks)} face masks")
    return face_masks


def _split_roof_faces(
    np_image: np.ndarray,
    building_mask: np.ndarray,
    min_face_ratio: float = 0.02,
) -> list[np.ndarray]:
    """
    건물 마스크 영역 내에서 Edge 기반 용마루(棟) 검출로 경사면 분리.
    Canny + HoughLinesP → 직선 병합 → 스코어링 선별 → Shapely split.

    Returns:
        face_masks: 각 경사면의 바이너리 마스크 리스트
    """
    h, w = np_image.shape[:2]

    if building_mask.sum() == 0:
        return []

    # A. 전처리
    preprocess = _preprocess_roof_region(np_image, building_mask)
    if preprocess is None:
        return [building_mask]

    edges, crop_bbox, diag = preprocess
    y1p, y2p, x1p, x2p = crop_bbox
    crop_mask = building_mask[y1p:y2p, x1p:x2p]

    # B. 직선 검출
    candidate_lines = _detect_candidate_lines(edges, diag)
    if not candidate_lines:
        return [building_mask]

    # C. 유사 직선 병합
    merged_lines = _merge_parallel_lines(candidate_lines, diag)
    if not merged_lines:
        return [building_mask]

    # 디버그: 후보 직선(blue) + 병합 직선(yellow)
    try:
        crop_img = np_image[y1p:y2p, x1p:x2p].copy()
        for x1, y1, x2, y2 in candidate_lines:
            cv2.line(crop_img, (x1, y1), (x2, y2), (100, 100, 255), 1)
        for ml in merged_lines:
            ex1, ey1, ex2, ey2 = ml["endpoints"]
            cv2.line(crop_img, (ex1, ey1), (ex2, ey2), (0, 255, 255), 2)
        Image.fromarray(crop_img).save("/tmp/debug_ridge_candidates.png")
        print("[DEBUG] saved /tmp/debug_ridge_candidates.png")
    except Exception as e:
        print(f"[DEBUG] candidates save failed: {e}")

    # D. 용마루 선별
    selected = _select_ridge_lines(merged_lines, edges, crop_mask, diag, min_face_ratio)
    if not selected:
        return [building_mask]

    # E. 마스크 분할 — 과분할 시 스코어 낮은 직선 제거 후 재시도
    face_masks = _split_mask_by_lines(building_mask, selected, crop_bbox, diag, min_face_ratio)

    # 과분할 보정은 하지 않음 — 교차 직선은 직선수보다 많은 면을 만드는 것이 정상
    # 소면적 면은 _split_mask_by_lines 내부에서 이웃에 병합됨

    # 디버그: 선별 용마루(red) + face 영역(color overlay)
    try:
        dbg = np_image.copy()
        face_colors = [
            (255, 80, 80), (80, 255, 80), (80, 80, 255),
            (255, 255, 80), (255, 80, 255), (80, 255, 255),
        ]
        overlay = np.zeros_like(dbg)
        for i, fm in enumerate(face_masks):
            c = face_colors[i % len(face_colors)]
            overlay[fm] = c
        mask_bool = building_mask.astype(bool)
        dbg[mask_bool] = (dbg[mask_bool] * 0.4 + overlay[mask_bool] * 0.6).astype(np.uint8)
        # 선별 용마루 (원본 좌표)
        for rl in selected:
            cx1, cy1, cx2, cy2 = rl["endpoints"]
            ox1, oy1 = cx1 + x1p, cy1 + y1p
            ox2, oy2 = cx2 + x1p, cy2 + y1p
            cv2.line(dbg, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
        Image.fromarray(dbg).save("/tmp/debug_ridge_selected.png")
        print("[DEBUG] saved /tmp/debug_ridge_selected.png")
    except Exception as e:
        print(f"[DEBUG] selected save failed: {e}")

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
    Step 2: building_mask로 Edge 기반 용마루 검출 면 분리. face predictions 반환.

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

    # face_id_map 기반 마스크는 겹침/갭 없음 → 단순 폴리곤 변환만
    face_entries = []  # (shapely_poly, pixel_area)
    for face_mask in face_masks:
        # snap 없이 윤곽 추출 (face_id_map에서 이미 깔끔한 영역)
        face_points = _mask_to_polygon(
            face_mask,
            epsilon_ratio=epsilon_ratio,
            snap_deg=0,  # snap 없음 — 원래 마스크 형태 보존
            min_area=0,
        )
        if face_points is None:
            continue

        face_poly = ShapelyPolygon([(p["x"], p["y"]) for p in face_points])
        if not face_poly.is_valid:
            face_poly = face_poly.buffer(0)

        # 건물 윤곽으로 클리핑
        if clip_poly is not None:
            face_poly = face_poly.intersection(clip_poly)

        face_poly = _largest_polygon(face_poly)
        if face_poly is None:
            continue

        face_entries.append((face_poly, int(face_mask.sum())))

    # Shapely → points dict 변환
    face_predictions = []
    for poly, px_area in face_entries:
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
