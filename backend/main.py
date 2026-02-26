"""
Roof Analyzer Backend (FastAPI)
===============================
좌표 → 위성이미지 → MobileSAM 세그멘테이션 → GeoJSON 응답

흐름:
1. /api/outline - 건물 윤곽만 추출 (Step 1)
2. /api/analyze-faces - 면 분리 + 오검출 보정 (Step 2+3)
3. /api/analyze - 전체 한번에 (호환성 유지)
"""

from dotenv import load_dotenv
load_dotenv()

import os
import uuid

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from satellite import fetch_satellite_image, get_image_geo_bounds
from sam_segmenter import segment_building, segment_outline, segment_faces_skeleton
from geo_converter import predictions_to_geojson, pixel_to_latlng, latlng_to_pixel, CLASS_META, calculate_polygon_area_m2

app = FastAPI(title="Roof Analyzer API", version="0.3.0")

# ---------------------------------------------------------------------------
# 세션 캐시 (in-memory)
# ---------------------------------------------------------------------------
_session_cache: dict[str, dict] = {}  # session_id → cached data

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["*"],
    allow_headers=["*"],
)

IMAGE_SIZE = 640
# 이미지 가장자리 마진 (px) - 건물이 이 안쪽에 있어야 "잘리지 않음"으로 판단
EDGE_MARGIN = 30


class AnalyzeRequest(BaseModel):
    lat: float = Field(..., description="클릭한 건물의 위도", examples=[35.6812])
    lng: float = Field(..., description="클릭한 건물의 경도", examples=[139.7671])


class ObstacleInfo(BaseModel):
    id: int
    class_name: str
    confidence: float
    area_m2: float
    bbox_m: dict
    azimuth_deg: float | None = None


class AnalyzeResponse(BaseModel):
    lat: float
    lng: float
    zoom: int
    total_roof_area_m2: float
    total_obstacle_area_m2: float
    installable_area_m2: float
    obstacle_count: int
    obstacles: list[ObstacleInfo]
    geojson: dict
    satellite_image_url: str
    warning: str | None = None


# ---------------------------------------------------------------------------
# 후처리 보정 함수
# ---------------------------------------------------------------------------

def _centroid(points: list[dict]) -> tuple[float, float]:
    """폴리곤 점 리스트의 중심(centroid) 계산."""
    cx = sum(p["x"] for p in points) / len(points)
    cy = sum(p["y"] for p in points) / len(points)
    return cx, cy


def mark_distant_faces(
    predictions: list[dict],
    meters_per_pixel: float,
    max_distance_m: float = 50.0,
) -> list[dict]:
    """
    [보정] 오브젝트 간 거리가 허용 범위 초과인 면을 오검출로 표시합니다.

    같은 건물의 경사면들은 서로 인접해 있으므로, 어떤 면이 다른 모든 면과
    허용 거리 이상 떨어져 있으면 해당 건물의 지붕이 아닌 오검출로 판단합니다.
    허용 거리는 건물 크기에 비례하여 동적으로 계산합니다.

    Args:
        predictions: segment_building() 반환값 (building_outline + roof_faces)
        meters_per_pixel: 줌 레벨별 픽셀당 미터
        max_distance_m: 오브젝트 간 최소 허용 거리 (미터, 건물 크기에 따라 확대)

    Returns:
        오검출 면이 "misdetected"로 표시된 predictions 리스트
    """
    # roof_face만 추출
    faces = [(i, p) for i, p in enumerate(predictions) if p["class"] == "roof_face"]

    if len(faces) <= 1:
        return predictions

    # 건물 크기 기반 동적 임계값: 대형 건물은 centroid 간 거리가 클 수 있음
    outline = next((p for p in predictions if p["class"] == "building_outline"), None)
    if outline:
        building_area_m2 = outline.get("pixel_area", 0) * (meters_per_pixel ** 2)
        building_size_m = building_area_m2 ** 0.5
        effective_max_dist = max(max_distance_m, building_size_m * 0.7)
    else:
        effective_max_dist = max_distance_m

    # 각 face의 centroid 계산
    centroids = []
    for idx, pred in faces:
        centroids.append((idx, _centroid(pred["points"])))

    # 각 face에 대해 가장 가까운 다른 face까지의 거리 계산
    misdetected_indices = set()
    for i, (idx_i, (cx_i, cy_i)) in enumerate(centroids):
        min_dist_m = float("inf")
        for j, (idx_j, (cx_j, cy_j)) in enumerate(centroids):
            if i == j:
                continue
            dist_px = ((cx_i - cx_j) ** 2 + (cy_i - cy_j) ** 2) ** 0.5
            dist_m = dist_px * meters_per_pixel
            min_dist_m = min(min_dist_m, dist_m)

        if min_dist_m > effective_max_dist:
            misdetected_indices.add(idx_i)

    # 오검출 면을 misdetected로 변경
    removed_pixel_area = 0
    for idx in misdetected_indices:
        pred = predictions[idx]
        pred["class"] = "misdetected"
        removed_pixel_area += pred.get("pixel_area", 0)

    if misdetected_indices:
        print(f"[보정] 오검출 면 {len(misdetected_indices)}개 표시 (오브젝트 간 >{effective_max_dist:.0f}m)")
        # building_outline의 pixel_area에서 오검출 면적 차감 (면적 정합성 유지)
        for pred in predictions:
            if pred["class"] == "building_outline" and removed_pixel_area > 0:
                pred["pixel_area"] = max(0, pred["pixel_area"] - removed_pixel_area)
                break

    return predictions


def is_building_clipped(predictions: list[dict], image_size: int, margin: int) -> bool:
    """
    건물 폴리곤이 이미지 가장자리에 잘렸는지 확인.
    building_outline (첫 번째 예측)의 점으로만 판단.
    """
    if not predictions:
        return False

    # building_outline = predictions[0] 기준으로 판단
    outline = predictions[0]
    for pt in outline.get("points", []):
        x, y = pt["x"], pt["y"]
        if x <= margin or x >= image_size - margin:
            return True
        if y <= margin or y >= image_size - margin:
            return True
    return False


def find_optimal_zoom_and_outline(lat: float, lng: float):
    """
    줌 레벨 21부터 시작해서, 건물이 완전히 보이는 최대 줌을 자동 탐색.
    Step 1만 수행 (윤곽 추출).

    Returns:
        (outline_pred, building_mask, image_bytes, bounds, zoom, image_url)
    """
    center = IMAGE_SIZE // 2

    for zoom in range(21, 17, -1):
        print(f"[ZOOM] 줌 레벨 {zoom} 시도 중...")

        try:
            image_bytes, image_url = fetch_satellite_image(
                lat=lat, lng=lng, zoom=zoom, size=IMAGE_SIZE,
            )
        except Exception:
            print(f"[WARN] 줌 {zoom} 위성 이미지 수집 실패")
            continue

        bounds = get_image_geo_bounds(lat=lat, lng=lng, zoom=zoom, size=IMAGE_SIZE)

        outline_pred, building_mask = segment_outline(image_bytes, click_x=center, click_y=center)

        if outline_pred is None:
            print(f"[ZOOM] 줌 {zoom}: 건물 감지 없음 → 줌 아웃")
            continue

        # 건물이 이미지 가장자리에 잘렸는지 확인
        if is_building_clipped([outline_pred], IMAGE_SIZE, EDGE_MARGIN):
            print(f"[ZOOM] 줌 {zoom}: 건물이 잘림 → 줌 아웃")
            continue

        print(f"[ZOOM] 최적 줌 레벨: {zoom}")
        return outline_pred, building_mask, image_bytes, bounds, zoom, image_url

    # 모든 줌에서 실패 → 줌 19 기본값
    print("[ZOOM] 최적 줌 탐색 실패 → 줌 19 기본값 사용")
    image_bytes, image_url = fetch_satellite_image(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    bounds = get_image_geo_bounds(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    outline_pred, building_mask = segment_outline(image_bytes, click_x=center, click_y=center)
    return outline_pred, building_mask, image_bytes, bounds, 19, image_url


def find_optimal_zoom_and_analyze(lat: float, lng: float):
    """
    줌 레벨 21부터 시작해서, 건물이 완전히 보이는 최대 줌을 자동 탐색.
    SAM 포인트 프롬프트는 항상 이미지 중앙 (= 클릭 좌표) 사용.

    Returns:
        (predictions, bounds, zoom, image_url)
    """
    center = IMAGE_SIZE // 2

    for zoom in range(21, 17, -1):
        print(f"[ZOOM] 줌 레벨 {zoom} 시도 중...")

        try:
            image_bytes, image_url = fetch_satellite_image(
                lat=lat, lng=lng, zoom=zoom, size=IMAGE_SIZE,
            )
        except Exception:
            print(f"[WARN] 줌 {zoom} 위성 이미지 수집 실패")
            continue

        bounds = get_image_geo_bounds(lat=lat, lng=lng, zoom=zoom, size=IMAGE_SIZE)

        # SAM 포인트 프롬프트: 이미지 중앙 = 클릭한 건물 위치
        predictions = segment_building(image_bytes, click_x=center, click_y=center)

        if not predictions:
            print(f"[ZOOM] 줌 {zoom}: 건물 감지 없음 → 줌 아웃")
            continue

        # 건물이 이미지 가장자리에 잘렸는지 확인
        if is_building_clipped(predictions, IMAGE_SIZE, EDGE_MARGIN):
            print(f"[ZOOM] 줌 {zoom}: 건물이 잘림 → 줌 아웃")
            continue

        # 건물이 온전히 보이는 최대 줌 찾음!
        print(f"[ZOOM] 최적 줌 레벨: {zoom}")
        return predictions, bounds, zoom, image_url

    # 모든 줌에서 실패 → 줌 19 기본값 사용
    print("[ZOOM] 최적 줌 탐색 실패 → 줌 19 기본값 사용")
    image_bytes, image_url = fetch_satellite_image(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    bounds = get_image_geo_bounds(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    predictions = segment_building(image_bytes, click_x=center, click_y=center)
    return predictions, bounds, 19, image_url


# ---------------------------------------------------------------------------
# POST /api/outline — Step 1: 건물 윤곽만 추출
# ---------------------------------------------------------------------------

class OutlineRequest(BaseModel):
    lat: float = Field(..., description="클릭한 건물의 위도")
    lng: float = Field(..., description="클릭한 건물의 경도")


class OutlineResponse(BaseModel):
    session_id: str
    zoom: int
    satellite_image_url: str
    outline_geojson: dict


@app.post("/api/outline", response_model=OutlineResponse)
async def get_outline(req: OutlineRequest):
    """클릭한 건물 좌표 → 최적 줌 탐색 → 건물 윤곽만 반환"""
    try:
        result = find_optimal_zoom_and_outline(req.lat, req.lng)
        outline_pred, building_mask, image_bytes, bounds, zoom, image_url = result

        if outline_pred is None:
            raise HTTPException(status_code=404, detail="건물을 감지하지 못했습니다")

        # outline_pred의 points → GeoJSON 변환
        outline_coords = []
        for p in outline_pred["points"]:
            lat, lng = pixel_to_latlng(p["x"], p["y"], bounds, IMAGE_SIZE)
            outline_coords.append([lng, lat])
        if outline_coords and outline_coords[0] != outline_coords[-1]:
            outline_coords.append(outline_coords[0])

        outline_geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [outline_coords],
                },
                "properties": {
                    "type": "outline",
                    "color": "#2196F3",
                    "confidence": outline_pred["confidence"],
                },
            }],
        }

        # 세션 캐시 저장
        session_id = str(uuid.uuid4())
        _session_cache[session_id] = {
            "image_bytes": image_bytes,
            "bounds": bounds,
            "zoom": zoom,
            "image_url": image_url,
            "building_mask": building_mask,
            "outline_pred": outline_pred,
            "lat": req.lat,
            "lng": req.lng,
        }

        return OutlineResponse(
            session_id=session_id,
            zoom=zoom,
            satellite_image_url=image_url,
            outline_geojson=outline_geojson,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"외부 API 연결 실패: {e}")


# ---------------------------------------------------------------------------
# POST /api/analyze-faces — Step 2+3: 면 분리 + 오검출 보정
# ---------------------------------------------------------------------------

class AnalyzeFacesRequest(BaseModel):
    session_id: str = Field(..., description="세션 ID")
    modified_points: list[list[float]] | None = Field(
        None, description="수정된 윤곽 좌표 [[lng, lat], ...]",
    )


@app.post("/api/analyze-faces", response_model=AnalyzeResponse)
async def analyze_faces(req: AnalyzeFacesRequest):
    """세션 캐시 → 면 분리 + 오검출 보정 → 전체 분석 결과 반환"""
    session = _session_cache.get(req.session_id, None)
    if session is None:
        raise HTTPException(status_code=404, detail="세션이 만료되었습니다")

    try:
        image_bytes = session["image_bytes"]
        bounds = session["bounds"]
        zoom = session["zoom"]
        image_url = session["image_url"]
        building_mask = session["building_mask"]
        outline_pred = session["outline_pred"]

        # 수정된 좌표가 있으면 새 마스크 생성
        if req.modified_points:
            h, w = building_mask.shape[:2]
            pixel_points = []
            for lng_val, lat_val in req.modified_points:
                px, py = latlng_to_pixel(lat_val, lng_val, bounds, IMAGE_SIZE)
                pixel_points.append([px, py])
            pts = np.array(pixel_points, dtype=np.int32)
            building_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(building_mask, [pts], 1)
            building_mask = building_mask.astype(bool)
            # outline_pred도 업데이트
            outline_pred["points"] = [{"x": p[0], "y": p[1]} for p in pixel_points]
            outline_pred["pixel_area"] = int(building_mask.sum())

        # Step 2: Skeleton 기반 면 분리
        face_predictions = segment_faces_skeleton(outline_pred, building_mask)

        # predictions 조합
        predictions = [outline_pred] + face_predictions

        # 후처리 보정: 오검출 면 표시
        predictions = mark_distant_faces(
            predictions,
            meters_per_pixel=bounds["meters_per_pixel"],
        )

        # 이하 기존 /api/analyze와 동일한 응답 구성
        building_outline = predictions[0]
        face_preds = [p for p in predictions if p["class"] == "roof_face"]
        misdetected = [p for p in predictions if p["class"] == "misdetected"]

        geojson_predictions = (
            ([building_outline] if building_outline else [])
            + face_preds
            + misdetected
        )

        geojson, stats = predictions_to_geojson(
            predictions=geojson_predictions,
            bounds=bounds,
            image_size=IMAGE_SIZE,
        )

        mpp = bounds["meters_per_pixel"]

        building_pixel_area = building_outline.get("pixel_area", 0) if building_outline else 0
        roof_area_m2 = building_pixel_area * (mpp ** 2) if building_pixel_area > 0 else 0.0

        if roof_area_m2 <= 0:
            roof_area_m2 = stats["image_area_m2"]

        warning = None
        face_pixel_sum = sum(p.get("pixel_area", 0) for p in face_preds)
        face_area_sum = face_pixel_sum * (mpp ** 2)
        if building_pixel_area > 0:
            diff_ratio = abs(face_pixel_sum - building_pixel_area) / building_pixel_area
            if diff_ratio > 0.10:
                warning = f"면적 오차범위 초과: 건물 전체 {roof_area_m2:.1f}m² vs 면 합산 {face_area_sum:.1f}m² (오차 {diff_ratio:.1%}, 허용 ±10%)"

        if misdetected:
            mis_msg = f"오검출 면 {len(misdetected)}개 감지 (노란색 표시)"
            warning = f"{warning} | {mis_msg}" if warning else mis_msg

        obstacles = []
        for i, pred in enumerate(geojson_predictions):
            obs = ObstacleInfo(
                id=i + 1,
                class_name=pred["class"],
                confidence=round(pred["confidence"], 3),
                area_m2=round(stats["obstacle_areas_m2"][i], 2),
                bbox_m=stats["obstacle_bboxes_m"][i],
                azimuth_deg=round(pred["azimuth_deg"], 1) if "azimuth_deg" in pred else None,
            )
            obstacles.append(obs)

        # lat/lng는 outline의 중심으로 계산
        center_px = sum(p["x"] for p in outline_pred["points"]) / len(outline_pred["points"])
        center_py = sum(p["y"] for p in outline_pred["points"]) / len(outline_pred["points"])
        center_lat, center_lng = pixel_to_latlng(center_px, center_py, bounds, IMAGE_SIZE)

        return AnalyzeResponse(
            lat=center_lat,
            lng=center_lng,
            zoom=zoom,
            total_roof_area_m2=round(roof_area_m2, 1),
            total_obstacle_area_m2=round(stats["total_obstacle_area_m2"], 2),
            installable_area_m2=round(
                roof_area_m2 - stats["total_obstacle_area_m2"], 2,
            ),
            obstacle_count=len(obstacles),
            obstacles=obstacles,
            geojson=geojson,
            satellite_image_url=image_url,
            warning=warning,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"외부 API 연결 실패: {e}")


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_roof(req: AnalyzeRequest):
    """클릭한 건물 좌표 → 최적 줌 탐색 → SAM 세그멘테이션 → 분석"""
    try:
        predictions, bounds, zoom, image_url = find_optimal_zoom_and_analyze(
            req.lat, req.lng,
        )

        # 후처리 보정: 오브젝트 간 50m 초과 면을 오검출(misdetected)로 표시
        predictions = mark_distant_faces(
            predictions,
            meters_per_pixel=bounds["meters_per_pixel"],
        )

        # building_outline / roof_face / misdetected 분리
        building_outline = predictions[0] if predictions else None
        face_predictions = [p for p in predictions if p["class"] == "roof_face"]
        misdetected = [p for p in predictions if p["class"] == "misdetected"]

        # GeoJSON에 building_outline + roof_face + misdetected 모두 포함
        geojson_predictions = (
            ([building_outline] if building_outline else [])
            + face_predictions
            + misdetected
        )

        # GeoJSON 변환 (roof_face + misdetected)
        geojson, stats = predictions_to_geojson(
            predictions=geojson_predictions,
            bounds=bounds,
            image_size=IMAGE_SIZE,
        )

        mpp = bounds["meters_per_pixel"]

        # 면적 계산: 픽셀 카운트 기반 (label map이 overlap 없이 보장)
        # building_outline pixel_area는 보정 후 오검출 면적이 차감된 값
        building_pixel_area = building_outline.get("pixel_area", 0) if building_outline else 0
        roof_area_m2 = building_pixel_area * (mpp ** 2) if building_pixel_area > 0 else 0.0

        # 폴리곤 없으면 이미지 면적 폴백
        if roof_area_m2 <= 0:
            roof_area_m2 = stats["image_area_m2"]

        # 검증: roof_face 픽셀 합산 vs building_outline 픽셀 ±10% 범위 확인
        warning = None
        face_pixel_sum = sum(p.get("pixel_area", 0) for p in face_predictions)
        face_area_sum = face_pixel_sum * (mpp ** 2)
        if building_pixel_area > 0:
            diff_ratio = abs(face_pixel_sum - building_pixel_area) / building_pixel_area
            print(f"[AREA] 건물 전체: {roof_area_m2:.1f}m² ({building_pixel_area}px) | 면 합산: {face_area_sum:.1f}m² ({face_pixel_sum}px) | 오차: {diff_ratio:.1%}")
            if diff_ratio > 0.10:
                warning = f"면적 오차범위 초과: 건물 전체 {roof_area_m2:.1f}m² vs 면 합산 {face_area_sum:.1f}m² (오차 {diff_ratio:.1%}, 허용 ±10%)"

        if misdetected:
            mis_msg = f"오검출 면 {len(misdetected)}개 감지 (노란색 표시)"
            warning = f"{warning} | {mis_msg}" if warning else mis_msg

        # 응답 구성 — 지붕면 + 오검출 + 장애물 모두 포함
        obstacles = []
        for i, pred in enumerate(geojson_predictions):
            obs = ObstacleInfo(
                id=i + 1,
                class_name=pred["class"],
                confidence=round(pred["confidence"], 3),
                area_m2=round(stats["obstacle_areas_m2"][i], 2),
                bbox_m=stats["obstacle_bboxes_m"][i],
                azimuth_deg=round(pred["azimuth_deg"], 1) if "azimuth_deg" in pred else None,
            )
            obstacles.append(obs)

        return AnalyzeResponse(
            lat=req.lat,
            lng=req.lng,
            zoom=zoom,
            total_roof_area_m2=round(roof_area_m2, 1),
            total_obstacle_area_m2=round(stats["total_obstacle_area_m2"], 2),
            installable_area_m2=round(
                roof_area_m2 - stats["total_obstacle_area_m2"], 2,
            ),
            obstacle_count=len(obstacles),
            obstacles=obstacles,
            geojson=geojson,
            satellite_image_url=image_url,
            warning=warning,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"외부 API 연결 실패: {e}")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
