"""
Roof Analyzer Backend (FastAPI)
===============================
좌표 → 위성이미지 → Roboflow 추론 → GeoJSON 응답

흐름:
1. /api/analyze - 클릭한 건물 좌표로 최적 줌 자동 탐색 후 분석
"""

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from satellite import fetch_satellite_image, get_image_geo_bounds
from roboflow_client import run_inference
from geo_converter import predictions_to_geojson, CLASS_META

app = FastAPI(title="Roof Analyzer API", version="0.1.0")

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


def is_building_clipped(predictions: list[dict], image_size: int, margin: int) -> bool:
    """
    건물 폴리곤이 이미지 가장자리에 잘렸는지 확인.
    폴리곤의 점이 마진 안쪽 가장자리에 닿아있으면 잘린 것으로 판단.
    """
    for pred in predictions:
        for pt in pred.get("points", []):
            x, y = pt["x"], pt["y"]
            if x <= margin or x >= image_size - margin:
                return True
            if y <= margin or y >= image_size - margin:
                return True
    return False


def find_optimal_zoom_and_analyze(lat: float, lng: float):
    """
    줌 레벨 21부터 시작해서, 건물이 완전히 보이는 최대 줌을 자동 탐색.

    Returns:
        (predictions, bounds, zoom, image_url)
    """
    # 줌 21 (최대 확대) → 18 (넓은 시야) 순서로 시도
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
        predictions = run_inference(image_bytes)

        if not predictions:
            print(f"[ZOOM] 줌 {zoom}: 건물 감지 없음 → 줌 아웃")
            continue

        # 클릭 좌표에 가장 가까운 건물 찾기
        target = find_closest_building(predictions, lat, lng, bounds)
        if not target:
            print(f"[ZOOM] 줌 {zoom}: 클릭 좌표 근처 건물 없음 → 줌 아웃")
            continue

        # 건물이 이미지 가장자리에 잘렸는지 확인
        if is_building_clipped([target], IMAGE_SIZE, EDGE_MARGIN):
            print(f"[ZOOM] 줌 {zoom}: 건물이 잘림 → 줌 아웃")
            continue

        # 건물이 온전히 보이는 최대 줌 찾음!
        print(f"[ZOOM] 최적 줌 레벨: {zoom}")
        return [target], bounds, zoom, image_url

    # 모든 줌에서 실패 → 마지막 시도 결과라도 반환
    print("[ZOOM] 최적 줌 탐색 실패 → 줌 19 기본값 사용")
    image_bytes, image_url = fetch_satellite_image(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    bounds = get_image_geo_bounds(lat=lat, lng=lng, zoom=19, size=IMAGE_SIZE)
    predictions = run_inference(image_bytes)
    fallback_target = find_closest_building(predictions, lat, lng, bounds)
    return [fallback_target] if fallback_target else predictions, bounds, 19, image_url


def find_closest_building(
    predictions: list[dict], lat: float, lng: float, bounds: dict,
) -> dict | None:
    """
    클릭 좌표와 가장 가까운 건물 예측을 찾습니다.
    클릭 좌표를 픽셀 좌표로 변환한 뒤 각 폴리곤 중심과의 거리를 비교.
    """
    # 클릭 좌표 → 픽셀
    click_px = (
        (lng - bounds["west"]) / (bounds["east"] - bounds["west"]) * IMAGE_SIZE,
        (bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * IMAGE_SIZE,
    )

    best = None
    best_dist = float("inf")

    for pred in predictions:
        points = pred.get("points", [])
        if not points:
            continue

        # 폴리곤 중심
        cx = sum(p["x"] for p in points) / len(points)
        cy = sum(p["y"] for p in points) / len(points)

        dist = (cx - click_px[0]) ** 2 + (cy - click_px[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best = pred

    return best


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_roof(req: AnalyzeRequest):
    """클릭한 건물 좌표 → 최적 줌 탐색 → 분석"""
    try:
        predictions, bounds, zoom, image_url = find_optimal_zoom_and_analyze(
            req.lat, req.lng,
        )

        # GeoJSON 변환
        geojson, stats = predictions_to_geojson(
            predictions=predictions,
            bounds=bounds,
            image_size=IMAGE_SIZE,
        )

        # 응답 구성 — 장애물만 분리
        obstacles = []
        for i, pred in enumerate(predictions):
            meta = CLASS_META.get(pred["class"], {"type": "unknown"})
            if meta["type"] == "roof":
                continue
            obstacles.append(ObstacleInfo(
                id=i + 1,
                class_name=pred["class"],
                confidence=round(pred["confidence"], 3),
                area_m2=round(stats["obstacle_areas_m2"][i], 2),
                bbox_m=stats["obstacle_bboxes_m"][i],
            ))

        return AnalyzeResponse(
            lat=req.lat,
            lng=req.lng,
            zoom=zoom,
            total_roof_area_m2=round(stats["image_area_m2"], 1),
            total_obstacle_area_m2=round(stats["total_obstacle_area_m2"], 2),
            installable_area_m2=round(
                stats["image_area_m2"] - stats["total_obstacle_area_m2"], 2,
            ),
            obstacle_count=len(obstacles),
            obstacles=obstacles,
            geojson=geojson,
            satellite_image_url=image_url,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=502, detail=f"외부 API 연결 실패: {e}")


@app.get("/api/health")
async def health():
    return {"status": "ok"}
