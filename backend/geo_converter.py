"""
GeoJSON 변환 모듈
==================
Roboflow의 픽셀 좌표 예측 결과를 GeoJSON으로 변환합니다.
"""

import math


# 경사면별 동적 색상 팔레트 (최대 8면)
FACE_COLORS = [
    "#4CAF50",  # Green
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#9C27B0",  # Purple
    "#00BCD4",  # Cyan
    "#E91E63",  # Pink
    "#8BC34A",  # Light Green
    "#FF5722",  # Deep Orange
]

KMEANS_COLORS = [
    "#E91E63",  # Pink
    "#9C27B0",  # Purple
    "#673AB7",  # Deep Purple
    "#3F51B5",  # Indigo
    "#2196F3",  # Blue
    "#00BCD4",  # Cyan
]

DT_COLORS = [
    "#FF9800",  # Orange
    "#FF5722",  # Deep Orange
    "#FFC107",  # Amber
    "#CDDC39",  # Lime
    "#8BC34A",  # Light Green
    "#009688",  # Teal
    "#00BCD4",  # Cyan
    "#795548",  # Brown
]

# 클래스별 색상 및 메타데이터
CLASS_META = {
    "building_outline": {"color": "#2196F3", "label": "건물 윤곽", "type": "outline"},
    "misdetected": {"color": "#FFEB3B", "label": "오검출", "type": "misdetected"},
    "roof_face": {"color": "#F44336", "label": "지붕면", "type": "roof"},
    "Roof": {"color": "#4CAF50", "label": "지붕", "type": "roof"},
    "skylight": {"color": "#2196F3", "label": "천창", "type": "obstacle"},
    "vent": {"color": "#FF9800", "label": "환기구", "type": "obstacle"},
    "chimney": {"color": "#F44336", "label": "굴뚝", "type": "obstacle"},
    "dormer": {"color": "#9C27B0", "label": "도머", "type": "obstacle"},
    "antenna": {"color": "#607D8B", "label": "안테나", "type": "obstacle"},
    "solar_panel": {"color": "#00BCD4", "label": "기존 태양광", "type": "obstacle"},
    "other_obstruction": {"color": "#795548", "label": "기타 장애물", "type": "obstacle"},
    "debug_kmeans": {"color": "#9C27B0", "label": "K-means", "type": "kmeans"},
    "debug_dt": {"color": "#FF9800", "label": "DT면", "type": "dt"},
}


def latlng_to_pixel(
    lat: float, lng: float,
    bounds: dict, image_size: int,
) -> tuple[int, int]:
    """
    위경도 → 픽셀 좌표 변환 (pixel_to_latlng의 역함수)
    """
    x = (lng - bounds["west"]) / (bounds["east"] - bounds["west"]) * image_size
    y = (bounds["north"] - lat) / (bounds["north"] - bounds["south"]) * image_size
    return int(round(x)), int(round(y))


def _azimuth_direction(deg: float) -> str:
    """방위각(0=북, 시계방향) → 방위 문자열"""
    dirs = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
    idx = int((deg + 22.5) % 360 / 45)
    return dirs[idx]


def pixel_to_latlng(
    px: float, py: float,
    bounds: dict, image_size: int,
) -> tuple[float, float]:
    """
    픽셀 좌표 → 위경도 변환

    이미지 좌상단 = (0, 0), 우하단 = (image_size, image_size)
    """
    # x축: 경도 (좌→우 = west→east)
    lng = bounds["west"] + (px / image_size) * (bounds["east"] - bounds["west"])

    # y축: 위도 (상→하 = north→south)
    lat = bounds["north"] - (py / image_size) * (bounds["north"] - bounds["south"])

    return lat, lng


def calculate_polygon_area_m2(
    points: list[dict], bounds: dict, image_size: int, meters_per_pixel: float,
) -> float:
    """
    폴리곤의 실제 면적을 계산합니다 (m²).

    Shoelace formula를 픽셀 좌표에 적용 후 meters_per_pixel로 변환합니다.
    """
    n = len(points)
    if n < 3:
        return 0.0

    # Shoelace formula (픽셀 단위)
    area_px = 0.0
    for i in range(n):
        j = (i + 1) % n
        area_px += points[i]["x"] * points[j]["y"]
        area_px -= points[j]["x"] * points[i]["y"]

    area_px = abs(area_px) / 2.0

    # 픽셀 → m²
    return area_px * (meters_per_pixel ** 2)


def calculate_bbox_meters(
    points: list[dict], meters_per_pixel: float,
) -> dict:
    """장애물의 바운딩 박스를 미터 단위로 계산"""
    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]

    width_px = max(xs) - min(xs)
    height_px = max(ys) - min(ys)

    return {
        "width_m": round(width_px * meters_per_pixel, 2),
        "height_m": round(height_px * meters_per_pixel, 2),
    }


def predictions_to_geojson(
    predictions: list[dict],
    bounds: dict,
    image_size: int,
) -> tuple[dict, dict]:
    """
    Roboflow 예측 결과를 GeoJSON FeatureCollection으로 변환합니다.

    Returns:
        (geojson, stats)
    """
    mpp = bounds["meters_per_pixel"]
    features = []
    obstacle_areas = []
    obstacle_bboxes = []
    total_obstacle_area = 0.0
    face_index = 0  # roof_face 색상 순환 인덱스
    kmeans_index = 0
    dt_index = 0

    for i, pred in enumerate(predictions):
        points = pred["points"]
        class_name = pred["class"]
        meta = CLASS_META.get(class_name, {
            "color": "#999999", "label": class_name, "type": "unknown",
        })

        # roof_face 넘버링 + 개별 색상
        if class_name == "roof_face":
            meta = dict(meta)  # 원본 변경 방지
            meta["label"] = f"지붕면 {face_index + 1}"
            meta["color"] = FACE_COLORS[face_index % len(FACE_COLORS)]
            face_index += 1
        elif class_name == "debug_kmeans":
            meta = dict(meta)
            meta["label"] = f"K-means {kmeans_index + 1}"
            meta["color"] = KMEANS_COLORS[kmeans_index % len(KMEANS_COLORS)]
            kmeans_index += 1
        elif class_name == "debug_dt":
            meta = dict(meta)
            meta["label"] = f"DT면 {dt_index + 1}"
            meta["color"] = DT_COLORS[dt_index % len(DT_COLORS)]
            dt_index += 1

        # 픽셀 → 위경도 변환
        coords = []
        for p in points:
            lat, lng = pixel_to_latlng(p["x"], p["y"], bounds, image_size)
            coords.append([lng, lat])  # GeoJSON은 [lng, lat] 순서

        # 폴리곤 닫기
        if coords and coords[0] != coords[-1]:
            coords.append(coords[0])

        # 면적 계산: pixel_area가 있으면 픽셀 기반, 없으면 Shoelace 폴백
        pixel_area = pred.get("pixel_area")
        if pixel_area is not None:
            area_m2 = pixel_area * (mpp ** 2)
        else:
            area_m2 = calculate_polygon_area_m2(points, bounds, image_size, mpp)
        bbox_m = calculate_bbox_meters(points, mpp)

        obstacle_areas.append(area_m2)
        obstacle_bboxes.append(bbox_m)

        if meta["type"] == "obstacle":
            total_obstacle_area += area_m2

        props = {
                "id": i + 1,
                "class": class_name,
                "label": meta["label"],
                "type": meta["type"],
                "confidence": round(pred["confidence"], 3),
                "color": meta["color"],
                "area_m2": round(area_m2, 2),
                "width_m": bbox_m["width_m"],
                "height_m": bbox_m["height_m"],
        }

        if "azimuth_deg" in pred:
            props["azimuth_deg"] = round(pred["azimuth_deg"], 1)
            props["azimuth_label"] = _azimuth_direction(pred["azimuth_deg"])

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords],
            },
            "properties": props,
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    # 이미지 전체 면적 (m²)
    image_area_m2 = (image_size * mpp) ** 2

    stats = {
        "image_area_m2": image_area_m2,
        "total_obstacle_area_m2": total_obstacle_area,
        "obstacle_areas_m2": obstacle_areas,
        "obstacle_bboxes_m": obstacle_bboxes,
    }

    return geojson, stats
