"""
위성 이미지 수집 모듈
====================
Google Maps Static API에서 위성 이미지를 가져오고,
이미지의 지리적 범위(bounds)를 계산합니다.
"""

import math
import os

import requests


def fetch_satellite_image(
    lat: float, lng: float, zoom: int = 20, size: int = 640,
) -> tuple[bytes, str]:
    """
    Google Maps Static API에서 위성 이미지를 가져옵니다.

    Returns:
        (image_bytes, image_url)
    """
    api_key = os.environ.get("GOOGLE_MAPS_API_KEY", "")

    url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lng}",
        "zoom": zoom,
        "size": f"{size}x{size}",
        "maptype": "satellite",
        "key": api_key,
    }

    # API 키 없으면 데모 모드 (워터마크 포함 이미지)
    if not api_key:
        # 키 없이도 저해상도 이미지는 받을 수 있음
        params.pop("key")

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    # URL 재구성 (키 제외)
    display_url = (
        f"{url}?center={lat},{lng}&zoom={zoom}"
        f"&size={size}x{size}&maptype=satellite"
    )

    return resp.content, display_url


def get_image_geo_bounds(
    lat: float, lng: float, zoom: int, size: int,
) -> dict:
    """
    Google Maps 타일 좌표 체계로 이미지의 지리적 범위를 계산합니다.

    Mercator 투영 기반으로 이미지의 4개 모서리 좌표를 반환합니다.

    Returns:
        {
            "north": float, "south": float,
            "east": float, "west": float,
            "meters_per_pixel": float,
        }
    """
    # Google Maps 타일 해상도: 256px per tile at zoom 0
    # 적도 기준 전체 둘레 = 2 * pi * 6378137m
    EARTH_CIRCUMFERENCE = 2 * math.pi * 6378137  # meters
    TILE_SIZE = 256

    # 해당 줌 레벨에서 1px당 미터 (적도 기준)
    meters_per_pixel_equator = EARTH_CIRCUMFERENCE / (TILE_SIZE * (2 ** zoom))

    # 위도 보정 (Mercator)
    lat_rad = math.radians(lat)
    meters_per_pixel = meters_per_pixel_equator * math.cos(lat_rad)

    # 이미지가 커버하는 실제 거리 (미터)
    half_size = size / 2
    half_meters_x = half_size * meters_per_pixel
    half_meters_y = half_size * meters_per_pixel

    # 미터 → 위경도 변환
    # 위도: 1도 ≈ 111,320m
    # 경도: 1도 ≈ 111,320m * cos(lat)
    delta_lat = half_meters_y / 111320
    delta_lng = half_meters_x / (111320 * math.cos(lat_rad))

    return {
        "north": lat + delta_lat,
        "south": lat - delta_lat,
        "east": lng + delta_lng,
        "west": lng - delta_lng,
        "meters_per_pixel": meters_per_pixel,
    }
