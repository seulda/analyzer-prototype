"""
Roboflow 추론 클라이언트
========================
Roboflow Inference API를 통해 지붕 이미지에서
장애물을 감지합니다.

실제 프로덕션에서는 커스텀 학습 모델을 사용하고,
프로토타입에서는 Roboflow Universe의 기존 모델 또는 데모 데이터를 사용합니다.
"""

import os
import base64

ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")
# Roboflow Universe에서 가져온 모델 (예시)
# 실제로는 커스텀 학습 모델 ID로 교체
ROBOFLOW_MODEL_ID = os.environ.get("ROBOFLOW_MODEL_ID", "")
ROBOFLOW_MODEL_VERSION = os.environ.get("ROBOFLOW_MODEL_VERSION", "1")


def run_inference(image_bytes: bytes) -> list[dict]:
    """
    Roboflow 모델로 이미지 추론을 실행합니다.

    Returns:
        predictions: [
            {
                "class": "skylight",
                "confidence": 0.92,
                "points": [{"x": 100, "y": 200}, {"x": 150, "y": 200}, ...],
            },
            ...
        ]
    """
    # Roboflow API 키와 모델이 설정되고 이미지가 있는 경우 실제 추론
    if ROBOFLOW_API_KEY and ROBOFLOW_MODEL_ID and image_bytes:
        try:
            return _run_roboflow_inference(image_bytes)
        except Exception as e:
            print(f"[ERROR] Roboflow 추론 실패: {e}")
            print("[FALLBACK] 데모 데이터로 대체합니다")
            return _get_demo_predictions()

    # 설정 안 된 경우 또는 이미지 없는 경우 데모 데이터 반환
    reason = []
    if not ROBOFLOW_API_KEY:
        reason.append("API_KEY 없음")
    if not ROBOFLOW_MODEL_ID:
        reason.append("MODEL_ID 없음")
    if not image_bytes:
        reason.append("이미지 없음")
    print(f"[DEMO] Roboflow 데모 모드 ({', '.join(reason)})")
    return _get_demo_predictions()


def _run_roboflow_inference(image_bytes: bytes) -> list[dict]:
    """Roboflow Hosted API로 실제 추론"""
    import requests

    # 이미지를 base64로 인코딩
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    url = (
        f"https://detect.roboflow.com/"
        f"{ROBOFLOW_MODEL_ID}/{ROBOFLOW_MODEL_VERSION}"
    )
    params = {
        "api_key": ROBOFLOW_API_KEY,
        "confidence": 30,  # 30% 이상
    }

    resp = requests.post(
        url,
        params=params,
        data=img_b64,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()

    predictions = []
    for pred in data.get("predictions", []):
        # Instance Segmentation 결과에는 points가 있음
        points = pred.get("points", [])

        # points가 없으면 bbox로 폴리곤 생성
        if not points:
            x, y = pred.get("x", 0), pred.get("y", 0)
            w, h = pred.get("width", 0), pred.get("height", 0)
            points = [
                {"x": x - w / 2, "y": y - h / 2},
                {"x": x + w / 2, "y": y - h / 2},
                {"x": x + w / 2, "y": y + h / 2},
                {"x": x - w / 2, "y": y + h / 2},
            ]

        predictions.append({
            "class": pred.get("class", "unknown"),
            "confidence": pred.get("confidence", 0),
            "points": points,
        })

    return predictions


def _get_demo_predictions() -> list[dict]:
    """
    데모용 예측 데이터 (일본 주택 지붕 시뮬레이션)

    640x640 이미지 기준으로 지붕 위 장애물을 시뮬레이션합니다.
    """
    return [
        {
            "class": "roof_face_south",
            "confidence": 0.95,
            "points": [
                {"x": 120, "y": 80},
                {"x": 520, "y": 80},
                {"x": 560, "y": 320},
                {"x": 80, "y": 320},
            ],
        },
        {
            "class": "roof_face_north",
            "confidence": 0.93,
            "points": [
                {"x": 80, "y": 320},
                {"x": 560, "y": 320},
                {"x": 520, "y": 560},
                {"x": 120, "y": 560},
            ],
        },
        {
            "class": "skylight",
            "confidence": 0.88,
            "points": [
                {"x": 200, "y": 150},
                {"x": 280, "y": 150},
                {"x": 280, "y": 220},
                {"x": 200, "y": 220},
            ],
        },
        {
            "class": "vent",
            "confidence": 0.82,
            "points": [
                {"x": 400, "y": 180},
                {"x": 440, "y": 180},
                {"x": 440, "y": 210},
                {"x": 400, "y": 210},
            ],
        },
        {
            "class": "antenna",
            "confidence": 0.76,
            "points": [
                {"x": 350, "y": 400},
                {"x": 380, "y": 390},
                {"x": 400, "y": 420},
                {"x": 390, "y": 450},
                {"x": 360, "y": 440},
            ],
        },
        {
            "class": "chimney",
            "confidence": 0.91,
            "points": [
                {"x": 480, "y": 100},
                {"x": 520, "y": 100},
                {"x": 520, "y": 145},
                {"x": 480, "y": 145},
            ],
        },
        {
            "class": "dormer",
            "confidence": 0.85,
            "points": [
                {"x": 150, "y": 370},
                {"x": 190, "y": 350},
                {"x": 250, "y": 350},
                {"x": 290, "y": 370},
                {"x": 290, "y": 430},
                {"x": 150, "y": 430},
            ],
        },
    ]
