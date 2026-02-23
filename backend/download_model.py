"""
MobileSAM 모델 다운로드 스크립트
================================
MobileSAM checkpoint (~40MB)을 backend/models/ 디렉토리에 다운로드합니다.

사용법:
    python download_model.py
"""

import os
import subprocess
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "mobile_sam.pt")

# MobileSAM 공식 체크포인트 URL (GitHub LFS)
CHECKPOINT_URL = (
    "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
)


def download_model():
    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(CHECKPOINT_PATH):
        size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
        print(f"[SAM] 모델 이미 존재: {CHECKPOINT_PATH} ({size_mb:.1f}MB)")
        return CHECKPOINT_PATH

    print("[SAM] MobileSAM 체크포인트 다운로드 중...")
    print(f"  URL: {CHECKPOINT_URL}")
    print(f"  저장: {CHECKPOINT_PATH}")

    result = subprocess.run(
        ["curl", "-L", "-o", CHECKPOINT_PATH, CHECKPOINT_URL],
        check=True,
    )

    size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    print(f"[SAM] 다운로드 완료 ({size_mb:.1f}MB)")
    return CHECKPOINT_PATH


if __name__ == "__main__":
    download_model()
