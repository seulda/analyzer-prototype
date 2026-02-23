# Roof Analyzer Prototype

좌표 입력 → 위성 이미지 → Roboflow 추론 → GeoJSON → 지도 표출

## 실행 방법

### 1. 백엔드 (FastAPI)

```bash
cd backend
cp .env.example .env  # API 키 설정

# 실행 (데모 모드: API 키 없어도 동작)
python3 -m uvicorn main:app --reload --port 8000
```

### 2. 프론트엔드 (Next.js)

```bash
cd frontend
pnpm install
pnpm dev  # http://localhost:3001
```

### 3. 브라우저에서 확인

http://localhost:3001 접속 → 좌표 입력 → 분석 시작

## 동작 모드

- **데모 모드**: Roboflow API 키 없이 실행 → 시뮬레이션 데이터로 GeoJSON 표출
- **실제 모드**: `.env`에 API 키 설정 → Roboflow 모델 추론 결과 표출

## 환경 변수

### backend/.env
```
GOOGLE_MAPS_API_KEY=...     # 위성 이미지 수집 (선택)
ROBOFLOW_API_KEY=...        # Roboflow 추론 (선택, 없으면 데모)
ROBOFLOW_MODEL_ID=...       # 커스텀 학습 모델 ID
ROBOFLOW_MODEL_VERSION=1    # 모델 버전
```
