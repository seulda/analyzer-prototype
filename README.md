# Roof Analyzer Prototype

좌표 입력 → 위성 이미지 → MobileSAM 세그멘테이션 → 건물 윤곽 편집 → 경사면 분리 → GeoJSON 지도 표출

## 분석 파이프라인

```
지도 클릭 → /api/outline (윤곽) → 꼭지점 편집 → "분석" 클릭 → /api/analyze-faces (면 분리) → 결과 표출
```

### Step 1. 건물 윤곽 추출 (`/api/outline`)

| 항목 | 내용 |
|------|------|
| 위성 이미지 | **Google Maps Static API** — 줌 21→18 자동 탐색, 640x640px |
| 건물 마스크 | **MobileSAM** (`mobile_sam` / `vit_t`) — 클릭 좌표를 point prompt로 SAM 추론, argmax(scores) 마스크 선택. 면적 부족 시 bbox prompt로 재추론 |
| 윤곽 폴리곤 | **OpenCV** `findContours` → `approxPolyDP` (Douglas-Peucker 단순화) → 15° 각도 스냅으로 직선화 |
| 줌 최적화 | 줌 21부터 내려가며 건물이 이미지 가장자리에 잘리지 않는 최대 줌 선택 |

프론트엔드에 건물 윤곽 GeoJSON + session_id 반환. 사용자가 꼭지점을 드래그/추가/삭제하여 윤곽 보정 가능.

### Step 2. 경사면 분리 (`/api/analyze-faces`)

| 항목 | 내용 |
|------|------|
| 면 분리 | **OpenCV** K-means — 건물 마스크 내 Lab 색공간 + 정규화 좌표 5D 피처 클러스터링 |
| 최적 K 선택 | **scikit-learn** `silhouette_score` + 경계 직선성(cv2.fitLine 잔차) — k=2~16 동적 탐색, 건물 크기 비례 임계값 |
| 소면적 처리 | connected components → 건물 면적 5% 미만 CC는 `cv2.distanceTransform`으로 최근접 면에 편입 |
| 면 폴리곤 | **Shapely** 능선 split — label map 경계를 직선 피팅 후 outline 폴리곤을 칼로 잘라 겹침 없는 퍼즐 조각 생성 |

수정된 윤곽 좌표가 있으면 `cv2.fillPoly`로 새 마스크를 생성하여 면 분리 수행.

### Step 3. 오검출 보정

| 항목 | 내용 |
|------|------|
| 오검출 판정 | 각 면의 centroid 간 nearest neighbor 거리 계산, 50m 초과 시 `misdetected`로 표시 |
| 면적 보정 | 오검출 면의 pixel_area를 building_outline에서 차감 |

### GeoJSON 변환 및 표출

| 항목 | 내용 |
|------|------|
| 좌표 변환 | Mercator 투영 기반 픽셀 → 위경도 변환 (`pixel_to_latlng`) |
| 면적 계산 | pixel_area 기반 (label map이 overlap 없이 보장), meters_per_pixel² 변환 |
| 지도 | **Leaflet** — 3개 레이어 토글 (1: 건물 윤곽, 2: 면 분리, 3: 오검출) |

## 사용자 워크플로

1. 좌표 입력 → "이동" → 지도 이동
2. 지도에서 건물 클릭 → 핑크색 마커 + 건물 윤곽 표출
3. 꼭지점 드래그로 윤곽 보정 (우클릭: 선 위 → 추가, 꼭지점 → 삭제)
4. "분석" 버튼 → 면 분리 + 오검출 결과 표출
5. 1/2/3 레이어 토글로 확인

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | Python, FastAPI, MobileSAM, OpenCV, scikit-learn, Shapely, NumPy, Pillow, PyTorch |
| Frontend | Next.js 15, TypeScript, Leaflet |
| 외부 API | Google Maps Static API |

## 실행 방법

### 1. 백엔드 (FastAPI)

```bash
cd backend
cp .env.example .env  # API 키 설정
.venv/bin/uvicorn main:app --reload --port 8000
```

### 2. 프론트엔드 (Next.js)

```bash
cd frontend
pnpm install
pnpm dev  # http://localhost:3001
```

### 3. 브라우저에서 확인

http://localhost:3001 접속 → 좌표 입력 → 분석 시작

## 환경 변수

### backend/.env
```
GOOGLE_MAPS_API_KEY=...     # 위성 이미지 수집 (선택)
SAM_CHECKPOINT=...          # MobileSAM 체크포인트 경로 (기본: models/mobile_sam.pt)
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/outline` | 건물 윤곽 추출 (Step 1) |
| POST | `/api/analyze-faces` | 면 분리 + 오검출 보정 (Step 2+3) |
| POST | `/api/analyze` | 전체 한번에 실행 (호환용) |
| GET | `/api/health` | 서버 상태 확인 |
