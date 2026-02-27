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

K-means(색상)와 Distance Transform(기하학)을 **병렬 실행 후 대조**하여 면을 분리한다.

```
건물 마스크 + 위성 이미지
  ├─ Path A: K-means → face_id_map (클러스터 ID)
  └─ Path B: Distance Transform → nearest_edge (가장 가까운 변 ID)

대조: DT 경계가 K-means 경계와 일치 → 진짜 → 유지
      일치 안 함 → 가짜 → Union-Find로 합침
```

| 항목 | 내용 |
|------|------|
| Path A: K-means | 건물 마스크 내 **Lab + 정규화좌표 5D** 피처, `silhouette_score`로 최적 k 선택 (k=2~6). k=1이면 단일면 즉시 반환 |
| Path B: DT | 윤곽 폴리곤 단순화(`_simplify_polygon`) → 각 변까지 `cv2.distanceTransform` → 가장 가까운 변에 픽셀 할당 |
| 대조 | 인접 DT 면 쌍의 **dominant K-means cluster** 비교. 같으면 합침, 다르면 유지 |
| 합침 | Union-Find로 그룹 결정 → 마스크 OR → `_mask_to_polygon(snap_deg=0)` |
| Azimuth | 그룹 내 가장 긴 변의 outward normal |

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
| 지도 | **Google Maps JavaScript API** — 5개 레이어 토글 (1: 건물 윤곽, 2: 면 분리, 3: 오검출, K: K-means, DT: 기하학) |

## 사용자 워크플로

1. 좌표 입력 → "이동" → 지도 이동
2. 지도에서 건물 클릭 → 핑크색 마커 + 건물 윤곽 표출
3. 꼭지점 드래그로 윤곽 보정 (우클릭: 선 위 → 추가, 꼭지점 → 삭제)
4. "분석" 버튼 → 면 분리 + 오검출 결과 표출
5. 1/2/3/K/DT 레이어 토글로 확인 (K: K-means 중간결과, DT: 기하학 중간결과)

## 기술 스택

| 구분 | 기술 |
|------|------|
| Backend | Python, FastAPI, MobileSAM, OpenCV, scikit-learn, NumPy, Pillow, PyTorch |
| Frontend | Next.js 15, TypeScript, Google Maps JavaScript API (`@vis.gl/react-google-maps`) |
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
