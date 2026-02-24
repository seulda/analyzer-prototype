"use client";

import type { AnalyzeResponse } from "@/lib/api";

interface Props {
  data: AnalyzeResponse | null;
}

const CLASS_LABELS: Record<string, string> = {
  roof_face_south: "지붕면 (남)",
  roof_face_north: "지붕면 (북)",
  roof_face_east: "지붕면 (동)",
  roof_face_west: "지붕면 (서)",
  roof_face: "지붕면",
  misdetected: "오검출",
  skylight: "천창",
  vent: "환기구",
  chimney: "굴뚝",
  dormer: "도머",
  antenna: "안테나",
  solar_panel: "기존 태양광",
  other_obstruction: "기타 장애물",
};

const CLASS_COLORS: Record<string, string> = {
  misdetected: "#FFEB3B",
  skylight: "#2196F3",
  vent: "#FF9800",
  chimney: "#F44336",
  dormer: "#9C27B0",
  antenna: "#607D8B",
  solar_panel: "#00BCD4",
};

export default function ResultPanel({ data }: Props) {
  if (!data) {
    return (
      <div style={styles.empty}>
        <p>좌표를 입력하고 분석을 시작하세요</p>
      </div>
    );
  }

  const obstacles = data.obstacles.filter(
    (o) => !o.class_name.startsWith("roof_face") && o.class_name !== "building_outline" && o.class_name !== "misdetected",
  );
  const roofFaces = data.obstacles.filter((o) =>
    o.class_name.startsWith("roof_face"),
  );

  const usableRatio =
    data.total_roof_area_m2 > 0
      ? (data.installable_area_m2 / data.total_roof_area_m2) * 100
      : 0;

  return (
    <div style={styles.container}>
      <h3 style={styles.sectionTitle}>분석 결과</h3>

      {/* 요약 카드 */}
      <div style={styles.cards}>
        <div style={styles.card}>
          <span style={styles.cardLabel}>전체 지붕</span>
          <span style={styles.cardValue}>
            {data.total_roof_area_m2.toFixed(1)} m²
          </span>
        </div>
        <div style={styles.card}>
          <span style={styles.cardLabel}>장애물</span>
          <span style={{ ...styles.cardValue, color: "#F44336" }}>
            {data.total_obstacle_area_m2.toFixed(2)} m²
          </span>
        </div>
        <div style={styles.card}>
          <span style={styles.cardLabel}>설치 가능</span>
          <span style={{ ...styles.cardValue, color: "#4CAF50" }}>
            {data.installable_area_m2.toFixed(2)} m²
          </span>
        </div>
        <div style={styles.card}>
          <span style={styles.cardLabel}>활용률</span>
          <span style={styles.cardValue}>{usableRatio.toFixed(1)}%</span>
        </div>
      </div>

      {/* 지붕면 목록 */}
      {roofFaces.length > 0 && (
        <>
          <h4 style={styles.subTitle}>지붕면 ({roofFaces.length})</h4>
          <div style={styles.list}>
            {roofFaces.map((r) => (
              <div key={r.id} style={styles.listItem}>
                <div style={styles.itemHeader}>
                  <span style={dotStyle("#F44336")} />
                  <span>{CLASS_LABELS[r.class_name] || r.class_name}</span>
                  <span style={styles.confidence}>
                    {(r.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div style={styles.itemDetail}>
                  {r.area_m2} m² / {r.bbox_m.width_m}m x {r.bbox_m.height_m}m
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {/* 장애물 목록 */}
      <h4 style={styles.subTitle}>장애물 ({obstacles.length})</h4>
      {obstacles.length === 0 ? (
        <p style={styles.noData}>감지된 장애물 없음</p>
      ) : (
        <div style={styles.list}>
          {obstacles.map((o) => (
            <div key={o.id} style={styles.listItem}>
              <div style={styles.itemHeader}>
                <span
                  style={dotStyle(
                    CLASS_COLORS[o.class_name] || "#999",
                  )}
                />
                <span>{CLASS_LABELS[o.class_name] || o.class_name}</span>
                <span style={styles.confidence}>
                  {(o.confidence * 100).toFixed(0)}%
                </span>
              </div>
              <div style={styles.itemDetail}>
                {o.area_m2} m² / {o.bbox_m.width_m}m x {o.bbox_m.height_m}m
              </div>
            </div>
          ))}
        </div>
      )}

      {/* GeoJSON 원본 토글 */}
      <details style={styles.details}>
        <summary style={styles.summary}>
          <span>GeoJSON 원본 데이터</span>
          <button
            style={styles.copyBtn}
            onClick={(e) => {
              e.preventDefault();
              navigator.clipboard.writeText(JSON.stringify(data.geojson, null, 2));
              const btn = e.currentTarget;
              btn.textContent = "copied!";
              setTimeout(() => { btn.textContent = "copy"; }, 1500);
            }}
          >
            copy
          </button>
        </summary>
        <pre style={styles.json}>
          {JSON.stringify(data.geojson, null, 2)}
        </pre>
      </details>
    </div>
  );
}

function dotStyle(color: string): React.CSSProperties {
  return {
    width: "10px",
    height: "10px",
    borderRadius: "50%",
    background: color,
    display: "inline-block",
    flexShrink: 0,
  };
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "16px 24px",
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  empty: {
    padding: "40px 24px",
    textAlign: "center",
    color: "#666",
  },
  sectionTitle: {
    fontSize: "16px",
    fontWeight: 700,
    color: "#fff",
  },
  subTitle: {
    fontSize: "13px",
    fontWeight: 600,
    color: "#aaa",
    marginTop: "8px",
  },
  cards: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: "8px",
  },
  card: {
    background: "#16213e",
    borderRadius: "8px",
    padding: "12px",
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  cardLabel: {
    fontSize: "11px",
    color: "#888",
  },
  cardValue: {
    fontSize: "18px",
    fontWeight: 700,
    color: "#fff",
  },
  list: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
  },
  listItem: {
    background: "#16213e",
    borderRadius: "6px",
    padding: "10px 12px",
  },
  itemHeader: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    fontSize: "13px",
    color: "#ddd",
  },
  itemDetail: {
    marginTop: "4px",
    fontSize: "12px",
    color: "#888",
    paddingLeft: "20px",
  },
  confidence: {
    marginLeft: "auto",
    fontSize: "11px",
    color: "#666",
  },
  noData: {
    fontSize: "13px",
    color: "#666",
    padding: "8px 0",
  },
  details: {
    marginTop: "12px",
  },
  summary: {
    fontSize: "12px",
    color: "#666",
    cursor: "pointer",
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  copyBtn: {
    padding: "2px 8px",
    borderRadius: "4px",
    border: "1px solid #444",
    background: "transparent",
    color: "#888",
    fontSize: "11px",
    cursor: "pointer",
  },
  json: {
    marginTop: "8px",
    background: "#0a0a1a",
    borderRadius: "6px",
    padding: "12px",
    fontSize: "11px",
    color: "#888",
    overflow: "auto",
    maxHeight: "300px",
    whiteSpace: "pre-wrap",
    wordBreak: "break-all",
  },
};
