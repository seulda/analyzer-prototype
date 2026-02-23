"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import CoordinateForm from "@/components/CoordinateForm";
import ResultPanel from "@/components/ResultPanel";
import { analyzeRoof, type AnalyzeResponse } from "@/lib/api";

const RoofMap = dynamic(() => import("@/components/RoofMap"), { ssr: false });

type Phase = "input" | "select" | "analyzing" | "result";

export default function Home() {
  const [phase, setPhase] = useState<Phase>("input");
  const [center, setCenter] = useState<{ lat: number; lng: number } | null>(null);
  const [data, setData] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // 1단계: 좌표 입력 → 지도 이동
  const handleNavigate = (lat: number, lng: number) => {
    setCenter({ lat, lng });
    setData(null);
    setError(null);
    setPhase("select");
  };

  // 2단계: 지도에서 건물 클릭 → 분석 시작
  const handleBuildingClick = async (lat: number, lng: number) => {
    setPhase("analyzing");
    setError(null);
    try {
      const result = await analyzeRoof(lat, lng);
      setData(result);
      setPhase("result");
      if (result.warning) {
        alert(result.warning);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "분석 실패");
      setPhase("select");
    }
  };

  // 다시 선택
  const handleReset = () => {
    setData(null);
    setError(null);
    setPhase("select");
  };

  return (
    <div style={styles.layout}>
      <aside style={styles.sidebar}>
        <CoordinateForm
          onSubmit={handleNavigate}
          loading={phase === "analyzing"}
        />

        {/* 단계별 안내 */}
        <div style={styles.phaseGuide}>
          {phase === "input" && (
            <p style={styles.guide}>좌표를 입력하고 이동하세요</p>
          )}
          {phase === "select" && (
            <p style={styles.guideActive}>
              지도에서 분석할 건물을 클릭하세요
            </p>
          )}
          {phase === "analyzing" && (
            <p style={styles.guideLoading}>
              최적 줌 레벨 탐색 + 분석 중...
            </p>
          )}
          {phase === "result" && data && (
            <div style={styles.zoomInfo}>
              <span>최적 줌 레벨: {data.zoom}</span>
              <button onClick={handleReset} style={styles.resetBtn}>
                다른 건물 선택
              </button>
            </div>
          )}
        </div>

        {error && <div style={styles.error}>{error}</div>}
        <div style={styles.divider} />
        <ResultPanel data={data} />
      </aside>

      <main style={styles.main}>
        <RoofMap
          center={center}
          data={data}
          selectable={phase === "select"}
          onBuildingClick={handleBuildingClick}
        />
        {phase === "analyzing" && (
          <div style={styles.overlay}>
            <div style={styles.spinner} />
            <p style={styles.overlayText}>지붕 분석 중...</p>
          </div>
        )}
      </main>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  layout: {
    display: "flex",
    height: "100vh",
    width: "100vw",
  },
  sidebar: {
    width: "380px",
    minWidth: "380px",
    background: "#1a1a2e",
    borderRight: "1px solid #2a2a4a",
    display: "flex",
    flexDirection: "column",
    overflowY: "auto",
  },
  main: {
    flex: 1,
    position: "relative",
  },
  divider: {
    height: "1px",
    background: "#2a2a4a",
    margin: "0 24px",
  },
  error: {
    margin: "0 24px",
    padding: "10px 12px",
    background: "#3d1111",
    borderRadius: "6px",
    color: "#ff6b6b",
    fontSize: "13px",
  },
  phaseGuide: {
    padding: "0 24px 12px",
  },
  guide: {
    fontSize: "13px",
    color: "#666",
  },
  guideActive: {
    fontSize: "13px",
    color: "#4CAF50",
    fontWeight: 600,
  },
  guideLoading: {
    fontSize: "13px",
    color: "#FF9800",
    fontWeight: 600,
  },
  zoomInfo: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    fontSize: "13px",
    color: "#aaa",
  },
  resetBtn: {
    padding: "4px 12px",
    borderRadius: "4px",
    border: "1px solid #444",
    background: "transparent",
    color: "#aaa",
    fontSize: "12px",
    cursor: "pointer",
  },
  overlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(0, 0, 0, 0.5)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    zIndex: 1000,
    gap: "16px",
  },
  spinner: {
    width: "48px",
    height: "48px",
    border: "4px solid rgba(255, 255, 255, 0.2)",
    borderTop: "4px solid #4CAF50",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  overlayText: {
    color: "#fff",
    fontSize: "16px",
    fontWeight: 600,
  },
};
