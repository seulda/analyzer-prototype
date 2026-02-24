"use client";

import { useState } from "react";

interface Props {
  onSubmit: (lat: number, lng: number) => void;
  loading: boolean;
  onReset?: () => void;
  showReset?: boolean;
}

// 일본 주요 좌표 프리셋
const PRESETS = [
  { label: "도쿄역", lat: 35.6812, lng: 139.7671 },
  { label: "오사카 주택가", lat: 34.6525, lng: 135.5062 },
  { label: "교토 주택가", lat: 35.0116, lng: 135.7681 },
  { label: "후쿠오카 주택가", lat: 33.5902, lng: 130.4017 },
];

export default function CoordinateForm({ onSubmit, loading, onReset, showReset }: Props) {
  const [lat, setLat] = useState("35.6812");
  const [lng, setLng] = useState("139.7671");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const latNum = parseFloat(lat);
    const lngNum = parseFloat(lng);
    if (isNaN(latNum) || isNaN(lngNum)) return;
    onSubmit(latNum, lngNum);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>Roof Analyzer</h2>
      <p style={styles.subtitle}>좌표를 입력하고 건물을 선택하세요</p>

      <div style={styles.presets}>
        <p style={styles.presetLabel}>빠른 선택:</p>
        <div style={styles.presetButtons}>
          {PRESETS.map((p) => (
            <button
              key={p.label}
              onClick={() => {
                setLat(String(p.lat));
                setLng(String(p.lng));
                onSubmit(p.lat, p.lng);
              }}
              style={styles.presetBtn}
            >
              {p.label}
            </button>
          ))}
        </div>
      </div>

      <form onSubmit={handleSubmit} style={styles.form}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>위도 (Latitude)</label>
          <input
            type="text"
            value={lat}
            onChange={(e) => setLat(e.target.value)}
            style={styles.input}
            placeholder="35.6812"
          />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>경도 (Longitude)</label>
          <input
            type="text"
            value={lng}
            onChange={(e) => setLng(e.target.value)}
            style={styles.input}
            placeholder="139.7671"
          />
        </div>
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? "분석 중..." : "이동"}
        </button>
        {showReset && onReset && (
          <button type="button" onClick={onReset} style={styles.resetButton}>
            다른 건물 선택
          </button>
        )}
      </form>
    </div>
  );
}

const styles: Record<string, React.CSSProperties> = {
  container: {
    padding: "24px",
    display: "flex",
    flexDirection: "column",
    gap: "16px",
  },
  title: {
    fontSize: "22px",
    fontWeight: 700,
    color: "#fff",
  },
  subtitle: {
    fontSize: "13px",
    color: "#888",
  },
  form: {
    display: "flex",
    flexDirection: "column",
    gap: "12px",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
  },
  label: {
    fontSize: "12px",
    color: "#aaa",
    fontWeight: 500,
  },
  input: {
    padding: "10px 12px",
    borderRadius: "6px",
    border: "1px solid #333",
    background: "#16213e",
    color: "#fff",
    fontSize: "14px",
    outline: "none",
  },
  button: {
    marginTop: "8px",
    padding: "12px",
    borderRadius: "6px",
    border: "none",
    background: "#0f3460",
    color: "#fff",
    fontSize: "15px",
    fontWeight: 600,
    cursor: "pointer",
  },
  resetButton: {
    padding: "12px",
    borderRadius: "6px",
    border: "none",
    background: "#e74c3c",
    color: "#fff",
    fontSize: "15px",
    fontWeight: 600,
    cursor: "pointer",
    marginBottom: "12px",
  },
  presets: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
  },
  presetLabel: {
    fontSize: "12px",
    color: "#666",
  },
  presetButtons: {
    display: "flex",
    flexWrap: "wrap",
    gap: "6px",
  },
  presetBtn: {
    padding: "6px 12px",
    borderRadius: "4px",
    border: "1px solid #333",
    background: "transparent",
    color: "#aaa",
    fontSize: "12px",
    cursor: "pointer",
  },
};
