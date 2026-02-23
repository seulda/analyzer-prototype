const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ObstacleInfo {
  id: number;
  class_name: string;
  confidence: number;
  area_m2: number;
  bbox_m: { width_m: number; height_m: number };
}

export interface AnalyzeResponse {
  lat: number;
  lng: number;
  zoom: number;
  total_roof_area_m2: number;
  total_obstacle_area_m2: number;
  installable_area_m2: number;
  obstacle_count: number;
  obstacles: ObstacleInfo[];
  geojson: GeoJSON.FeatureCollection;
  satellite_image_url: string;
  warning: string | null;
}

export interface OutlineResponse {
  session_id: string;
  zoom: number;
  satellite_image_url: string;
  outline_geojson: GeoJSON.FeatureCollection;
}

export async function analyzeRoof(
  lat: number,
  lng: number,
): Promise<AnalyzeResponse> {
  const resp = await fetch(`${API_BASE}/api/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lng }),
  });

  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `API error: ${resp.status}`);
  }

  return resp.json();
}

export async function getOutline(
  lat: number,
  lng: number,
): Promise<OutlineResponse> {
  const resp = await fetch(`${API_BASE}/api/outline`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ lat, lng }),
  });

  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `API error: ${resp.status}`);
  }

  return resp.json();
}

export async function analyzeFaces(
  session_id: string,
  modified_points?: [number, number][],
): Promise<AnalyzeResponse> {
  const resp = await fetch(`${API_BASE}/api/analyze-faces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id, modified_points }),
  });

  if (!resp.ok) {
    const error = await resp.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `API error: ${resp.status}`);
  }

  return resp.json();
}
