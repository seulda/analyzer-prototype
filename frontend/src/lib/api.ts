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
