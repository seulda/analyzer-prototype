"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import type { AnalyzeResponse } from "@/lib/api";

interface Props {
  center: { lat: number; lng: number } | null;
  data: AnalyzeResponse | null;
  selectable: boolean;
  onBuildingClick: (lat: number, lng: number) => void;
}

// 파이프라인 단계 정의
const PIPELINE_LAYERS = [
  { key: "outline", typeMatch: "outline", label: "1", dotColor: "#2196F3", title: "건물 윤곽" },
  { key: "face", typeMatch: "roof", label: "2", dotColor: "#F44336", title: "면 분리" },
  { key: "misdetected", typeMatch: "misdetected", label: "3", dotColor: "#FFEB3B", title: "오검출" },
] as const;

function createGeoJSONLayer(
  features: GeoJSON.Feature[],
): L.GeoJSON {
  return L.geoJSON(
    { type: "FeatureCollection", features } as GeoJSON.FeatureCollection,
    {
      style: (feature) => {
        const color = feature?.properties?.color || "#ff0000";
        const type = feature?.properties?.type;
        return {
          color: color,
          weight: type === "outline" ? 2 : 2,
          opacity: 1,
          fillColor: color,
          fillOpacity: type === "outline" ? 0.05 : 0.35,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        if (!p) return;

        const typeLabel =
          p.type === "obstacle" ? "장애물" :
          p.type === "misdetected" ? "오검출" :
          p.type === "outline" ? "건물 윤곽" : "지붕면";

        const popup = `
          <div style="font-size:13px; line-height:1.6; min-width:160px;">
            <strong style="font-size:14px;">${p.label}</strong>
            <span style="background:${p.color}; color:${p.type === "misdetected" || p.type === "outline" ? "#000" : "#fff"}; padding:1px 6px; border-radius:3px; font-size:11px; margin-left:6px;">
              ${typeLabel}
            </span>
            <hr style="margin:6px 0; border-color:#eee;" />
            <div>면적: <strong>${p.area_m2} m²</strong></div>
            <div>크기: ${p.width_m}m x ${p.height_m}m</div>
            <div>신뢰도: ${(p.confidence * 100).toFixed(1)}%</div>
          </div>
        `;
        layer.bindPopup(popup);
      },
    },
  );
}

export default function RoofMap({ center, data, selectable, onBuildingClick }: Props) {
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 단계별 레이어 그룹 refs
  const outlineLayerRef = useRef<L.LayerGroup | null>(null);
  const faceLayerRef = useRef<L.LayerGroup | null>(null);
  const misdetectedLayerRef = useRef<L.LayerGroup | null>(null);
  const controlRef = useRef<L.Control | null>(null);

  // 맵 초기화
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = L.map(containerRef.current, {
      center: [35.6812, 139.7671],
      zoom: 18,
      zoomControl: true,
    });

    // Google Maps 위성 타일
    L.tileLayer(
      "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
      {
        attribution: "&copy; Google Maps",
        maxZoom: 22,
      },
    ).addTo(map);

    const outlineLayer = L.layerGroup().addTo(map);
    const faceLayer = L.layerGroup().addTo(map);
    const misdetectedLayer = L.layerGroup().addTo(map);

    outlineLayerRef.current = outlineLayer;
    faceLayerRef.current = faceLayer;
    misdetectedLayerRef.current = misdetectedLayer;

    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // center 변경 시 지도 이동
  useEffect(() => {
    if (!mapRef.current || !center) return;
    mapRef.current.setView([center.lat, center.lng], 19, { animate: true });
  }, [center]);

  // 클릭 핸들러 (selectable 상태일 때만)
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;

    const handleClick = (e: L.LeafletMouseEvent) => {
      if (!selectable) return;

      const { lat, lng } = e.latlng;

      // 클릭 마커 표시
      if (markerRef.current) {
        markerRef.current.setLatLng([lat, lng]);
      } else {
        markerRef.current = L.marker([lat, lng], {
          icon: L.divIcon({
            className: "",
            html: `<div style="
              width:20px; height:20px; border-radius:50%;
              background:rgba(255,87,34,0.8); border:3px solid #fff;
              box-shadow:0 2px 6px rgba(0,0,0,0.4);
              transform:translate(-10px,-10px);
            "></div>`,
          }),
        }).addTo(map);
      }

      onBuildingClick(lat, lng);
    };

    map.on("click", handleClick);

    // 커서 스타일
    if (selectable) {
      map.getContainer().style.cursor = "crosshair";
    } else {
      map.getContainer().style.cursor = "";
    }

    return () => {
      map.off("click", handleClick);
    };
  }, [selectable, onBuildingClick]);

  // 데이터 변경 시 GeoJSON 렌더링 + 토글 컨트롤 갱신
  useEffect(() => {
    if (!mapRef.current) return;
    const map = mapRef.current;

    const layerRefs = {
      outline: outlineLayerRef,
      face: faceLayerRef,
      misdetected: misdetectedLayerRef,
    };

    // 기존 레이어 클리어
    Object.values(layerRefs).forEach((ref) => ref.current?.clearLayers());

    // 기존 컨트롤 제거
    if (controlRef.current) {
      map.removeControl(controlRef.current);
      controlRef.current = null;
    }

    // 클릭 마커 제거
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }

    if (!data) return;

    const features = (data.geojson as GeoJSON.FeatureCollection).features;

    // 타입별 feature 분류 및 레이어 생성
    const layerMap: Record<string, L.LayerGroup> = {};
    const featureCounts: Record<string, number> = {};
    const allBounds = L.latLngBounds([]);

    for (const cfg of PIPELINE_LAYERS) {
      const matched = features.filter(
        (f) => f.properties?.type === cfg.typeMatch,
      );
      featureCounts[cfg.key] = matched.length;

      const ref = layerRefs[cfg.key as keyof typeof layerRefs];
      if (!ref.current) continue;

      if (matched.length > 0) {
        const geoLayer = createGeoJSONLayer(matched);
        geoLayer.addTo(ref.current);
        // bounds 확장
        const b = geoLayer.getBounds();
        if (b.isValid()) allBounds.extend(b);
      }

      layerMap[cfg.key] = ref.current;
    }

    // 지도 맞춤
    if (allBounds.isValid()) {
      map.fitBounds(allBounds, { padding: [50, 50] });
    }

    // 토글 상태 추적
    const layerVisible: Record<string, boolean> = {};
    for (const cfg of PIPELINE_LAYERS) {
      layerVisible[cfg.key] = featureCounts[cfg.key] > 0;
    }

    // 커스텀 컨트롤 생성
    const PipelineControl = L.Control.extend({
      options: { position: "topleft" as L.ControlPosition },

      onAdd() {
        const container = L.DomUtil.create("div", "pipeline-control leaflet-bar");
        L.DomEvent.disableClickPropagation(container);
        L.DomEvent.disableScrollPropagation(container);

        for (const cfg of PIPELINE_LAYERS) {
          const btn = L.DomUtil.create("button", "pipeline-btn", container);
          const hasData = featureCounts[cfg.key] > 0;

          btn.innerHTML = `<span class="pipeline-dot" style="background:${cfg.dotColor};${cfg.dotColor === "#FFFFFF" ? "border:1px solid #999;" : ""}"></span><span class="pipeline-label">${cfg.label}</span>`;
          btn.title = cfg.title;
          btn.disabled = !hasData;

          if (hasData) {
            btn.classList.add("active");
          } else {
            btn.classList.add("disabled");
          }

          btn.addEventListener("click", () => {
            if (!hasData) return;
            const layer = layerMap[cfg.key];
            if (!layer) return;

            if (layerVisible[cfg.key]) {
              map.removeLayer(layer);
              layerVisible[cfg.key] = false;
              btn.classList.remove("active");
            } else {
              map.addLayer(layer);
              layerVisible[cfg.key] = true;
              btn.classList.add("active");
            }
          });
        }

        return container;
      },
    });

    const control = new PipelineControl();
    control.addTo(map);
    controlRef.current = control;
  }, [data]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
