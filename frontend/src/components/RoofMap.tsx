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

export default function RoofMap({ center, data, selectable, onBuildingClick }: Props) {
  const mapRef = useRef<L.Map | null>(null);
  const layerRef = useRef<L.LayerGroup | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

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

    const layerGroup = L.layerGroup().addTo(map);

    mapRef.current = map;
    layerRef.current = layerGroup;

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

  // 데이터 변경 시 GeoJSON 렌더링
  useEffect(() => {
    if (!mapRef.current || !layerRef.current) return;
    const map = mapRef.current;
    const layerGroup = layerRef.current;

    layerGroup.clearLayers();

    // 클릭 마커 제거
    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }

    if (!data) return;

    const geoLayer = L.geoJSON(data.geojson as GeoJSON.FeatureCollection, {
      style: () => {
        return {
          color: "#ff0000",
          weight: 4,
          opacity: 1,
          fillColor: "#ff0000",
          fillOpacity: 0.4,
        };
      },
      onEachFeature: (feature, layer) => {
        const p = feature.properties;
        if (!p) return;

        const popup = `
          <div style="font-size:13px; line-height:1.6; min-width:160px;">
            <strong style="font-size:14px;">${p.label}</strong>
            <span style="background:${p.color}; color:#fff; padding:1px 6px; border-radius:3px; font-size:11px; margin-left:6px;">
              ${p.type === "obstacle" ? "장애물" : "지붕면"}
            </span>
            <hr style="margin:6px 0; border-color:#eee;" />
            <div>면적: <strong>${p.area_m2} m²</strong></div>
            <div>크기: ${p.width_m}m x ${p.height_m}m</div>
            <div>신뢰도: ${(p.confidence * 100).toFixed(1)}%</div>
          </div>
        `;
        layer.bindPopup(popup);
      },
    });

    geoLayer.addTo(layerGroup);

    const bounds = geoLayer.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }
  }, [data]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
