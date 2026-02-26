"use client";

import { useEffect, useRef, useCallback, useState } from "react";
import {
  APIProvider,
  Map,
  MapControl,
  ControlPosition,
  useMap,
  MapMouseEvent,
} from "@vis.gl/react-google-maps";
import type { AnalyzeResponse, OutlineResponse } from "@/lib/api";
import type { FeatureCollection, Feature, Polygon } from "geojson";

interface Props {
  center: { lat: number; lng: number } | null;
  data: AnalyzeResponse | null;
  outlineData: OutlineResponse | null;
  selectable: boolean;
  editable: boolean;
  selectedFaceId: number | null;
  onBuildingClick: (lat: number, lng: number) => void;
  onOutlineEdit: (points: [number, number][]) => void;
}

// 파이프라인 단계 정의
const PIPELINE_LAYERS = [
  { key: "outline", typeMatch: "outline", label: "1", dotColor: "#2196F3", title: "건물 윤곽" },
  { key: "face", typeMatch: "roof", label: "2", dotColor: "#F44336", title: "면 분리" },
  { key: "misdetected", typeMatch: "misdetected", label: "3", dotColor: "#FFEB3B", title: "오검출" },
] as const;

const API_KEY = process.env.NEXT_PUBLIC_GOOGLE_MAPS_API_KEY || "";

function MapInner({
  center,
  data,
  outlineData,
  selectable,
  editable,
  selectedFaceId,
  onBuildingClick,
  onOutlineEdit,
}: Props) {
  const map = useMap();

  const clickMarkerRef = useRef<google.maps.Marker | null>(null);
  const editPolygonRef = useRef<google.maps.Polygon | null>(null);
  const dataLayersRef = useRef<Record<string, google.maps.Data>>({});
  const infoWindowRef = useRef<google.maps.InfoWindow | null>(null);

  const [layerVisible, setLayerVisible] = useState<Record<string, boolean>>({
    outline: true,
    face: true,
    misdetected: true,
  });
  const [featureCounts, setFeatureCounts] = useState<Record<string, number>>({
    outline: 0,
    face: 0,
    misdetected: 0,
  });

  // Stable callback refs
  const onBuildingClickRef = useRef(onBuildingClick);
  onBuildingClickRef.current = onBuildingClick;
  const onOutlineEditRef = useRef(onOutlineEdit);
  onOutlineEditRef.current = onOutlineEdit;

  // center 변경 시 지도 이동
  useEffect(() => {
    if (!map || !center) return;
    map.panTo({ lat: center.lat, lng: center.lng });
    map.setZoom(19);
  }, [map, center]);

  // 커서 스타일
  useEffect(() => {
    if (!map) return;
    map.setOptions({
      draggableCursor: selectable ? "crosshair" : undefined,
    });
  }, [map, selectable]);

  // 클릭 마커 (핑크 원) — cleanup on unmount
  useEffect(() => {
    return () => {
      clickMarkerRef.current?.setMap(null);
      clickMarkerRef.current = null;
    };
  }, []);

  // 편집 가능한 윤곽 (editable polygon)
  const clearOutlineEdit = useCallback(() => {
    if (editPolygonRef.current) {
      editPolygonRef.current.setMap(null);
      editPolygonRef.current = null;
    }
  }, []);

  useEffect(() => {
    if (!map) return;

    clearOutlineEdit();

    if (!editable || !outlineData) return;

    const feature = outlineData.outline_geojson.features[0];
    if (!feature || feature.geometry.type !== "Polygon") return;

    const rawCoords = (feature.geometry as Polygon).coordinates[0];
    const coords = rawCoords.slice(0, -1); // 마지막 중복 점 제거

    const path = coords.map((c: number[]) => ({ lat: c[1], lng: c[0] }));

    const polygon = new google.maps.Polygon({
      paths: path,
      editable: true,
      draggable: false,
      strokeColor: "#2196F3",
      strokeWeight: 2,
      fillColor: "#2196F3",
      fillOpacity: 0.15,
    });
    polygon.setMap(map);
    editPolygonRef.current = polygon;

    const emitEdit = () => {
      const p = polygon.getPath();
      const points: [number, number][] = [];
      for (let i = 0; i < p.getLength(); i++) {
        const ll = p.getAt(i);
        points.push([ll.lng(), ll.lat()]);
      }
      onOutlineEditRef.current(points);
    };

    // 경로 변경 감지
    const gPath = polygon.getPath();
    const listeners = [
      google.maps.event.addListener(gPath, "set_at", emitEdit),
      google.maps.event.addListener(gPath, "insert_at", emitEdit),
      google.maps.event.addListener(gPath, "remove_at", emitEdit),
    ];

    // 우클릭 꼭지점 삭제
    const rightClickListener = polygon.addListener(
      "rightclick",
      (e: google.maps.PolyMouseEvent) => {
        if (e.vertex != null && gPath.getLength() > 3) {
          gPath.removeAt(e.vertex);
        }
      },
    );
    listeners.push(rightClickListener);

    // 지도를 폴리곤에 맞춤
    const bounds = new google.maps.LatLngBounds();
    path.forEach((pt) => bounds.extend(pt));
    map.fitBounds(bounds, 50);

    // 우클릭 메뉴 방지
    const preventContext = (e: Event) => e.preventDefault();
    map.getDiv().addEventListener("contextmenu", preventContext);

    return () => {
      listeners.forEach((l) => google.maps.event.removeListener(l));
      map.getDiv().removeEventListener("contextmenu", preventContext);
      clearOutlineEdit();
    };
  }, [map, outlineData, editable, clearOutlineEdit]);

  // GeoJSON 렌더링 + 레이어 관리
  useEffect(() => {
    if (!map) return;

    // 기존 레이어 정리
    Object.values(dataLayersRef.current).forEach((dl) => dl.setMap(null));
    dataLayersRef.current = {};
    infoWindowRef.current?.close();

    // 결과가 있으면 편집 폴리곤 정리
    if (data) {
      clearOutlineEdit();
    }

    if (!data) {
      setFeatureCounts({ outline: 0, face: 0, misdetected: 0 });
      return;
    }

    const features = (data.geojson as FeatureCollection).features;
    const newCounts: Record<string, number> = {};
    const allBounds = new google.maps.LatLngBounds();

    const infoWindow = new google.maps.InfoWindow();
    infoWindowRef.current = infoWindow;

    for (const cfg of PIPELINE_LAYERS) {
      const matched = features.filter(
        (f: Feature) => f.properties?.type === cfg.typeMatch,
      );
      newCounts[cfg.key] = matched.length;

      const dataLayer = new google.maps.Data({ map });

      if (matched.length > 0) {
        const fc: FeatureCollection = {
          type: "FeatureCollection",
          features: matched,
        };
        dataLayer.addGeoJson(fc);

        // 스타일 설정
        dataLayer.setStyle((feature) => {
          const color = feature.getProperty("color") as string || "#ff0000";
          const type = feature.getProperty("type") as string;
          return {
            fillColor: color,
            fillOpacity: type === "outline" ? 0.05 : 0.35,
            strokeColor: color,
            strokeWeight: 2,
            strokeOpacity: 1,
          };
        });

        // 클릭 → InfoWindow 팝업
        dataLayer.addListener("click", (e: google.maps.Data.MouseEvent) => {
          const f = e.feature;
          const p = {
            label: f.getProperty("label") as string,
            type: f.getProperty("type") as string,
            color: f.getProperty("color") as string,
            area_m2: f.getProperty("area_m2"),
            width_m: f.getProperty("width_m"),
            height_m: f.getProperty("height_m"),
            confidence: f.getProperty("confidence") as number,
            azimuth_deg: f.getProperty("azimuth_deg") as number | undefined,
            azimuth_label: f.getProperty("azimuth_label") as string | undefined,
          };

          const typeLabel =
            p.type === "obstacle" ? "장애물" :
            p.type === "misdetected" ? "오검출" :
            p.type === "outline" ? "건물 윤곽" : "지붕면";

          const textColor = p.type === "misdetected" || p.type === "outline" ? "#000" : "#fff";

          const azimuthInfo = p.azimuth_deg != null
            ? `<div>방위: <strong>${p.azimuth_deg}° (${p.azimuth_label ?? ""})</strong></div>`
            : "";

          const content = `
            <div style="font-size:13px; line-height:1.6; min-width:160px; color:#000;">
              <strong style="font-size:14px;">${p.label}</strong>
              <span style="background:${p.color}; color:${textColor}; padding:1px 6px; border-radius:3px; font-size:11px; margin-left:6px;">
                ${typeLabel}
              </span>
              <hr style="margin:6px 0; border-color:#eee;" />
              <div>면적: <strong>${p.area_m2} m²</strong></div>
              <div>크기: ${p.width_m}m x ${p.height_m}m</div>
              ${azimuthInfo}
              <div>신뢰도: ${((p.confidence ?? 0) * 100).toFixed(1)}%</div>
            </div>
          `;

          infoWindow.setContent(content);
          infoWindow.setPosition(e.latLng!);
          infoWindow.open(map);
        });

        // bounds 확장
        dataLayer.forEach((feature) => {
          const geo = feature.getGeometry();
          geo?.forEachLatLng((ll) => allBounds.extend(ll));
        });
      }

      dataLayersRef.current[cfg.key] = dataLayer;
    }

    setFeatureCounts(newCounts);
    setLayerVisible({
      outline: newCounts.outline > 0,
      face: newCounts.face > 0,
      misdetected: newCounts.misdetected > 0,
    });

    // 지도 맞춤
    if (!allBounds.isEmpty()) {
      map.fitBounds(allBounds, 50);
    }

    return () => {
      Object.values(dataLayersRef.current).forEach((dl) => dl.setMap(null));
      dataLayersRef.current = {};
      infoWindow.close();
    };
  }, [map, data, clearOutlineEdit]);

  // 레이어 토글 동기화
  useEffect(() => {
    if (!map) return;
    for (const cfg of PIPELINE_LAYERS) {
      const dl = dataLayersRef.current[cfg.key];
      if (!dl) continue;
      dl.setMap(layerVisible[cfg.key] ? map : null);
    }
  }, [map, layerVisible]);

  // 선택된 지붕면 하이라이트
  useEffect(() => {
    const dl = dataLayersRef.current["face"];
    if (!dl) return;

    dl.forEach((feature) => {
      const fid = feature.getProperty("id") as number;
      const color = feature.getProperty("color") as string || "#ff0000";
      if (selectedFaceId != null && fid === selectedFaceId) {
        dl.overrideStyle(feature, {
          fillOpacity: 0.7,
          strokeWeight: 3,
          strokeColor: "#fff",
        });
      } else {
        dl.overrideStyle(feature, {
          fillColor: color,
          fillOpacity: 0.35,
          strokeWeight: 2,
          strokeColor: color,
        });
      }
    });
  }, [selectedFaceId]);

  // 맵 클릭 핸들러
  const handleMapClick = useCallback(
    (e: MapMouseEvent) => {
      if (!selectable || !map) return;

      const latLng = e.detail.latLng;
      if (!latLng) return;
      const { lat, lng } = latLng;

      // 핑크색 클릭 마커
      if (clickMarkerRef.current) {
        clickMarkerRef.current.setPosition({ lat, lng });
      } else {
        clickMarkerRef.current = new google.maps.Marker({
          position: { lat, lng },
          icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: "rgba(255,105,180,0.6)",
            fillOpacity: 0.6,
            strokeColor: "#fff",
            strokeWeight: 3,
          },
          map,
        });
      }

      onBuildingClickRef.current(lat, lng);
    },
    [selectable, map],
  );

  // 토글 핸들러
  const handleToggle = useCallback((key: string) => {
    setLayerVisible((prev) => ({ ...prev, [key]: !prev[key] }));
  }, []);

  return (
    <>
      <Map
        defaultCenter={{ lat: 35.6812, lng: 139.7671 }}
        defaultZoom={18}
        mapTypeId="satellite"
        defaultTilt={0}
        tilt={0}
        gestureHandling="greedy"
        disableDefaultUI={false}
        zoomControl={true}
        mapTypeControl={false}
        streetViewControl={false}
        fullscreenControl={false}
        onClick={handleMapClick}
        style={{ width: "100%", height: "100%" }}
      >
        {/* 파이프라인 토글 컨트롤 — 데이터가 있을 때만 */}
        {data && (
          <MapControl position={ControlPosition.TOP_LEFT}>
            <div className="pipeline-control">
              {PIPELINE_LAYERS.map((cfg) => {
                const hasData = featureCounts[cfg.key] > 0;
                const active = layerVisible[cfg.key];
                const className = [
                  "pipeline-btn",
                  !hasData ? "disabled" : active ? "active" : "",
                ]
                  .filter(Boolean)
                  .join(" ");

                return (
                  <button
                    key={cfg.key}
                    className={className}
                    title={cfg.title}
                    disabled={!hasData}
                    onClick={() => hasData && handleToggle(cfg.key)}
                  >
                    <span
                      className="pipeline-dot"
                      style={{ background: cfg.dotColor }}
                    />
                    <span className="pipeline-label">{cfg.label}</span>
                  </button>
                );
              })}
            </div>
          </MapControl>
        )}
      </Map>
    </>
  );
}

export default function RoofMap(props: Props) {
  return (
    <APIProvider apiKey={API_KEY}>
      <MapInner {...props} />
    </APIProvider>
  );
}
