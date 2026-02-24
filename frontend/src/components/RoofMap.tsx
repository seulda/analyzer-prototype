"use client";

import { useEffect, useRef, useCallback } from "react";
import L from "leaflet";
import type { AnalyzeResponse, OutlineResponse } from "@/lib/api";

interface Props {
  center: { lat: number; lng: number } | null;
  data: AnalyzeResponse | null;
  outlineData: OutlineResponse | null;
  selectable: boolean;
  editable: boolean;
  onBuildingClick: (lat: number, lng: number) => void;
  onOutlineEdit: (points: [number, number][]) => void;
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

export default function RoofMap({
  center, data, outlineData, selectable, editable, onBuildingClick, onOutlineEdit,
}: Props) {
  const mapRef = useRef<L.Map | null>(null);
  const clickMarkerRef = useRef<L.Marker | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 단계별 레이어 그룹 refs
  const outlineLayerRef = useRef<L.LayerGroup | null>(null);
  const faceLayerRef = useRef<L.LayerGroup | null>(null);
  const misdetectedLayerRef = useRef<L.LayerGroup | null>(null);
  const controlRef = useRef<L.Control | null>(null);

  // Outline editing refs
  const outlineEditLayerRef = useRef<L.LayerGroup | null>(null);
  const vertexMarkersRef = useRef<L.CircleMarker[]>([]);
  const outlinePolygonRef = useRef<L.Polygon | null>(null);

  // Stable callback refs
  const onBuildingClickRef = useRef(onBuildingClick);
  onBuildingClickRef.current = onBuildingClick;
  const onOutlineEditRef = useRef(onOutlineEdit);
  onOutlineEditRef.current = onOutlineEdit;

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
    const outlineEditLayer = L.layerGroup().addTo(map);

    outlineLayerRef.current = outlineLayer;
    faceLayerRef.current = faceLayer;
    misdetectedLayerRef.current = misdetectedLayer;
    outlineEditLayerRef.current = outlineEditLayer;

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

      // 핑크색 클릭 마커 표시 (반투명)
      if (clickMarkerRef.current) {
        clickMarkerRef.current.setLatLng([lat, lng]);
      } else {
        clickMarkerRef.current = L.marker([lat, lng], {
          icon: L.divIcon({
            className: "",
            html: `<div style="
              width:20px; height:20px; border-radius:50%;
              background:rgba(255,105,180,0.6); border:3px solid #fff;
              box-shadow:0 2px 6px rgba(0,0,0,0.4);
              transform:translate(-10px,-10px);
            "></div>`,
          }),
        }).addTo(map);
      }

      onBuildingClickRef.current(lat, lng);
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
  }, [selectable]);

  // 편집 가능한 윤곽 표출 (outlining 모드)
  const clearOutlineEdit = useCallback(() => {
    outlineEditLayerRef.current?.clearLayers();
    vertexMarkersRef.current = [];
    outlinePolygonRef.current = null;
  }, []);

  useEffect(() => {
    if (!mapRef.current || !outlineEditLayerRef.current) return;

    clearOutlineEdit();

    // outlining 모드: editable이고 outlineData가 있을 때
    if (!editable || !outlineData) return;

    const map = mapRef.current;
    const editLayer = outlineEditLayerRef.current;

    const feature = outlineData.outline_geojson.features[0];
    if (!feature || feature.geometry.type !== "Polygon") return;

    // 폴리곤 좌표 추출 (마지막 중복 점 제거)
    const rawCoords = feature.geometry.coordinates[0];
    const coords = rawCoords.slice(0, -1);

    // 현재 꼭지점 좌표 (mutable)
    const currentLatLngs: L.LatLng[] = coords.map(
      (c: number[]) => L.latLng(c[1], c[0]),
    );

    // 폴리곤 생성 (파란색)
    const polygon = L.polygon(currentLatLngs, {
      color: "#2196F3",
      weight: 2,
      fillColor: "#2196F3",
      fillOpacity: 0.15,
    }).addTo(editLayer);
    outlinePolygonRef.current = polygon;

    // 콜백: 현재 좌표 → onOutlineEdit 전달
    const emitEdit = () => {
      const points: [number, number][] = currentLatLngs.map(
        (ll) => [ll.lng, ll.lat],
      );
      onOutlineEditRef.current(points);
    };

    // 점 P에서 선분 A-B까지의 거리 (픽셀 기반)
    const distToSegment = (p: L.LatLng, a: L.LatLng, b: L.LatLng): number => {
      const pp = map.latLngToContainerPoint(p);
      const pa = map.latLngToContainerPoint(a);
      const pb = map.latLngToContainerPoint(b);
      const dx = pb.x - pa.x;
      const dy = pb.y - pa.y;
      const lenSq = dx * dx + dy * dy;
      if (lenSq === 0) return pp.distanceTo(pa);
      let t = ((pp.x - pa.x) * dx + (pp.y - pa.y) * dy) / lenSq;
      t = Math.max(0, Math.min(1, t));
      const proj = L.point(pa.x + t * dx, pa.y + t * dy);
      return pp.distanceTo(proj);
    };

    // 마커 이벤트 클린업 추적
    const cleanupFns: (() => void)[] = [];

    // 마커 전체 재구성 (꼭지점 추가/삭제 후 호출)
    const rebuildMarkers = () => {
      // 기존 마커 이벤트 정리 + 제거
      cleanupFns.forEach((fn) => fn());
      cleanupFns.length = 0;
      vertexMarkersRef.current.forEach((m) => editLayer.removeLayer(m));
      vertexMarkersRef.current = [];

      // 폴리곤 업데이트
      polygon.setLatLngs(currentLatLngs);

      const markers: L.CircleMarker[] = [];

      currentLatLngs.forEach((ll, idx) => {
        const marker = L.circleMarker(ll, {
          radius: 6,
          color: "#2196F3",
          weight: 2,
          fillColor: "#ffffff",
          fillOpacity: 1,
          className: "vertex-marker",
        });
        marker.addTo(editLayer);

        // --- 드래그 ---
        let isDragging = false;

        const onMouseDown = (e: L.LeafletMouseEvent) => {
          isDragging = true;
          map.dragging.disable();
          L.DomEvent.stop(e.originalEvent);
        };
        const onMouseMove = (e: L.LeafletMouseEvent) => {
          if (!isDragging) return;
          marker.setLatLng(e.latlng);
          currentLatLngs[idx] = e.latlng;
          polygon.setLatLngs(currentLatLngs);
        };
        const onMouseUp = () => {
          if (!isDragging) return;
          isDragging = false;
          map.dragging.enable();
          emitEdit();
        };

        // --- 우클릭 → 꼭지점 삭제 (최소 3개 유지) ---
        const onVertexContext = (e: L.LeafletMouseEvent) => {
          L.DomEvent.stop(e.originalEvent);
          if (currentLatLngs.length <= 3) return;
          currentLatLngs.splice(idx, 1);
          rebuildMarkers();
          emitEdit();
        };

        marker.on("mousedown", onMouseDown);
        marker.on("contextmenu", onVertexContext);
        map.on("mousemove", onMouseMove);
        map.on("mouseup", onMouseUp);

        cleanupFns.push(() => {
          marker.off("mousedown", onMouseDown);
          marker.off("contextmenu", onVertexContext);
          map.off("mousemove", onMouseMove);
          map.off("mouseup", onMouseUp);
        });

        markers.push(marker);
      });

      vertexMarkersRef.current = markers;
    };

    // --- 폴리곤 선 위 우클릭 → 가장 가까운 edge에 꼭지점 추가 ---
    const onPolygonContext = (e: L.LeafletMouseEvent) => {
      L.DomEvent.stop(e.originalEvent);
      const clickLL = e.latlng;

      let bestInsertIdx = 0;
      let bestDist = Infinity;

      for (let i = 0; i < currentLatLngs.length; i++) {
        const j = (i + 1) % currentLatLngs.length;
        const d = distToSegment(clickLL, currentLatLngs[i], currentLatLngs[j]);
        if (d < bestDist) {
          bestDist = d;
          bestInsertIdx = j === 0 ? currentLatLngs.length : j;
        }
      }

      currentLatLngs.splice(bestInsertIdx, 0, clickLL);
      rebuildMarkers();
      emitEdit();
    };

    polygon.on("contextmenu", onPolygonContext);

    // 브라우저 기본 우클릭 메뉴 방지
    map.getContainer().addEventListener("contextmenu", (e) => e.preventDefault());

    // 초기 마커 생성
    rebuildMarkers();

    // 지도를 폴리곤에 맞춤
    const bounds = polygon.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50] });
    }

    return () => {
      cleanupFns.forEach((fn) => fn());
      polygon.off("contextmenu", onPolygonContext);
      map.dragging.enable();
    };
  }, [outlineData, editable, clearOutlineEdit]);

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

    // 결과가 있으면 outline edit 레이어 클리어
    if (data) {
      clearOutlineEdit();
    }

    // 주의: 핑크색 클릭 마커는 제거하지 않음 (분석 후에도 유지)

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
  }, [data, clearOutlineEdit]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
