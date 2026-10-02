"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { countMask, decodeMaskRle, filterOccludedProjectionPoints, ImageMask, MaskDirtyRect, mergeImageMasks, normalizeSamObjectMask, smoothSamMask, NormalizedRect, paintMaskStroke } from "./segmentation";
import { createWaymoCameraProjector, WaymoTemporalProjection } from "./waymo-camera-model";
import { type CameraFrameInfo, type CloudData, type ProjectionScope, type ImageSegmentationTool, type SynchronizedImageView, type SegmentationGranularity, type BrushMaskChange, type RectDragMode, type ProjectedPoint, isGroundPoint, projectionColor, renderMaskCanvas, renderMaskCanvasRegion, cursorForRectDragMode } from './workspace-core';

export function ImageCanvas({
  frame,
  cloud,
  matrix,
  intrinsic,
  distortion,
  distortionModel = "rational",
  temporalProjection = null,
  showProjection,
  stableOnly,
  hideGround,
  projectionSize,
  projectionOpacity,
  maskOpacity,
  backgroundOpacity = 1,
  selectedOnly = false,
  isolationMask = null,
  diagnosticHighlight = null,
  mask,
  secondaryMask = null,
  selectedPointIndices,
  projectionPointIndices,
  occlusionFiltering = true,
  projectionScope = "all",
  highlightSelectedProjection = false,
  selectable,
  segmentationTool,
  onSegmentationToolChange,
  brushSize,
  zoom: controlledZoom,
  localZoom = false,
  zoomResetToken = 0,
  onZoomChange,
  synchronizedView,
  onSynchronizedViewChange,
  granularity,
  preserveMaskLayers = false,
  onMaskEditStart,
  onMaskChange,
  onProjectedCount,
  projectionPicking = false,
  onProjectedPointPick,
}: {
  frame: CameraFrameInfo;
  cloud: CloudData | null;
  matrix: number[];
  intrinsic: number[];
  distortion: number[];
  distortionModel?: string;
  temporalProjection?: WaymoTemporalProjection | null;
  showProjection: boolean;
  stableOnly: boolean;
  hideGround: boolean;
  projectionSize: number;
  projectionOpacity: number;
  maskOpacity: number;
  backgroundOpacity?: number;
  selectedOnly?: boolean;
  isolationMask?: ImageMask | null;
  diagnosticHighlight?: { indices: number[]; color: number } | null;
  mask: ImageMask | null;
  secondaryMask?: ImageMask | null;
  selectedPointIndices: Uint32Array;
  projectionPointIndices?: Uint32Array;
  occlusionFiltering?: boolean;
  projectionScope?: ProjectionScope;
  highlightSelectedProjection?: boolean;
  selectable: boolean;
  segmentationTool?: ImageSegmentationTool;
  onSegmentationToolChange?: (tool: ImageSegmentationTool) => void;
  brushSize: number;
  zoom: number;
  localZoom?: boolean;
  zoomResetToken?: number;
  onZoomChange?: (zoom: number) => void;
  synchronizedView?: SynchronizedImageView;
  onSynchronizedViewChange?: (view: SynchronizedImageView) => void;
  granularity: SegmentationGranularity;
  preserveMaskLayers?: boolean;
  onMaskEditStart?: (source: "sam" | "brush") => void;
  onMaskChange?: (mask: ImageMask | null, change?: BrushMaskChange) => void;
  onProjectedCount?: (count: number) => void;
  projectionPicking?: boolean;
  onProjectedPointPick?: (pointIndex: number | null) => void;
}) {
  const [localZoomValue, setLocalZoomValue] = useState(controlledZoom);
  const zoom = localZoom ? localZoomValue : controlledZoom;
  useEffect(() => { if (localZoom) setLocalZoomValue(controlledZoom); }, [controlledZoom, localZoom, zoomResetToken]);
  const baseCanvasRef = useRef<HTMLCanvasElement>(null);
  const selectedProjectionCanvasRef = useRef<HTMLCanvasElement>(null);
  const maskCanvasRef = useRef<HTMLCanvasElement>(null);
  const interactionCanvasRef = useRef<HTMLCanvasElement>(null);
  const fitViewportRef = useRef<HTMLDivElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const panDragRef = useRef<{ pointerId: number; clientX: number; clientY: number; scrollLeft: number; scrollTop: number } | null>(null);
  const rectDragRef = useRef<{ mode: RectDragMode; start: { x: number; y: number }; initial: NormalizedRect } | null>(null);
  const lastBrushPointRef = useRef<{ x: number; y: number } | null>(null);
  const brushMaskRef = useRef<ImageMask | null>(null);
  const skipNextMaskRenderRef = useRef(false);
  const brushDirtyRectsRef = useRef<MaskDirtyRect[]>([]);
  const brushPixelDeltaRef = useRef(0);
  const latestMaskRef = useRef<ImageMask | null>(mask);
  const candidateBaseMaskRef = useRef<ImageMask | null>(null);
  const activePointerIdRef = useRef<number | null>(null);
  const onMaskChangeRef = useRef(onMaskChange);
  const onMaskEditStartRef = useRef(onMaskEditStart);
  const projectedPointsRef = useRef<ProjectedPoint[]>([]);
  const projectedPointByIndexRef = useRef<Array<ProjectedPoint | undefined>>([]);
  const pendingZoomAnchorRef = useRef<{
    frameUrl: string; imageX: number; imageY: number; pointerX: number; pointerY: number;
  } | null>(null);
  const synchronizedScrollTargetRef = useRef<{ left: number; top: number } | null>(null);
  const [imageVersion, setImageVersion] = useState(0);
  const [draft, setDraft] = useState<NormalizedRect | null>(null);
  const [candidates, setCandidates] = useState<Array<{ mask: ImageMask; score: number; area: number }>>([]);
  const [candidateIndex, setCandidateIndex] = useState(0);
  const [candidateMergeMode, setCandidateMergeMode] = useState<"replace" | "add">("replace");
  const [segmenting, setSegmenting] = useState(false);
  const [segmentationError, setSegmentationError] = useState<string | null>(null);
  const [fitScale, setFitScale] = useState(1);
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number } | null>(null);
  const brushCursorFrameRef = useRef<number | null>(null);
  const pendingBrushCursorRef = useRef<{ x: number; y: number } | null>(null);
  const [rectCursor, setRectCursor] = useState<React.CSSProperties["cursor"]>("crosshair");
  const [panning, setPanning] = useState(false);
  const draftRef = useRef<NormalizedRect | null>(null);

  const updateDraft = (value: NormalizedRect | null) => {
    draftRef.current = value;
    setDraft(value);
  };

  const scheduleBrushCursor = (value: { x: number; y: number } | null) => {
    pendingBrushCursorRef.current = value;
    if (brushCursorFrameRef.current !== null) return;
    brushCursorFrameRef.current = requestAnimationFrame(() => {
      brushCursorFrameRef.current = null;
      setBrushCursor(pendingBrushCursorRef.current);
    });
  };

  useEffect(() => () => {
    if (brushCursorFrameRef.current !== null) cancelAnimationFrame(brushCursorFrameRef.current);
  }, []);

  useEffect(() => {
    onMaskChangeRef.current = onMaskChange;
  }, [onMaskChange]);

  useEffect(() => {
    onMaskEditStartRef.current = onMaskEditStart;
  }, [onMaskEditStart]);

  useEffect(() => {
    latestMaskRef.current = mask;
  }, [mask]);

  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      setImageVersion((value) => value + 1);
    };
    image.src = frame.url;
    return () => { image.onload = null; };
  }, [frame.url]);

  useLayoutEffect(() => {
    // The outer viewport does not shrink when the inner scroller gains bars.
    // Observing the scroller creates a zoom -> scrollbar -> fitScale feedback loop.
    const viewport = fitViewportRef.current;
    if (!viewport) return;
    const update = () => setFitScale(Math.min(
      Math.max((viewport.clientWidth - 2) / frame.width, 0.05),
      Math.max((viewport.clientHeight - 2) / frame.height, 0.05),
      1,
    ));
    const observer = new ResizeObserver(update);
    observer.observe(viewport);
    update();
    return () => observer.disconnect();
  }, [frame.height, frame.width]);

  useLayoutEffect(() => {
    const viewport = scrollRef.current;
    const shell = interactionCanvasRef.current?.parentElement;
    if (!viewport || !shell) return;
    // Apply the scroll correction after DOM sizing, but before the browser paints.
    if (synchronizedView) {
      const scale = fitScale * synchronizedView.zoom;
      const renderedWidth = frame.width * scale;
      const renderedHeight = frame.height * scale;
      const desiredLeft = renderedWidth <= viewport.clientWidth
        ? 0
        : synchronizedView.centerX * renderedWidth - viewport.clientWidth * 0.5;
      const desiredTop = renderedHeight <= viewport.clientHeight
        ? 0
        : synchronizedView.centerY * renderedHeight - viewport.clientHeight * 0.5;
      viewport.scrollLeft = desiredLeft;
      viewport.scrollTop = desiredTop;
      synchronizedScrollTargetRef.current = { left: viewport.scrollLeft, top: viewport.scrollTop };
      pendingZoomAnchorRef.current = null;
      return;
    }
    const anchor = pendingZoomAnchorRef.current;
    pendingZoomAnchorRef.current = null;
    if (!anchor || anchor.frameUrl !== frame.url) return;
    const viewportRect = viewport.getBoundingClientRect();
    const shellRect = shell.getBoundingClientRect();
    const left = viewport.scrollLeft + shellRect.left - viewportRect.left;
    const top = viewport.scrollTop + shellRect.top - viewportRect.top;
    viewport.scrollLeft = left + anchor.imageX * (shellRect.width / frame.width) - anchor.pointerX;
    viewport.scrollTop = top + anchor.imageY * (shellRect.height / frame.height) - anchor.pointerY;
  }, [fitScale, frame.url, frame.height, frame.width, synchronizedView, zoom]);

  const projectionSelection = projectionScope === "all" ? null : projectionPointIndices ?? selectedPointIndices;

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, frame.width, frame.height);
    if (selectedOnly) {
      // Only the isolated view needs a raster cutout. The normal photograph
      // is a native image underneath this transparent projection canvas.
      context.globalAlpha = backgroundOpacity;
      context.drawImage(image, 0, 0, frame.width, frame.height);
      context.globalAlpha = 1;
      // Mask only the photograph; retain selected outlying projections for review.
      const cutout = document.createElement("canvas");
      cutout.width = isolationMask?.width ?? frame.width;
      cutout.height = isolationMask?.height ?? frame.height;
      const cutoutContext = cutout.getContext("2d");
      if (cutoutContext) {
        const pixels = cutoutContext.createImageData(cutout.width, cutout.height);
        isolationMask?.data.forEach((value, index) => { pixels.data[index * 4 + 3] = value ? 255 : 0; });
        cutoutContext.putImageData(pixels, 0, 0);
        context.globalCompositeOperation = "destination-in";
        context.drawImage(cutout, 0, 0, frame.width, frame.height);
        context.globalCompositeOperation = "source-over";
      }
    }

    let projected = 0;
    const projectedPoints: ProjectedPoint[] = [];
    if ((showProjection || highlightSelectedProjection || diagnosticHighlight) && cloud) {
      const waymoProjector = temporalProjection
        ? createWaymoCameraProjector(temporalProjection, matrix, intrinsic, distortion)
        : null;
      const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0] = distortion;
      const fullProjectionPoints: ProjectedPoint[] = [];
      const isolatedIndices = selectedOnly ? new Set(selectedPointIndices) : null;
      const diagnosticIndices = new Set(diagnosticHighlight?.indices ?? []);
      const selectedFlags = projectionSelection ? new Uint8Array(cloud.stable.length) : null;
      const highlightedFlags = highlightSelectedProjection ? new Uint8Array(cloud.stable.length) : null;
      projectionSelection?.forEach((index) => { if (selectedFlags && index < selectedFlags.length) selectedFlags[index] = 1; });
      selectedPointIndices.forEach((index) => { if (highlightedFlags && index < highlightedFlags.length) highlightedFlags[index] = 1; });
      for (let index = 0; index < cloud.stable.length; index += 1) {
      if (cloud.stable[index] === 2) continue;
        if (isolatedIndices && !isolatedIndices.has(index)) continue;
        const annotatedPoint = selectedFlags?.[index] === 1;
        const inFullProjection = showProjection && (projectionScope === "all" || annotatedPoint);
        const inSelectedProjection = highlightedFlags?.[index] === 1 || diagnosticIndices.has(index);
        const explicitPoint = annotatedPoint || inSelectedProjection;
        if (!inFullProjection && !inSelectedProjection) continue;
        // Explicitly annotated or currently selected points remain visible even
        // if display filters later classify them as dynamic or ground.
        if (!explicitPoint && stableOnly && cloud.stable[index] !== 1) continue;
        if (!explicitPoint && hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
        const x = cloud.positions[index * 3];
        const y = cloud.positions[index * 3 + 1];
        const z = cloud.positions[index * 3 + 2];
        let pixelX: number, pixelY: number, cameraZ: number;
        if (waymoProjector) {
          const projected = waymoProjector(x, y, z);
          if (!projected) continue;
          pixelX = projected.x;
          pixelY = projected.y;
          cameraZ = projected.depth;
        } else {
          const cameraX = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
          const cameraY = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
          cameraZ = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
          if (cameraZ <= 0.1) continue;
          const normalizedX = cameraX / cameraZ;
          const normalizedY = cameraY / cameraZ;
          const radius2 = normalizedX * normalizedX + normalizedY * normalizedY;
          let distortedX: number, distortedY: number;
          if (distortionModel === "fisheye") {
            const radius = Math.sqrt(radius2), theta = Math.atan(radius), theta2 = theta * theta;
            const thetaDistorted = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 + p1 * theta2 ** 3 + p2 * theta2 ** 4);
            const scale = radius > 1e-12 ? thetaDistorted / radius : 1;
            distortedX = normalizedX * scale;
            distortedY = normalizedY * scale;
          } else {
            const radius4 = radius2 * radius2;
            const radius6 = radius4 * radius2;
            const denominator = 1 + k4 * radius2 + k5 * radius4 + k6 * radius6;
            if (Math.abs(denominator) < 1e-8) continue;
            const radial = (1 + k1 * radius2 + k2 * radius4 + k3 * radius6) / denominator;
            distortedX = normalizedX * radial + 2 * p1 * normalizedX * normalizedY + p2 * (radius2 + 2 * normalizedX * normalizedX);
            distortedY = normalizedY * radial + p1 * (radius2 + 2 * normalizedY * normalizedY) + 2 * p2 * normalizedX * normalizedY;
          }
          pixelX = intrinsic[0] * distortedX + intrinsic[2];
          pixelY = intrinsic[4] * distortedY + intrinsic[5];
        }
        if (!Number.isFinite(pixelX) || !Number.isFinite(pixelY) ||
            pixelX < 0 || pixelX >= frame.width || pixelY < 0 || pixelY >= frame.height) continue;
        const projectedPoint = { index, x: pixelX, y: pixelY, depth: cameraZ };
        projectedPoints.push(projectedPoint);
        if (inFullProjection) fullProjectionPoints.push(projectedPoint);
      }
      const visibleProjectionPoints = occlusionFiltering && projectionScope === "all"
        ? filterOccludedProjectionPoints(fullProjectionPoints, Math.max(6, projectionSize * 1.4), 0.35, 1)
        : fullProjectionPoints;
      visibleProjectionPoints.sort((left, right) => right.depth - left.depth);
      projected = visibleProjectionPoints.length;
      context.globalAlpha = projectionOpacity;
      let activeColor = "";
      visibleProjectionPoints.forEach((point) => {
        const color = projectionColor(point.depth);
        if (color !== activeColor) {
          context.fillStyle = color;
          activeColor = color;
        }
        const size = projectionSize * (point.depth < 20 ? 1.2 : 1);
        context.fillRect(point.x - size * 0.5, point.y - size * 0.5, size, size);
      });
      context.globalAlpha = 1;
    }

    projectedPointsRef.current = projectedPoints;
    const projectedByIndex: Array<ProjectedPoint | undefined> = [];
    for (const point of projectedPoints) projectedByIndex[point.index] = point;
    projectedPointByIndexRef.current = projectedByIndex;
    onProjectedCount?.(projected);
  }, [diagnosticHighlight, selectedOnly, isolationMask, backgroundOpacity, cloud, distortion, distortionModel, frame, imageVersion, intrinsic, matrix, temporalProjection,
    hideGround, highlightSelectedProjection, occlusionFiltering, onProjectedCount, projectionOpacity, projectionScope, projectionSelection,
    projectionSize, selectedPointIndices, showProjection, stableOnly]);

  useEffect(() => {
    const canvas = selectedProjectionCanvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, frame.width, frame.height);
    const projectedByIndex = projectedPointByIndexRef.current;
    const outerSize = Math.max(projectionSize * 1.9, projectionSize + 3);
    const innerSize = Math.max(projectionSize * 1.05, 2);
    context.globalAlpha = Math.max(0.82, projectionOpacity);
    context.fillStyle = "#050a09";
    (highlightSelectedProjection ? selectedPointIndices : []).forEach((index) => {
      const point = projectedByIndex[index];
      if (point) context.fillRect(point.x - outerSize * 0.5, point.y - outerSize * 0.5, outerSize, outerSize);
    });
    context.fillStyle = "#ffffff";
    (highlightSelectedProjection ? selectedPointIndices : []).forEach((index) => {
      const point = projectedByIndex[index];
      if (point) context.fillRect(point.x - innerSize * 0.5, point.y - innerSize * 0.5, innerSize, innerSize);
    });
    if (diagnosticHighlight) {
      context.fillStyle = `#${diagnosticHighlight.color.toString(16).padStart(6, "0")}`;
      diagnosticHighlight.indices.forEach((index) => {
        const point = projectedByIndex[index];
        if (!point) return;
        context.beginPath();
        context.arc(point.x, point.y, Math.max(3, projectionSize * 1.5), 0, Math.PI * 2);
        context.fill();
      });
    }
    context.globalAlpha = 1;
  }, [diagnosticHighlight, selectedOnly, hideGround, temporalProjection, backgroundOpacity, cloud, distortion, distortionModel, frame.height, frame.width, highlightSelectedProjection,
    imageVersion, intrinsic, matrix, projectionOpacity, projectionScope, projectionSelection, projectionSize,
    selectedPointIndices, showProjection, stableOnly]);

  useEffect(() => {
    const canvas = maskCanvasRef.current;
    if (!canvas || brushMaskRef.current) return;
    if (skipNextMaskRenderRef.current) {
      skipNextMaskRenderRef.current = false;
      return;
    }
    renderMaskCanvas(canvas, mask, maskOpacity, secondaryMask);
  }, [frame.height, frame.width, mask, maskOpacity, secondaryMask]);

  useEffect(() => {
    const canvas = interactionCanvasRef.current;
    if (!canvas) return;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, frame.width, frame.height);
    if (!draft) return;
    const x = draft.x1 * frame.width;
    const y = draft.y1 * frame.height;
    const width = (draft.x2 - draft.x1) * frame.width;
    const height = (draft.y2 - draft.y1) * frame.height;
    context.setLineDash([14, 9]);
    context.lineWidth = 7;
    context.strokeStyle = "rgba(0,0,0,.82)";
    context.strokeRect(x, y, width, height);
    context.lineWidth = 3;
    context.strokeStyle = "#ffffff";
    context.strokeRect(x, y, width, height);
    context.setLineDash([]);
    const radius = Math.max(5, 7 / Math.max(fitScale * zoom, 0.05));
    const handles = [
      [x, y], [x + width * 0.5, y], [x + width, y],
      [x, y + height * 0.5], [x + width, y + height * 0.5],
      [x, y + height], [x + width * 0.5, y + height], [x + width, y + height],
    ];
    for (const [handleX, handleY] of handles) {
      context.beginPath();
      context.arc(handleX, handleY, radius, 0, Math.PI * 2);
      context.fillStyle = "rgba(0,0,0,.86)";
      context.fill();
      context.beginPath();
      context.arc(handleX, handleY, radius * 0.58, 0, Math.PI * 2);
      context.fillStyle = "#ffffff";
      context.fill();
    }
  }, [draft, fitScale, frame.height, frame.width, zoom]);

  const canvasPosition = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(frame.width, (event.clientX - rect.left) * frame.width / rect.width)),
      y: Math.max(0, Math.min(frame.height, (event.clientY - rect.top) * frame.height / rect.height)),
    };
  };

  const pickProjectedPoint = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (!projectionPicking || !onProjectedPointPick) return false;
    if (event.button !== 0 && event.pointerType === "mouse") return true;
    event.preventDefault();
    const rect = event.currentTarget.getBoundingClientRect();
    const position = canvasPosition(event);
    const intrinsicRadius = Math.max(
      projectionSize * 1.5,
      10 * frame.width / Math.max(rect.width, 1),
    );
    const radiusSquared = intrinsicRadius * intrinsicRadius;
    let nearest: ProjectedPoint | null = null;
    let nearestDistance = radiusSquared;
    for (const point of projectedPointsRef.current) {
      const distance = (point.x - position.x) ** 2 + (point.y - position.y) ** 2;
      if (distance > nearestDistance) continue;
      if (distance < nearestDistance - 0.01 || !nearest || point.depth < nearest.depth) {
        nearest = point;
        nearestDistance = distance;
      }
    }
    onProjectedPointPick(nearest?.index ?? null);
    return true;
  };

  const makeBlankMask = () => {
    const width = frame.width;
    const height = frame.height;
    return { width, height, data: new Uint8Array(width * height) };
  };

  const paintAt = (position: { x: number; y: number }) => {
    const target = brushMaskRef.current;
    if (!target || !segmentationTool) return;
    const point = {
      x: Math.max(0, Math.min(target.width - Number.EPSILON, position.x / frame.width * target.width)),
      y: Math.max(0, Math.min(target.height - Number.EPSILON, position.y / frame.height * target.height)),
    };
    const last = lastBrushPointRef.current ?? point;
    const dirty = paintMaskStroke(
      target,
      last,
      point,
      brushSize,
      segmentationTool === "brush-remove" ? 0 : 1,
    );
    lastBrushPointRef.current = point;
    if (dirty && maskCanvasRef.current) renderMaskCanvasRegion(maskCanvasRef.current, target, dirty, maskOpacity);
    if (dirty) {
      latestMaskRef.current = target;
      brushDirtyRectsRef.current.push(dirty);
      brushPixelDeltaRef.current += dirty.changedPixels * (segmentationTool === "brush-remove" ? -1 : 1);
    }
  };

  const releasePointer = (target: HTMLCanvasElement, pointerId: number) => {
    if (activePointerIdRef.current === pointerId) activePointerIdRef.current = null;
    if (target.hasPointerCapture(pointerId)) target.releasePointerCapture(pointerId);
  };

  const commitBrush = (publish = true) => {
    const editedMask = brushMaskRef.current;
    if (editedMask) {
      latestMaskRef.current = editedMask;
      if (publish && brushDirtyRectsRef.current.length) {
        skipNextMaskRenderRef.current = true;
        onMaskChangeRef.current?.(editedMask, {
          dirtyRects: brushDirtyRectsRef.current,
          pixelDelta: brushPixelDeltaRef.current,
        });
      }
    }
    brushMaskRef.current = null;
    brushDirtyRectsRef.current = [];
    brushPixelDeltaRef.current = 0;
    lastBrushPointRef.current = null;
  };

  const resetInteraction = (commit: boolean) => {
    if (commit) commitBrush(true);
    else {
      brushMaskRef.current = null;
      brushDirtyRectsRef.current = [];
      brushPixelDeltaRef.current = 0;
      lastBrushPointRef.current = null;
    }
    dragStartRef.current = null;
    rectDragRef.current = null;
    updateDraft(null);
  };

  const rectDragModeAt = (position: { x: number; y: number }, value: NormalizedRect, canvas: HTMLCanvasElement): RectDragMode => {
    const bounds = canvas.getBoundingClientRect();
    const thresholdX = 12 * frame.width / Math.max(bounds.width, 1);
    const thresholdY = 12 * frame.height / Math.max(bounds.height, 1);
    const left = value.x1 * frame.width;
    const right = value.x2 * frame.width;
    const top = value.y1 * frame.height;
    const bottom = value.y2 * frame.height;
    const nearLeft = Math.abs(position.x - left) <= thresholdX;
    const nearRight = Math.abs(position.x - right) <= thresholdX;
    const nearTop = Math.abs(position.y - top) <= thresholdY;
    const nearBottom = Math.abs(position.y - bottom) <= thresholdY;
    if (nearLeft && nearTop) return "top-left";
    if (nearRight && nearTop) return "top-right";
    if (nearLeft && nearBottom) return "bottom-left";
    if (nearRight && nearBottom) return "bottom-right";
    if (nearLeft && position.y >= top - thresholdY && position.y <= bottom + thresholdY) return "left";
    if (nearRight && position.y >= top - thresholdY && position.y <= bottom + thresholdY) return "right";
    if (nearTop && position.x >= left - thresholdX && position.x <= right + thresholdX) return "top";
    if (nearBottom && position.x >= left - thresholdX && position.x <= right + thresholdX) return "bottom";
    if (position.x >= left && position.x <= right && position.y >= top && position.y <= bottom) return "move";
    return "new";
  };

  const adjustedDraft = (drag: NonNullable<typeof rectDragRef.current>, position: { x: number; y: number }): NormalizedRect => {
    if (drag.mode === "new") return {
      x1: Math.min(drag.start.x, position.x) / frame.width,
      y1: Math.min(drag.start.y, position.y) / frame.height,
      x2: Math.max(drag.start.x, position.x) / frame.width,
      y2: Math.max(drag.start.y, position.y) / frame.height,
    };
    const minimumX = 3 / frame.width;
    const minimumY = 3 / frame.height;
    const pointX = Math.max(0, Math.min(1, position.x / frame.width));
    const pointY = Math.max(0, Math.min(1, position.y / frame.height));
    if (drag.mode === "move") {
      const width = drag.initial.x2 - drag.initial.x1;
      const height = drag.initial.y2 - drag.initial.y1;
      const x1 = Math.max(0, Math.min(1 - width, drag.initial.x1 + (position.x - drag.start.x) / frame.width));
      const y1 = Math.max(0, Math.min(1 - height, drag.initial.y1 + (position.y - drag.start.y) / frame.height));
      return { x1, y1, x2: x1 + width, y2: y1 + height };
    }
    const next = { ...drag.initial };
    if (drag.mode.includes("left")) next.x1 = Math.min(pointX, next.x2 - minimumX);
    if (drag.mode.includes("right")) next.x2 = Math.max(pointX, next.x1 + minimumX);
    if (drag.mode.includes("top")) next.y1 = Math.min(pointY, next.y2 - minimumY);
    if (drag.mode.includes("bottom")) next.y2 = Math.max(pointY, next.y1 + minimumY);
    return next;
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const shouldPan = Boolean(onZoomChange) && (segmentationTool === "pan" || (event.pointerType === "mouse" && event.button === 2));
    if (shouldPan) {
      event.preventDefault();
      event.stopPropagation();
      if (event.pointerType === "mouse" && event.button === 2) onSegmentationToolChange?.("pan");
      if (activePointerIdRef.current !== null) resetInteraction(true);
      const viewport = scrollRef.current;
      if (!viewport) return;
      activePointerIdRef.current = event.pointerId;
      panDragRef.current = { pointerId: event.pointerId, clientX: event.clientX, clientY: event.clientY, scrollLeft: viewport.scrollLeft, scrollTop: viewport.scrollTop };
      setPanning(true);
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }
    if (selectable) event.currentTarget.focus({ preventScroll: true });
    if (pickProjectedPoint(event)) return;
    if (!selectable || !segmentationTool) return;
    if (event.button !== 0 && event.pointerType === "mouse") return;
    event.preventDefault();
    if (segmenting && !segmentationTool.startsWith("brush")) {
      setSegmentationError("上一次 SAM2 请求仍在处理中，请稍候再框选");
      return;
    }
    if (activePointerIdRef.current !== null) resetInteraction(true);
    activePointerIdRef.current = event.pointerId;
    const position = canvasPosition(event);
    if (segmentationTool.startsWith("brush")) {
      const editableMask = latestMaskRef.current;
      setSegmentationError(null);
      setCandidates([]);
      candidateBaseMaskRef.current = null;
      onMaskEditStartRef.current?.("brush");
      brushMaskRef.current = editableMask
        ? { width: editableMask.width, height: editableMask.height, data: editableMask.data.slice() }
        : makeBlankMask();
      if (maskCanvasRef.current && !editableMask) {
        maskCanvasRef.current.getContext("2d")?.clearRect(0, 0, frame.width, frame.height);
      }
      lastBrushPointRef.current = null;
      brushDirtyRectsRef.current = [];
      brushPixelDeltaRef.current = 0;
      event.currentTarget.setPointerCapture(event.pointerId);
      paintAt(position);
      return;
    }
    event.currentTarget.setPointerCapture(event.pointerId);
    dragStartRef.current = position;
    const existingDraft = draftRef.current;
    const mode = existingDraft ? rectDragModeAt(position, existingDraft, event.currentTarget) : "new";
    setRectCursor(cursorForRectDragMode(mode));
    const initial = existingDraft ?? {
      x1: position.x / frame.width, y1: position.y / frame.height,
      x2: position.x / frame.width, y2: position.y / frame.height,
    };
    rectDragRef.current = { mode, start: position, initial };
    setSegmentationError(null);
    if (mode === "new") updateDraft(initial);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const pan = panDragRef.current;
    if (pan?.pointerId === event.pointerId) {
      event.preventDefault();
      const viewport = scrollRef.current;
      if (viewport) {
        viewport.scrollLeft = pan.scrollLeft - (event.clientX - pan.clientX);
        viewport.scrollTop = pan.scrollTop - (event.clientY - pan.clientY);
      }
      return;
    }
    if (!selectable || !segmentationTool) return;
    if (segmentationTool.startsWith("brush")) {
      const rect = event.currentTarget.getBoundingClientRect();
      scheduleBrushCursor({ x: event.clientX - rect.left, y: event.clientY - rect.top });
    }
    if (activePointerIdRef.current !== event.pointerId) {
      if (!segmentationTool.startsWith("brush") && draftRef.current) {
        setRectCursor(cursorForRectDragMode(rectDragModeAt(canvasPosition(event), draftRef.current, event.currentTarget)));
      } else if (!segmentationTool.startsWith("brush")) setRectCursor("crosshair");
      return;
    }
    if (brushMaskRef.current && event.pointerType === "mouse" && event.buttons === 0) {
      activePointerIdRef.current = null;
      commitBrush(true);
      return;
    }
    const current = canvasPosition(event);
    if (brushMaskRef.current) {
      paintAt(current);
      return;
    }
    const rectDrag = rectDragRef.current;
    if (!dragStartRef.current || !rectDrag) {
      setSegmentationError("没有识别到框选起点，请重新按住左键拖动");
      return;
    }
    updateDraft(adjustedDraft(rectDrag, current));
  };

  const generateMaskFromDraft = async (rect: NormalizedRect) => {
    if (!segmentationTool || segmentationTool === "pan" || segmentationTool.startsWith("brush") || segmenting) return;
    const width = (rect.x2 - rect.x1) * frame.width;
    const height = (rect.y2 - rect.y1) * frame.height;
    if (width < 3 || height < 3) {
      setSegmentationError(`框选区域过小（${Math.round(width)}×${Math.round(height)} 原图像素），请框得稍大一些`);
      return;
    }
    if (!imageRef.current) {
      setSegmentationError("原图尚未加载完成，请稍候再按回车");
      return;
    }
    setSegmenting(true);
    setSegmentationError(null);
    setCandidates([]);
    candidateBaseMaskRef.current = latestMaskRef.current;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 45000);
    try {
      const response = await fetch("/api/segment/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_url: frame.url, box: rect, granularity }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `SAM2 请求失败：${response.status}`);
      if (!Array.isArray(payload.candidates)) throw new Error("SAM2 返回格式异常，请重试");
      const decoded = (payload.candidates as Array<{ width: number; height: number; area: number; score: number; rle: number[] }>).map(
        (candidate) => {
          const normalized = smoothSamMask(normalizeSamObjectMask(
            decodeMaskRle(candidate.width, candidate.height, candidate.rle),
            rect,
          ));
          return { mask: normalized, score: candidate.score, area: countMask(normalized) };
        },
      );
      if (!decoded.length) throw new Error("SAM2 没有返回候选，请扩大框选范围后重试");
      const mergeMode = segmentationTool === "box-add" ? "add" : "replace";
      setCandidateIndex(0);
      setCandidateMergeMode(mergeMode);
      setCandidates(decoded);
      updateDraft(null);
      if (decoded[0]) {
        onMaskEditStartRef.current?.("sam");
        const applied = preserveMaskLayers
          ? decoded[0].mask
          : mergeImageMasks(candidateBaseMaskRef.current, decoded[0].mask, mergeMode);
        latestMaskRef.current = applied;
        onMaskChangeRef.current?.(applied);
      }
    } catch (reason) {
      setSegmentationError(reason instanceof DOMException && reason.name === "AbortError"
        ? "SAM2 请求超过 45 秒，已取消；可调整框后重新按回车"
        : reason instanceof Error ? reason.message : "SAM2 服务不可用");
    } finally {
      window.clearTimeout(timeout);
      setSegmenting(false);
    }
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (panDragRef.current?.pointerId === event.pointerId) {
      releasePointer(event.currentTarget, event.pointerId);
      panDragRef.current = null;
      setPanning(false);
      return;
    }
    if (!selectable || !segmentationTool) return;
    if (activePointerIdRef.current !== event.pointerId) return;
    releasePointer(event.currentTarget, event.pointerId);
    if (brushMaskRef.current) {
      commitBrush(true);
      return;
    }
    if (!dragStartRef.current || !rectDragRef.current) return;
    const rect = adjustedDraft(rectDragRef.current, canvasPosition(event));
    dragStartRef.current = null;
    rectDragRef.current = null;
    setRectCursor(cursorForRectDragMode(rectDragModeAt(canvasPosition(event), rect, event.currentTarget)));
    updateDraft(rect);
    const width = (rect.x2 - rect.x1) * frame.width;
    const height = (rect.y2 - rect.y1) * frame.height;
    if (width < 3 || height < 3) {
      setSegmentationError(`框选区域过小（${Math.round(width)}×${Math.round(height)} 原图像素），请框得稍大一些`);
      return;
    }
    setSegmentationError(null);
  };

  const handlePointerCancel = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (panDragRef.current?.pointerId === event.pointerId) {
      releasePointer(event.currentTarget, event.pointerId);
      panDragRef.current = null;
      setPanning(false);
      return;
    }
    const hadBox = Boolean(dragStartRef.current && !brushMaskRef.current);
    releasePointer(event.currentTarget, event.pointerId);
    resetInteraction(false);
    if (hadBox) setSegmentationError("框选被浏览器中断，请重新框选");
  };

  const handleLostPointerCapture = (event: React.PointerEvent<HTMLCanvasElement>) => {
    if (activePointerIdRef.current !== event.pointerId) return;
    if (panDragRef.current?.pointerId === event.pointerId) {
      activePointerIdRef.current = null;
      panDragRef.current = null;
      setPanning(false);
      return;
    }
    const hadBox = Boolean(dragStartRef.current && !brushMaskRef.current);
    activePointerIdRef.current = null;
    resetInteraction(false);
    if (hadBox) setSegmentationError("框选结束事件丢失，请重新框选");
  };

  useEffect(() => {
    if (segmentationTool?.startsWith("brush")) updateDraft(null);
  }, [segmentationTool]);

  useEffect(() => {
    const handleDraftKey = (event: KeyboardEvent) => {
      if (!selectable || !draftRef.current || segmentationTool === "pan" || segmentationTool?.startsWith("brush")) return;
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ||
          target instanceof HTMLSelectElement || (target instanceof HTMLElement && target.isContentEditable)) return;
      if (event.key === "Escape") {
        event.preventDefault();
        updateDraft(null);
        setSegmentationError(null);
        return;
      }
      if (event.key !== "Enter" || segmenting) return;
      event.preventDefault();
      void generateMaskFromDraft(draftRef.current);
    };
    window.addEventListener("keydown", handleDraftKey);
    return () => window.removeEventListener("keydown", handleDraftKey);
  }, [frame.url, granularity, preserveMaskLayers, segmenting, segmentationTool, selectable]);

  const hideBrushCursor = () => scheduleBrushCursor(null);

  useEffect(() => {
    const finishBrush = (pointerId?: number) => {
      if (!brushMaskRef.current) return;
      if (pointerId !== undefined && activePointerIdRef.current !== pointerId) return;
      activePointerIdRef.current = null;
      commitBrush(true);
    };
    const handleWindowPointerUp = (event: PointerEvent) => finishBrush(event.pointerId);
    const handleWindowPointerCancel = (event: PointerEvent) => finishBrush(event.pointerId);
    const handleWindowMouseUp = () => finishBrush();
    const handleWindowBlur = () => finishBrush();
    const handleVisibilityChange = () => {
      if (document.visibilityState !== "visible") finishBrush();
    };
    window.addEventListener("pointerup", handleWindowPointerUp);
    window.addEventListener("pointercancel", handleWindowPointerCancel);
    window.addEventListener("mouseup", handleWindowMouseUp);
    window.addEventListener("blur", handleWindowBlur);
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      window.removeEventListener("pointerup", handleWindowPointerUp);
      window.removeEventListener("pointercancel", handleWindowPointerCancel);
      window.removeEventListener("mouseup", handleWindowMouseUp);
      window.removeEventListener("blur", handleWindowBlur);
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  const handleWheel = (event: Pick<WheelEvent, "clientX" | "clientY" | "deltaY">, steps = event.deltaY < 0 ? 1 : -1) => {
    const synchronized = Boolean(synchronizedView && onSynchronizedViewChange);
    if (!synchronized && (!selectable || (!localZoom && !onZoomChange))) return;
    const viewport = scrollRef.current;
    const interactionCanvas = interactionCanvasRef.current;
    const shell = interactionCanvas?.parentElement;
    if (!viewport || !shell) return;
    const rect = viewport.getBoundingClientRect();
    const pointerX = event.clientX - rect.left;
    const pointerY = event.clientY - rect.top;
    const shellRect = shell.getBoundingClientRect();
    if (!shellRect.width || !shellRect.height) return;
    const imageX = (event.clientX - shellRect.left) * frame.width / shellRect.width;
    const imageY = (event.clientY - shellRect.top) * frame.height / shellRect.height;
    const maxZoom = Math.max(1, 24 / fitScale);
    const nextZoom = Math.max(1, Math.min(maxZoom, zoom * 1.22 ** steps));
    if (Math.abs(nextZoom - zoom) < 0.001) return;
    if (synchronized && synchronizedView && onSynchronizedViewChange) {
      const nextScale = fitScale * nextZoom;
      const renderedWidth = frame.width * nextScale;
      const renderedHeight = frame.height * nextScale;
      onSynchronizedViewChange({
        zoom: nextZoom,
        centerX: renderedWidth <= viewport.clientWidth
          ? 0.5
          : Math.max(0, Math.min(1, (imageX + (viewport.clientWidth * 0.5 - pointerX) / nextScale) / frame.width)),
        centerY: renderedHeight <= viewport.clientHeight
          ? 0.5
          : Math.max(0, Math.min(1, (imageY + (viewport.clientHeight * 0.5 - pointerY) / nextScale) / frame.height)),
      });
      return;
    }
    pendingZoomAnchorRef.current = { frameUrl: frame.url, imageX, imageY, pointerX, pointerY };
    if (localZoom) setLocalZoomValue(nextZoom);
    else onZoomChange?.(nextZoom);
  };

  // Read the latest committed view when the coalesced wheel callback runs.
  const wheelHandlerRef = useRef(handleWheel);
  useLayoutEffect(() => { wheelHandlerRef.current = handleWheel; });
  useEffect(() => {
    const viewport = scrollRef.current;
    if (!viewport) return;
    let animation = 0;
    let steps = 0;
    let pointer = { clientX: 0, clientY: 0, deltaY: 0 };
    const onNativeWheel = (event: WheelEvent) => {
      if (!(synchronizedView && onSynchronizedViewChange) && (!selectable || (!localZoom && !onZoomChange))) return;
      event.preventDefault();
      event.stopPropagation();
      if (!event.deltaY) return;
      steps += event.deltaY < 0 ? 1 : -1;
      pointer = { clientX: event.clientX, clientY: event.clientY, deltaY: event.deltaY };
      if (animation) return;
      animation = requestAnimationFrame(() => {
        animation = 0;
        const pending = steps;
        steps = 0;
        if (pending) wheelHandlerRef.current(pointer, pending);
      });
    };
    viewport.addEventListener("wheel", onNativeWheel, { passive: false });
    return () => {
      viewport.removeEventListener("wheel", onNativeWheel);
      cancelAnimationFrame(animation);
    };
  }, [frame.url, localZoom, onSynchronizedViewChange, onZoomChange, selectable, Boolean(synchronizedView), zoomResetToken]);

  const handleSynchronizedScroll = () => {
    const viewport = scrollRef.current;
    const shell = interactionCanvasRef.current?.parentElement;
    if (!viewport || !shell || !synchronizedView || !onSynchronizedViewChange) return;
    const target = synchronizedScrollTargetRef.current;
    if (target && Math.abs(target.left - viewport.scrollLeft) < 1 && Math.abs(target.top - viewport.scrollTop) < 1) return;
    synchronizedScrollTargetRef.current = null;
    const scale = fitScale * synchronizedView.zoom;
    if (scale <= 0) return;
    const renderedWidth = frame.width * scale;
    const renderedHeight = frame.height * scale;
    const nextView = {
      zoom: synchronizedView.zoom,
      centerX: renderedWidth <= viewport.clientWidth
        ? 0.5
        : Math.max(0, Math.min(1, (viewport.scrollLeft + viewport.clientWidth * 0.5) / renderedWidth)),
      centerY: renderedHeight <= viewport.clientHeight
        ? 0.5
        : Math.max(0, Math.min(1, (viewport.scrollTop + viewport.clientHeight * 0.5) / renderedHeight)),
    };
    if (Math.abs(nextView.centerX - synchronizedView.centerX) < 0.00005 &&
        Math.abs(nextView.centerY - synchronizedView.centerY) < 0.00005) return;
    onSynchronizedViewChange(nextView);
  };

  const selectCandidate = (index: number) => {
    setCandidateIndex(index);
    const candidate = candidates[index];
    if (!candidate) return;
    const applied = preserveMaskLayers
      ? candidate.mask
      : mergeImageMasks(candidateBaseMaskRef.current, candidate.mask, candidateMergeMode);
    latestMaskRef.current = applied;
    onMaskChangeRef.current?.(applied);
  };

  const closeCandidates = () => {
    candidateBaseMaskRef.current = null;
    setCandidates([]);
  };

  const acceptCandidate = () => {
    const candidate = candidates[candidateIndex];
    if (!candidate) return;
    selectCandidate(candidateIndex);
    closeCandidates();
  };

  return (
    <div className="image-canvas-viewport" ref={fitViewportRef}>
    <div className="image-canvas-scroll" ref={scrollRef} onScroll={handleSynchronizedScroll}>
    <div className="image-canvas-shell" style={{ width: `${frame.width * fitScale * zoom}px`, height: `${frame.height * fitScale * zoom}px` }}>
      {!selectedOnly && <img
        className="image-canvas image-canvas-photograph"
        src={frame.url}
        width={frame.width}
        height={frame.height}
        style={{ opacity: backgroundOpacity }}
        alt=""
        aria-hidden="true"
        draggable={false}
      />}
      <canvas
        ref={baseCanvasRef}
        className="image-canvas image-canvas-base"
        width={frame.width}
        height={frame.height}
        aria-hidden="true"
      />
      <canvas
        ref={maskCanvasRef}
        className="image-canvas image-canvas-mask"
        width={frame.width}
        height={frame.height}
        aria-hidden="true"
      />
      <canvas
        ref={selectedProjectionCanvasRef}
        className="image-canvas image-canvas-selected-projection"
        width={frame.width}
        height={frame.height}
        aria-hidden="true"
      />
      <canvas
        ref={interactionCanvasRef}
        className={selectable
          ? `image-canvas image-canvas-interaction selectable ${segmentationTool === "pan" ? "pan-tool" : segmentationTool?.startsWith("brush") ? "brush-tool" : "box-tool"}`
          : `image-canvas image-canvas-interaction ${projectionPicking ? "projection-picker" : ""}`}
        style={segmentationTool === "pan" ? { cursor: panning ? "grabbing" : "grab" } : selectable && !segmentationTool?.startsWith("brush") ? { cursor: rectCursor } : undefined}
        width={frame.width}
        height={frame.height}
        tabIndex={selectable ? 0 : -1}
        aria-label={showProjection || highlightSelectedProjection ? "雷达投影与分割结果" : "图像分割视图"}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
        onPointerCancel={handlePointerCancel}
        onLostPointerCapture={handleLostPointerCapture}
        onContextMenu={(event) => {
          if (!onZoomChange) return;
          event.preventDefault();
          onSegmentationToolChange?.("pan");
        }}
        onPointerEnter={(event) => {
          if (segmentationTool === "pan") return;
          if (!segmentationTool?.startsWith("brush")) {
            if (draftRef.current) setRectCursor(cursorForRectDragMode(rectDragModeAt(canvasPosition(event), draftRef.current, event.currentTarget)));
            return;
          }
          const rect = event.currentTarget.getBoundingClientRect();
          scheduleBrushCursor({ x: event.clientX - rect.left, y: event.clientY - rect.top });
        }}
        onPointerLeave={() => { hideBrushCursor(); if (!rectDragRef.current) setRectCursor("crosshair"); }}
      />
      {selectable && segmentationTool?.startsWith("brush") && brushCursor && <div
        className={`brush-outline ${brushSize * fitScale * zoom < 6 ? "tiny" : ""}`}
        style={{
          left: brushCursor.x,
          top: brushCursor.y,
          width: `${Math.max(brushSize * fitScale * zoom, 1)}px`,
          height: `${Math.max(brushSize * fitScale * zoom, 1)}px`,
        }}
      />}
      {selectable && segmentationTool?.startsWith("brush") && <div className="brush-mode-badge">
        {segmentationTool === "brush-remove" ? "擦除" : "添加"} · {brushSize}px · {fitScale * zoom >= 1 ? `${(fitScale * zoom).toFixed(1)} 屏幕像素 / 原图像素` : "适应窗口"}
      </div>}
      {selectable && draft && segmentationTool !== "pan" && !segmentationTool?.startsWith("brush") && <div className="sam-draft-status">
        <span>拖动边缘、角点或框内微调</span><b>Enter 生成 Mask</b><small>Esc 取消</small>
      </div>}
      {selectable && segmenting && <div className="sam-status">SAM2 正在生成候选…</div>}
      {selectable && segmentationError && <div className="sam-status error">{segmentationError}</div>}
      {projectionPicking && <div className="projection-pick-badge">点击雷达投影点 · 自动补全三维连通物体</div>}
    </div>
    </div>
      {selectable && candidates.length > 0 && !segmentationTool?.startsWith("brush") && <div className="candidate-bar">
        <span>SAM2 已自动应用 · 可切换候选</span>
        {candidates.map((candidate, index) => (
          <button key={index} className={candidateIndex === index ? "active" : ""} onClick={() => selectCandidate(index)}>
            {(["精细", "均衡", "粗略"] as const)[index] ?? `候选 ${index + 1}`}
          </button>
        ))}
        <button className="accept" onClick={acceptCandidate}>完成</button>
      </div>}
    </div>
  );
}
