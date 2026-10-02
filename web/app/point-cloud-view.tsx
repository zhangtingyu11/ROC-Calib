"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { filterForegroundScreenPoints } from "./segmentation";
import { EMPTY_CLOUD_OVERLAYS, type CloudData, type PointInteractionMode, type ViewRequest, type FitMode, type RectDragMode, isGroundPoint, estimateDisplayGroundPlane, cursorForRectDragMode, activeChordPanViewport } from './workspace-core';

export function PointCloudViewport({
  clouds,
  frameIndex,
  stableOnly,
  hideGround,
  pointSize,
  brushSize,
  brushDepthTolerance,
  interactionMode,
  selectedIndices,
  selectedOnly = false,
  structureLine = null,
  inspectionLines = [],
  inspectionFocus = 0,
  overlayClouds = EMPTY_CLOUD_OVERLAYS,
  overlayOpacity = .5,
  clearSelectionToken,
  viewRequest,
  onSelection,
  onPointEdit,
  onInteractionModeChange,
  preserveViewOnCloudChange = false,
  gridReferenceCloud = null,
  comparisonMode = false,
  comparisonVisibleLayer = "both",
  selectedPointColor = 0xffffff,
  uniformPointColor,
}: {
  clouds: Array<CloudData | null>;
  frameIndex: number;
  stableOnly: boolean;
  hideGround: boolean;
  pointSize: number;
  brushSize: number;
  brushDepthTolerance: number;
  interactionMode: PointInteractionMode;
  selectedIndices: Uint32Array;
  selectedOnly?: boolean;
  structureLine?: number[][] | null;
  inspectionLines?: number[][][];
  inspectionFocus?: number;
  overlayClouds?: Array<{ cloud: CloudData; color: number; size?:number }>;
  overlayOpacity?: number;
  clearSelectionToken: number;
  viewRequest: ViewRequest;
  onSelection: (indices: Uint32Array) => void;
  onPointEdit: (indices: Uint32Array, mode: "add" | "remove") => void;
  onInteractionModeChange: (mode: PointInteractionMode) => void;
  preserveViewOnCloudChange?: boolean;
  gridReferenceCloud?: CloudData | null;
  comparisonMode?: boolean;
  comparisonVisibleLayer?: "both" | "source" | "target";
  selectedPointColor?: number;
  uniformPointColor?: number;
}) {
  const invalidateRenderRef = useRef<() => void>(() => {});
  useEffect(() => { invalidateRenderRef.current(); });
  const hostRef = useRef<HTMLDivElement>(null);
  const [graphicsError, setGraphicsError] = useState("");
  const visibleSelection = useMemo(() => new Set(selectedIndices), [selectedIndices]);
  const cloudGroupRef = useRef<THREE.Group | null>(null);
  const selectionGroupRef = useRef<THREE.Group | null>(null);
  const selectionPointsRef = useRef<THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> | null>(null);
  const overlayPointsRef = useRef<Array<THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial>>>([]);
  const selectionSlotByIndexRef = useRef<Map<number, number>>(new Map());
  const selectionIndexBySlotRef = useRef<number[]>([]);
  const selectionVisibleCountRef = useRef(0);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const gridRef = useRef<THREE.GridHelper | null>(null);
  const navigationBaseDistanceRef = useRef(1);
  const navigationFocusDistanceRef = useRef(1);
  const currentCloudRef = useRef<CloudData | null>(null);
  const fitViewRef = useRef<(mode: FitMode) => void>(() => undefined);
  const fittedCloudsRef = useRef<Array<CloudData | null> | null>(null);
  const handledViewTokenRef = useRef(viewRequest.token);
  const dragStartRef = useRef<{ x: number; y: number } | null>(null);
  const boxDragRef = useRef<{
    mode: RectDragMode;
    start: { x: number; y: number };
    initial: { x: number; y: number; width: number; height: number };
  } | null>(null);
  const dragRectRef = useRef<{ x: number; y: number; width: number; height: number } | null>(null);
  const brushScreenIndexRef = useRef<{
    cellSize: number;
    cells: Map<string, Array<{ index: number; x: number; y: number; depth: number }>>;
  } | null>(null);
  const lastBrushApplyRef = useRef<{ x: number; y: number } | null>(null);
  const pendingBrushPointIndicesRef = useRef<Set<number>>(new Set());
  const chordPanRef = useRef(false);
  const chordPanPositionRef = useRef<{ x: number; y: number } | null>(null);
  const controlsResumeFrameRef = useRef<number | null>(null);
  const chordDampingRef = useRef<boolean | null>(null);
  const chordAwaitReleaseRef = useRef(false);
  const suppressChordContextMenuRef = useRef(false);
  const [dragRect, setDragRect] = useState<{ x: number; y: number; width: number; height: number } | null>(null);
  const [boxCursor, setBoxCursor] = useState<React.CSSProperties["cursor"]>("crosshair");
  const [brushCursor, setBrushCursor] = useState<{ x: number; y: number } | null>(null);
  const [chordPanning, setChordPanning] = useState(false);

  const clearSelection = () => {
    const group = selectionGroupRef.current;
    if (group) {
      for (const child of [...group.children]) {
        group.remove(child);
        if (child instanceof THREE.Points) {
          child.geometry.dispose();
          (child.material as THREE.Material).dispose();
        }
      }
    }
    selectionPointsRef.current = null;
    overlayPointsRef.current = [];
    selectionSlotByIndexRef.current.clear();
    selectionIndexBySlotRef.current = [];
    selectionVisibleCountRef.current = 0;
  };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x07100f);
    scene.fog = new THREE.FogExp2(0x07100f, 0.006);
    const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 1200);
    camera.up.set(0, 0, 1);
    let renderer: THREE.WebGLRenderer;
    try { renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" }); }
    catch { setGraphicsError("无法启动三维视图，请检查浏览器硬件加速或更换浏览器。"); return; }
    const lost = (event: Event) => { event.preventDefault(); setGraphicsError("三维视图暂时不可用，正在等待显卡恢复。请先保存标注。"); };
    const restored = () => { setGraphicsError(""); invalidateRenderRef.current(); };
    renderer.domElement.addEventListener("webglcontextlost", lost);
    renderer.domElement.addEventListener("webglcontextrestored", restored);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.screenSpacePanning = true;
    // Wheel navigation is implemented below as cursor-directed camera travel.
    // Disable OrbitControls' radius-based dolly, which cannot pass its target.
    controls.enableZoom = false;
    const grid = new THREE.GridHelper(160, 80, 0x31534c, 0x142825);
    grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1));
    grid.position.z = -1.6;
    scene.add(grid);
    scene.add(new THREE.AxesHelper(3));
    const cloudGroup = new THREE.Group();
    const selectionGroup = new THREE.Group();
    scene.add(cloudGroup, selectionGroup);
    cloudGroupRef.current = cloudGroup;
    selectionGroupRef.current = selectionGroup;
    cameraRef.current = camera;
    controlsRef.current = controls;
    gridRef.current = grid;

    const resize = () => {
      const width = host.clientWidth;
      const height = host.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / Math.max(height, 1);
      camera.updateProjectionMatrix();
      invalidateRenderRef.current();
    };
    const observer = new ResizeObserver(resize);
    observer.observe(host);
    resize();
    let animation = 0;
    const render = () => {
      animation = 0;
      const currentDistance = Math.max(camera.position.distanceTo(controls.target), 0.01);
      // Wheel travel moves camera and orbit target together, so their mutual
      // distance no longer describes the depth of the cloud under the cursor.
      // Scale OrbitControls panning by the actual focus depth and add a modest
      // screen-space gain so a half-screen drag produces a useful translation.
      const focusDepthRatio = navigationFocusDistanceRef.current / currentDistance;
      controls.panSpeed = Math.min(20, Math.max(.35, focusDepthRatio * 1.8));
      controls.update();
      renderer.render(scene, camera);
    };
    const invalidate = () => {
      if (!animation) animation = requestAnimationFrame(render);
    };
    invalidateRenderRef.current = invalidate;
    controls.addEventListener("change", invalidate);
    invalidate();

    return () => {
      cancelAnimationFrame(animation);
      invalidateRenderRef.current = () => {};
      controls.removeEventListener("change", invalidate);
      observer.disconnect();
      controls.dispose();
      scene.traverse((object) => {
        if (object instanceof THREE.Points) {
          object.geometry.dispose();
          (object.material as THREE.Material).dispose();
        }
      });
      renderer.domElement.removeEventListener("webglcontextlost", lost);
      renderer.domElement.removeEventListener("webglcontextrestored", restored);
      renderer.dispose();
      renderer.domElement.remove();
      cloudGroupRef.current = null;
      selectionGroupRef.current = null;
      cameraRef.current = null;
      controlsRef.current = null;
      gridRef.current = null;
      fittedCloudsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const controls = controlsRef.current;
    const host = hostRef.current;
    if (!controls || !host) return;
    controls.enabled = interactionMode === "navigate" || interactionMode === "pan";
    controls.mouseButtons.LEFT = interactionMode === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    const handlePointerDown = (event: PointerEvent) => {
      if (interactionMode !== "navigate") return;
      if (event.shiftKey) {
        controls.mouseButtons.LEFT = THREE.MOUSE.PAN;
        return;
      }
      if (event.button !== 0 || event.buttons !== 1) return;
      const camera = cameraRef.current;
      const cloud = currentCloudRef.current;
      if (!camera || !cloud) return;
      // Retarget along the current optical axis without moving the camera.
      // Visible points near the screen center provide the orbit depth.
      camera.updateMatrixWorld();
      const rect = host.getBoundingClientRect();
      const point = new THREE.Vector3();
      const local = new THREE.Vector3();
      let bestDistance = Number.POSITIVE_INFINITY;
      let depth = Math.max(navigationFocusDistanceRef.current, 0.01);
      for (let i = 0; i < cloud.stable.length; i += 1) {
      if (cloud.stable[i] === 2) continue;
        if (selectedOnly && !visibleSelection.has(i)) continue;
        if (stableOnly && cloud.stable[i] !== 1) continue;
        if (hideGround && isGroundPoint(cloud, i, stableOnly)) continue;
        point.fromArray(cloud.positions, i * 3);
        local.copy(point).applyMatrix4(camera.matrixWorldInverse);
        if (-local.z <= camera.near) continue;
        point.project(camera);
        if (Math.abs(point.x) > 1 || Math.abs(point.y) > 1 || point.z < -1 || point.z > 1) continue;
        const distance = (point.x * rect.width) ** 2 + (point.y * rect.height) ** 2;
        if (distance < bestDistance || (distance === bestDistance && -local.z < depth)) {
          bestDistance = distance;
          depth = -local.z;
        }
      }
      const forward = camera.getWorldDirection(new THREE.Vector3());
      controls.target.copy(camera.position).addScaledVector(forward, depth);
      navigationFocusDistanceRef.current = depth;
    };
    const restoreRotate = () => {
      if (interactionMode === "navigate") controls.mouseButtons.LEFT = THREE.MOUSE.ROTATE;
    };
    host.addEventListener("pointerdown", handlePointerDown, true);
    host.addEventListener("pointerup", restoreRotate, true);
    host.addEventListener("pointercancel", restoreRotate, true);
    host.addEventListener("lostpointercapture", restoreRotate, true);
    return () => {
      host.removeEventListener("pointerdown", handlePointerDown, true);
      host.removeEventListener("pointerup", restoreRotate, true);
      host.removeEventListener("pointercancel", restoreRotate, true);
      host.removeEventListener("lostpointercapture", restoreRotate, true);
    };
  }, [interactionMode, selectedOnly, visibleSelection, stableOnly, hideGround]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const handleTravelWheel = (event: WheelEvent) => {
      const camera = cameraRef.current;
      const controls = controlsRef.current;
      if (!camera || !controls || !currentCloudRef.current) return;
      event.preventDefault();
      event.stopImmediatePropagation();

      const rect = host.getBoundingClientRect();
      const pointerX = event.clientX - rect.left;
      const pointerY = event.clientY - rect.top;
      const cloud = currentCloudRef.current;
      const projected = new THREE.Vector3();
      const cameraSpace = new THREE.Vector3();
      const focus = new THREE.Vector3();
      let nearestScreenDistance = 30 * 30;
      let nearestDepth = Number.POSITIVE_INFINITY;
      let found = false;
      camera.updateMatrixWorld();
      for (let index = 0; index < cloud.stable.length; index += 1) {
      if (cloud.stable[index] === 2) continue;
        if (stableOnly && cloud.stable[index] !== 1) continue;
        if (hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
        projected.set(cloud.positions[index * 3], cloud.positions[index * 3 + 1], cloud.positions[index * 3 + 2]);
        cameraSpace.copy(projected).applyMatrix4(camera.matrixWorldInverse);
        const depth = -cameraSpace.z;
        if (depth <= 0) continue;
        projected.project(camera);
        if (projected.z < -1 || projected.z > 1) continue;
        const screenX = (projected.x + 1) * rect.width * 0.5;
        const screenY = (1 - projected.y) * rect.height * 0.5;
        const screenDistance = (screenX - pointerX) ** 2 + (screenY - pointerY) ** 2;
        if (screenDistance < nearestScreenDistance ||
            (screenDistance === nearestScreenDistance && depth < nearestDepth)) {
          nearestScreenDistance = screenDistance;
          nearestDepth = depth;
          focus.set(cloud.positions[index * 3], cloud.positions[index * 3 + 1], cloud.positions[index * 3 + 2]);
          found = true;
        }
      }
      if (!found) {
        const ray = new THREE.Vector3(
          pointerX / Math.max(rect.width, 1) * 2 - 1,
          1 - pointerY / Math.max(rect.height, 1) * 2,
          0.5,
        ).unproject(camera).sub(camera.position).normalize();
        focus.copy(camera.position).addScaledVector(
          ray, Math.max(camera.position.distanceTo(controls.target), 1),
        );
      }

      const towardFocus = focus.sub(camera.position);
      const focusDistance = towardFocus.length();
      if (focusDistance < 1e-6) return;
      const wheelPixels = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? event.deltaY * 16 : event.deltaY;
      const fraction = 1 - Math.exp(-Math.min(Math.abs(wheelPixels), 240) * 0.0025);
      const maxStep = Math.max(0.25, navigationBaseDistanceRef.current * 0.3);
      const signedStep = Math.min(focusDistance * fraction, maxStep) * (wheelPixels < 0 ? 1 : -1);
      const shift = towardFocus.normalize().multiplyScalar(signedStep);
      // Move camera and orbit target together: travel has no fixed-center wall.
      camera.position.add(shift);
      controls.target.add(shift);
      navigationFocusDistanceRef.current = Math.max(.01, focusDistance - signedStep);
      camera.near = 0.002;
      camera.updateProjectionMatrix();
      controls.update();
    };
    host.addEventListener("wheel", handleTravelWheel, { capture: true, passive: false });
    return () => host.removeEventListener("wheel", handleTravelWheel, { capture: true });
  }, [hideGround, stableOnly]);

  useEffect(() => {
    const group = cloudGroupRef.current;
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!group || !camera || !controls) return;
    for (const child of [...group.children]) {
      group.remove(child);
      if (child instanceof THREE.Points) {
        child.geometry.dispose();
        (child.material as THREE.Material).dispose();
      }
    }
    clearSelection();
    const cloud = clouds[frameIndex];
    currentCloudRef.current = cloud;
    const grid = gridRef.current;
    if (grid) {
      const plane = estimateDisplayGroundPlane(gridReferenceCloud ?? cloud);
      const normal = plane ? new THREE.Vector3(...plane.normal).normalize() : new THREE.Vector3(0, 0, 1);
      grid.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
      grid.position.copy(normal.multiplyScalar(plane?.offset ?? -1.6));
    }
    if (cloud) {
      const makeLayer = (stableLayer: boolean) => {
        const count = cloud.stable.reduce(
          (total, value, index) => total + (
            value !== 2 && (value === 1) === stableLayer && (!hideGround || !isGroundPoint(cloud, index, stableOnly)) ? 1 : 0
          ), 0,
        );
        const positions = new Float32Array(count * 3);
        let output = 0;
        for (let index = 0; index < cloud.stable.length; index += 1) {
      if (cloud.stable[index] === 2) continue;
          if ((cloud.stable[index] === 1) !== stableLayer) continue;
          if (hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
          positions.set(cloud.positions.subarray(index * 3, index * 3 + 3), output * 3);
          output += 1;
        }
        return positions;
      };
      if (!stableOnly) {
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute("position", new THREE.BufferAttribute(makeLayer(false), 3));
        const targetPoints = new THREE.Points(geometry, new THREE.PointsMaterial({
          color: uniformPointColor ?? (comparisonMode ? 0x648fff : 0x4ea5d9),
          size: pointSize * (uniformPointColor === undefined ? (comparisonMode ? 1.25 : .72) : 1.75),
          sizeAttenuation: true,
          transparent: true,
          opacity: uniformPointColor === undefined ? (comparisonMode ? .88 : .48) : 1,
          depthWrite: false,
          fog: false,
          blending: comparisonMode ? THREE.AdditiveBlending : THREE.NormalBlending,
        }));
        targetPoints.userData.comparisonLayer = "target";
        group.add(targetPoints);
      }
      const stableGeometry = new THREE.BufferGeometry();
      stableGeometry.setAttribute("position", new THREE.BufferAttribute(makeLayer(true), 3));
      const stablePoints = new THREE.Points(stableGeometry, new THREE.PointsMaterial({
        color: uniformPointColor ?? (comparisonMode ? 0xffb000 : 0xffd84d),
        size: pointSize * (comparisonMode ? 1.25 : 1.75),
        sizeAttenuation: true,
        transparent: true,
        opacity: comparisonMode ? .88 : 1,
        depthWrite: false,
        fog: false,
        blending: comparisonMode ? THREE.AdditiveBlending : THREE.NormalBlending,
      }));
      stablePoints.renderOrder = 2;
      stablePoints.userData.comparisonLayer = "source";
      group.add(stablePoints);
    }

    fitViewRef.current = (mode) => {
      const box = new THREE.Box3();
      const sample = new THREE.Vector3();
      const ranges: number[] = [];
      if (mode === "near") {
        clouds.forEach((item) => {
          if (!item) return;
          for (let index = 0; index < item.positions.length; index += 3) {
          if (item.stable[index / 3] === 2) continue;
            ranges.push(Math.hypot(item.positions[index], item.positions[index + 1], item.positions[index + 2]));
          }
        });
        ranges.sort((left, right) => left - right);
      }
      const nearLimit = ranges.length
        ? ranges[Math.floor((ranges.length - 1) * 0.8)]
        : Number.POSITIVE_INFINITY;
      clouds.forEach((item) => {
        if (!item) return;
        for (let index = 0; index < item.positions.length; index += 3) {
          if (item.stable[index / 3] === 2) continue;
          if (stableOnly && item.stable[index / 3] !== 1) continue;
          if (hideGround && isGroundPoint(item, index / 3, stableOnly)) continue;
          sample.set(item.positions[index], item.positions[index + 1], item.positions[index + 2]);
          if (mode === "near" && sample.length() > nearLimit) continue;
          box.expandByPoint(sample);
        }
      });
      if (box.isEmpty()) return;
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const span = Math.max(size.x, size.y, size.z, 8);
      controls.target.copy(center);
      camera.position.set(center.x + span * 0.42, center.y - span * 0.52, center.z + span * 0.3);
      navigationBaseDistanceRef.current = camera.position.distanceTo(controls.target);
      navigationFocusDistanceRef.current = navigationBaseDistanceRef.current;
      camera.near = 0.002;
      camera.far = Math.max(span * 12, 1200);
      camera.updateProjectionMatrix();
      controls.update();
    };
    if (cloud && fittedCloudsRef.current !== clouds) {
      const shouldFitView = !preserveViewOnCloudChange || fittedCloudsRef.current === null;
      fittedCloudsRef.current = clouds;
      if (shouldFitView) fitViewRef.current("near");
    }
  }, [clouds, comparisonMode, frameIndex, gridReferenceCloud, hideGround, preserveViewOnCloudChange, stableOnly, pointSize, uniformPointColor]);

  useEffect(() => {
    if (!comparisonMode) return;
    cloudGroupRef.current?.children.forEach((child) => {
      const layer = child.userData.comparisonLayer as "source" | "target" | undefined;
      child.visible = comparisonVisibleLayer === "both" || layer === comparisonVisibleLayer;
    });
  }, [clouds, comparisonMode, comparisonVisibleLayer, pointSize]);

  useEffect(() => {
    if (cloudGroupRef.current) cloudGroupRef.current.visible = !selectedOnly;
  }, [selectedOnly, clouds]);

  useEffect(() => {
    if (viewRequest.token === handledViewTokenRef.current) return;
    handledViewTokenRef.current = viewRequest.token;
    fitViewRef.current(viewRequest.mode);
  }, [viewRequest]);

  useEffect(() => {
    clearSelection();
  }, [clearSelectionToken]);

  useEffect(() => {
    const cloud = clouds[frameIndex];
    const group = selectionGroupRef.current;
    if (!cloud || !group) return;
    let points = selectionPointsRef.current;
    if (!points) {
      const geometry = new THREE.BufferGeometry();
      const initialCapacity = Math.max(3, 2 ** Math.ceil(Math.log2(Math.max(selectedIndices.length * 3, 3))));
      const attribute = new THREE.BufferAttribute(new Float32Array(initialCapacity), 3);
      attribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute("position", attribute);
      geometry.setDrawRange(0, 0);
      points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color: selectedPointColor,
        // Scale exactly like the cloud points. A fixed six-pixel marker becomes
        // visually buried by the enlarged source points after zooming in.
        size: pointSize * 2.6,
        sizeAttenuation: true,
        depthTest: false,
        depthWrite: false,
        fog: false,
      }));
      points.renderOrder = 4;
      points.frustumCulled = false;
      group.add(points);
      selectionPointsRef.current = points;
    }
    points.material.color.setHex(selectedPointColor);
    points.material.size = pointSize * 2.6;
    const required = Math.max(selectedIndices.length * 3, 3);
    let attribute = points.geometry.getAttribute("position") as THREE.BufferAttribute;
    if (attribute.array.length < required) {
      const capacity = 2 ** Math.ceil(Math.log2(required));
      points.geometry.dispose();
      const geometry = new THREE.BufferGeometry();
      attribute = new THREE.BufferAttribute(new Float32Array(capacity), 3);
      attribute.setUsage(THREE.DynamicDrawUsage);
      geometry.setAttribute("position", attribute);
      points.geometry = geometry;
    }
    const positions = attribute.array as Float32Array;
    const slotByIndex = new Map<number, number>();
    const indexBySlot: number[] = [];
    let visibleCount = 0;
    selectedIndices.forEach((index) => {
      if (cloud.stable[index] === 2) return;
      if ((stableOnly && cloud.stable[index] !== 1) || (hideGround && isGroundPoint(cloud, index, stableOnly))) return;
      positions.set(cloud.positions.subarray(index * 3, index * 3 + 3), visibleCount * 3);
      slotByIndex.set(index, visibleCount);
      indexBySlot.push(index);
      visibleCount += 1;
    });
    selectionSlotByIndexRef.current = slotByIndex;
    selectionIndexBySlotRef.current = indexBySlot;
    selectionVisibleCountRef.current = visibleCount;
    attribute.clearUpdateRanges();
    if (visibleCount) attribute.addUpdateRange(0, visibleCount * 3);
    attribute.needsUpdate = true;
    points.geometry.setDrawRange(0, visibleCount);
    invalidateRenderRef.current();
  }, [clouds, frameIndex, hideGround, stableOnly, pointSize, selectedIndices, selectedPointColor]);

  useEffect(() => {
    const group = selectionGroupRef.current;
    if (!group) return;
    overlayPointsRef.current.forEach((points) => {
      group.remove(points);
      points.geometry.dispose();
      points.material.dispose();
    });
    overlayPointsRef.current = overlayClouds.filter(({ cloud }) => cloud.positions.length > 0).map(({ cloud, color, size }, index) => {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(cloud.positions, 3));
      const points = new THREE.Points(geometry, new THREE.PointsMaterial({
        color,
        size: size??7,
        sizeAttenuation: false,
        depthTest: false,
        depthWrite: false,
        transparent: true,
        opacity: overlayOpacity,
        fog: false,
      }));
      points.renderOrder = 5 + index;
      points.frustumCulled = false;
      group.add(points);
      return points;
    });
  }, [clouds, overlayClouds, overlayOpacity, pointSize]);

  useEffect(() => {
    const group=selectionGroupRef.current;
    const segments=inspectionLines.length?inspectionLines:structureLine?.length===2?[structureLine]:[];
    if(!group||!segments.length)return;
    const geometry=new THREE.BufferGeometry().setFromPoints(segments.flatMap(segment=>segment.map(p=>new THREE.Vector3(p[0],p[1],p[2]))));
    const material=new THREE.LineBasicMaterial({color:0xce68ae,depthTest:false,transparent:true});
    const line=new THREE.LineSegments(geometry,material);line.renderOrder=20;group.add(line);
    return()=>{group.remove(line);geometry.dispose();material.dispose();};
  },[structureLine,inspectionLines,clouds,frameIndex]);
  const focusedInspectionRef=useRef(0);
  useEffect(()=>{
    const camera=cameraRef.current,controls=controlsRef.current;
    if(!camera||!controls||!clouds[frameIndex]||!inspectionFocus||focusedInspectionRef.current===inspectionFocus||!inspectionLines.length)return;
    const box=new THREE.Box3().setFromPoints(inspectionLines.flatMap(line=>line.map(p=>new THREE.Vector3(...p as [number,number,number]))));
    if(box.isEmpty())return;
    const center=box.getCenter(new THREE.Vector3()),size=box.getSize(new THREE.Vector3());
    const span=Math.max(size.length()*1.5,3);
    const direction=camera.position.clone().sub(controls.target).normalize();
    controls.target.copy(center);camera.position.copy(center).addScaledVector(direction,span);
    controls.update();focusedInspectionRef.current=inspectionFocus;
  },[inspectionFocus,inspectionLines,clouds,frameIndex]);

  const pointerPosition = (event: React.PointerEvent<HTMLDivElement>) => {
    const rect = event.currentTarget.getBoundingClientRect();
    return { x: event.clientX - rect.left, y: event.clientY - rect.top };
  };

  const updateDragRect = (value: { x: number; y: number; width: number; height: number } | null) => {
    dragRectRef.current = value;
    setDragRect(value);
  };

  const boxDragModeAt = (position: { x: number; y: number }, rect: NonNullable<typeof dragRect>): RectDragMode => {
    const threshold = 11;
    const right = rect.x + rect.width;
    const bottom = rect.y + rect.height;
    const nearLeft = Math.abs(position.x - rect.x) <= threshold;
    const nearRight = Math.abs(position.x - right) <= threshold;
    const nearTop = Math.abs(position.y - rect.y) <= threshold;
    const nearBottom = Math.abs(position.y - bottom) <= threshold;
    if (nearLeft && nearTop) return "top-left";
    if (nearRight && nearTop) return "top-right";
    if (nearLeft && nearBottom) return "bottom-left";
    if (nearRight && nearBottom) return "bottom-right";
    if (nearLeft && position.y >= rect.y - threshold && position.y <= bottom + threshold) return "left";
    if (nearRight && position.y >= rect.y - threshold && position.y <= bottom + threshold) return "right";
    if (nearTop && position.x >= rect.x - threshold && position.x <= right + threshold) return "top";
    if (nearBottom && position.x >= rect.x - threshold && position.x <= right + threshold) return "bottom";
    if (position.x >= rect.x && position.x <= right && position.y >= rect.y && position.y <= bottom) return "move";
    return "new";
  };

  const adjustedBoxRect = (position: { x: number; y: number }) => {
    const drag = boxDragRef.current;
    const host = hostRef.current;
    if (!drag || !host) return dragRectRef.current;
    if (drag.mode === "new") return {
      x: Math.min(drag.start.x, position.x),
      y: Math.min(drag.start.y, position.y),
      width: Math.abs(position.x - drag.start.x),
      height: Math.abs(position.y - drag.start.y),
    };
    if (drag.mode === "move") {
      const x = Math.max(0, Math.min(host.clientWidth - drag.initial.width, drag.initial.x + position.x - drag.start.x));
      const y = Math.max(0, Math.min(host.clientHeight - drag.initial.height, drag.initial.y + position.y - drag.start.y));
      return { ...drag.initial, x, y };
    }
    let left = drag.initial.x;
    let right = drag.initial.x + drag.initial.width;
    let top = drag.initial.y;
    let bottom = drag.initial.y + drag.initial.height;
    if (drag.mode.includes("left")) left = Math.min(position.x, right - 4);
    if (drag.mode.includes("right")) right = Math.max(position.x, left + 4);
    if (drag.mode.includes("top")) top = Math.min(position.y, bottom - 4);
    if (drag.mode.includes("bottom")) bottom = Math.max(position.y, top + 4);
    left = Math.max(0, left); right = Math.min(host.clientWidth, right);
    top = Math.max(0, top); bottom = Math.min(host.clientHeight, bottom);
    return { x: left, y: top, width: right - left, height: bottom - top };
  };

  const commitBoxSelection = () => {
    const finalRect = dragRectRef.current;
    const cloud = currentCloudRef.current;
    const camera = cameraRef.current;
    const host = hostRef.current;
    if (!finalRect || !cloud || !camera || !host || finalRect.width < 4 || finalRect.height < 4) return;
    const candidates: Array<{ index: number; x: number; y: number; depth: number }> = [];
    const sample = new THREE.Vector3();
    const cameraSpace = new THREE.Vector3();
    camera.updateMatrixWorld();
    for (let index = 0; index < cloud.stable.length; index += 1) {
      if (cloud.stable[index] === 2) continue;
      if (selectedOnly && !visibleSelection.has(index)) continue;
      if (stableOnly && cloud.stable[index] !== 1) continue;
      if (hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
      sample.set(cloud.positions[index * 3], cloud.positions[index * 3 + 1], cloud.positions[index * 3 + 2]);
      cameraSpace.copy(sample).applyMatrix4(camera.matrixWorldInverse);
      const depth = -cameraSpace.z;
      sample.project(camera);
      if (sample.z < -1 || sample.z > 1 || depth <= 0) continue;
      const x = (sample.x + 1) * host.clientWidth * 0.5;
      const y = (1 - sample.y) * host.clientHeight * 0.5;
      if (x >= finalRect.x && x <= finalRect.x + finalRect.width &&
          y >= finalRect.y && y <= finalRect.y + finalRect.height) candidates.push({ index, x, y, depth });
    }
    onSelection(filterForegroundScreenPoints(candidates, 10, brushDepthTolerance));
    updateDragRect(null);
    boxDragRef.current = null;
    setBoxCursor("crosshair");
  };

  useEffect(() => {
    if (interactionMode !== "box") {
      updateDragRect(null);
      boxDragRef.current = null;
      return;
    }
    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement || target instanceof HTMLSelectElement) return;
      if (event.key === "Enter" && dragRectRef.current) {
        event.preventDefault();
        commitBoxSelection();
      } else if (event.key === "Escape" && dragRectRef.current) {
        event.preventDefault();
        updateDragRect(null);
        boxDragRef.current = null;
        setBoxCursor("crosshair");
      }
    };
    const host = hostRef.current;
    if (!host) return;
    host.addEventListener("keydown", onKeyDown);
    return () => host.removeEventListener("keydown", onKeyDown);
  }, [brushDepthTolerance, interactionMode, stableOnly, hideGround, onSelection]);

  useEffect(() => {
    updateDragRect(null);
    boxDragRef.current = null;
  }, [frameIndex, viewRequest.token]);

  const buildBrushScreenIndex = () => {
    const cloud = currentCloudRef.current;
    const camera = cameraRef.current;
    const host = hostRef.current;
    if (!cloud || !camera || !host) return null;
    camera.updateMatrixWorld();
    const cellSize = Math.max(8, brushSize * 0.5);
    const cells = new Map<string, Array<{ index: number; x: number; y: number; depth: number }>>();
    const sample = new THREE.Vector3();
    const cameraSpace = new THREE.Vector3();
    const removeMode = interactionMode === "brush-remove";
    const pointIndices: Iterable<number> = removeMode || selectedOnly
      ? selectedIndices
      : { *[Symbol.iterator]() { for (let index = 0; index < cloud.stable.length; index += 1) yield index; } };
    for (const index of pointIndices) {
      if (cloud.stable[index] === 2) continue;
      if (!removeMode && stableOnly && cloud.stable[index] !== 1) continue;
      if (!removeMode && hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
      sample.set(cloud.positions[index * 3], cloud.positions[index * 3 + 1], cloud.positions[index * 3 + 2]);
      cameraSpace.copy(sample).applyMatrix4(camera.matrixWorldInverse);
      const depth = -cameraSpace.z;
      sample.project(camera);
      if (sample.z < -1 || sample.z > 1 || depth <= 0) continue;
      const x = (sample.x + 1) * host.clientWidth * 0.5;
      const y = (1 - sample.y) * host.clientHeight * 0.5;
      const key = `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
      const cell = cells.get(key);
      const point = { index, x, y, depth };
      if (cell) cell.push(point);
      else cells.set(key, [point]);
    }
    return { cellSize, cells };
  };

  const previewPointRemoval = (indices: number[]) => {
    const points = selectionPointsRef.current;
    if (!points || !indices.length) return;
    const attribute = points.geometry.getAttribute("position") as THREE.BufferAttribute;
    const positions = attribute.array as Float32Array;
    const slotByIndex = selectionSlotByIndexRef.current;
    const indexBySlot = selectionIndexBySlotRef.current;
    let visibleCount = selectionVisibleCountRef.current;
    let firstChangedSlot = Number.POSITIVE_INFINITY;
    let lastChangedSlot = -1;
    for (const index of indices) {
      const slot = slotByIndex.get(index);
      if (slot === undefined || visibleCount === 0) continue;
      const finalSlot = visibleCount - 1;
      if (slot !== finalSlot) {
        positions.copyWithin(slot * 3, finalSlot * 3, finalSlot * 3 + 3);
        const movedIndex = indexBySlot[finalSlot];
        indexBySlot[slot] = movedIndex;
        slotByIndex.set(movedIndex, slot);
        firstChangedSlot = Math.min(firstChangedSlot, slot);
        lastChangedSlot = Math.max(lastChangedSlot, slot);
      }
      slotByIndex.delete(index);
      indexBySlot.pop();
      visibleCount -= 1;
    }
    selectionVisibleCountRef.current = visibleCount;
    if (lastChangedSlot >= firstChangedSlot) {
      attribute.clearUpdateRanges();
      attribute.addUpdateRange(firstChangedSlot * 3, (lastChangedSlot - firstChangedSlot + 1) * 3);
      attribute.needsUpdate = true;
    }
    points.geometry.setDrawRange(0, visibleCount);
  };

  const previewPointAddition = (indices: number[]) => {
    const points = selectionPointsRef.current;
    const cloud = currentCloudRef.current;
    if (!points || !cloud || !indices.length) return;
    const slotByIndex = selectionSlotByIndexRef.current;
    const additions = indices.filter((index) =>
      !slotByIndex.has(index) &&
      (!stableOnly || cloud.stable[index] === 1) &&
      (!hideGround || !isGroundPoint(cloud, index, stableOnly)),
    );
    if (!additions.length) return;
    let attribute = points.geometry.getAttribute("position") as THREE.BufferAttribute;
    const visibleCount = selectionVisibleCountRef.current;
    const requiredFloats = (visibleCount + additions.length) * 3;
    if (attribute.array.length < requiredFloats) {
      const capacity = 2 ** Math.ceil(Math.log2(Math.max(requiredFloats, 3)));
      const expanded = new Float32Array(capacity);
      expanded.set((attribute.array as Float32Array).subarray(0, visibleCount * 3));
      attribute = new THREE.BufferAttribute(expanded, 3);
      attribute.setUsage(THREE.DynamicDrawUsage);
      points.geometry.setAttribute("position", attribute);
    }
    const positions = attribute.array as Float32Array;
    additions.forEach((index, offset) => {
      const slot = visibleCount + offset;
      positions.set(cloud.positions.subarray(index * 3, index * 3 + 3), slot * 3);
      slotByIndex.set(index, slot);
      selectionIndexBySlotRef.current.push(index);
    });
    selectionVisibleCountRef.current = visibleCount + additions.length;
    attribute.clearUpdateRanges();
    attribute.addUpdateRange(visibleCount * 3, additions.length * 3);
    attribute.needsUpdate = true;
    points.geometry.setDrawRange(0, selectionVisibleCountRef.current);
    invalidateRenderRef.current();
  };

  const applyBrushAt = (position: { x: number; y: number }) => {
    const screenIndex = brushScreenIndexRef.current;
    if (!screenIndex) return;
    const radius = brushSize * 0.5;
    const radiusSquared = radius * radius;
    const minCellX = Math.floor((position.x - radius) / screenIndex.cellSize);
    const maxCellX = Math.floor((position.x + radius) / screenIndex.cellSize);
    const minCellY = Math.floor((position.y - radius) / screenIndex.cellSize);
    const maxCellY = Math.floor((position.y + radius) / screenIndex.cellSize);
    const candidates: Array<{ index: number; depth: number }> = [];
    let nearestDepth = Number.POSITIVE_INFINITY;
    for (let cellX = minCellX; cellX <= maxCellX; cellX += 1) {
      for (let cellY = minCellY; cellY <= maxCellY; cellY += 1) {
        for (const point of screenIndex.cells.get(`${cellX},${cellY}`) ?? []) {
          if ((point.x - position.x) ** 2 + (point.y - position.y) ** 2 > radiusSquared) continue;
          candidates.push(point);
          nearestDepth = Math.min(nearestDepth, point.depth);
        }
      }
    }
    const painted = (interactionMode === "brush-remove"
      ? candidates
      : candidates.filter((candidate) => candidate.depth <= nearestDepth + brushDepthTolerance))
      .map((candidate) => candidate.index);
    const newlyPainted: number[] = [];
    for (const index of painted) {
      if (pendingBrushPointIndicesRef.current.has(index)) continue;
      pendingBrushPointIndicesRef.current.add(index);
      newlyPainted.push(index);
    }
    if (interactionMode === "brush-add") previewPointAddition(newlyPainted);
    else if (interactionMode === "brush-remove") previewPointRemoval(newlyPainted);
  };

  const commitPointBrush = () => {
    const pending = pendingBrushPointIndicesRef.current;
    if (pending.size) {
      const sorted = Uint32Array.from(pending).sort();
      onPointEdit(
        sorted,
        interactionMode === "brush-add" ? "add" : "remove",
      );
    }
    pending.clear();
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (chordPanRef.current || event.buttons === 3) return;
    if (event.button === 2) return;
    if (interactionMode === "navigate" || interactionMode === "pan") return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const position = pointerPosition(event);
    dragStartRef.current = position;
    if (interactionMode === "box") {
      const existing = dragRectRef.current;
      const mode = existing ? boxDragModeAt(position, existing) : "new";
      const initial = existing ?? { ...position, width: 0, height: 0 };
      boxDragRef.current = { mode, start: position, initial };
      setBoxCursor(cursorForRectDragMode(mode));
      if (mode === "new") updateDragRect(initial);
    }
    if (interactionMode === "brush-add" || interactionMode === "brush-remove") {
      pendingBrushPointIndicesRef.current.clear();
      brushScreenIndexRef.current = buildBrushScreenIndex();
      lastBrushApplyRef.current = position;
      applyBrushAt(position);
    }
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const current = pointerPosition(event);
    if (interactionMode === "brush-add" || interactionMode === "brush-remove") {
      setBrushCursor(current);
      if (dragStartRef.current) {
        const previous = lastBrushApplyRef.current;
        if (!previous || Math.hypot(current.x - previous.x, current.y - previous.y) >= Math.max(2, brushSize * 0.15)) {
          lastBrushApplyRef.current = current;
          applyBrushAt(current);
        }
      }
      return;
    }
    if (interactionMode !== "box") return;
    if (!dragStartRef.current) {
      const existing = dragRectRef.current;
      setBoxCursor(existing ? cursorForRectDragMode(boxDragModeAt(current, existing)) : "crosshair");
      return;
    }
    updateDragRect(adjustedBoxRect(current));
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button === 2) return;
    if (interactionMode === "navigate" || interactionMode === "pan" || !dragStartRef.current) return;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    const current = pointerPosition(event);
    if (interactionMode === "brush-add" || interactionMode === "brush-remove") {
      applyBrushAt(current);
      commitPointBrush();
      dragStartRef.current = null;
      brushScreenIndexRef.current = null;
      lastBrushApplyRef.current = null;
      return;
    }
    if (interactionMode === "point-add" || interactionMode === "point-remove") {
      const cloud = currentCloudRef.current;
      const camera = cameraRef.current;
      const host = hostRef.current;
      dragStartRef.current = null;
      if (!cloud || !camera || !host) return;
      const sample = new THREE.Vector3();
      const cameraSpace = new THREE.Vector3();
      camera.updateMatrixWorld();
      const hitRadiusSquared = 14 * 14;
      const removeMode = interactionMode === "point-remove";
      const pointIndices: Iterable<number> = removeMode || selectedOnly
        ? selectedIndices
        : { *[Symbol.iterator]() { for (let index = 0; index < cloud.stable.length; index += 1) yield index; } };
      const hits: Array<{ index: number; distance: number; depth: number }> = [];
      for (const index of pointIndices) {
      if (cloud.stable[index] === 2) continue;
        if (!removeMode && stableOnly && cloud.stable[index] !== 1) continue;
        if (!removeMode && hideGround && isGroundPoint(cloud, index, stableOnly)) continue;
        sample.set(cloud.positions[index * 3], cloud.positions[index * 3 + 1], cloud.positions[index * 3 + 2]);
        cameraSpace.copy(sample).applyMatrix4(camera.matrixWorldInverse);
        const depth = -cameraSpace.z;
        sample.project(camera);
        if (sample.z < -1 || sample.z > 1 || depth <= 0) continue;
        const x = (sample.x + 1) * host.clientWidth * 0.5;
        const y = (1 - sample.y) * host.clientHeight * 0.5;
        const distance = (x - current.x) ** 2 + (y - current.y) ** 2;
        if (distance <= hitRadiusSquared) hits.push({ index, distance, depth });
      }
      if (removeMode) {
        if (hits.length) onPointEdit(Uint32Array.from(hits.map((hit) => hit.index)), "remove");
      } else {
        // Addition is visibility-aware: among nearby screen hits, choose the
        // nearest surface first and only then the closest projected point.
        hits.sort((left, right) => left.depth - right.depth || left.distance - right.distance);
        if (hits[0]) onPointEdit(Uint32Array.of(hits[0].index), "add");
      }
      return;
    }
    updateDragRect(adjustedBoxRect(current));
    dragStartRef.current = null;
    boxDragRef.current = null;
    boxDragRef.current = null;
    const finalRect = dragRectRef.current;
    setBoxCursor(finalRect ? cursorForRectDragMode(boxDragModeAt(current, finalRect)) : "crosshair");
  };

  const handlePointerCancel = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (brushMode && pendingBrushPointIndicesRef.current.size) commitPointBrush();
    dragStartRef.current = null;
    boxDragRef.current = null;
    brushScreenIndexRef.current = null;
    lastBrushApplyRef.current = null;
    pendingBrushPointIndicesRef.current.clear();
    updateDragRect(null);
  };

  const brushMode = interactionMode === "brush-add" || interactionMode === "brush-remove";
  const beginChordPan = (clientX: number, clientY: number) => {
    const host = hostRef.current;
    if (!host || host.clientWidth < 2 || host.clientHeight < 2 || host.getClientRects().length === 0) return false;
    if (activeChordPanViewport.current && activeChordPanViewport.current !== host) return false;
    if (chordPanRef.current) return true;
    activeChordPanViewport.current = host;
    chordPanRef.current = true;
    chordAwaitReleaseRef.current = true;
    chordPanPositionRef.current = { x: clientX, y: clientY };
    setChordPanning(true);
    suppressChordContextMenuRef.current = true;
    if (brushMode && pendingBrushPointIndicesRef.current.size) commitPointBrush();
    dragStartRef.current = null;
    brushScreenIndexRef.current = null;
    lastBrushApplyRef.current = null;
    updateDragRect(null);
    const controls = controlsRef.current;
    const camera = cameraRef.current;
    if (controls && camera) {
      const position = camera.position.clone();
      const target = controls.target.clone();
      chordDampingRef.current = controls.enableDamping;
      controls.enableDamping = false;
      controls.update();
      camera.position.copy(position);
      controls.target.copy(target);
      controls.update();
      controls.enabled = false;
    }
    return true;
  };

  const finishChordPan = (remainingButtons = 0) => {
    if (chordPanRef.current) {
      chordPanRef.current = false;
      chordPanPositionRef.current = null;
      setChordPanning(false);
    }
    if (!chordAwaitReleaseRef.current) return;
    const controls = controlsRef.current;
    if (!controls) return;
    const damping = chordDampingRef.current ?? controls.enableDamping;
    // OrbitControls damping keeps angular velocity after a drag. Flush that
    // velocity without moving the camera, otherwise releasing the chord can
    // make the cloud continue spinning.
    const camera = cameraRef.current;
    if (camera) {
      const position = camera.position.clone();
      const target = controls.target.clone();
      controls.enableDamping = false;
      controls.update();
      camera.position.copy(position);
      controls.target.copy(target);
      controls.update();
      controls.enableDamping = false;
    }
    controls.mouseButtons.LEFT = interactionMode === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    controls.enabled = false;
    // Re-enabling while either mouse button is still held revives the stale
    // OrbitControls ROTATE gesture. Wait for the final release instead.
    if ((remainingButtons & 3) !== 0) return;
    if (activeChordPanViewport.current === hostRef.current) activeChordPanViewport.current = null;
    if (controlsResumeFrameRef.current !== null) cancelAnimationFrame(controlsResumeFrameRef.current);
    controlsResumeFrameRef.current = requestAnimationFrame(() => {
      controlsResumeFrameRef.current = null;
      if (!chordPanRef.current && controlsRef.current === controls) {
        chordAwaitReleaseRef.current = false;
        controls.enabled = interactionMode === "navigate" || interactionMode === "pan";
        controls.update();
        controls.enableDamping = damping;
        chordDampingRef.current = null;
      }
    });
  };

  const moveChordPan = (clientX: number, clientY: number) => {
    if (!chordPanRef.current) return;
    const previous = chordPanPositionRef.current;
    chordPanPositionRef.current = { x: clientX, y: clientY };
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    const host = hostRef.current;
    if (!previous || !camera || !controls || !host) return;
    camera.updateMatrixWorld();
    const focusDistance = Math.max(navigationFocusDistanceRef.current, 0.01);
    const worldPerPixel = 2 * focusDistance
      * Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)) / Math.max(host.clientHeight, 1);
    const shift = new THREE.Vector3()
      .setFromMatrixColumn(camera.matrixWorld, 0)
      .multiplyScalar(-(clientX - previous.x) * worldPerPixel * 1.8)
      .add(new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 1)
        .multiplyScalar((clientY - previous.y) * worldPerPixel * 1.8));
    camera.position.add(shift);
    controls.target.add(shift);
    // Chord pan bypasses OrbitControls events; explicitly request a frame.
    invalidateRenderRef.current();
  };

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return;
    const begin = (event: MouseEvent) => {
      if ((event.buttons & 3) !== 3 || !beginChordPan(event.clientX, event.clientY)) return;
      host.focus({ preventScroll: true });
      event.preventDefault();
      event.stopImmediatePropagation();
    };
    const move = (event: PointerEvent) => {
      if (activeChordPanViewport.current !== host) return;
      if (chordPanRef.current && (event.buttons & 3) !== 3) {
        finishChordPan(event.buttons);
        return;
      }
      if (!chordPanRef.current) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      moveChordPan(event.clientX, event.clientY);
    };
    const release = (event: PointerEvent | MouseEvent) => {
      if (activeChordPanViewport.current !== host && !chordAwaitReleaseRef.current) return;
      finishChordPan(event.buttons);
    };
    const blur = () => {
      if (activeChordPanViewport.current === host || chordAwaitReleaseRef.current) finishChordPan(0);
    };
    host.addEventListener("mousedown", begin, true);
    window.addEventListener("pointermove", move, true);
    window.addEventListener("pointerup", release, true);
    window.addEventListener("pointercancel", release, true);
    window.addEventListener("mouseup", release, true);
    window.addEventListener("blur", blur);
    return () => {
      host.removeEventListener("mousedown", begin, true);
      window.removeEventListener("pointermove", move, true);
      window.removeEventListener("pointerup", release, true);
      window.removeEventListener("pointercancel", release, true);
      window.removeEventListener("mouseup", release, true);
      window.removeEventListener("blur", blur);
      finishChordPan(0);
      if (activeChordPanViewport.current === host) activeChordPanViewport.current = null;
      if (controlsResumeFrameRef.current !== null) {
        cancelAnimationFrame(controlsResumeFrameRef.current);
        controlsResumeFrameRef.current = null;
      }
    };
  }, [interactionMode]);

  const blockPointCloudContextMenu = (event: React.MouseEvent<HTMLDivElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (brushMode && pendingBrushPointIndicesRef.current.size) commitPointBrush();
    if (suppressChordContextMenuRef.current) {
      suppressChordContextMenuRef.current = false;
      return;
    }
    onInteractionModeChange("navigate");
  };

  return (
    <div
      className={`viewport ${interactionMode !== "navigate" && interactionMode !== "pan" && !chordPanning ? "selecting" : ""} ${interactionMode === "pan" || chordPanning ? "panning" : ""} ${brushMode && !chordPanning ? "point-brushing" : ""}`}
      ref={hostRef}
      tabIndex={0}
      style={interactionMode === "box" ? { cursor: boxCursor } : undefined}
      aria-label="三维点云视图"
      onPointerDownCapture={(event) => event.currentTarget.focus({ preventScroll: true })}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerCancel}
      onContextMenuCapture={blockPointCloudContextMenu}
      onContextMenu={blockPointCloudContextMenu}
      onPointerEnter={(event) => { if (brushMode) setBrushCursor(pointerPosition(event)); }}
      onPointerLeave={() => { if (!dragStartRef.current) setBrushCursor(null); }}
    >
      {graphicsError && <div className="graphics-error" role="alert">{graphicsError}</div>}
      {dragRect && <span className="selection-rect" style={{
        left: dragRect.x,
        top: dragRect.y,
        width: dragRect.width,
        height: dragRect.height,
      }}>
        {(["top-left", "top", "top-right", "left", "right", "bottom-left", "bottom", "bottom-right"] as const).map((handle) => <i key={handle} className={`selection-handle ${handle}`} />)}
        <b className="selection-rect-status">拖动边界微调 · Enter 聚类 · Esc 取消</b>
      </span>}
      {brushMode && brushCursor && <span className="point-brush-outline" style={{
        left: brushCursor.x,
        top: brushCursor.y,
        width: brushSize,
        height: brushSize,
      }} />}
    </div>
  );
}
