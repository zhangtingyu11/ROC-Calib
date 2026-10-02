"use client";
import { useDialogFocus } from "./dialog-focus";

import { useEffect, useMemo, useRef, useState } from "react";
import { resolvePreviewVersion } from "./preview-version";
import { annotationContentChanged } from "./annotation-dirty";
import { buildPointCloudConnectivityIndex, clusterPointSelection, combinePointIndices, countMask, decodeMaskRle, encodeMaskRle, growConnectedPointCloud, ImageMask, mergeImageMasks, PointCloudConnectivityIndex } from "./segmentation";
import CalibrationReviewPanel from "./calibration-review-panel";
import { type Manifest, type CalibrationWorkspaceMode, type CalibrationTaskGroup, type BagBrowserResult, type PairedDatasetBrowserResult, type PairedDatasetBrowserEntry, type LocalBagUploadFile, type LocalBagUploadProgress, type BagBrowserEntry, type ManualFrameTimeline, type BagPreparationProgress, type IntrinsicsProfile, type IntrinsicsBrowserResult, type ProjectionModel, type FrameLabel, type CloudData, type PointInteractionMode, type ImageMaskLayer, type ImageMaskSnapshot, type ImageSegmentationTool, type SegmentationGranularity, type SynchronizedImageView, type PointSegmentationMode, type PointGrowFeedback, type SavedAnnotation, type LidarPairAnnotation, type LidarPairCalibration, type LidarFramePairSet, type LidarExtrinsicGraph, type ViewRequest, type StageMode, type OptimizationResult, type ExtrinsicVersion, type ExtrinsicInitializationMode, type ProjectionScope, type FullProjectionScope, loadStoredAnnotations, loadStoredCalibrations, defaultTaskGroupId, calibrationModelKey, CALIBRATION_ALGORITHM_VERSION, frameIndexByLabel, type ProjectionModelChoice, temporalProjectionFor, EMPTY_CLOUD_OVERLAYS, applyManualMaskOverrides, combineMaskLayers, copyMask, copyMaskLayers, type BrushMaskChange, updateManualMaskOverridesRegion, type BagUploadSession, type DatasetInfo, multiply4, invertRigid4, type ManualSelectionState, normalizeManualSelections, validManualSelections, type ManualFrameCandidate, calibrationGroupKey, type StoredExtrinsicVersion, normalizeStoredExtrinsic, readCloud, formatPoints, rawProjectionCalibration, FORWARD_FACING_LIDAR_TO_CAMERA, isGroundPoint, RAW_COORDINATE_SYSTEM, storeAnnotation, deleteStoredAnnotation, PAIRS_PER_PAGE, type CalibrationObservation, storeCalibration, lidarDisplayLabel, formatTimestamp, canvasMaskView, formatFileSize, ManualLidarCloudPreview } from './workspace-core';
import { LidarLidarWorkspace } from './lidar-workspace';
import { PointCloudViewport } from './point-cloud-view';
import { ImageCanvas } from './image-canvas';

export default function Home() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [calibrationWorkspaceMode, setCalibrationWorkspaceMode] = useState<CalibrationWorkspaceMode>("lidar-camera");
  const [lidarCalibrationAvailable, setLidarCalibrationAvailable] = useState(true);
  const [taskGroups, setTaskGroups] = useState<CalibrationTaskGroup[]>([]);
  const [activeTaskGroupId, setActiveTaskGroupId] = useState("");
  const [groupDeleting, setGroupDeleting] = useState(false);

  useEffect(() => {
    let active = true;
    fetch("/api/segment/health")
      .then((response) => response.ok ? response.json() : null)
      .then((health: { lidar_calibration_enabled?: boolean } | null) => {
        if (active && typeof health?.lidar_calibration_enabled === "boolean") {
          setLidarCalibrationAvailable(health.lidar_calibration_enabled);
        }
      })
      .catch(() => undefined);
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!lidarCalibrationAvailable && calibrationWorkspaceMode === "lidar-lidar") setCalibrationWorkspaceMode("lidar-camera");
  }, [calibrationWorkspaceMode, lidarCalibrationAvailable]);
  const [groupStateReady, setGroupStateReady] = useState(false);
  const [groupStateRefreshToken, setGroupStateRefreshToken] = useState(0);
  const domainRevisionRef = useRef({ camera: 0, lidar: 0 });
  const domainSavePendingRef = useRef({ camera: false, lidar: false });
  const domainRefreshDeferredRef = useRef({ camera: false, lidar: false });
  const skipDomainSaveRef = useRef({ camera: false, lidar: false });
  useEffect(() => {
    domainRevisionRef.current = { camera: 0, lidar: 0 };
    domainSavePendingRef.current = { camera: false, lidar: false };
    domainRefreshDeferredRef.current = { camera: false, lidar: false };
    skipDomainSaveRef.current = { camera: false, lidar: false };
    saveErrorsRef.current = { camera: false, lidar: false };
    setSaveStates({camera: "已同步", lidar: "已同步"});
  }, [activeTaskGroupId]);
  const [localStateLoaded, setLocalStateLoaded] = useState(false);
  const [reviewStatuses, setReviewStatuses] = useState<Record<string,string>>({});
  const [reviewHighlight, setReviewHighlight] = useState<{id:string;indices:number[];color:number}|null>(null);
  const [reviewEditRevision, setReviewEditRevision] = useState(0);
  const [reviewPairEdits, setReviewPairEdits] = useState<Record<string, number>>({});
  const [structurePreview, setStructurePreview] = useState<{datasetId:string;frame:string;indicesA:number[];indicesB?:number[]}|null>(null);
  const [structureLine, setStructureLine] = useState<number[][]|null>(null);
  const [extractionView,setExtractionView]=useState<Record<string,any>|null>(null);
  const [extractionFocus,setExtractionFocus]=useState(0);
  const [bagBrowserOpen, setBagBrowserOpen] = useState(false);
  const [editingDatasetId, setEditingDatasetId] = useState<string | null>(null);
  const [bagBrowser, setBagBrowser] = useState<BagBrowserResult | null>(null);
  const [bagBrowserLoading, setBagBrowserLoading] = useState(false);
  const [bagImportSource, setBagImportSource] = useState<"local" | "server" | "paired">("local");
  const [pairedBrowser, setPairedBrowser] = useState<PairedDatasetBrowserResult | null>(null);
  const [pairedBrowserLoading, setPairedBrowserLoading] = useState(false);
  const [selectedPairedDatasets, setSelectedPairedDatasets] = useState<Record<string, PairedDatasetBrowserEntry>>({});
  const [localBagFiles, setLocalBagFiles] = useState<LocalBagUploadFile[]>([]);
  const [localBagPackageName, setLocalBagPackageName] = useState("");
  const [localBagUpload, setLocalBagUpload] = useState<LocalBagUploadProgress>({ active: false, paused: false, percent: 0, message: "", uploadedBytes: 0, totalBytes: 0 });
  const [localBagUploadError, setLocalBagUploadError] = useState("");
  const localBagFileInputRef = useRef<HTMLInputElement | null>(null);
  const localBagFolderInputRef = useRef<HTMLInputElement | null>(null);
  const localBagUploadAbortRef = useRef<AbortController | null>(null);
  const [selectedBags, setSelectedBags] = useState<Record<string, BagBrowserEntry>>({});
  const [manualTimeline, setManualTimeline] = useState<ManualFrameTimeline | null>(null);
  const [manualTimelineLoading, setManualTimelineLoading] = useState(false);
  const [manualFrameIndex, setManualFrameIndex] = useState(0);
  const [manualCameraTopic, setManualCameraTopic] = useState("");
  const [manualLidarTopic, setManualLidarTopic] = useState("");
  const [selectedManualFrames, setSelectedManualFrames] = useState<Record<string, string[]>>({});
  const [allManualFrameSelections, setAllManualFrameSelections] = useState<Record<string, string[]>>({});
  const [manualSelectionVersion, setManualSelectionVersion] = useState(0);
  const [manualSelectionDirty, setManualSelectionDirty] = useState(false);
  const [manualSelectionSync, setManualSelectionSync] = useState<"connecting" | "synced" | "error">("connecting");
  const manualSelectionClientId = useRef("");
  const [manualPreviewLoading, setManualPreviewLoading] = useState(false);
  const [manualImagePreviewLoadedKey, setManualImagePreviewLoadedKey] = useState("");
  const [bagPreparation, setBagPreparation] = useState<BagPreparationProgress>({ active: false, percent: 0, message: "" });
  const [intrinsicsProfile, setIntrinsicsProfile] = useState<IntrinsicsProfile | null>(null);
  const [intrinsicsImporting, setIntrinsicsImporting] = useState(false);
  const [intrinsicsBrowserOpen, setIntrinsicsBrowserOpen] = useState(false);
  const [intrinsicsBrowser, setIntrinsicsBrowser] = useState<IntrinsicsBrowserResult | null>(null);
  const [intrinsicsBrowserLoading, setIntrinsicsBrowserLoading] = useState(false);
  const [selectedIntrinsicsPath, setSelectedIntrinsicsPath] = useState("");
  const [intrinsicsImportError, setIntrinsicsImportError] = useState("");
  const [datasetIndex, setDatasetIndex] = useState(0);
  const [lidarId, setLidarId] = useState("main");
  const [cameraId, setCameraId] = useState("front120");
  const [projectionModelOverrides, setProjectionModelOverrides] = useState<Record<string, ProjectionModel>>({});
  const [frameLabel, setFrameLabel] = useState<FrameLabel>("1");
  const [loadedFrameCloud, setLoadedFrameCloud] = useState<{ key: string; cloud: CloudData } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [stableOnly, setStableOnly] = useState(false);
  const [pointSize, setPointSize] = useState(0.04);
  const [projectionSize, setProjectionSize] = useState(1);
  const [projectionOpacity, setProjectionOpacity] = useState(0.7);
  const [maskOpacity, setMaskOpacity] = useState(0.25);
  const [projectionBackgroundOpacity, setProjectionBackgroundOpacity] = useState(1);
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [pointInteractionMode, setPointInteractionMode] = useState<PointInteractionMode>("navigate");
  const [pointSegment, setPointSegment] = useState<Uint32Array>(() => new Uint32Array());
  const pointSegmentRef = useRef<Uint32Array>(pointSegment);
  const pointSegmentHistoryRef = useRef<Uint32Array[]>([]);
  const pointSegmentRedoRef = useRef<Uint32Array[]>([]);
  const lastAnnotationEditRef = useRef<"image" | "point">("image");
  const [imageMask, setImageMask] = useState<ImageMask | null>(null);
  const [maskPixelCount, setMaskPixelCount] = useState(0);
  const imageMaskRef = useRef<ImageMask | null>(null);
  const [imageMaskLayers, setImageMaskLayers] = useState<ImageMaskLayer[]>([]);
  const imageMaskLayersRef = useRef<ImageMaskLayer[]>([]);
  const [activeMaskLayerId, setActiveMaskLayerId] = useState<string | null>(null);
  const activeMaskLayerIdRef = useRef<string | null>(null);
  const [manualAddedMask, setManualAddedMask] = useState<ImageMask | null>(null);
  const [manualRemovedMask, setManualRemovedMask] = useState<ImageMask | null>(null);
  const manualAddedMaskRef = useRef<ImageMask | null>(null);
  const manualRemovedMaskRef = useRef<ImageMask | null>(null);
  const imageMaskHistoryRef = useRef<ImageMaskSnapshot[]>([]);
  const imageMaskRedoRef = useRef<ImageMaskSnapshot[]>([]);
  const pendingAddedMaskLayerRef = useRef<string | null>(null);
  const currentMaskEditSourceRef = useRef<"sam" | "brush" | null>(null);
  const [imageTool, setImageTool] = useState<ImageSegmentationTool>("box-replace");
  const granularity: SegmentationGranularity = "fine";
  const [brushSize, setBrushSize] = useState(35);
  const [pointBrushSize, setPointBrushSize] = useState(28);
  const [pointBrushDepthTolerance, setPointBrushDepthTolerance] = useState(0.8);
  const [imageZoomResetToken, setImageZoomResetToken] = useState(0);
  const [imageRefinementView, setImageRefinementView] = useState<SynchronizedImageView>({ zoom: 1, centerX: 0.5, centerY: 0.5 });
  const [focusedImagePanel, setFocusedImagePanel] = useState<"image" | "projection" | null>(null);
  const [projectionZoom, setProjectionZoom] = useState(1);
  const [pointMode, setPointMode] = useState<PointSegmentationMode>("replace");
  const [projectionPicking, setProjectionPicking] = useState(false);
  const [clusterVoxelSize, setClusterVoxelSize] = useState(0.25);
  const [surfaceAngle, setSurfaceAngle] = useState(35);
  const [surfaceThickness, setSurfaceThickness] = useState(0.12);
  const [removeGround, setRemoveGround] = useState(true);
  const [hideGround, setHideGround] = useState(false);
  const [pointGrowFeedback, setPointGrowFeedback] = useState<PointGrowFeedback>({
    kind: "success",
    text: "在右下投影图点击目标上的雷达点",
  });

  const [savedAnnotations, setSavedAnnotations] = useState<SavedAnnotation[]>([]);
  const [lidarPairAnnotations, setLidarPairAnnotations] = useState<LidarPairAnnotation[]>([]);
  const [lidarPairCalibrations, setLidarPairCalibrations] = useState<LidarPairCalibration[]>([]);
  const [lidarFramePairSets, setLidarFramePairSets] = useState<LidarFramePairSet[]>([]);
  const [lidarPairNoAnnotationKeys, setLidarPairNoAnnotationKeys] = useState<string[]>([]);
  const [lidarTopicRemarks, setLidarTopicRemarks] = useState<Record<string, string>>({});
  const [lidarExtrinsicGraph, setLidarExtrinsicGraph] = useState<LidarExtrinsicGraph>({});
  const [activeAnnotationId, setActiveAnnotationId] = useState<string | null>(null);
  const [savingAnnotation, setSavingAnnotation] = useState(false);
  const savingAnnotationRef = useRef(false);
  const [pairPage, setPairPage] = useState(0);
  const [pairFrameFilter, setPairFrameFilter] = useState("all");
  const [annotationFeedback, setAnnotationFeedback] = useState<string>("标注自动同步到当前任务组");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [saveStates, setSaveStates] = useState<Record<string, string>>({camera: "已同步", lidar: "已同步"});
  const [saveRetry, setSaveRetry] = useState(0);
  const saveErrorsRef = useRef<Record<string, boolean>>({});
  useEffect(() => {
    const guard = (event: BeforeUnloadEvent) => {
      if (Object.values(saveStates).some(state => state !== "已同步")) { event.preventDefault(); event.returnValue = ""; }
    };
    window.addEventListener("beforeunload", guard);
    return () => window.removeEventListener("beforeunload", guard);
  }, [saveStates]);
  const [clearSelectionToken, setClearSelectionToken] = useState(0);
  const [projectedCount, setProjectedCount] = useState(0);
  const [viewRequest, setViewRequest] = useState<ViewRequest>({ mode: "near", token: 0 });
  const [stageMode, setStageMode] = useState<StageMode>("annotate");
  const [calibrations, setCalibrations] = useState<Record<string, OptimizationResult>>({});
  const [importedExtrinsics, setImportedExtrinsics] = useState<Record<string, ExtrinsicVersion[]>>({});
  const extrinsicImportRef = useRef<HTMLInputElement | null>(null);
  const [extrinsicInitializationMode, setExtrinsicInitializationMode] = useState<ExtrinsicInitializationMode>("forward");
  const [optimizing, setOptimizing] = useState(false);
  const [optimizationProgress, setOptimizationProgress] = useState(0);
  const [optimizationFeedback, setOptimizationFeedback] = useState("保存一个完整配对后即可优化；多配对通常更稳定");
  const [comparisonView, setComparisonView] = useState<SynchronizedImageView>({
    zoom: 1,
    centerX: 0.5,
    centerY: 0.5,
  });
  const [comparisonProjectionScope, setComparisonProjectionScope] = useState<ProjectionScope>("all");
  const [showSelectedPointProjection, setShowSelectedPointProjection] = useState(true);
  const [showFullPointProjection, setShowFullPointProjection] = useState(true);
  const [filterProjectionOcclusion, setFilterProjectionOcclusion] = useState(false);
  const [fullProjectionScope, setFullProjectionScope] = useState<FullProjectionScope>("all");
  const [singleProjectionAnnotationId, setSingleProjectionAnnotationId] = useState("");
  const [comparisonLeftVersionId, setComparisonLeftVersionId] = useState("");
  const [comparisonRightVersionId, setComparisonRightVersionId] = useState("");
  const [exportVersionId, setExportVersionId] = useState("");
  const pendingAnnotationRef = useRef<SavedAnnotation | null>(null);
  const connectivityCacheRef = useRef<{
    cloud: CloudData;
    voxelSize: number;
    stableOnly: boolean;
    removeGround: boolean;
    index: PointCloudConnectivityIndex;
  } | null>(null);

  const refreshManifest = async () => {
    const value = await fetch(`/api/calibration/manifest?updated=${Date.now()}`)
      .then((response) => {
        if (!response.ok) throw new Error("尚未生成标定数据");
        return response.json() as Promise<Manifest>;
      });
    setManifest(value);
    setDatasetIndex((current) => value.datasets.length ? Math.min(Math.max(0, current), value.datasets.length - 1) : 0);
    setError(null);
    return value;
  };

  useEffect(() => {
    refreshManifest().catch((reason: Error) => setError(reason.message));
  }, []);

  const refreshTaskGroups = async () => {
    const response = await fetch("/api/calibration/groups");
    if (!response.ok) throw new Error(`读取标定任务组失败：${response.status}`);
    const payload = await response.json() as { groups?: CalibrationTaskGroup[] };
    setTaskGroups(payload.groups ?? []);
    return payload.groups ?? [];
  };

  useEffect(() => {
    refreshTaskGroups().catch((reason: Error) => setAnnotationFeedback(reason.message));
  }, []);

  useEffect(() => {
    setIntrinsicsProfile(null);
    if (!activeTaskGroupId) return;
    let active = true;
    fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/intrinsics`)
      .then(async (response) => {
        if (!response.ok) throw new Error(`读取相机内参失败：${response.status}`);
        return response.json() as Promise<IntrinsicsProfile>;
      })
      .then((profile) => { if (active) setIntrinsicsProfile(profile); })
      .catch((reason: Error) => { if (active) setAnnotationFeedback(reason.message); });
    return () => { active = false; };
  }, [activeTaskGroupId, taskGroups]);

  useEffect(() => {
    // Load legacy local data once, after the manifest is ready. A second local
    // read must not replace newer server state and autosave an empty group.
    if (!manifest || localStateLoaded) return;
    let active = true;
    Promise.all([loadStoredAnnotations(), loadStoredCalibrations()])
      .then(([annotations, storedCalibrations]) => {
        if (!active) return;
        const migrated = annotations.map((annotation) => {
          const annotationDataset = manifest?.datasets.find((item) => item.id === annotation.datasetId);
          const rigId = annotationDataset?.rigId ?? "truck5-legacy";
          const taskGroupId = annotation.taskGroupId ?? defaultTaskGroupId(rigId);
          return annotation.taskGroupId ? annotation : { ...annotation, taskGroupId };
        });
        setSavedAnnotations(migrated);
        const normalized: Record<string, OptimizationResult> = {};
        storedCalibrations.forEach((item) => {
          if (!item.distortionModel) return;
          const key = calibrationModelKey(item.groupKey, item.distortionModel);
          if (key.includes(":e005-20260813:") && item.algorithmVersion !== CALIBRATION_ALGORITHM_VERSION) return;
          if (!normalized[key] || normalized[key].generatedAt < item.generatedAt) normalized[key] = item;
        });
        setCalibrations(normalized);
        setLocalStateLoaded(true);
      })
      .catch((reason: Error) => { if (active) setAnnotationFeedback(reason.message); });
    return () => { active = false; };
  }, [manifest, localStateLoaded]);

  const activeTaskGroup = taskGroups.find((group) => group.id === activeTaskGroupId) ?? null;
  const selectedDataset = manifest?.datasets[datasetIndex];
  const modeDatasets = manifest?.datasets.filter((item) =>
    activeTaskGroup?.datasetIds.includes(item.id) &&
    (calibrationWorkspaceMode === "lidar-lidar"
      ? item.lidars.length >= 2
      : item.lidars.length >= 1 && item.cameras.length >= 1),
  ) ?? [];
  const dataset = selectedDataset && modeDatasets.some((item) => item.id === selectedDataset.id)
    ? selectedDataset
    : modeDatasets[0];
  const archivedDatasets = modeDatasets
    .map((item) => ({ item, index: manifest?.datasets.findIndex((candidate) => candidate.id === item.id) ?? -1 }));
  const lidar = dataset?.lidars.find((item) => item.id === lidarId) ?? dataset?.lidars[0];
  const camera = dataset?.cameras.find((item) => item.id === cameraId) ?? dataset?.cameras[0];
  const changeLidarTopicRemark = (topic: string, remark: string) => {
    setLidarTopicRemarks((current) => {
      const next = { ...current };
      if (remark.trim()) next[topic] = remark.trim();
      else delete next[topic];
      return next;
    });
  };
  const promptLidarTopicRemark = (topic: string | undefined) => {
    if (!topic) return;
    const value = window.prompt(`设置 ${topic} 的备注（留空则删除）`, lidarTopicRemarks[topic] ?? "");
    if (value !== null) changeLidarTopicRemark(topic, value);
  };
  const pairedAnnotationsForSensors = useMemo(() => savedAnnotations.filter((annotation) =>
    annotation.taskGroupId === activeTaskGroupId && annotation.status === "paired" &&
    annotation.lidarId === lidar?.id && annotation.cameraId === camera?.id,
  ), [activeTaskGroupId, camera?.id, lidar?.id, savedAnnotations]);
  const pairedDatasetIds = useMemo(() => new Set(pairedAnnotationsForSensors
    .map((annotation) => annotation.datasetId)), [pairedAnnotationsForSensors]);
  const pairedFrameKeys = useMemo(() => new Set(pairedAnnotationsForSensors
    .map((annotation) => `${annotation.datasetId}:${annotation.frame}`)), [pairedAnnotationsForSensors]);
  const frameIndex = frameIndexByLabel(lidar?.frames, frameLabel);
  const cameraFrameIndex = frameIndexByLabel(camera?.frames, frameLabel);
  const lidarFrame = lidar?.frames[frameIndex];
  const cameraFrame = camera?.frames[cameraFrameIndex];
  const importedCameraIntrinsics = dataset && camera
    ? intrinsicsProfile?.datasets[dataset.id]?.cameras[camera.id] ?? null
    : null;
  const projectionModelChoice: ProjectionModelChoice = camera
    ? projectionModelOverrides[camera.id] ?? "auto"
    : "auto";
  const projectionModelOverride = projectionModelChoice === "auto" ? undefined : projectionModelChoice;
  const displayedCameraFrame = cameraFrame;
  const temporalProjection = temporalProjectionFor(dataset, lidar?.id, camera?.id, frameLabel);
  const imageAnnotationEnabled = true;
  const cloudRequestKey = `${dataset?.id}:${lidar?.id}:${lidarFrame?.url}`;
  const currentCloud = loadedFrameCloud?.key === cloudRequestKey ? loadedFrameCloud.cloud : null;
  const clouds = useMemo(() => {
    const frames: Array<CloudData | null> = [];
    if (currentCloud) frames[frameIndex] = currentCloud;
    return frames;
  }, [currentCloud, frameIndex]);
  const activeExtraction=extractionView?.datasetId===dataset?.id&&extractionView?.frame===frameLabel?extractionView:null;
  const extractionOverlays=useMemo(()=>{
    if(!activeExtraction||!currentCloud)return EMPTY_CLOUD_OVERLAYS;
    return activeExtraction.planes.map((p:{indices:number[];color:number})=>{
      const indices=p.indices.filter(i=>i>=0&&i<currentCloud.stable.length);
      const positions=new Float32Array(indices.length*3);
      indices.forEach((id,i)=>positions.set(currentCloud.positions.subarray(id*3,id*3+3),i*3));
      return {cloud:{...currentCloud,positions,stable:new Uint8Array(indices.length).fill(1),ground:new Uint8Array(indices.length)},color:p.color,size:2};
    });
  },[activeExtraction,currentCloud]);
  const visibleReviewHighlight = useMemo(() => {
    const annotation = savedAnnotations.find((item) => item.id === reviewHighlight?.id);
    if (!currentCloud || !reviewHighlight || annotation?.datasetId !== dataset?.id || annotation?.frame !== frameLabel || annotation?.lidarId !== lidar?.id || annotation?.cameraId !== camera?.id) return null;
    const selected = new Set(activeAnnotationId === annotation.id ? pointSegment : annotation.pointIndices);
    if (selectedOnly && activeAnnotationId !== annotation.id) return null;
    return {...reviewHighlight, indices: reviewHighlight.indices.filter((i) => i >= 0 && i < currentCloud.stable.length && (!selected || selected.has(i)))};
  }, [currentCloud, reviewHighlight, savedAnnotations, dataset?.id, frameLabel, lidar?.id, camera?.id, selectedOnly, pointSegment, activeAnnotationId]);
  const reviewOverlayClouds = useMemo(() => {
    if (!currentCloud || !visibleReviewHighlight) return EMPTY_CLOUD_OVERLAYS;
    const ids = visibleReviewHighlight.indices;
    const positions = new Float32Array(ids.length*3);
    ids.forEach((id,i)=>positions.set(currentCloud.positions.subarray(id*3,id*3+3),i*3));
    return [{cloud:{...currentCloud,positions,stable:new Uint8Array(ids.length).fill(1),ground:new Uint8Array(ids.length)},color:visibleReviewHighlight.color}];
  }, [currentCloud, visibleReviewHighlight]);
  const currentRigId = dataset?.rigId ?? "truck5-legacy";

  const applyMaskLayers = (
    layers: ImageMaskLayer[],
    activeId: string | null,
    added: ImageMask | null = manualAddedMaskRef.current,
    removed: ImageMask | null = manualRemovedMaskRef.current,
  ) => {
    const combined = applyManualMaskOverrides(combineMaskLayers(layers), added, removed);
    imageMaskLayersRef.current = layers;
    activeMaskLayerIdRef.current = activeId;
    manualAddedMaskRef.current = added;
    manualRemovedMaskRef.current = removed;
    imageMaskRef.current = combined;
    setImageMaskLayers(layers);
    setActiveMaskLayerId(activeId);
    setManualAddedMask(added);
    setManualRemovedMask(removed);
    setImageMask(combined);
    setMaskPixelCount(countMask(combined));
  };

  const loadImageMaskState = (
    value: ImageMask | null,
    serialized?: SavedAnnotation["imageMaskLayers"],
    serializedAdded?: SavedAnnotation["manualAddedMask"],
    serializedRemoved?: SavedAnnotation["manualRemovedMask"],
  ) => {
    const layers: ImageMaskLayer[] = serialized?.length
      ? serialized.map((layer) => ({
          id: layer.id,
          mask: decodeMaskRle(layer.width, layer.height, layer.rle),
          granularity: layer.granularity,
          brushModified: layer.brushModified,
        }))
      : value ? [{ id: `mask-${Date.now()}`, mask: copyMask(value), granularity: "fine", brushModified: true }] : [];
    const activeId = layers[0]?.id ?? null;
    const added = serializedAdded
      ? decodeMaskRle(serializedAdded.width, serializedAdded.height, serializedAdded.rle)
      : null;
    const removed = serializedRemoved
      ? decodeMaskRle(serializedRemoved.width, serializedRemoved.height, serializedRemoved.rle)
      : null;
    applyMaskLayers(layers, activeId, added, removed);
    imageMaskHistoryRef.current = [];
    imageMaskRedoRef.current = [];
  };

  const loadPointSegmentState = (value: Uint32Array) => {
    pointSegmentRef.current = value;
    setPointSegment(value);
    pointSegmentHistoryRef.current = [];
    pointSegmentRedoRef.current = [];
  };

  const updatePointSegment = (updater: (current: Uint32Array) => Uint32Array) => {
    const current = pointSegmentRef.current;
    const next = updater(current);
    if (next.length === current.length && next.every((value, index) => value === current[index])) return;
    setReviewEditRevision((value)=>value+1);
    if (activeAnnotationId) setReviewPairEdits((all)=>({...all,[activeAnnotationId]:(all[activeAnnotationId]??0)+1}));
    if (activeAnnotationId) setOptimizationFeedback('当前配对已修改，请保存；可继续处理其他配对，最后统一复查。');
    pointSegmentHistoryRef.current.push(current.slice());
    if (pointSegmentHistoryRef.current.length > 30) pointSegmentHistoryRef.current.shift();
    pointSegmentRedoRef.current = [];
    pointSegmentRef.current = next;
    lastAnnotationEditRef.current = "point";
    setPointSegment(next);
  };

  const beginImageMaskEdit = (source: "sam" | "brush") => {
    setReviewEditRevision((value)=>value+1);
    if (activeAnnotationId) setReviewPairEdits((all)=>({...all,[activeAnnotationId]:(all[activeAnnotationId]??0)+1}));
    if (activeAnnotationId) setOptimizationFeedback('当前配对已修改，请保存；可继续处理其他配对，最后统一复查。');
    imageMaskHistoryRef.current.push({
      layers: copyMaskLayers(imageMaskLayersRef.current),
      activeLayerId: activeMaskLayerIdRef.current,
      manualAddedMask: manualAddedMaskRef.current ? copyMask(manualAddedMaskRef.current) : null,
      manualRemovedMask: manualRemovedMaskRef.current ? copyMask(manualRemovedMaskRef.current) : null,
    });
    if (imageMaskHistoryRef.current.length > 30) imageMaskHistoryRef.current.shift();
    imageMaskRedoRef.current = [];
    lastAnnotationEditRef.current = "image";
    currentMaskEditSourceRef.current = source;
    pendingAddedMaskLayerRef.current = source === "sam" && imageTool === "box-add"
      ? `mask-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`
      : null;
  };

  const updateImageMask = (value: ImageMask | null, change?: BrushMaskChange) => {
    if (!value) return;
    const source = currentMaskEditSourceRef.current;
    const current = imageMaskLayersRef.current;
    const activeId = activeMaskLayerIdRef.current;
    if (source === "brush" && change) {
      const overrides = updateManualMaskOverridesRegion(
        imageMaskRef.current,
        value,
        manualAddedMaskRef.current,
        manualRemovedMaskRef.current,
        change.dirtyRects,
      );
      manualAddedMaskRef.current = overrides.added;
      manualRemovedMaskRef.current = overrides.removed;
      imageMaskRef.current = value;
      setManualAddedMask(overrides.added);
      setManualRemovedMask(overrides.removed);
      setImageMask(value);
      setMaskPixelCount((count) => Math.max(0, Math.min(value.width * value.height, count + change.pixelDelta)));
      return;
    }
    const addedId = source === "sam" && imageTool === "box-add" ? pendingAddedMaskLayerRef.current : null;
    const targetId = addedId ?? activeId ?? `mask-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
    const existing = current.find((layer) => layer.id === targetId);
    const nextLayer: ImageMaskLayer = {
      id: targetId,
      mask: copyMask(value),
      granularity: source === "sam" ? granularity : existing?.granularity ?? granularity,
      brushModified: source === "brush" || Boolean(existing?.brushModified && source !== "sam"),
    };
    const next = existing
      ? current.map((layer) => layer.id === targetId ? nextLayer : layer)
      : [...current, nextLayer];
    applyMaskLayers(next, targetId);
  };

  useEffect(() => {
    imageMaskRef.current = imageMask;
  }, [imageMask]);

  useEffect(() => {
    imageMaskHistoryRef.current = [];
    imageMaskRedoRef.current = [];
  }, [dataset?.id, camera?.id, frameLabel]);

  useEffect(() => {
    const handleHistoryShortcut = (event: KeyboardEvent) => {
      if (!event.ctrlKey && !event.metaKey) return;
      const target = event.target;
      if (target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement ||
          target instanceof HTMLSelectElement || (target instanceof HTMLElement && target.isContentEditable)) return;
      const key = event.key.toLowerCase();
      const redo = key === "y" || (key === "z" && event.shiftKey);
      const undo = key === "z" && !event.shiftKey;
      if (!undo && !redo) return;
      if (lastAnnotationEditRef.current === "point") {
        const source = redo ? pointSegmentRedoRef.current : pointSegmentHistoryRef.current;
        const destination = redo ? pointSegmentHistoryRef.current : pointSegmentRedoRef.current;
        const snapshot = source.pop();
        if (!snapshot) return;
        event.preventDefault();
        destination.push(pointSegmentRef.current.slice());
        const restored = snapshot.slice();
        pointSegmentRef.current = restored;
        setPointSegment(restored);
        return;
      }
      if (!imageAnnotationEnabled) return;
      const source = redo ? imageMaskRedoRef.current : imageMaskHistoryRef.current;
      const destination = redo ? imageMaskHistoryRef.current : imageMaskRedoRef.current;
      const snapshot = source.pop();
      if (!snapshot) return;
      event.preventDefault();
      destination.push({
        layers: copyMaskLayers(imageMaskLayersRef.current),
        activeLayerId: activeMaskLayerIdRef.current,
        manualAddedMask: manualAddedMaskRef.current ? copyMask(manualAddedMaskRef.current) : null,
        manualRemovedMask: manualRemovedMaskRef.current ? copyMask(manualRemovedMaskRef.current) : null,
      });
      applyMaskLayers(
        copyMaskLayers(snapshot.layers),
        snapshot.activeLayerId,
        snapshot.manualAddedMask ? copyMask(snapshot.manualAddedMask) : null,
        snapshot.manualRemovedMask ? copyMask(snapshot.manualRemovedMask) : null,
      );
    };
    window.addEventListener("keydown", handleHistoryShortcut);
    return () => window.removeEventListener("keydown", handleHistoryShortcut);
  }, [imageAnnotationEnabled]);

  const selectTaskGroup = (groupId: string) => {
    setGroupStateReady(false);
    setActiveTaskGroupId(groupId);
    setActiveAnnotationId(null);
    loadImageMaskState(null);
    loadPointSegmentState(new Uint32Array());
    const group = taskGroups.find((item) => item.id === groupId);
    const nextIndex = manifest?.datasets.findIndex((item) => group?.datasetIds.includes(item.id)) ?? -1;
    if (nextIndex >= 0) setDatasetIndex(nextIndex);
  };
  const [loadingDemo, setLoadingDemo] = useState(false);
  const openDemo = async () => {
    setLoadingDemo(true);
    try {
      const response = await fetch("/api/calibration/demo", {method:"POST"});
      const group = await response.json();
      if (!response.ok) throw new Error(group.detail ?? "示例载入失败");
      await refreshManifest();
      await refreshTaskGroups();
      selectTaskGroup(group.id);
      setAnnotationFeedback("CARLA 示例已载入：19 帧、20 对标注");
    } catch (error) { setAnnotationFeedback(error instanceof Error ? error.message : "示例载入失败"); }
    finally { setLoadingDemo(false); }
  };

  const createTaskGroup = async () => {
    const now = new Date();
    const pad = (value: number) => String(value).padStart(2, "0");
    const defaultName = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}_${pad(now.getHours())}-${pad(now.getMinutes())}-${pad(now.getSeconds())}`;
    const name = window.prompt("请输入新标定组名称（该名称将直接作为文件夹名）", defaultName)?.trim();
    if (!name) return;
    try {
      const response = await fetch("/api/calibration/groups", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, rig_id: "unassigned", dataset_ids: [] }),
      });
      if (!response.ok) throw new Error(`新建标定组失败：${response.status}`);
      const created = await response.json() as CalibrationTaskGroup;
      await refreshTaskGroups();
      selectTaskGroup(created.id);
      setAnnotationFeedback(`已新建标定组“${created.name}”，目录：${created.storagePath}`);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "新建标定组失败");
    }
  };
  const deleteTaskGroup = async () => {
    if (!activeTaskGroup || groupDeleting) return;
    const confirmed = window.confirm(
      `确定删除标定组“${activeTaskGroup.name}”吗？\n\n该组内的标注、设置和标定结果会被永久删除。公共数据包、内参和预处理缓存不会删除。`,
    );
    if (!confirmed) return;
    const deletedId = activeTaskGroup.id;
    const deletedName = activeTaskGroup.name;
    setGroupDeleting(true);
    try {
      const response = await fetch(`/api/calibration/groups/${encodeURIComponent(deletedId)}`, { method: "DELETE" });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `删除标定组失败：${response.status}`);
      setGroupStateReady(false);
      setActiveTaskGroupId("");
      setActiveAnnotationId(null);
      setIntrinsicsProfile(null);
      setSavedAnnotations((current) => current.filter((item) => item.taskGroupId !== deletedId));
      setCalibrations((current) => Object.fromEntries(
        Object.entries(current).filter(([key]) => !key.startsWith(`shared:${deletedId}:`)),
      ));
      setImportedExtrinsics((current) => Object.fromEntries(
        Object.entries(current).filter(([key]) => !key.startsWith(`shared:${deletedId}:`)),
      ));
      loadImageMaskState(null);
      loadPointSegmentState(new Uint32Array());
      await refreshTaskGroups();
      setAnnotationFeedback(`已删除标定组“${deletedName}”；公共数据包和缓存已保留`);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "删除标定组失败");
    } finally {
      setGroupDeleting(false);
    }
  };
  const browseBagDirectory = async (path: string) => {
    setBagBrowserLoading(true);
    try {
      const response = await fetch(`/api/calibration/bags?path=${encodeURIComponent(path)}`);
      if (!response.ok) throw new Error((await response.json()).detail ?? `读取目录失败：${response.status}`);
      setBagBrowser(await response.json() as BagBrowserResult);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "读取数据包目录失败");
    } finally {
      setBagBrowserLoading(false);
    }
  };
  const browsePairedDirectory = async (path: string) => {
    setPairedBrowserLoading(true);
    try {
      const response = await fetch(`/api/calibration/paired?path=${encodeURIComponent(path)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `读取成对数据目录失败：${response.status}`);
      setPairedBrowser(payload as PairedDatasetBrowserResult);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "读取成对数据目录失败");
    } finally {
      setPairedBrowserLoading(false);
    }
  };
  const openBagBrowser = () => {
    setEditingDatasetId(null);
    setSelectedBags({});
    setManualTimeline(null);
    setManualFrameIndex(0);
    setManualCameraTopic("");
    setSelectedManualFrames({});
    setAllManualFrameSelections({});
    setManualSelectionVersion(0);
    setManualSelectionDirty(false);
    setBagImportSource("local");
    setSelectedPairedDatasets({});
    setPairedBrowser(null);
    setLocalBagFiles([]);
    setLocalBagPackageName("");
    setLocalBagUpload({ active: false, paused: false, percent: 0, message: "", uploadedBytes: 0, totalBytes: 0 });
    setLocalBagUploadError("");
    setBagBrowserOpen(true);
    void browseBagDirectory("");
  };
  const switchBagImportSource = (source: "local" | "server" | "paired") => {
    if (source === bagImportSource) return;
    setBagImportSource(source);
    setSelectedBags({});
    setSelectedPairedDatasets({});
    setManualTimeline(null);
    setManualTimelineLoading(false);
    setManualFrameIndex(0);
    setManualCameraTopic("");
    setManualLidarTopic("");
    setSelectedManualFrames({});
    setAllManualFrameSelections({});
    setManualSelectionVersion(0);
    setManualSelectionDirty(false);
    setManualSelectionSync("connecting");
    if (source === "server") void browseBagDirectory("");
    if (source === "paired") void browsePairedDirectory("");
  };

  const selectLocalBagFiles = (files: FileList | null) => {
    if (!files?.length) return;
    const selected = [...files];
    const firstRelative = selected[0].webkitRelativePath?.replaceAll("\\", "/") ?? "";
    const rootName = firstRelative.split("/")[0];
    const allowed = new Set([".db3", ".mcap", ".yaml", ".yml", ".json"]);
    const normalized = selected.flatMap((file) => {
      const browserPath = file.webkitRelativePath?.replaceAll("\\", "/") ?? "";
      const relativePath = browserPath
        ? browserPath.split("/").slice(1).join("/")
        : file.name;
      const suffix = `.${file.name.split(".").at(-1)?.toLowerCase() ?? ""}`;
      return relativePath && allowed.has(suffix) ? [{ file, relativePath }] : [];
    });
    if (!normalized.some((item) => /\.(db3|mcap)$/i.test(item.relativePath))) {
      setLocalBagFiles([]);
      setLocalBagUploadError("请选择包含 .db3 或 .mcap 文件的 rosbag 文件夹");
      return;
    }
    const fallbackName = selected.find((file) => /\.(db3|mcap)$/i.test(file.name))?.name.replace(/\.(db3|mcap)$/i, "") ?? "rosbag";
    setLocalBagFiles(normalized);
    setLocalBagPackageName(rootName || fallbackName);
    setLocalBagUploadError("");
    const totalBytes = normalized.reduce((sum, item) => sum + item.file.size, 0);
    setLocalBagUpload({ active: false, paused: false, percent: 0, message: "已选择，等待上传", uploadedBytes: 0, totalBytes });
  };

  const pauseLocalBagUpload = () => {
    localBagUploadAbortRef.current?.abort();
    setLocalBagUpload((current) => ({ ...current, active: false, paused: true, message: "已暂停，可继续上传" }));
  };

  const uploadLocalBag = async () => {
    if (!activeTaskGroup || !localBagFiles.length || localBagUpload.active) return;
    const totalBytes = localBagFiles.reduce((sum, item) => sum + item.file.size, 0);
    const controller = new AbortController();
    localBagUploadAbortRef.current = controller;
    setLocalBagUploadError("");
    setLocalBagUpload((current) => ({ ...current, active: true, paused: false, totalBytes, message: "正在检查服务器续传进度…" }));
    const putChunk = (
      url: string,
      body: Blob,
      offset: number,
      onProgress: (uploaded: number) => void,
    ) => new Promise<{ status: number; payload: { offset?: number; detail?: string | { offset?: number } } }>((resolve, reject) => {
      const request = new XMLHttpRequest();
      const abortRequest = () => request.abort();
      const cleanup = () => controller.signal.removeEventListener("abort", abortRequest);
      request.open("PUT", url);
      request.setRequestHeader("Content-Type", "application/octet-stream");
      request.setRequestHeader("X-Upload-Offset", String(offset));
      request.upload.onprogress = (event) => onProgress(Math.min(body.size, event.loaded));
      request.onload = () => {
        cleanup();
        try {
          resolve({ status: request.status, payload: JSON.parse(request.responseText || "{}") });
        } catch {
          reject(new Error(`服务器返回了无效响应：${request.status}`));
        }
      };
      request.onerror = () => { cleanup(); reject(new Error("上传分片时网络连接中断")); };
      request.onabort = () => { cleanup(); reject(new DOMException("上传已暂停", "AbortError")); };
      controller.signal.addEventListener("abort", abortRequest, { once: true });
      if (controller.signal.aborted) abortRequest();
      else request.send(body);
    });
    try {
      const initResponse = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/uploads`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          package_name: localBagPackageName,
          files: localBagFiles.map((item) => ({ relative_path: item.relativePath, size: item.file.size, last_modified: item.file.lastModified })),
        }),
        signal: controller.signal,
      });
      const initPayload = await initResponse.json();
      if (!initResponse.ok) throw new Error(initPayload.detail ?? `创建上传任务失败：${initResponse.status}`);
      let session = initPayload as BagUploadSession;
      if (!session.complete) {
        let uploadedBytes = session.files.reduce((sum, item) => sum + item.offset, 0);
        setLocalBagUpload({ active: true, paused: false, percent: totalBytes ? Math.floor(uploadedBytes * 100 / totalBytes) : 100, message: uploadedBytes ? "已找到未完成上传，正在续传…" : "开始分片上传…", uploadedBytes, totalBytes });
        for (const remoteFile of session.files) {
          const localFile = localBagFiles.find((item) => item.relativePath === remoteFile.relativePath);
          if (!localFile) throw new Error(`重新选择的文件中缺少：${remoteFile.relativePath}`);
          let offset = remoteFile.offset;
          while (offset < localFile.file.size) {
            const end = Math.min(offset + session.chunkSize, localFile.file.size);
            let result: Awaited<ReturnType<typeof putChunk>> | null = null;
            for (let attempt = 0; attempt < 4; attempt += 1) {
              try {
                result = await putChunk(
                  `/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/uploads/${session.uploadId}/files/${remoteFile.index}`,
                  localFile.file.slice(offset, end),
                  offset,
                  (chunkUploaded) => {
                    const visibleBytes = Math.min(totalBytes, uploadedBytes + chunkUploaded);
                    setLocalBagUpload({
                      active: true,
                      paused: false,
                      percent: totalBytes ? Math.min(100, Math.floor(visibleBytes * 100 / totalBytes)) : 100,
                      message: `正在上传 ${localFile.relativePath}`,
                      uploadedBytes: visibleBytes,
                      totalBytes,
                    });
                  },
                );
                break;
              } catch (reason) {
                if (controller.signal.aborted) throw reason;
                if (attempt === 3) throw reason;
                await new Promise((resolve) => window.setTimeout(resolve, 700 * 2 ** attempt));
              }
            }
            if (!result) throw new Error("上传请求未完成");
            const { status, payload } = result;
            const mismatchOffset = typeof payload.detail === "object" ? payload.detail.offset : undefined;
            if (status === 409 && typeof mismatchOffset === "number") {
              const serverOffset = Number(mismatchOffset);
              uploadedBytes += serverOffset - offset;
              offset = serverOffset;
              continue;
            }
            if (status < 200 || status >= 300) throw new Error(typeof payload.detail === "string" ? payload.detail : `上传分片失败：${status}`);
            const previousOffset = offset;
            offset = Number(payload.offset);
            uploadedBytes += offset - previousOffset;
            setLocalBagUpload({
              active: true,
              paused: false,
              percent: totalBytes ? Math.min(100, Math.floor(uploadedBytes * 100 / totalBytes)) : 100,
              message: `正在上传 ${localFile.relativePath}`,
              uploadedBytes,
              totalBytes,
            });
          }
        }
        const finishResponse = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/uploads/${session.uploadId}/complete`, { method: "POST", signal: controller.signal });
        const finishPayload = await finishResponse.json();
        if (!finishResponse.ok) throw new Error(finishPayload.detail ?? `完成上传失败：${finishResponse.status}`);
        session = finishPayload as BagUploadSession;
      }
      if (!session.entry) throw new Error("上传完成，但服务器没有返回 rosbag 目录");
      setLocalBagUpload({ active: false, paused: false, percent: 100, message: "上传完成，正在读取可选帧…", uploadedBytes: totalBytes, totalBytes });
      setBagImportSource("server");
      await browseBagDirectory(session.entry.path.split("/").slice(0, -1).join("/"));
      await toggleSelectedBag(session.entry);
    } catch (reason) {
      if (controller.signal.aborted) {
        setLocalBagUpload((current) => ({ ...current, active: false, paused: true, message: "已暂停，可继续上传" }));
      } else {
        const message = reason instanceof Error ? reason.message : "上传数据包失败";
        setLocalBagUploadError(message);
        setLocalBagUpload((current) => ({ ...current, active: false, paused: true, message: "上传中断，点击继续会从断点恢复" }));
      }
    } finally {
      if (localBagUploadAbortRef.current === controller) localBagUploadAbortRef.current = null;
    }
  };
  const browseIntrinsicsDirectory = async (path: string) => {
    setIntrinsicsBrowserLoading(true);
    try {
      const response = await fetch(`/api/calibration/intrinsics?path=${encodeURIComponent(path)}`);
      if (!response.ok) throw new Error((await response.json()).detail ?? `读取内参目录失败：${response.status}`);
      setIntrinsicsBrowser(await response.json() as IntrinsicsBrowserResult);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "读取内参目录失败");
    } finally {
      setIntrinsicsBrowserLoading(false);
    }
  };
  const openIntrinsicsBrowser = () => {
    setSelectedIntrinsicsPath("");
    setIntrinsicsImportError("");
    setIntrinsicsBrowserOpen(true);
    void browseIntrinsicsDirectory("");
  };
  const importCameraIntrinsics = async () => {
    if (!selectedIntrinsicsPath || !activeTaskGroup) return;
    const fileName = selectedIntrinsicsPath.split("/").at(-1) ?? selectedIntrinsicsPath;
    setIntrinsicsImporting(true);
    setIntrinsicsImportError("");
    setAnnotationFeedback(`正在导入原始图像投影参数：${fileName}`);
    try {
      const response = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/intrinsics`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ source_path: selectedIntrinsicsPath }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `导入相机内参失败：${response.status}`);
      const profile = payload as IntrinsicsProfile;
      setIntrinsicsProfile(profile);
      loadImageMaskState(null);
      setImageZoomResetToken((value) => value + 1);
      setProjectionZoom(1);
      await refreshTaskGroups();
      setIntrinsicsBrowserOpen(false);
      setAnnotationFeedback(`内参导入成功：${fileName}；标注与投影均使用原始畸变图坐标`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "导入相机内参失败";
      setIntrinsicsImportError(message);
      setAnnotationFeedback(message);
    } finally {
      setIntrinsicsImporting(false);
    }
  };
  const currentManualSelectionClientId = () => {
    if (!manualSelectionClientId.current) {
      manualSelectionClientId.current = `browser-${globalThis.crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2)}`;
    }
    return manualSelectionClientId.current;
  };
  const loadBagForFrameSelection = async (entry: BagBrowserEntry) => {
    setSelectedBags({ [entry.path]: entry });
    setManualTimeline(null);
    setManualFrameIndex(0);
    setManualCameraTopic("");
    setManualLidarTopic("");
    setSelectedManualFrames({});
    setAllManualFrameSelections({});
    setManualSelectionVersion(0);
    setManualSelectionDirty(false);
    setManualSelectionSync("connecting");
    setManualTimelineLoading(true);
    try {
      const response = await fetch(`/api/calibration/bags/manual-frames?path=${encodeURIComponent(entry.path)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail ?? `读取 1 秒帧列表失败：${response.status}`);
      const timeline = payload as ManualFrameTimeline;
      setManualCameraTopic(timeline.previewTopic || timeline.cameras[0]?.topic || "");
      setManualLidarTopic(timeline.lidars[0]?.topic || "");
      setManualTimeline(timeline);
      setManualTimelineLoading(false);
      setSelectedManualFrames({});
      setAllManualFrameSelections({});
      // The group-scoped selection is loaded after the timeline is ready.
    } catch (reason) {
      setSelectedBags({});
      setAnnotationFeedback(reason instanceof Error ? reason.message : "读取 1 秒帧列表失败");
    } finally {
      setManualTimelineLoading(false);
    }
  };
  const toggleSelectedBag = async (entry: BagBrowserEntry) => {
    if (selectedBags[entry.path]) {
      setSelectedBags({});
      setManualTimeline(null);
      setManualCameraTopic("");
      setManualLidarTopic("");
      setSelectedManualFrames({});
      setManualSelectionSync("connecting");
      return;
    }
    await loadBagForFrameSelection(entry);
  };
  const toggleSelectedPairedDataset = (entry: PairedDatasetBrowserEntry) => {
    if (entry.kind !== "dataset" || !entry.valid) return;
    setSelectedPairedDatasets((current) => {
      const next = { ...current };
      if (next[entry.path]) delete next[entry.path];
      else next[entry.path] = entry;
      return next;
    });
  };
  const openSourceFrameEditor = async (sourcePath: string, item?: DatasetInfo) => {
    if (!sourcePath) {
      setAnnotationFeedback("这个历史数据包没有保存源路径，无法重新选帧");
      return;
    }
    const entry: BagBrowserEntry = {
      name: sourcePath.split("/").at(-1) || item?.name || "rosbag",
      path: sourcePath,
      kind: "bag",
      sizeBytes: 0,
      datasetIds: item ? [item.id] : [],
      rigIds: item?.rigId ? [item.rigId] : [],
      prepared: true,
    };
    setEditingDatasetId(item?.id ?? sourcePath);
    setBagImportSource("server");
    setBagBrowserOpen(true);
    void browseBagDirectory(sourcePath.split("/").slice(0, -1).join("/"));
    await loadBagForFrameSelection(entry);
  };
  const openDatasetFrameEditor = async (item: DatasetInfo) => {
    await openSourceFrameEditor(item.sourcePath ?? "", item);
  };
  const activeManualTopic = calibrationWorkspaceMode === "lidar-lidar" ? manualLidarTopic : manualCameraTopic;
  const currentManualFrame = manualTimeline?.candidates[manualFrameIndex] ?? null;
  const manualImagePreviewKey = manualTimeline && currentManualFrame && activeManualTopic
    ? `${manualTimeline.path}:${activeManualTopic}:${currentManualFrame.timestampNs}`
    : "";
  const visibleManualPreviewLoading = calibrationWorkspaceMode === "lidar-lidar"
    ? manualPreviewLoading
    : Boolean(manualImagePreviewKey && manualImagePreviewLoadedKey !== manualImagePreviewKey);
  const manualLidarDisplayTransform = useMemo(() => {
    if (calibrationWorkspaceMode !== "lidar-lidar" || !manifest || !manualTimeline) return null;
    const lidarId = manualTimeline.lidars.find((item) => item.topic === activeManualTopic)?.id;
    if (!lidarId) return null;
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    if (lidarId === "main") return identity;
    const carFromMain = manifest.calibration.lidars.main?.carFromLidar;
    const carFromLidar = manifest.calibration.lidars[lidarId]?.carFromLidar;
    return carFromMain && carFromLidar ? multiply4(invertRigid4(carFromMain), carFromLidar) : null;
  }, [activeManualTopic, calibrationWorkspaceMode, manifest, manualTimeline]);
  const currentManualSelection = selectedManualFrames[activeManualTopic] ?? [];
  const selectedManualFrameCount = Object.values(selectedManualFrames).reduce((total, frames) => total + frames.length, 0);
  const selectedManualCameraCount = Object.values(selectedManualFrames).filter((frames) => frames.length > 0).length;
  const allManualFrameCount = Object.values(allManualFrameSelections).reduce((total, frames) => total + frames.length, 0);
  const protectedManualFrameKeys = useMemo(() => {
    const keys = new Set<string>();
    if (!manualTimeline || !manifest) return keys;
    const add = (topic: string | undefined, timestamp: string | undefined) => {
      if (topic && timestamp) keys.add(`${topic.replace(/^\/+/, "")}\n${timestamp}`);
    };
    savedAnnotations.filter((item) => item.taskGroupId === activeTaskGroupId).forEach((annotation) => {
      const itemDataset = manifest.datasets.find((item) => item.id === annotation.datasetId && item.sourcePath === manualTimeline.path);
      if (!itemDataset) return;
      const topic = itemDataset.cameras.find((item) => item.id === annotation.cameraId)?.topic;
      const timestamp = itemDataset.anchorFrames.find((item) => item.label === annotation.frame)?.timestampNs;
      add(topic, timestamp);
    });
    lidarPairAnnotations.filter((item) => item.taskGroupId === activeTaskGroupId).forEach((annotation) => {
      const itemDataset = manifest.datasets.find((item) => item.id === annotation.datasetId && item.sourcePath === manualTimeline.path);
      if (!itemDataset) return;
      add(itemDataset.lidars.find((item) => item.id === annotation.sourceLidarId)?.topic, itemDataset.anchorFrames.find((item) => item.label === annotation.sourceFrame)?.timestampNs);
      add(itemDataset.lidars.find((item) => item.id === annotation.targetLidarId)?.topic, itemDataset.anchorFrames.find((item) => item.label === annotation.targetFrame)?.timestampNs);
    });
    return keys;
  }, [activeTaskGroupId, lidarPairAnnotations, manifest, manualTimeline, savedAnnotations]);
  const currentManualFrameProtected = Boolean(currentManualFrame && activeManualTopic && protectedManualFrameKeys.has(`${activeManualTopic.replace(/^\/+/, "")}\n${currentManualFrame.timestampNs}`));
  useEffect(() => {
    if (!manualTimeline || !activeTaskGroupId) return;
    setManualSelectionSync("connecting");
    const query = new URLSearchParams({ source_path: manualTimeline.path });
    let active = true;
    fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/manual-selection?${query}`)
      .then(async (response) => {
        const state = await response.json() as ManualSelectionState & { detail?: string };
        if (!response.ok) throw new Error(state.detail ?? `读取选帧失败：${response.status}`);
        return state;
      })
      .then((state) => {
        if (!active) return;
        setAllManualFrameSelections(normalizeManualSelections(state.selections, manualTimeline));
        setManualSelectionVersion(state.version);
        setManualSelectionDirty(false);
        setManualSelectionSync("synced");
      })
      .catch(() => { if (active) setManualSelectionSync("error"); });
    return () => { active = false; };
  }, [activeTaskGroupId, manualTimeline]);
  useEffect(() => {
    if (!manualTimeline) return;
    setSelectedManualFrames(validManualSelections({ version: manualSelectionVersion, sourcePath: manualTimeline.path, selections: allManualFrameSelections, updatedAt: null, updatedBy: null }, manualTimeline, calibrationWorkspaceMode));
  }, [allManualFrameSelections, calibrationWorkspaceMode, manualSelectionVersion, manualTimeline]);
  useEffect(() => {
    if (!manualTimeline || !currentManualFrame || !activeManualTopic) return;
    if (calibrationWorkspaceMode === "lidar-lidar") {
      setManualPreviewLoading(true);
      return;
    }
    const previewUrl = (candidate: ManualFrameCandidate) =>
      `/api/calibration/bags/manual-preview?path=${encodeURIComponent(manualTimeline.path)}&timestamp_ns=${candidate.timestampNs}&camera_topic=${encodeURIComponent(activeManualTopic)}`;
    const neighbors = [manualTimeline.candidates[manualFrameIndex - 1], manualTimeline.candidates[manualFrameIndex + 1]]
      .filter((candidate): candidate is ManualFrameCandidate => Boolean(candidate));
    const timer = window.setTimeout(() => {
      neighbors.forEach((candidate) => { const image = new window.Image(); image.src = previewUrl(candidate); });
    }, 150);
    return () => window.clearTimeout(timer);
  }, [activeManualTopic, calibrationWorkspaceMode, currentManualFrame?.timestampNs, manualFrameIndex, manualTimeline]);
  const toggleCurrentManualFrame = () => {
    if (!currentManualFrame || !activeManualTopic || !manualTimeline) return;
    const selected = !currentManualSelection.includes(currentManualFrame.timestampNs);
    setAllManualFrameSelections((current) => {
      const values = new Set(current[activeManualTopic] ?? []);
      if (selected) values.add(currentManualFrame.timestampNs);
      else values.delete(currentManualFrame.timestampNs);
      const next = { ...current };
      if (values.size) next[activeManualTopic] = [...values].sort((left, right) => Number(BigInt(left) - BigInt(right)));
      else delete next[activeManualTopic];
      return next;
    });
    setManualSelectionDirty(true);
  };
  const importDatasetIntoGroup = async () => {
    if (!activeTaskGroup || bagPreparation.active) return;
    const bags = Object.values(selectedBags);
    if (!bags.length) return;
    const cameraTopics = new Set(manualTimeline?.cameras.map((item) => item.topic) ?? []);
    const lidarTopics = new Set(manualTimeline?.lidars.map((item) => item.topic) ?? []);
    const cameraFrameSelections = Object.fromEntries(Object.entries(allManualFrameSelections).filter(([topic, values]) => cameraTopics.has(topic) && values.length));
    const lidarFrameSelections = Object.fromEntries(Object.entries(allManualFrameSelections).filter(([topic, values]) => lidarTopics.has(topic) && values.length));
    try {
      setBagPreparation({ active: true, percent: 0, message: "正在提交原始数据包…" });
      if (manualSelectionDirty && manualTimeline) {
        const selectionResponse = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/manual-selection`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_path: manualTimeline.path,
            selections: allManualFrameSelections,
            expected_version: manualSelectionVersion,
            client_id: currentManualSelectionClientId(),
          }),
        });
        const selectionState = await selectionResponse.json() as ManualSelectionState & { detail?: string };
        if (!selectionResponse.ok) throw new Error(selectionState.detail ?? `保存选帧失败：${selectionResponse.status}`);
        setManualSelectionVersion(selectionState.version);
        setManualSelectionDirty(false);
      }
      const response = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_paths: bags.map((item) => item.path),
          selected_frames: {},
          selected_camera_frames: Object.fromEntries(bags.map((item) => [item.path, cameraFrameSelections])),
          selected_lidar_frames: Object.fromEntries(bags.map((item) => [item.path, lidarFrameSelections])),
          replace_frame_selection: Boolean(editingDatasetId),
        }),
      });
      if (!response.ok || !response.body) throw new Error(`处理服务返回 ${response.status}`);
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let datasetIds: string[] = [];
      let cachedCount = 0;
      let resultMessage = "";
      const consume = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as { type: "progress" | "result" | "error"; percent?: number; message?: string; bagIndex?: number; bagCount?: number; datasetIds?: string[]; cachedCount?: number };
        if (event.type === "error") throw new Error(event.message || "数据包处理失败");
        setBagPreparation({ active: event.type !== "result", percent: event.percent ?? 0, message: event.message ?? "处理中", bagIndex: event.bagIndex, bagCount: event.bagCount });
        if (event.type === "result") {
          datasetIds = event.datasetIds ?? [];
          cachedCount = event.cachedCount ?? 0;
          resultMessage = event.message ?? "";
        }
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(consume);
        if (done) break;
      }
      consume(buffer);
      // Prevent the autosave effect from writing stale, pre-migration dataset
      // IDs while the refreshed group state is being loaded below.
      setGroupStateReady(false);
      const nextManifest = await refreshManifest();
      const groups = await refreshTaskGroups();
      setTaskGroups(groups);
      const nextIndex = nextManifest.datasets.findIndex((item) => datasetIds.includes(item.id));
      if (nextIndex >= 0) setDatasetIndex(nextIndex);
      setBagBrowserOpen(false);
      setAnnotationFeedback(resultMessage || `已处理并导入 ${bags.length} 个数据包${cachedCount ? `，其中 ${cachedCount} 个命中缓存` : ""}`);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "导入数据包失败");
      setBagPreparation((current) => ({ ...current, active: false }));
    }
  };
  const importPairedDatasetsIntoGroup = async () => {
    if (!activeTaskGroup || bagPreparation.active) return;
    const sources = Object.values(selectedPairedDatasets);
    if (!sources.length) return;
    try {
      setBagPreparation({ active: true, percent: 0, message: "正在校验成对数据集…" });
      const response = await fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroup.id)}/paired-import`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ paired_source_paths: sources.map((item) => item.path) }),
      });
      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({})) as { detail?: string };
        throw new Error(payload.detail ?? `成对数据导入服务返回 ${response.status}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let datasetIds: string[] = [];
      let resultMessage = "";
      const consume = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as { type: "progress" | "result" | "error"; percent?: number; message?: string; datasetIndex?: number; datasetCount?: number; datasetIds?: string[] };
        if (event.type === "error") throw new Error(event.message || "成对数据集导入失败");
        setBagPreparation({
          active: event.type !== "result",
          percent: event.percent ?? 0,
          message: event.message ?? "处理中",
          bagIndex: event.datasetIndex,
          bagCount: event.datasetCount,
        });
        if (event.type === "result") {
          datasetIds = event.datasetIds ?? [];
          resultMessage = event.message ?? "";
        }
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(consume);
        if (done) break;
      }
      consume(buffer);
      setGroupStateReady(false);
      const nextManifest = await refreshManifest();
      const groups = await refreshTaskGroups();
      setTaskGroups(groups);
      const nextIndex = nextManifest.datasets.findIndex((item) => datasetIds.includes(item.id));
      if (nextIndex >= 0) setDatasetIndex(nextIndex);
      setBagBrowserOpen(false);
      setAnnotationFeedback(resultMessage || `已导入 ${sources.length} 个成对数据集`);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "成对数据集导入失败");
      setBagPreparation((current) => ({ ...current, active: false }));
    }
  };
  const currentCalibrationKey = lidar && camera
    ? calibrationGroupKey(activeTaskGroupId, currentRigId, lidar.id, camera.id)
    : null;
  const allCurrentExtrinsicVersions = currentCalibrationKey ? importedExtrinsics[currentCalibrationKey] ?? [] : [];

  useEffect(() => {
    if (!manifest || !localStateLoaded || !taskGroups.some((group) => group.id === activeTaskGroupId)) return;
    let active = true;
    setGroupStateReady(false);
    setProjectionModelOverrides({});
    Promise.all([
      fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/state/camera`).then(async (response) => {
        if (!response.ok) throw new Error(`读取相机标定状态失败：${response.status}`);
        return response.json() as Promise<{
          revision?: number;
          annotations?: SavedAnnotation[];
          calibrations?: OptimizationResult[];
          initialExtrinsics?: StoredExtrinsicVersion[];
          projectionModelOverrides?: Record<string, ProjectionModel>;
        }>;
      }),
      fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/state/lidar`).then(async (response) => {
        if (!response.ok) throw new Error(`读取雷达标定状态失败：${response.status}`);
        return response.json() as Promise<{
          revision?: number;
          lidarPairAnnotations?: LidarPairAnnotation[];
          lidarPairCalibrations?: LidarPairCalibration[];
          lidarFramePairSets?: LidarFramePairSet[];
          lidarPairNoAnnotationKeys?: string[];
          lidarTopicRemarks?: Record<string, string>;
          extrinsicGraph?: LidarExtrinsicGraph;
        }>;
      }),
    ])
      .then(([cameraState, lidarState]) => {
        if (!active) return;
        domainRevisionRef.current = {
          camera: cameraState.revision ?? 0,
          lidar: lidarState.revision ?? 0,
        };
        skipDomainSaveRef.current = { camera: true, lidar: true };
        const serverAnnotations = (cameraState.annotations ?? []).map((item) => ({ ...item, taskGroupId: activeTaskGroupId }));
        setSavedAnnotations((current) => [
          ...current.filter((item) => item.taskGroupId !== activeTaskGroupId),
          ...serverAnnotations,
        ].sort((left, right) => left.savedAt.localeCompare(right.savedAt)));
        const serverCalibrations = cameraState.calibrations ?? [];
        setCalibrations((current) => ({
          ...Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith(`shared:${activeTaskGroupId}:`))),
          ...Object.fromEntries(serverCalibrations
            .filter((item) => item.algorithmVersion === CALIBRATION_ALGORITHM_VERSION && item.distortionModel)
            .map((item) => [calibrationModelKey(item.groupKey, item.distortionModel!), item])),
        }));
        const serverInitialExtrinsics = (cameraState.initialExtrinsics ?? []).filter((item) =>
          !item.id?.includes(":sparse-containment:"),
        );
        const normalizedVersions: Record<string, ExtrinsicVersion[]> = {};
        serverInitialExtrinsics.forEach((item, index) => {
          const version = normalizeStoredExtrinsic(item, index);
          normalizedVersions[version.groupKey] = [...(normalizedVersions[version.groupKey] ?? []), version];
        });
        setImportedExtrinsics((current) => ({
          ...Object.fromEntries(Object.entries(current).filter(([key]) => !key.startsWith("shared:" + activeTaskGroupId + ":"))),
          ...normalizedVersions,
        }));
        setProjectionModelOverrides(cameraState.projectionModelOverrides ?? {});
        setLidarPairAnnotations(lidarState.lidarPairAnnotations ?? []);
        setLidarPairCalibrations(lidarState.lidarPairCalibrations ?? []);
        setLidarFramePairSets(lidarState.lidarFramePairSets ?? []);
        setLidarPairNoAnnotationKeys(lidarState.lidarPairNoAnnotationKeys ?? []);
        setLidarTopicRemarks(lidarState.lidarTopicRemarks ?? {});
        setLidarExtrinsicGraph(lidarState.extrinsicGraph ?? {});
        setGroupStateReady(true);
      })
      .catch((reason: Error) => { if (active) setAnnotationFeedback(reason.message); });
    return () => { active = false; };
  }, [activeTaskGroupId, groupStateRefreshToken, localStateLoaded, manifest, taskGroups]);

  useEffect(() => {
    if (!groupStateReady) return;
    if (skipDomainSaveRef.current.camera) {
      skipDomainSaveRef.current.camera = false;
      return;
    }
    const timer = window.setTimeout(() => {
      const annotations = savedAnnotations.filter((item) => item.taskGroupId === activeTaskGroupId);
      const groupCalibrations = Object.entries(calibrations)
        .filter(([key]) => key.startsWith(`shared:${activeTaskGroupId}:`))
        .map(([, value]) => value);
      const groupInitialExtrinsics = Object.entries(importedExtrinsics)
        .filter(([key]) => key.startsWith("shared:" + activeTaskGroupId + ":"))
        .flatMap(([, value]) => value);
      domainSavePendingRef.current.camera = true;
      setSaveStates(current => ({...current, camera: "保存中"}));
      fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/state/camera`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          revision: domainRevisionRef.current.camera,
          updatedBy: currentManualSelectionClientId(),
          annotations,
          calibrations: groupCalibrations,
          initialExtrinsics: groupInitialExtrinsics,
          projectionModelOverrides,
        }),
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(response.status === 409 ? "另一窗口已修改此组，请备份当前标注后刷新" : `同步失败：${response.status}`);
          const result = await response.json() as { revision?: number };
          domainRevisionRef.current.camera = result.revision ?? domainRevisionRef.current.camera;
          saveErrorsRef.current.camera = false;
          setSaveStates(current => ({...current, camera: "已同步"}));
        })
        .catch((reason: Error) => { saveErrorsRef.current.camera = true; setSaveStates(current => ({...current, camera: reason.message})); setAnnotationFeedback(reason.message); })
        .finally(() => {
          domainSavePendingRef.current.camera = false;
          if (domainRefreshDeferredRef.current.camera && !saveErrorsRef.current.camera) {
            domainRefreshDeferredRef.current.camera = false;
            setGroupStateRefreshToken((value) => value + 1);
          }
        });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [activeTaskGroupId, calibrations, groupStateReady, importedExtrinsics, projectionModelOverrides, savedAnnotations, saveRetry]);

  useEffect(() => {
    if (!groupStateReady) return;
    if (skipDomainSaveRef.current.lidar) {
      skipDomainSaveRef.current.lidar = false;
      return;
    }
    const timer = window.setTimeout(() => {
      domainSavePendingRef.current.lidar = true;
      setSaveStates(current => ({...current, lidar: "保存中"}));
      fetch(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/state/lidar`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          revision: domainRevisionRef.current.lidar,
          updatedBy: currentManualSelectionClientId(),
          lidarPairAnnotations,
          lidarPairCalibrations,
          lidarFramePairSets,
          lidarPairNoAnnotationKeys,
          lidarTopicRemarks,
          extrinsicGraph: lidarExtrinsicGraph,
        }),
      })
        .then(async (response) => {
          if (!response.ok) throw new Error(response.status === 409 ? "另一窗口已修改此组，请备份当前标注后刷新" : `同步失败：${response.status}`);
          const result = await response.json() as { revision?: number };
          domainRevisionRef.current.lidar = result.revision ?? domainRevisionRef.current.lidar;
          saveErrorsRef.current.lidar = false;
          setSaveStates(current => ({...current, lidar: "已同步"}));
        })
        .catch((reason: Error) => { saveErrorsRef.current.lidar = true; setSaveStates(current => ({...current, lidar: reason.message})); setAnnotationFeedback(reason.message); })
        .finally(() => {
          domainSavePendingRef.current.lidar = false;
          if (domainRefreshDeferredRef.current.lidar && !saveErrorsRef.current.lidar) {
            domainRefreshDeferredRef.current.lidar = false;
            setGroupStateRefreshToken((value) => value + 1);
          }
        });
    }, 500);
    return () => window.clearTimeout(timer);
  }, [activeTaskGroupId, groupStateReady, lidarExtrinsicGraph, lidarFramePairSets, lidarPairAnnotations, lidarPairCalibrations, lidarPairNoAnnotationKeys, lidarTopicRemarks, saveRetry]);

  useEffect(() => {
    if (!groupStateReady || !activeTaskGroupId) return;
    const streams = (["camera", "lidar"] as const).map((domain) => {
      const events = new EventSource(`/api/calibration/groups/${encodeURIComponent(activeTaskGroupId)}/state/${domain}/events`);
      events.onmessage = (event) => {
        const state = JSON.parse(event.data) as { revision?: number; updatedBy?: string | null };
        const revision = state.revision ?? 0;
        if (revision <= domainRevisionRef.current[domain]) return;
        if (state.updatedBy === currentManualSelectionClientId()) {
          domainRevisionRef.current[domain] = revision;
          return;
        }
        if (saveErrorsRef.current[domain]) return;
        if (domainSavePendingRef.current[domain]) domainRefreshDeferredRef.current[domain] = true;
        else setGroupStateRefreshToken((value) => value + 1);
      };
      return events;
    });
    return () => streams.forEach((events) => events.close());
  }, [activeTaskGroupId, groupStateReady]);

  useEffect(() => {
    const labels = camera?.frames.map((frame) => frame.label) ?? [];
    if (labels.length && !labels.includes(frameLabel)) setFrameLabel(labels[0]);
  }, [dataset?.id, frameLabel, camera?.frames]);

  useEffect(() => {
    // Only the selected frame is fetched; switching frames cancels the old request.
    const controller = new AbortController();
    let active = true;
    setLoadedFrameCloud(null);
    setError(null);
    if (!lidarFrame?.url) {
      setLoading(false);
      return () => { active = false; controller.abort(); };
    }
    setLoading(true);
    readCloud(lidarFrame.url, controller.signal)
      .then((cloud) => { if (active) setLoadedFrameCloud({ key: cloudRequestKey, cloud }); })
      .catch((reason: Error) => { if (active) setError(reason.message); })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; controller.abort(); };
  }, [cloudRequestKey, lidarFrame?.url]);

  useEffect(() => {
    loadImageMaskState(null);
    loadPointSegmentState(new Uint32Array());
    setPointGrowFeedback({ kind: "success", text: "在右下投影图点击目标上的雷达点" });
    connectivityCacheRef.current = null;
    setClearSelectionToken((value) => value + 1);
    const pending = pendingAnnotationRef.current;
    if (pending && pending.datasetId === dataset?.id && pending.lidarId === lidar?.id &&
        pending.cameraId === camera?.id && pending.frame === frameLabel) {
      loadPointSegmentState(Uint32Array.from(pending.pointIndices));
      loadImageMaskState(pending.imageMask
        ? decodeMaskRle(pending.imageMask.width, pending.imageMask.height, pending.imageMask.rle)
        : null, pending.imageMaskLayers, pending.manualAddedMask, pending.manualRemovedMask);
      setActiveAnnotationId(pending.id);
      setPointGrowFeedback({ kind: "success", text: `正在预览已保存标注 · ${formatPoints(pending.pointCount)} 点` });
      pendingAnnotationRef.current = null;
    } else if (!pending) setActiveAnnotationId(null);
  }, [dataset?.id, lidar?.id, camera?.id, frameLabel]);

  useEffect(() => {
    if (!structurePreview || structurePreview.datasetId !== dataset?.id || structurePreview.frame !== frameLabel) return;
    loadPointSegmentState(Uint32Array.from(new Set([...structurePreview.indicesA,...(structurePreview.indicesB??[])])));
    loadImageMaskState(null);
    setActiveAnnotationId(null);
    setSelectedOnly(false);
    setStructurePreview(null);
  }, [structurePreview,dataset?.id,frameLabel]);

  const builtInExtrinsicMatrix = useMemo(() => {
    if (!manifest || !dataset || !lidar || !camera || dataset.calibrationCompatibility === "unknown-extrinsic") return null;
    const lidarCalibration = manifest.calibration.lidars[lidar.id];
    const legacyCameraCalibration = manifest.calibration.cameras[camera.id];
    return lidarCalibration && legacyCameraCalibration
      ? multiply4(legacyCameraCalibration.cameraFromCar, lidarCalibration.carFromLidar)
      : null;
  }, [camera, dataset, lidar, manifest]);
  const baseCameraIntrinsics = useMemo(() => {
    if (!manifest || !camera) return null;
    const configured = importedCameraIntrinsics ?? dataset?.cameraCalibration?.[camera.id] ?? manifest.calibration.cameras[camera.id];
    return configured ? rawProjectionCalibration(configured) : null;
  }, [camera, dataset, importedCameraIntrinsics, manifest]);
  const cameraIntrinsics = useMemo(() => baseCameraIntrinsics
    ? rawProjectionCalibration(baseCameraIntrinsics, projectionModelOverride)
    : null, [baseCameraIntrinsics, projectionModelOverride]);
  const displayIntrinsics = cameraIntrinsics ?? {
    intrinsic: [1, 0, 0, 0, 1, 0, 0, 0, 1], distortion: [], distortionModel: "rational",
  };
  const currentProjectionModel = displayIntrinsics.distortionModel as ProjectionModel;
  const currentCalibrationStorageKey = currentCalibrationKey
    ? calibrationModelKey(currentCalibrationKey, currentProjectionModel)
    : null;
  const optimization = currentCalibrationStorageKey ? calibrations[currentCalibrationStorageKey] ?? null : null;
  const currentExtrinsicVersions = allCurrentExtrinsicVersions.filter((item) =>
    item.source !== "calculated" || item.distortionModel === currentProjectionModel,
  );
  const initialExtrinsic = currentExtrinsicVersions.find((item) => item.isInitial) ?? null;
  const projection = useMemo(() => {
    if (!cameraIntrinsics || !lidar) return null;
    const matrix = initialExtrinsic?.matrix ?? builtInExtrinsicMatrix;
    if (!matrix) return null;
    return {
      matrix,
      intrinsic: cameraIntrinsics.intrinsic,
      distortion: cameraIntrinsics.distortion,
      distortionModel: cameraIntrinsics.distortionModel ?? "rational",
    };
  }, [builtInExtrinsicMatrix, cameraIntrinsics, initialExtrinsic, lidar]);

  const initialMatrix = extrinsicInitializationMode === "forward"
    ? FORWARD_FACING_LIDAR_TO_CAMERA
    : projection?.matrix ?? FORWARD_FACING_LIDAR_TO_CAMERA;
  // The next solve's initialization mode must not hide an already saved pose.
  const optimizationMatchesModel = optimization &&
    optimization.distortionModel === displayIntrinsics.distortionModel &&
    optimization.algorithmVersion === CALIBRATION_ALGORITHM_VERSION;
  const comparisonIntrinsics = cameraIntrinsics;

  const clearSelections = () => {
    loadImageMaskState(null);
    loadPointSegmentState(new Uint32Array());
    setClearSelectionToken((value) => value + 1);
  };

  const clearCurrentAnnotation = () => {
    pendingAnnotationRef.current = null;
    clearSelections();
    setActiveAnnotationId(null);
    setProjectionPicking(false);
    setStageMode("annotate");
    setPointGrowFeedback({ kind: "success", text: "当前标注已清空；已保存记录不受影响" });
    setAnnotationFeedback("当前标注已清空");
  };

  const previewAnnotation = (annotation: SavedAnnotation) => {
    const targetDataset = manifest?.datasets.findIndex((item) => item.id === annotation.datasetId) ?? -1;
    const sameContext = annotation.datasetId === dataset?.id && annotation.lidarId === lidar?.id &&
      annotation.cameraId === camera?.id && annotation.frame === frameLabel;
    if (!sameContext) pendingAnnotationRef.current = annotation;
    if (targetDataset >= 0) setDatasetIndex(targetDataset);
    setLidarId(annotation.lidarId);
    setCameraId(annotation.cameraId);
    setFrameLabel(annotation.frame);
    if (sameContext) {
      loadPointSegmentState(Uint32Array.from(annotation.pointIndices));
      loadImageMaskState(annotation.imageMask
        ? decodeMaskRle(annotation.imageMask.width, annotation.imageMask.height, annotation.imageMask.rle)
        : null, annotation.imageMaskLayers, annotation.manualAddedMask, annotation.manualRemovedMask);
      setActiveAnnotationId(annotation.id);
    }
    setPointGrowFeedback({ kind: "success", text: `正在预览已保存标注 · ${formatPoints(annotation.pointCount)} 点` });
    setAnnotationFeedback("已加载，可继续编辑后点击“更新当前标注”");
  };

  const toggleAnnotationPreview = (annotation: SavedAnnotation) => {
    if (activeAnnotationId !== annotation.id) {
      previewAnnotation(annotation);
      return;
    }
    clearCurrentAnnotation();
    setPointGrowFeedback({ kind: "success", text: "已退出当前配对，可开始新建标注" });
    setAnnotationFeedback("已取消配对选择，可开始新建标注");
  };

  const handlePointCandidate = (indices: Uint32Array) => {
    if (!currentCloud) return;
    const candidates = removeGround
      ? Uint32Array.from(indices.filter((index) => !isGroundPoint(currentCloud, index, stableOnly)))
      : indices;
    const clustered = clusterPointSelection(
      currentCloud.positions, candidates, clusterVoxelSize, false,
    );
    updatePointSegment((current) => combinePointIndices(current, clustered, pointMode));
    setPointGrowFeedback({
      kind: clustered.length ? "success" : "error",
      text: clustered.length
        ? `框内 ${formatPoints(indices.length)} 点 · 自适应密度聚类保留 ${formatPoints(clustered.length)} 点`
        : "框内没有可用点，请调整边界或关闭地面过滤",
    });
  };

  const handlePointEdit = (indices: Uint32Array, mode: "add" | "remove") => {
    updatePointSegment((current) => combinePointIndices(current, indices, mode));
  };

  const handleProjectedPointPick = (seedIndex: number | null) => {
    if (!currentCloud) return;
    if (seedIndex === null) {
      setPointGrowFeedback({ kind: "error", text: "附近没有雷达投影点，请点在可见方块上" });
      return;
    }
    const startedAt = performance.now();
    let cached = connectivityCacheRef.current;
    let builtIndex = false;
    if (!cached || cached.cloud !== currentCloud || cached.voxelSize !== clusterVoxelSize ||
        cached.stableOnly !== stableOnly || cached.removeGround !== removeGround) {
      cached = {
        cloud: currentCloud,
        voxelSize: clusterVoxelSize,
        stableOnly,
        removeGround,
        index: buildPointCloudConnectivityIndex(
          currentCloud.positions,
          currentCloud.stable,
          clusterVoxelSize,
          stableOnly,
          removeGround,
          currentCloud.hasGroundLabels ? currentCloud.ground : null,
        ),
      };
      connectivityCacheRef.current = cached;
      builtIndex = true;
    }
    const result = growConnectedPointCloud(cached.index, seedIndex, {
      surfaceAware: true,
      maxNormalAngleDegrees: surfaceAngle,
      maxPlaneDistance: surfaceThickness,
    });
    const elapsed = performance.now() - startedAt;
    if (result.reason !== "ok") {
      const reason = result.reason === "ground"
        ? "这个点被判定为地面；可关闭“抑制地面”后再点"
        : result.reason === "not-static"
          ? "这个点不是静止点；可关闭“仅显示静止点”后再点"
          : result.reason === "surface-unavailable"
            ? "这个位置过于稀疏，无法可靠估计局部表面；请换一个邻近投影点"
          : "这个点无法形成可用的三维连通区域";
      setPointGrowFeedback({ kind: "error", text: reason });
      return;
    }
    updatePointSegment((current) => combinePointIndices(current, result.indices, pointMode));
    setPointGrowFeedback({
      kind: "success",
      text: `种子 #${seedIndex} · 同一表面 ${formatPoints(result.indices.length)} 点 · ${elapsed.toFixed(1)} ms${builtIndex ? "（含首次索引）" : ""}`,
    });
  };

  const activeMaskLayer = activeMaskLayerId
    ? imageMaskLayers.find((layer) => layer.id === activeMaskLayerId) ?? null
    : null;
  const inactiveMask = useMemo(() => activeMaskLayerId
    ? combineMaskLayers(imageMaskLayers.filter((layer) => layer.id !== activeMaskLayerId))
    : null, [activeMaskLayerId, imageMaskLayers]);
  const canEditDisplayedMask = imageAnnotationEnabled;
  const brushEditingWholeImage = imageTool === "brush-add" || imageTool === "brush-remove";
  const displayedPrimaryMask = useMemo(() => {
    if (brushEditingWholeImage || !activeMaskLayer) return imageMask;
    return applyManualMaskOverrides(activeMaskLayer.mask, manualAddedMask, manualRemovedMask);
  }, [activeMaskLayer, brushEditingWholeImage, imageMask, manualAddedMask, manualRemovedMask]);
  const displayedSecondaryMask = useMemo(() => {
    if (brushEditingWholeImage || !activeMaskLayer) return null;
    return applyManualMaskOverrides(inactiveMask, null, manualRemovedMask);
  }, [activeMaskLayer, brushEditingWholeImage, inactiveMask, manualRemovedMask]);
  const selectMaskLayer = (layerId: string) => {
    if (!layerId) {
      activeMaskLayerIdRef.current = null;
      setActiveMaskLayerId(null);
      return;
    }
    const layer = imageMaskLayersRef.current.find((item) => item.id === layerId);
    if (!layer) return;
    activeMaskLayerIdRef.current = layer.id;
    setActiveMaskLayerId(layer.id);
  };
  const annotationContent = useMemo(() => {
    const serialize = (mask: ImageMask | null) => mask ? {
      width: mask.width, height: mask.height, rle: encodeMaskRle(mask),
    } : null;
    return {
      pointIndices: Array.from(pointSegment),
      imageMask: serialize(imageMask),
      imageMaskLayers: imageMaskLayers.map((layer) => ({
        id: layer.id, ...serialize(layer.mask)!,
        granularity: layer.granularity, brushModified: layer.brushModified,
      })),
      manualAddedMask: serialize(manualAddedMask),
      manualRemovedMask: serialize(manualRemovedMask),
    };
  }, [pointSegment, imageMask, imageMaskLayers, manualAddedMask, manualRemovedMask]);
  const annotationHasChanges = useMemo(() => {
    const saved = savedAnnotations.find((item) => item.id === activeAnnotationId);
    return !saved || annotationContentChanged(annotationContent, saved);
  }, [annotationContent, savedAnnotations, activeAnnotationId]);
  const canSaveAnnotation = Boolean(
    !savingAnnotation && annotationHasChanges && dataset && lidar && camera &&
    lidarFrame && cameraFrame && (maskPixelCount || pointSegment.length),
  );

  const saveAnnotation = async () => {
    if (savingAnnotationRef.current || !canSaveAnnotation) return;
    if (!manifest || !dataset || !lidar || !camera || !lidarFrame || !cameraFrame ||
        (!maskPixelCount && !pointSegment.length)) return;
    const previousAnnotation = activeAnnotationId
      ? savedAnnotations.find((item) => item.id === activeAnnotationId)
      : null;
    const annotation: SavedAnnotation = {
      id: activeAnnotationId ?? `${dataset.id}-${frameLabel}-${lidar.id}-${camera.id}-${Date.now()}`,
      datasetId: dataset.id,
      frame: frameLabel,
      lidarId: lidar.id,
      cameraId: camera.id,
      lidarTimestampNs: lidarFrame.timestampNs,
      cameraTimestampNs: cameraFrame.timestampNs,
      calibrationSource: dataset.rigId ? `${dataset.rigId}:intrinsics-only` : manifest.calibration.source,
      status: maskPixelCount && pointSegment.length ? "paired" : maskPixelCount ? "image-draft" : "point-draft",
      pointCount: pointSegment.length,
      maskPixelCount,
      pointIndices: Array.from(pointSegment),
      imageMask: imageMask ? {
        width: imageMask.width,
        height: imageMask.height,
        rle: encodeMaskRle(imageMask),
      } : null,
      imageMaskLayers: imageMaskLayers.map((layer) => ({
        id: layer.id,
        width: layer.mask.width,
        height: layer.mask.height,
        rle: encodeMaskRle(layer.mask),
        granularity: layer.granularity,
        brushModified: layer.brushModified,
      })),
      manualAddedMask: manualAddedMask ? {
        width: manualAddedMask.width,
        height: manualAddedMask.height,
        rle: encodeMaskRle(manualAddedMask),
      } : null,
      manualRemovedMask: manualRemovedMask ? {
        width: manualRemovedMask.width,
        height: manualRemovedMask.height,
        rle: encodeMaskRle(manualRemovedMask),
      } : null,
      useForOptimization: previousAnnotation?.useForOptimization ?? true,
      taskGroupId: activeTaskGroupId,
      imageCoordinateSystem: RAW_COORDINATE_SYSTEM,
      savedAt: new Date().toISOString(),
    };
    savingAnnotationRef.current = true;
    setSavingAnnotation(true);
    try {
      await storeAnnotation(annotation);
      setSavedAnnotations((current) => {
        const existing = current.findIndex((item) => item.id === annotation.id);
        if (existing < 0) return [...current, annotation];
        const updated = current.slice();
        updated[existing] = annotation;
        return updated;
      });
      if (currentCalibrationStorageKey && calibrations[currentCalibrationStorageKey]?.annotationIds.includes(annotation.id)) {
        setOptimizationFeedback("此配对已更新；当前外参保持不变，可继续修改其他配对，最后统一复查。");
      }
      // Updating a pair is part of the same review round: retain its selection
      // and mask, especially when the rest of the scene is hidden.
      if (!previousAnnotation) {
        clearSelections();
        setActiveAnnotationId(null);
        setSelectedOnly(false);
      }
      setProjectionPicking(false);
      setImageTool("box-replace");
      setPointInteractionMode("navigate");
      setPointMode("replace");
      setStageMode("annotate");
      setPointGrowFeedback({ kind: "success", text: previousAnnotation ? "标注已更新，保留当前选区，可继续复查" : "配对已保存，已清空当前选择，可继续下一组" });
      setAnnotationFeedback(previousAnnotation ? "当前标注已更新，选点和 Mask 保持显示" : "配对已保存，已进入空白标注");
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? reason.message : "保存标注失败");
    } finally {
      savingAnnotationRef.current = false;
      setSavingAnnotation(false);
    }
  };

  const deleteAnnotation = async (annotation: SavedAnnotation, index: number) => {
    const label = `配对 ${index + 1}（帧 ${annotation.frame}，${formatPoints(annotation.pointCount)} 点）`;
    if (!window.confirm(`确定删除${label}吗？此操作会删除当前任务组中的配对，并同步到服务器。`)) return;
    try {
      await deleteStoredAnnotation(annotation.id);
      setSavedAnnotations((current) => current.filter((item) => item.id !== annotation.id));
      if (pendingAnnotationRef.current?.id === annotation.id) pendingAnnotationRef.current = null;
      if (activeAnnotationId === annotation.id) {
        clearSelections();
        setActiveAnnotationId(null);
        setStageMode("annotate");
        setPointGrowFeedback({ kind: "success", text: "当前配对已删除，可新建或预览其他配对" });
      }
      const affectedCalibration = Object.values(calibrations).find((item) => item.annotationIds.includes(annotation.id));
      if (affectedCalibration) setOptimizationFeedback("已删除参与联合优化的配对；现有共享外参仅供对比，请重新联合优化");
      setAnnotationFeedback(`${label}已删除`);
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? `删除失败：${reason.message}` : "删除失败");
    }
  };

  const setAnnotationOptimizationEnabled = async (annotation: SavedAnnotation, enabled: boolean) => {
    const updated = { ...annotation, useForOptimization: enabled };
    try {
      await storeAnnotation(updated);
      setSavedAnnotations((current) => current.map((item) => item.id === annotation.id ? updated : item));
      if (Object.values(calibrations).some((item) => item.annotationIds.includes(annotation.id))) {
        setOptimizationFeedback("参与优化的配对已改变；现有共享外参仅供对比，请重新优化");
      }
      setAnnotationFeedback(enabled ? "此配对将参与外参优化" : "此配对仍会保存和预览，但不参与外参优化");
    } catch (reason) {
      setAnnotationFeedback(reason instanceof Error ? `更新失败：${reason.message}` : "更新失败");
    }
  };

  const visibleAnnotations = useMemo(() => {
    if (!lidar || !camera) return [];
    return savedAnnotations.filter((annotation) =>
      annotation.taskGroupId === activeTaskGroupId &&
      (manifest?.datasets.find((item) => item.id === annotation.datasetId)?.rigId ?? "truck5-legacy") === currentRigId &&
      annotation.lidarId === lidar.id && annotation.cameraId === camera.id,
    );
  }, [activeTaskGroupId, camera, currentRigId, lidar, manifest, savedAnnotations]);
  const pairFrameOptions = useMemo(() => {
    const frames = new Map<string, { key: string; datasetId: string; frame: string; count: number }>();
    visibleAnnotations.forEach((annotation) => {
      const key = JSON.stringify([annotation.datasetId, annotation.frame]);
      const current = frames.get(key);
      if (current) current.count += 1;
      else frames.set(key, { key, datasetId: annotation.datasetId, frame: annotation.frame, count: 1 });
    });
    const multipleDatasets = new Set(visibleAnnotations.map((annotation) => annotation.datasetId)).size > 1;
    return [...frames.values()].sort((left, right) => {
      const leftDataset = manifest?.datasets.find((item) => item.id === left.datasetId);
      const rightDataset = manifest?.datasets.find((item) => item.id === right.datasetId);
      const datasetOrder = (leftDataset?.name ?? left.datasetId).localeCompare(
        rightDataset?.name ?? right.datasetId,
        "zh-CN",
        { numeric: true },
      );
      return datasetOrder || left.frame.localeCompare(right.frame, "zh-CN", { numeric: true });
    }).map((item) => ({
      ...item,
      label: `${multipleDatasets ? `${manifest?.datasets.find((datasetItem) => datasetItem.id === item.datasetId)?.name ?? item.datasetId} · ` : ""}帧 ${item.frame}（${item.count} 对）`,
    }));
  }, [manifest, visibleAnnotations]);
  const filteredAnnotations = pairFrameFilter === "all"
    ? visibleAnnotations
    : visibleAnnotations.filter((annotation) => JSON.stringify([annotation.datasetId, annotation.frame]) === pairFrameFilter);
  const visibleAnnotationNumbers = useMemo(() => new Map(
    visibleAnnotations.map((annotation, index) => [annotation.id, index + 1]),
  ), [visibleAnnotations]);
  const pairPageCount = Math.max(1, Math.ceil(filteredAnnotations.length / PAIRS_PER_PAGE));
  const currentPairPage = Math.min(pairPage, pairPageCount - 1);
  const pagedAnnotations = filteredAnnotations.slice(currentPairPage * PAIRS_PER_PAGE, (currentPairPage + 1) * PAIRS_PER_PAGE);

  useEffect(() => {
    setPairPage((page) => Math.min(page, pairPageCount - 1));
  }, [pairPageCount]);

  useEffect(() => {
    setPairPage(0);
    setPairFrameFilter("all");
    // Previewing a pair synchronizes the raw selection IDs after reload.
    // Reset only when the effective sensor/list context actually changes.
  }, [activeTaskGroupId, currentRigId, camera?.id, lidar?.id]);

  useEffect(() => {
    if (pairFrameFilter === "all" || pairFrameOptions.some((item) => item.key === pairFrameFilter)) return;
    setPairFrameFilter("all");
    setPairPage(0);
  }, [pairFrameFilter, pairFrameOptions]);

  const eligibleAnnotations = useMemo(() => {
    if (!lidar || !camera) return [];
    return savedAnnotations.filter((annotation) =>
      annotation.status === "paired" && annotation.imageMask && annotation.pointIndices.length > 0 &&
      annotation.useForOptimization !== false &&
      annotation.taskGroupId === activeTaskGroupId &&
      (manifest?.datasets.find((item) => item.id === annotation.datasetId)?.rigId ?? "truck5-legacy") === currentRigId &&
      annotation.lidarId === lidar.id && annotation.cameraId === camera.id,
    );
  }, [activeTaskGroupId, camera, currentRigId, lidar, manifest, savedAnnotations]);
  const optimizationRequiredPairs = extrinsicInitializationMode === "forward" ? 4 : 1;
  const optimizationBlockedReason = !baseCameraIntrinsics
    ? `当前相机 ${camera?.topic ?? ""} 尚未导入内参，请先点击顶部“导入内参”`
    : !currentCalibrationKey
      ? "当前雷达与相机组合尚未就绪"
      : eligibleAnnotations.length < optimizationRequiredPairs
        ? `当前只有 ${eligibleAnnotations.length} 个可用配对，${extrinsicInitializationMode === "forward" ? "无初值 PnP" : "基于初值精调"}至少需要 ${optimizationRequiredPairs} 个`
        : extrinsicInitializationMode === "existing" && !projection
          ? "尚未导入或生成当前组合的初始外参"
          : null;
  const optimizationAnnotationStats = useMemo(() => {
    const complete = savedAnnotations.filter((annotation) =>
      annotation.status === "paired" && annotation.imageMask && annotation.pointIndices.length > 0,
    );
    const sameSensors = complete.filter((annotation) =>
      annotation.taskGroupId === activeTaskGroupId &&
      (manifest?.datasets.find((item) => item.id === annotation.datasetId)?.rigId ?? "truck5-legacy") === currentRigId &&
      annotation.lidarId === lidar?.id && annotation.cameraId === camera?.id,
    );
    return {
      totalSaved: visibleAnnotations.length,
      complete: sameSensors.length,
      unchecked: sameSensors.filter((annotation) => annotation.useForOptimization === false).length,
    };
  }, [activeTaskGroupId, camera, currentRigId, lidar, manifest, savedAnnotations, visibleAnnotations.length]);
  const currentFrameAnnotations = useMemo(() => {
    if (!dataset || !lidar || !camera) return [];
    return savedAnnotations.filter((annotation) =>
      annotation.taskGroupId === activeTaskGroupId && annotation.status === "paired" && annotation.datasetId === dataset.id &&
      annotation.frame === frameLabel && annotation.lidarId === lidar.id && annotation.cameraId === camera.id,
    );
  }, [activeTaskGroupId, camera, dataset, frameLabel, lidar, savedAnnotations]);
  const currentFrameAnnotationPoints = useMemo(() => {
    if (!currentCloud) return new Uint32Array();
    const indices = new Set<number>();
    currentFrameAnnotations.forEach((annotation) => {
      annotation.pointIndices.forEach((index) => {
        if (index >= 0 && index < currentCloud.stable.length) indices.add(index);
      });
    });
    return Uint32Array.from([...indices].sort((left, right) => left - right));
  }, [currentCloud, currentFrameAnnotations]);
  useEffect(() => {
    if (currentFrameAnnotations.some((annotation) => annotation.id === singleProjectionAnnotationId)) return;
    setSingleProjectionAnnotationId(currentFrameAnnotations[0]?.id ?? "");
  }, [currentFrameAnnotations, singleProjectionAnnotationId]);
  const singleProjectionAnnotation = currentFrameAnnotations.find((annotation) => annotation.id === singleProjectionAnnotationId) ?? null;
  const fullProjectionPointIndices = fullProjectionScope === "frame-annotations"
    ? currentFrameAnnotationPoints
    : fullProjectionScope === "single-annotation"
      ? Uint32Array.from(singleProjectionAnnotation?.pointIndices ?? [])
      : new Uint32Array();
  const currentFrameAnnotationMask = useMemo(() => {
    if (!dataset || !lidar || !camera || !cameraFrame) return null;
    let combined: ImageMask | null = null;
    savedAnnotations.forEach((annotation) => {
      if (annotation.taskGroupId !== activeTaskGroupId || annotation.status !== "paired" || !annotation.imageMask || annotation.datasetId !== dataset.id ||
          annotation.frame !== frameLabel || annotation.lidarId !== lidar.id || annotation.cameraId !== camera.id) return;
      const decoded = decodeMaskRle(annotation.imageMask.width, annotation.imageMask.height, annotation.imageMask.rle);
      combined = mergeImageMasks(combined, decoded, "add");
    });
    return combined;
  }, [activeTaskGroupId, camera, cameraFrame, dataset, frameLabel, lidar, savedAnnotations]);
  const comparisonPointIndices = comparisonProjectionScope === "frame-annotations"
    ? currentFrameAnnotationPoints
    : pointSegment;
  const comparisonMask = comparisonProjectionScope === "frame-annotations"
    ? currentFrameAnnotationMask
    : imageMask;
  const optimizationIsStale = Boolean(optimization && (
    optimization.distortionModel !== displayIntrinsics.distortionModel ||
    optimization.annotationIds.length !== eligibleAnnotations.length ||
    optimization.annotationIds.some((id, index) =>
      id !== eligibleAnnotations[index]?.id ||
      (optimization.annotationSavedAts?.[index] ?? "") !== eligibleAnnotations[index]?.savedAt,
    )
  ));
  const currentOptimization = optimizationIsStale ? null : optimization;
  const builtInVersionId = currentCalibrationKey ? currentCalibrationKey + ":builtin" : "";
  const currentResultVersion = currentOptimization
    ? currentExtrinsicVersions.find((item) => item.optimization?.generatedAt === currentOptimization.generatedAt) ?? null
    : null;
  const currentResultVersionId = currentOptimization
    ? currentResultVersion?.id ?? currentCalibrationKey + ":current:" + currentOptimization.generatedAt
    : "";
  const comparisonVersions = useMemo(() => {
    const versions = [...currentExtrinsicVersions];
    if (currentCalibrationKey && builtInExtrinsicMatrix) versions.unshift({
      id: builtInVersionId,
      groupKey: currentCalibrationKey,
      matrix: builtInExtrinsicMatrix,
      createdAt: "",
      label: "数据内置初始外参",
      source: "builtin",
      isInitial: !initialExtrinsic,
    });
    if (currentOptimization && !versions.some((item) => item.optimization?.generatedAt === currentOptimization.generatedAt)) versions.push({
      id: currentResultVersionId,
      groupKey: currentCalibrationKey ?? currentOptimization.groupKey,
      matrix: currentOptimization.matrix,
      createdAt: currentOptimization.generatedAt,
      label: "当前计算结果 · " + new Date(currentOptimization.generatedAt).toLocaleString("zh-CN"),
      source: "calculated",
      isInitial: false,
      optimization: currentOptimization,
      distortionModel: currentProjectionModel,
    });
    return versions;
  }, [builtInExtrinsicMatrix, builtInVersionId, currentCalibrationKey, currentExtrinsicVersions, currentOptimization, currentProjectionModel, currentResultVersionId, initialExtrinsic]);
  const currentInitialVersionId = initialExtrinsic?.id ?? (builtInExtrinsicMatrix ? builtInVersionId : "");
  const selectedLeftVersion = comparisonVersions.find((item) => item.id === comparisonLeftVersionId) ?? null;
  const selectedRightVersion = comparisonVersions.find((item) => item.id === comparisonRightVersionId) ?? null;
  const selectedExportVersion = resolvePreviewVersion(comparisonVersions, exportVersionId,
    currentResultVersionId || currentInitialVersionId);
  const exportablePackageVersions = useMemo(() => {
    if (!activeTaskGroup || !dataset) return [];
    const rigId = dataset.rigId ?? currentRigId;
    return dataset.lidars.flatMap((lidarInfo) => dataset.cameras.flatMap((cameraInfo) => {
      const groupKey = calibrationGroupKey(activeTaskGroup.id, rigId, lidarInfo.id, cameraInfo.id);
      const preferredModel = projectionModelOverrides[cameraInfo.id]
        ?? intrinsicsProfile?.datasets[dataset.id]?.cameras[cameraInfo.id]?.distortionModel
        ?? dataset.cameraCalibration?.[cameraInfo.id]?.distortionModel;
      const versions = (importedExtrinsics[groupKey] ?? []).filter((version) =>
        version.source !== "builtin"
        && version.matrix.length === 16
        && version.matrix.every(Number.isFinite)
        && !version.id.includes(":sparse-containment:"),
      );
      const calculated = versions.filter((version) => version.source === "calculated");
      const candidates = calculated.length ? calculated : versions;
      const newest = (items: ExtrinsicVersion[]) => [...items].sort((left, right) =>
        right.createdAt.localeCompare(left.createdAt),
      )[0];
      const version = newest(candidates.filter((item) => preferredModel && item.distortionModel === preferredModel))
        ?? newest(candidates);
      return version ? [{ groupKey, rigId, lidarInfo, cameraInfo, version }] : [];
    }));
  }, [activeTaskGroup, currentRigId, dataset, importedExtrinsics, intrinsicsProfile, projectionModelOverrides]);
  const workingMatrix = selectedExportVersion?.matrix ??
    (currentOptimization && optimizationMatchesModel ? currentOptimization.matrix : projection?.matrix ?? null);
  const canPickProjectedPoint = Boolean(workingMatrix && currentCloud && projectedCount > 0);
  const comparisonOptimization = selectedRightVersion?.optimization ?? null;
  const currentResultIsInitial = Boolean(initialExtrinsic && currentOptimization &&
    initialExtrinsic.optimization?.generatedAt === currentOptimization.generatedAt);
  const canCompareExtrinsics = Boolean(currentInitialVersionId && currentOptimization);

  useEffect(() => {
    const ids = new Set(comparisonVersions.map((item) => item.id));
    setComparisonLeftVersionId((current) => ids.has(current) ? current : currentInitialVersionId);
    setComparisonRightVersionId((current) => ids.has(current) ? current : currentResultVersionId || currentInitialVersionId);
    setExportVersionId((current) => resolvePreviewVersion(comparisonVersions, current,
      currentResultVersionId || currentInitialVersionId)?.id ?? "");
  }, [comparisonVersions, currentInitialVersionId, currentResultVersionId]);

  useEffect(() => {
    if (stageMode === "compare" && !canCompareExtrinsics) setStageMode("annotate");
  }, [canCompareExtrinsics, stageMode]);

  const loadCalibrationObservation = async (annotation: SavedAnnotation): Promise<CalibrationObservation> => {
    if (!manifest || !annotation.imageMask) throw new Error("配对缺少图像掩码");
    const observationDataset = manifest.datasets.find((item) => item.id === annotation.datasetId);
    const observationLidar = observationDataset?.lidars.find((item) => item.id === annotation.lidarId);
    const observationCamera = observationDataset?.cameras.find((item) => item.id === annotation.cameraId);
    const observationFrameIndex = frameIndexByLabel(observationLidar?.frames, annotation.frame);
    const cloudFrame = observationLidar?.frames[observationFrameIndex];
    const configuredCalibration = intrinsicsProfile?.datasets[annotation.datasetId]?.cameras[annotation.cameraId] ??
      observationDataset?.cameraCalibration?.[annotation.cameraId] ?? manifest.calibration.cameras[annotation.cameraId];
    const annotationModelOverride = projectionModelOverrides[annotation.cameraId];
    const cameraCalibration = configuredCalibration
      ? rawProjectionCalibration(configuredCalibration, annotationModelOverride)
      : null;
    if (!cloudFrame || !observationCamera?.frames[observationFrameIndex] || !cameraCalibration) {
      throw new Error(`配对 ${annotation.id} 的原始帧已不可用`);
    }
    const cloud = currentCloud && annotation.datasetId === dataset?.id && annotation.lidarId === lidar?.id &&
      annotation.frame === frameLabel
      ? currentCloud
      : await readCloud(cloudFrame.url);
    const validIndices = annotation.pointIndices.filter((index) => index >= 0 && index < cloud.stable.length);
    if (validIndices.length === 0) throw new Error(`配对 ${annotation.id} 没有有效点`);
    return {
      annotationId: annotation.id,
      cloud,
      indices: Uint32Array.from(validIndices),
      mask: decodeMaskRle(annotation.imageMask.width, annotation.imageMask.height, annotation.imageMask.rle),
      intrinsic: cameraCalibration.intrinsic,
      distortion: cameraCalibration.distortion,
      distortionModel: cameraCalibration.distortionModel ?? "rational",
    };
  };

  const importExtrinsicFile = async (file: File | null, input: HTMLInputElement) => {
    if (!file || !activeTaskGroup || !currentCalibrationKey || !lidar || !camera) return;
    const parseMatrix = (value: unknown): number[] | null => {
      if (!Array.isArray(value)) return null;
      const flat = value.length === 4 && value.every((row) => Array.isArray(row))
        ? value.flatMap((row) => row as unknown[])
        : value;
      if (flat.length !== 16) return null;
      const matrix = flat.map(Number);
      return matrix.every(Number.isFinite) ? matrix : null;
    };
    try {
      const payload = JSON.parse(await file.text()) as any;
      const entries = Array.isArray(payload?.extrinsics) ? payload.extrinsics : [payload];
      const imported: Record<string, ExtrinsicVersion> = {};
      entries.forEach((entry: any, index: number) => {
        const matrix = parseMatrix(entry?.matrix ?? entry);
        if (!matrix) return;
        const targetLidarId = String(entry?.lidar?.id ?? lidar.id);
        const targetCameraId = String(entry?.camera?.id ?? camera.id);
        const targetRigId = String(entry?.rigId ?? currentRigId);
        const groupKey = calibrationGroupKey(activeTaskGroup.id, targetRigId, targetLidarId, targetCameraId);
        const createdAt = new Date().toISOString();
        imported[groupKey] = {
          id: [groupKey, "imported", createdAt, index].join(":"),
          groupKey,
          matrix,
          createdAt,
          label: "导入 · " + file.name + " · " + new Date(createdAt).toLocaleString("zh-CN"),
          source: "imported",
          sourceName: file.name,
          isInitial: true,
        };
      });
      const importedCount = Object.keys(imported).length;
      if (!importedCount) throw new Error("未找到有效的 camera_from_lidar 4×4 矩阵");
      setImportedExtrinsics((current) => {
        const next = { ...current };
        Object.entries(imported).forEach(([groupKey, version]) => {
          const previous = (next[groupKey] ?? []).map((item) => ({ ...item, isInitial: false }));
          next[groupKey] = [...previous.filter((item) => item.id !== version.id), version];
        });
        return next;
      });
      if (imported[currentCalibrationKey]) {
        setComparisonLeftVersionId(imported[currentCalibrationKey].id);
        setExportVersionId(imported[currentCalibrationKey].id);
      }
      setExtrinsicInitializationMode("existing");
      setStageMode("annotate");
      setOptimizationFeedback(`已导入并记录 ${importedCount} 组初始外参，可直接查看投影或继续精调`);
      setAnnotationFeedback(`已从 ${file.name} 导入 ${importedCount} 组初始外参版本`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "文件格式无法识别";
      setOptimizationFeedback(`导入已有外参失败：${message}`);
    } finally {
      input.value = "";
    }
  };

  const setCurrentResultAsInitial = () => {
    if (!currentCalibrationKey || !currentOptimization) return;
    const version: ExtrinsicVersion = currentResultVersion ?? {
      id: [currentCalibrationKey, "calculated", currentOptimization.generatedAt].join(":"),
      groupKey: currentCalibrationKey,
      matrix: currentOptimization.matrix,
      createdAt: currentOptimization.generatedAt,
      label: "计算结果 · " + new Date(currentOptimization.generatedAt).toLocaleString("zh-CN"),
      source: "calculated",
      isInitial: false,
      optimization: currentOptimization,
      distortionModel: currentProjectionModel,
    };
    setImportedExtrinsics((current) => ({
      ...current,
      [currentCalibrationKey]: [
        ...(current[currentCalibrationKey] ?? []).filter((item) => item.id !== version.id).map((item) => ({ ...item, isInitial: false })),
        { ...version, isInitial: true },
      ],
    }));
    setComparisonLeftVersionId(version.id);
    setExportVersionId(version.id);
    setExtrinsicInitializationMode("existing");
    setStageMode("annotate");
    setOptimizationFeedback("已将当前计算结果记录并设为新的初始外参");
  };

  const buildCalibrationPairs = (modelToRun: ProjectionModel) => eligibleAnnotations.map((annotation) => {
        if (!annotation.imageMask) throw new Error(`配对 ${annotation.id} 缺少图像 Mask`);
        const observationDataset = manifest?.datasets.find((item) => item.id === annotation.datasetId);
        const observationLidar = observationDataset?.lidars.find((item) => item.id === annotation.lidarId);
        const cloudFrame = observationLidar?.frames[frameIndexByLabel(observationLidar?.frames, annotation.frame)];
        const configuredCalibration = intrinsicsProfile?.datasets[annotation.datasetId]?.cameras[annotation.cameraId] ??
          observationDataset?.cameraCalibration?.[annotation.cameraId] ?? manifest?.calibration.cameras[annotation.cameraId];
        const cameraCalibration = configuredCalibration
          ? rawProjectionCalibration(configuredCalibration, modelToRun)
          : null;
        if (!cloudFrame || !cameraCalibration) throw new Error(`配对 ${annotation.id} 的本机数据不可用`);
        const annotationTemporalProjection = temporalProjectionFor(
          observationDataset, annotation.lidarId, annotation.cameraId, annotation.frame,
        );
        if (annotationTemporalProjection && modelToRun !== "radtan") {
          throw new Error("Waymo 官方 CameraModel 只允许使用数据声明的 RadTan 模型");
        }
        return {
          annotation_id: annotation.id,
          image_url: observationDataset?.cameras.find((item) => item.id === annotation.cameraId)?.frames.find((item) => item.label === annotation.frame)?.url,
          dataset_id: annotation.datasetId,
          frame: annotation.frame,
          cloud_url: cloudFrame.url,
          point_indices: annotation.pointIndices,
          mask_width: annotation.imageMask.width,
          mask_height: annotation.imageMask.height,
          mask_rle: annotation.imageMask.rle,
          intrinsic: cameraCalibration.intrinsic,
          distortion: cameraCalibration.distortion,
          distortionModel: (cameraCalibration.distortionModel ?? "rational") as ProjectionModel,
          temporal_projection: annotationTemporalProjection ? {
            model: annotationTemporalProjection.model,
            vehicle_from_lidar: annotationTemporalProjection.vehicleFromLidar,
            reference_world_from_vehicle: annotationTemporalProjection.referenceWorldFromVehicle,
            camera_world_from_vehicle: annotationTemporalProjection.cameraWorldFromVehicle,
            linear_velocity_world: annotationTemporalProjection.linearVelocityWorld,
            angular_velocity_vehicle: annotationTemporalProjection.angularVelocityVehicle,
            pose_timestamp: annotationTemporalProjection.poseTimestamp,
            shutter: annotationTemporalProjection.shutter,
            trigger_time: annotationTemporalProjection.triggerTime,
            readout_done_time: annotationTemporalProjection.readoutDoneTime,
            rolling_shutter_direction: annotationTemporalProjection.rollingShutterDirection,
            image_width: annotationTemporalProjection.imageWidth,
            image_height: annotationTemporalProjection.imageHeight,
          } : null,
        };
      });

  const runOptimization = async (modelsToRun: ProjectionModel[] = [currentProjectionModel]) => {
    if (optimizing) return;
    if (optimizationBlockedReason || !baseCameraIntrinsics || !currentCalibrationKey) {
      setOptimizationFeedback(optimizationBlockedReason ?? "当前标定条件不完整");
      return;
    }
    setOptimizing(true);
    setOptimizationProgress(1);
    const completedModels: ProjectionModel[] = [];
    const failedModels: string[] = [];
    const optimizationBatchId = globalThis.crypto?.randomUUID?.() ?? `optimization-${Date.now()}`;
    const modelProgress = modelsToRun.map(() => 0);
    const updateModelProgress = (modelIndex: number, percent: number) => {
      modelProgress[modelIndex] = percent;
      setOptimizationProgress(Math.round(modelProgress.reduce((sum, value) => sum + value, 0) / modelsToRun.length));
    };
    setOptimizationFeedback(`正在并行计算 ${modelsToRun.length} 种投影模型…`);
    try {
      await Promise.all(modelsToRun.map(async (modelToRun, modelIndex) => {
      try {
      const pairs = buildCalibrationPairs(modelToRun);
      const requestBody = JSON.stringify({
        group_key: currentCalibrationKey,
        optimization_batch_id: optimizationBatchId,
        optimization_batch_size: modelsToRun.length,
        initialization_mode: extrinsicInitializationMode,
        base_matrix: initialMatrix,
        // Strict no-initial-value PnP must never reuse a browser-cached pose.
        previous_matrix: extrinsicInitializationMode === "forward"
          ? null
          : calibrations[calibrationModelKey(currentCalibrationKey, modelToRun)]?.matrix ?? null,
        pairs,
      });
      let response: Response | null = null;
      for (let attempt = 0; attempt < 3; attempt += 1) {
        try {
          response = await fetch("/api/calibration/optimize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: requestBody,
          });
        } catch (error) {
          if (attempt === 2) throw error;
          response = null;
        }
        if (response && response.status !== 502 && response.status !== 503) break;
        if (attempt < 2) {
          setOptimizationFeedback(`优化服务正在启动，${attempt + 1}/2 次自动重试…`);
          await new Promise((resolve) => window.setTimeout(resolve, 800 * (attempt + 1)));
        }
      }
      if (!response) throw new Error("优化服务暂时无法连接，请稍后重试");
      if (!response.ok) {
        const raw = await response.text();
        let detail = raw;
        try { detail = (JSON.parse(raw) as { detail?: string }).detail ?? raw; } catch { /* keep raw response */ }
        if (response.status === 502 || response.status === 503 || /<html[\s>]/i.test(raw)) {
          detail = `优化服务暂时未就绪（${response.status}），请稍后重试`;
        }
        throw new Error(detail || `本机求解器返回 ${response.status}`);
      }
      if (!response.body) throw new Error("本机求解器没有返回结果流");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const streamResult: { payload: Partial<OptimizationResult> | null } = { payload: null };
      const consumeEvent = (line: string) => {
        if (!line.trim()) return;
        const event = JSON.parse(line) as {
          type: "progress" | "result" | "error";
          percent?: number;
          message?: string;
          pnpInlierCount?: number;
          pnpReprojectionError?: number;
          result?: Partial<OptimizationResult>;
        };
        if (event.percent != null) updateModelProgress(modelIndex, event.percent);
        if (event.message) setOptimizationFeedback(`${modelIndex + 1}/${modelsToRun.length} · ${modelToRun}：${event.message}`);
        if (event.type === "error") throw new Error(event.message || "本机求解器失败");
        if (event.type === "result") streamResult.payload = event.result ?? null;
      };
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";
        lines.forEach(consumeEvent);
        if (done) break;
      }
      consumeEvent(buffer);
      const payload = streamResult.payload;
      if (!payload?.matrix || payload.matrix.length !== 16) throw new Error("本机求解器未返回有效外参");
      const result: OptimizationResult = {
        matrix: payload.matrix,
        delta: payload.delta ?? [0, 0, 0, 0, 0, 0],
        originalError: payload.originalError ?? 0,
        optimizedError: payload.optimizedError ?? 0,
        fullMatchMatrix: payload.fullMatchMatrix,
        fullMatchError: payload.fullMatchError,
        originalIoU: payload.originalIoU,
        fullMatchIoU: payload.fullMatchIoU,
        optimizedIoU: payload.optimizedIoU,
        silhouetteMetric: payload.silhouetteMetric,
        envelopePrecision: payload.envelopePrecision,
        maskCoverage: payload.maskCoverage,
        originalInsideRatio: payload.originalInsideRatio,
        fullMatchInsideRatio: payload.fullMatchInsideRatio,
        optimizedInsideRatio: payload.optimizedInsideRatio,
        originalContainmentError: payload.originalContainmentError,
        fullMatchContainmentError: payload.fullMatchContainmentError,
        optimizedContainmentError: payload.optimizedContainmentError,
        backgroundIntrusionRatio: payload.backgroundIntrusionRatio,
        backgroundIntrusionCount: payload.backgroundIntrusionCount,
        occludedBackgroundCount: payload.occludedBackgroundCount,
        medianEdgeError: payload.medianEdgeError,
        p90EdgeError: payload.p90EdgeError,
        partialMatchRatio: payload.partialMatchRatio,
        partialRefinementAccepted: payload.partialRefinementAccepted,
        fullMatchIterationCount: payload.fullMatchIterationCount,
        pairCount: payload.pairCount ?? pairs.length,
        annotationIds: eligibleAnnotations.map((item) => item.id),
        annotationSavedAts: [],
        groupKey: currentCalibrationKey,
        storageKey: calibrationModelKey(currentCalibrationKey, modelToRun),
        generatedAt: new Date().toISOString(),
        iterationCount: payload.iterationCount,
        converged: payload.converged,
        pnpInlierCount: payload.pnpInlierCount,
        pnpReprojectionError: payload.pnpReprojectionError,
        qualityStatus: payload.qualityStatus,
        qualityWarnings: payload.qualityWarnings,
        pairDiagnostics: payload.pairDiagnostics,
        suggestedReviewAnnotationIds: payload.suggestedReviewAnnotationIds,
        algorithmVersion: payload.algorithmVersion ?? CALIBRATION_ALGORITHM_VERSION,
        distortionModel: modelToRun,
        distortionParameterCount: payload.distortionParameterCount ?? pairs[0]?.distortion.length,
      };
      result.initializationMode = extrinsicInitializationMode;
      result.annotationSavedAts = eligibleAnnotations.map((item) => item.savedAt);
      setCalibrations((current) => ({ ...current, [result.storageKey!]: result }));
      const resultTime = new Date(result.generatedAt).toLocaleString("zh-CN");
      const resultModelLabel = result.distortionModel === "fisheye" ? "鱼眼" : result.distortionModel === "radtan" ? "RadTan" : "Rational";
      const resultVersion: ExtrinsicVersion = {
        id: [currentCalibrationKey, result.distortionModel, "visible-edge", result.generatedAt].join(":"),
        groupKey: currentCalibrationKey,
        matrix: result.matrix,
        createdAt: result.generatedAt,
        label: `${resultModelLabel} · 最终结果 · ${resultTime}`,
        source: "calculated",
        isInitial: false,
        optimization: result,
        distortionModel: result.distortionModel,
      };
      setImportedExtrinsics((current) => ({
        ...current,
        [currentCalibrationKey]: [
          ...(current[currentCalibrationKey] ?? []).filter((item) =>
            item.id !== resultVersion.id && !item.id.includes(":sparse-containment:"),
          ),
          resultVersion,
        ],
      }));
      if (modelToRun === currentProjectionModel) {
        setComparisonRightVersionId(resultVersion.id);
        setExportVersionId(resultVersion.id);
      }
      await storeCalibration(result);
      setComparisonView({ zoom: 1, centerX: 0.5, centerY: 0.5 });
      const reviewPairNumbers = (result.suggestedReviewAnnotationIds ?? [])
        .map((id) => eligibleAnnotations.findIndex((annotation) => annotation.id === id) + 1)
        .filter((index) => index > 0);
      const diagnosis = [
        ...(result.qualityWarnings ?? []),
        ...(reviewPairNumbers.length ? [`建议复核配对 ${reviewPairNumbers.map((index) => `#${index}`).join("、")}`] : []),
      ];
      setOptimizationFeedback(`${modelIndex + 1}/${modelsToRun.length} · ${modelToRun} 已生成：Mask 内点 ${Math.round((result.optimizedInsideRatio ?? 0) * 100)}%，平均越界 ${(result.optimizedContainmentError ?? 0).toFixed(2)} px，可见背景误入 ${Math.round((result.backgroundIntrusionRatio ?? 0) * 100)}%，忽略后方遮挡点 ${result.occludedBackgroundCount ?? 0}${result.partialRefinementAccepted === false ? "，局部精调退化已自动回退" : ""} · ${result.pairCount} 个等权配对${diagnosis.length ? `；诊断：${diagnosis.join("；")}` : "；质量检查通过"}`);
      updateModelProgress(modelIndex, 100);
      // Keep the point-cloud annotation view after calibration. The comparison
      // page remains available as an explicit user action, regardless of whether
      // this run started from an existing extrinsic or from forward-facing PnP.
      setStageMode("annotate");
      completedModels.push(modelToRun);
      } catch (reason) {
        const message = reason instanceof Error ? reason.message : "未知错误";
        failedModels.push(`${modelToRun}：${message}`);
        updateModelProgress(modelIndex, 100);
        setOptimizationFeedback(`${modelToRun} 计算失败，其他模型继续计算：${message}`);
      }
      }));
      if (!completedModels.length) throw new Error(failedModels.join("；"));
      if (modelsToRun.length > 1) {
        setOptimizationFeedback(
          `已完成 ${completedModels.map((model) => model === "fisheye" ? "鱼眼" : model === "radtan" ? "RadTan" : "Rational").join("、")}` +
          (failedModels.length ? `；未成功：${failedModels.join("；")}` : "；三种模型均计算成功") +
          "；切换投影模型可查看对应结果",
        );
      }
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "未知错误";
      setOptimizationFeedback(`联合优化失败：${message}${optimizationIsStale ? "；旧配对结果已失效，不会继续用于投影或对比" : ""}`);
    } finally {
      setOptimizing(false);
    }
  };

  const exportExtrinsics = () => {
    if (!activeTaskGroup || !dataset || !exportablePackageVersions.length) return;
    const extrinsics = exportablePackageVersions.map(({ rigId, lidarInfo, cameraInfo, version }) => ({
      taskGroupId: activeTaskGroup.id,
      rigId,
      lidar: { id: lidarInfo.id, name: lidarInfo.name, topic: lidarInfo.topic ?? null },
      camera: { id: cameraInfo.id, name: cameraInfo.name, topic: cameraInfo.topic ?? null },
      transform: "camera_from_lidar",
      convention: { vectors: "column", translationUnit: "meter", cameraAxes: "x-right, y-down, z-forward", matrixOrder: "row-major", equation: "p_camera = T_camera_from_lidar @ p_lidar" },
      algorithmVersion: version.optimization?.algorithmVersion ?? "imported",
      sourceAnnotations: version.optimization ? { ids: version.optimization.annotationIds, savedAt: version.optimization.annotationSavedAts } : null,
      projectionModel: version.distortionModel ?? version.optimization?.distortionModel ?? null,
      version: {
        id: version.id,
        label: version.label,
        source: version.source,
        createdAt: version.createdAt,
        isInitial: version.isInitial,
      },
      matrix: Array.from({ length: 4 }, (_, row) => version.matrix.slice(row * 4, row * 4 + 4)),
    }));
    const blob = new Blob([JSON.stringify({
      version: 2,
      type: "autocalib-extrinsics",
      taskGroup: { id: activeTaskGroup.id, name: activeTaskGroup.name },
      dataset: { id: dataset.id, name: dataset.name, sourceFile: dataset.sourceFile },
      exportedAt: new Date().toISOString(),
      extrinsics,
    }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    const safeDatasetName = dataset.name.replace(/[^\p{L}\p{N}._-]+/gu, "-");
    anchor.download = `autocalib-extrinsics-${safeDatasetName}-${new Date().toISOString().replaceAll(":", "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setAnnotationFeedback(`已导出当前数据包的 ${extrinsics.length} 组外参结果`);
  };

  useDialogFocus(bagBrowserOpen || intrinsicsBrowserOpen, () => {
    if (!bagPreparation.active && !intrinsicsImporting) { setBagBrowserOpen(false); setIntrinsicsBrowserOpen(false); }
  });

  return (
    <>
    <main className={`app-shell ${activeTaskGroup ? "" : "no-active-group"} ${sidebarOpen ? "sidebar-open" : ""}`}>
      <header className="topbar">
        <div className="brand">
          <button className="brand-mark menu-toggle" aria-label="数据与传感器" aria-expanded={sidebarOpen} onClick={() => setSidebarOpen(v => !v)}>☰</button>
          <div><h1>ROC-Calib</h1><p>{calibrationWorkspaceMode === "lidar-lidar" ? "激光雷达—激光雷达标定工作台" : "激光雷达—相机标定工作台"}</p></div>
        </div>
        <div className="top-group-controls">
          <div className="top-calibration-mode" role="group" aria-label="选择标定类型">
            <button className={calibrationWorkspaceMode === "lidar-camera" ? "active" : ""} onClick={() => setCalibrationWorkspaceMode("lidar-camera")}>雷达 × 相机</button>
            {lidarCalibrationAvailable && <button className={calibrationWorkspaceMode === "lidar-lidar" ? "active" : ""} onClick={() => setCalibrationWorkspaceMode("lidar-lidar")}>雷达 × 雷达</button>}
          </div>
          <label><span>标定任务组</span>
            <select value={activeTaskGroupId} onChange={(event) => {
              selectTaskGroup(event.currentTarget.value);
              event.currentTarget.blur();
            }}>
              <option value="">请选择标定组</option>
              {taskGroups.map((group) => <option key={group.id} value={group.id}>{group.name}{group.status === "draft" ? "（空组）" : ""}</option>)}
            </select>
          </label>
          <button className="new-group-button" onClick={createTaskGroup}>新建组</button>
          <button className="delete-group-button" disabled={!activeTaskGroup || groupDeleting} onClick={deleteTaskGroup}>{groupDeleting ? "正在删除…" : "删除组"}</button>
          <button className="top-import-button" disabled={!activeTaskGroup} onClick={openBagBrowser}>添加数据</button>
          {calibrationWorkspaceMode === "lidar-camera" && <button className="top-intrinsics-button" disabled={!activeTaskGroup || intrinsicsImporting} onClick={openIntrinsicsBrowser}>{intrinsicsImporting ? "正在处理…" : intrinsicsProfile?.importedAt ? "重新导入内参" : "导入内参"}</button>}
        </div>
        <div className="top-status"><a href="https://github.com/zhangtingyu11/ROC-Calib/blob/main/docs/usage.md" target="_blank" rel="noreferrer">使用说明 ↗</a></div>
        {activeTaskGroup && <div className="save-summary" role="status"><span>{annotationFeedback}</span><span>{Object.values(saveStates).every(v => v === "已同步") ? "已同步" : Object.values(saveStates).filter(v => v !== "已同步").join(" · ")}</span>{Object.values(saveStates).some(v => v !== "已同步" && v !== "保存中") && <><button onClick={() => setSaveRetry(v => v + 1)}>重试同步</button><button onClick={() => { const blob = new Blob([JSON.stringify({group:activeTaskGroupId,annotations:savedAnnotations.filter(a => a.taskGroupId === activeTaskGroupId),calibrations,lidarPairAnnotations,lidarPairCalibrations},null,2)],{type:"application/json"});const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download="roc-calib-unsynced.json";a.click();URL.revokeObjectURL(url); }}>备份当前标注</button></>}</div>}
      </header>

      {activeTaskGroup && manifest && !dataset ? <section className="mode-empty-workspace">
        <div><b>{calibrationWorkspaceMode === "lidar-lidar" ? "雷达 × 雷达" : "雷达 × 相机"}</b><h2>当前模式还没有可用帧</h2><p>添加 ROS bag 或包含当前传感器组合的成对数据集。</p></div>
        <section>{activeTaskGroup.sourcePaths?.map((sourcePath) => <button key={sourcePath} onClick={() => void openSourceFrameEditor(sourcePath, manifest.datasets.find((item) => item.sourcePath === sourcePath && activeTaskGroup.datasetIds.includes(item.id)))}><span>{sourcePath.split("/").at(-1)}</span><b>选帧 / 改帧</b></button>)}</section>
        <button className="mode-empty-add" onClick={openBagBrowser}>添加数据</button>
      </section> : lidarCalibrationAvailable && calibrationWorkspaceMode === "lidar-lidar" && manifest && dataset && activeTaskGroup ? <LidarLidarWorkspace
        manifest={manifest}
        dataset={dataset}
        datasetIds={modeDatasets.map((item) => item.id)}
        sourcePaths={activeTaskGroup.sourcePaths ?? []}
        taskGroupId={activeTaskGroup.id}
        annotations={lidarPairAnnotations}
        calibrations={lidarPairCalibrations}
        framePairSets={lidarFramePairSets}
        noAnnotationPairKeys={lidarPairNoAnnotationKeys}
        topicRemarks={lidarTopicRemarks}
        graph={lidarExtrinsicGraph}
        onDatasetChange={(datasetId) => {
          const nextIndex = manifest.datasets.findIndex((item) => item.id === datasetId);
          if (nextIndex >= 0) setDatasetIndex(nextIndex);
        }}
        onEditDatasetFrames={(item) => void openDatasetFrameEditor(item)}
        onEditSourceFrames={(sourcePath) => void openSourceFrameEditor(sourcePath, manifest.datasets.find((item) => item.sourcePath === sourcePath && activeTaskGroup.datasetIds.includes(item.id)))}
        onAnnotationsChange={setLidarPairAnnotations}
        onCalibrationsChange={setLidarPairCalibrations}
        onFramePairSetsChange={setLidarFramePairSets}
        onNoAnnotationPairKeysChange={setLidarPairNoAnnotationKeys}
        onTopicRemarkChange={changeLidarTopicRemark}
        onGraphChange={setLidarExtrinsicGraph}
      /> : <section className={`workspace ${activeTaskGroup ? "" : "no-active-group"} ${focusedImagePanel ? "image-refinement-focus" : ""}`}>
        <aside className="sidebar">
          {activeTaskGroup && <>
          <section className="side-section">
            <div className="section-heading"><span>01</span><div><strong>组内数据</strong><small>{activeTaskGroup.name}</small></div></div>
            <div className="dataset-picker">
              <label className="select-field"><span>数据集</span>
                <select
                  aria-label="选择数据集"
                  value={dataset?.id ?? ""}
                  onChange={(event) => {
                    const nextIndex = manifest?.datasets.findIndex((item) => item.id === event.target.value) ?? -1;
                    if (nextIndex >= 0) setDatasetIndex(nextIndex);
                  }}
                >
                  {archivedDatasets.map(({ item }) => (
                    <option key={item.id} value={item.id}>{pairedDatasetIds.has(item.id) ? "🟢" : "🔴"} {item.name} · {item.selectedInterval.durationSeconds.toFixed(2)} s</option>
                  ))}
                </select>
              </label>
              {dataset && <small title={dataset.sourceFile}>{dataset.sourceFile}</small>}
              {!!activeTaskGroup.sourcePaths?.length && <div className="group-source-frame-list">{activeTaskGroup.sourcePaths.map((sourcePath) => <button key={sourcePath} title={sourcePath} onClick={() => void openSourceFrameEditor(sourcePath, manifest?.datasets.find((item) => item.sourcePath === sourcePath && activeTaskGroup.datasetIds.includes(item.id)))}><span>{sourcePath.split("/").at(-1)}</span><b>选帧</b></button>)}</div>}
            </div>
          </section>

          <section className="side-section">
            <div className="section-heading"><span>02</span><div><strong>传感器组合</strong><small>任意雷达与相机</small></div></div>
            <label className="select-field"><span>激光雷达 Topic</span>
              <select value={lidar?.id ?? lidarId} onChange={(event) => setLidarId(event.target.value)}>
                {dataset?.lidars.map((item) => <option key={item.id} value={item.id}>{lidarDisplayLabel(item.topic, item.name, lidarTopicRemarks)}</option>)}
              </select>
            </label>
            <button className="lidar-topic-remark-button" disabled={!lidar?.topic} onClick={() => promptLidarTopicRemark(lidar?.topic)}>备注</button>
            <label className="select-field"><span>相机 Topic</span>
              <select value={camera?.id ?? cameraId} onChange={(event) => setCameraId(event.target.value)}>
                {dataset?.cameras.map((item) => <option key={item.id} value={item.id}>{item.topic}</option>)}
              </select>
            </label>
            {camera && baseCameraIntrinsics && <label className="select-field"><span>投影模型</span>
              <select
                aria-label="选择相机投影模型"
                value={projectionModelChoice}
                onChange={(event) => {
                  const choice = event.target.value as ProjectionModelChoice;
                  setProjectionModelOverrides((current) => {
                    const next = { ...current };
                    if (choice === "auto") delete next[camera.id];
                    else next[camera.id] = choice;
                    return next;
                  });
                  setStageMode("annotate");
                  const modelLabel = choice === "fisheye" ? "鱼眼" : choice === "radtan" ? "针孔 RadTan" : "针孔 Rational";
                  const yamlLabel = baseCameraIntrinsics.distortionModel === "fisheye"
                    ? "鱼眼"
                    : baseCameraIntrinsics.distortionModel === "radtan" ? "针孔 RadTan" : "针孔 Rational";
                  const effectiveChoice = (choice === "auto" ? baseCameraIntrinsics.distortionModel : choice) as ProjectionModel;
                  const hasModelResult = Boolean(currentCalibrationKey && calibrations[calibrationModelKey(currentCalibrationKey, effectiveChoice)]);
                  setOptimizationFeedback(choice === "auto"
                    ? `已恢复 YAML 模型：${yamlLabel}；${hasModelResult ? "已加载该模型自己的外参结果" : "该模型尚无计算结果，当前显示初始外参"}`
                    : `已切换为${modelLabel}模型；${hasModelResult ? "已加载该模型自己的外参结果" : "该模型尚无计算结果，当前显示初始外参"}`);
                }}
              >
                <option value="auto">自动（YAML：{baseCameraIntrinsics.distortionModel === "fisheye" ? "鱼眼" : baseCameraIntrinsics.distortionModel === "radtan" ? "针孔 RadTan" : "针孔 Rational"}）</option>
                <option value="fisheye">鱼眼（OpenCV Fisheye · 4参数）</option>
                <option value="radtan">针孔 RadTan（4/5参数）</option>
                <option value="rational">针孔 Rational（8参数{baseCameraIntrinsics.distortion.length < 8 ? " · 缺失项补0" : ""}）</option>
              </select>
            </label>}
            {dataset && <label className="select-field sensor-frame-select"><span>静止场景帧</span>
              <select aria-label="选择静止场景帧" value={frameLabel} onChange={(event) => setFrameLabel(event.target.value)}>
                {camera?.frames.map(({ label, timestampNs }, index) => (
                  <option key={label} value={label}>{pairedFrameKeys.has(`${dataset.id}:${label}`) ? "🟢" : "🔴"} 帧 {label} · 场景 {index + 1} · {formatTimestamp(timestampNs)}</option>
                ))}
              </select>
            </label>}
          </section>

          <section className="side-section settings-section">
            <div className="section-heading"><span>03</span><div><strong>全局过滤与选择</strong><small>三视图实时联动</small></div></div>
            <label className="toggle-row"><span><strong>仅显示静止点</strong><small>三维与投影同步过滤</small></span><input type="checkbox" checked={stableOnly} onChange={(event) => setStableOnly(event.target.checked)} /></label>
            <label className="toggle-row"><span><strong>隐藏地面点</strong><small>三维与所有投影同步过滤</small></span><input type="checkbox" checked={hideGround} onChange={(event) => setHideGround(event.target.checked)} /></label>
          </section>

          </>}
        </aside>

        <section className="calibration-stage">
          {!activeTaskGroup ? <section className="welcome-panel"><span>ROC-Calib</span><h2>从物体配对开始标定</h2><p>在图像和点云中选中同一物体，计算相机与雷达之间的外参。</p><div className="welcome-actions"><button className="primary" disabled={loadingDemo} onClick={openDemo}>{loadingDemo ? "正在载入示例…" : "打开 CARLA 示例"}</button><button onClick={createTaskGroup}>新建标定组</button><a href="https://github.com/zhangtingyu11/ROC-Calib/blob/main/docs/usage.md">查看教程</a></div><ol><li><b>导入数据</b><span>添加图像、点云与相机内参。</span></li><li><b>保存配对</b><span>选择同一物体的点云和图像区域。</span></li><li><b>计算并检查</b><span>检查投影，导出 4×4 外参矩阵。</span></li></ol><small>已有任务可从顶部列表打开。示例首次下载约 56 MiB。</small><p role="status">{annotationFeedback}</p></section> : !activeTaskGroup.datasetIds.length ? <div className="empty-group-console">
            <span>EMPTY CALIBRATION GROUP</span>
            <h2>{activeTaskGroup.name}</h2>
            <p>当前组还没有数据，请使用顶部的“添加数据”导入 ROS bag 或成对数据集。</p>
            <small>大文件不会复制进组目录；标注、中间过程和结果写入当前组。</small>
          </div> : <>
          <header className="stage-toolbar">
            <div className="stage-pair-actions">
              <button className="clear-annotation" onClick={clearCurrentAnnotation}>清空当前标注</button>
              <button className="save-annotation" disabled={!canSaveAnnotation} onClick={saveAnnotation}>{savingAnnotation ? "保存中…" : activeAnnotationId ? "更新标注" : "保存配对"}</button>
            </div>
            <div className="stage-pair-center">
              <strong>{lidarDisplayLabel(lidar?.topic, lidar?.name, lidarTopicRemarks)} × {camera?.name ?? "相机"}</strong>
              <section className="embedded-pair-list" aria-label="当前雷达和相机的已保存配对">
                <div className="embedded-pair-title"><i aria-hidden="true" /><span>配对</span><b>{filteredAnnotations.length}</b></div>
                <label className="embedded-pair-frame-filter">
                  <span>帧</span>
                  <select aria-label="按帧筛选配对" value={pairFrameFilter} onChange={(event) => {
                    setPairFrameFilter(event.target.value);
                    setPairPage(0);
                  }}>
                    <option value="all">全部帧（{visibleAnnotations.length} 对）</option>
                    {pairFrameOptions.map((item) => <option key={item.key} value={item.key}>{item.label}</option>)}
                  </select>
                </label>
                <div className="embedded-pair-items">
                  {pagedAnnotations.length ? pagedAnnotations.map((annotation, index) => {
                    const annotationNumber = visibleAnnotationNumbers.get(annotation.id) ?? currentPairPage * PAIRS_PER_PAGE + index + 1;
                    return <div key={annotation.id} className={`embedded-pair-item ${activeAnnotationId === annotation.id ? "active" : ""}`}>
                      <button className="embedded-pair-preview" aria-pressed={activeAnnotationId === annotation.id} title={activeAnnotationId === annotation.id ? `取消选择配对 ${annotationNumber}` : `预览配对 ${annotationNumber}：帧 ${annotation.frame}，${formatPoints(annotation.pointCount)} 点`} onClick={() => toggleAnnotationPreview(annotation)}>
                        <b>#{annotationNumber}</b><small>帧 {annotation.frame}</small>
                        {reviewStatuses[annotation.id] && <small className={reviewStatuses[annotation.id]==='建议复查'?'review-badge warning':'review-badge'}>{reviewStatuses[annotation.id]}</small>}
                      </button>
                      <label className="embedded-pair-opt" title={annotation.status === "paired" ? "控制此配对是否参与外参优化" : "草稿不参与外参优化"}>
                        <input type="checkbox" aria-label={`配对 ${annotationNumber} 参与优化`} checked={annotation.status === "paired" && annotation.useForOptimization !== false} disabled={annotation.status !== "paired"} onChange={(event) => setAnnotationOptimizationEnabled(annotation, event.target.checked)} />
                      </label>
                      <button className="embedded-pair-delete" aria-label={`删除配对 ${annotationNumber}`} title={`删除配对 ${annotationNumber}`} onClick={() => deleteAnnotation(annotation, annotationNumber - 1)}>×</button>
                    </div>;
                  }) : <span className="embedded-pair-empty">当前帧暂无配对</span>}
                </div>
                <nav className="embedded-pair-pager" aria-label="配对列表分页">
                  <button aria-label="上一页" disabled={currentPairPage === 0} onClick={() => setPairPage((page) => Math.max(0, page - 1))}>‹</button>
                  <span>{currentPairPage + 1}/{pairPageCount}</span>
                  <button aria-label="下一页" disabled={currentPairPage >= pairPageCount - 1} onClick={() => setPairPage((page) => Math.min(pairPageCount - 1, page + 1))}>›</button>
                </nav>
              </section>
            </div>
            <div className="stage-view-actions">
              {stageMode === "annotate" && <label className="selected-region-toggle" title="点云仅显示当前选点，图像仅显示当前 Mask；关闭恢复全景，不删除数据。"><input type="checkbox" checked={selectedOnly} onChange={(event) => setSelectedOnly(event.target.checked)} /><span>仅显示选中区域</span></label>}
              <nav className="annotation-frame-nav" aria-label="标注帧切换">
                <button type="button" disabled={!camera || camera.frames.findIndex((frame) => frame.label === frameLabel) <= 0}
                  onClick={() => setFrameLabel((current) => {
                    const frames = camera?.frames ?? [];
                    const index = frames.findIndex((frame) => frame.label === current);
                    return index > 0 ? frames[index - 1].label : current;
                  })}>← 上一帧</button>
                <span className="annotation-frame-position">
                  <input
                    key={`${dataset?.id}:${camera?.id}:${frameLabel}`}
                    className="annotation-frame-input"
                    aria-label="跳转到第几帧"
                    title={`输入帧序号（1–${camera?.frames.length ?? 0}），按回车跳转`}
                    type="number" min={1} max={camera?.frames.length ?? 0} step={1}
                    disabled={!camera?.frames.length}
                    defaultValue={camera ? Math.max(0, camera.frames.findIndex((frame) => frame.label === frameLabel) + 1) : 0}
                    onFocus={(event) => event.currentTarget.select()}
                    onBlur={(event) => {
                      const frames = camera?.frames ?? [];
                      const requested = event.currentTarget.valueAsNumber;
                      if (!frames.length || !Number.isInteger(requested)) {
                        event.currentTarget.value = event.currentTarget.defaultValue;
                        return;
                      }
                      const index = Math.max(0, Math.min(frames.length - 1, requested - 1));
                      event.currentTarget.value = String(index + 1);
                      setFrameLabel(frames[index].label);
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        event.preventDefault();
                        event.currentTarget.blur();
                      } else if (event.key === "Escape") {
                        event.preventDefault();
                        event.currentTarget.value = event.currentTarget.defaultValue;
                        event.currentTarget.blur();
                      }
                    }}
                  /> / {camera?.frames.length ?? 0}
                </span>
                <button type="button" disabled={!camera || !camera.frames.some((frame) => frame.label === frameLabel) || cameraFrameIndex >= camera.frames.length - 1}
                  onClick={() => setFrameLabel((current) => {
                    const frames = camera?.frames ?? [];
                    const index = frames.findIndex((frame) => frame.label === current);
                    return index >= 0 && index + 1 < frames.length ? frames[index + 1].label : current;
                  })}>下一帧 →</button>
              </nav>
              <button className={stageMode === "annotate" ? "stage-mode active" : "stage-mode"} onClick={() => setStageMode("annotate")}>标注</button>
              <button
                className={stageMode === "compare" ? "stage-mode compare active" : "stage-mode compare"}
                disabled={!canCompareExtrinsics}
                title={canCompareExtrinsics ? "选择两个已记录的外参版本进行投影对比" : "需要先设置初始外参并生成当前外参"}
                onClick={() => setStageMode("compare")}
              >外参对比</button>
            </div>
          </header>
          <section className="calibration-workflow-bar" aria-label="外参标定工作流">
            <div className="calibration-workflow-status">
              <strong>{lidarDisplayLabel(lidar?.topic, lidar?.name, lidarTopicRemarks)} → {camera?.name ?? "相机"}</strong>
            </div>
            <div className="calibration-workflow-actions">
              <input ref={extrinsicImportRef} className="extrinsic-file-input" type="file" accept="application/json,.json" aria-label="选择已有外参文件" onChange={(event) => void importExtrinsicFile(event.currentTarget.files?.[0] ?? null, event.currentTarget)} />
              <button className="import-extrinsic-button" onClick={() => extrinsicImportRef.current?.click()}>导入初始外参</button>
              <label className="calibration-mode-select"><span>生成方式</span><select aria-label="标定结果生成方式" value={extrinsicInitializationMode} onChange={(event) => setExtrinsicInitializationMode(event.target.value as ExtrinsicInitializationMode)}>
                <option value="forward">无初值 PnP</option>
                <option value="existing" disabled={!projection}>基于初始外参精调</option>
              </select></label>
              <button className="generate-calibration-button" title={optimizationBlockedReason ?? "使用当前相机模型运行 ROC-Calib"} disabled={optimizing || !!optimizationBlockedReason} onClick={() => void runOptimization()}>
                {optimizing ? `正在计算 ${optimizationProgress}%` : !baseCameraIntrinsics ? "请先导入当前相机内参" : `计算外参${eligibleAnnotations.length ? `（${eligibleAnnotations.length} 对）` : ""}`}
              </button>
              <button className="set-initial-extrinsic-button" disabled={!currentOptimization || currentResultIsInitial} onClick={setCurrentResultAsInitial}>
                {currentResultIsInitial ? "当前结果已是初值" : "设当前结果为初值"}
              </button>
              <label className="extrinsic-version-select"><span>外参版本</span><select aria-label="选择预览和导出外参" value={selectedExportVersion?.id ?? ""} disabled={!comparisonVersions.length} onChange={(event) => setExportVersionId(event.target.value)}>
                {comparisonVersions.map((item) => <option key={item.id} value={item.id}>{item.isInitial ? "当前初值 · " : ""}{item.label}</option>)}
              </select>{optimizationIsStale && selectedExportVersion?.source === "calculated" && <span className="extrinsic-stale-badge" role="status" title="标注已更新，当前仍预览已保存的历史外参；可直接复查，需要新结果时再重新计算。">待重算</span>}</label>
              <button className="export-extrinsic-button" disabled={!exportablePackageVersions.length} onClick={exportExtrinsics}>
                {exportablePackageVersions.length ? `导出当前包结果（${exportablePackageVersions.length}组）` : "当前包暂无结果"}
              </button>
            </div>
            {optimizing && <div className="workflow-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={optimizationProgress}><i style={{ width: `${optimizationProgress}%` }} /></div>}
          </section>

          {activeTaskGroupId && currentCalibrationKey && <CalibrationReviewPanel
            group={activeTaskGroupId} scope={`${currentCalibrationKey}:${currentProjectionModel}`}
            reviewBasis={JSON.stringify([workingMatrix,currentProjectionModel,displayIntrinsics])}
            annotationStamps={Object.fromEntries(eligibleAnnotations.map(a=>[a.id,`${a.savedAt}:${reviewPairEdits[a.id]??0}`]))}
            version={JSON.stringify([eligibleAnnotations.map((a)=>[a.id,a.savedAt]),workingMatrix,currentProjectionModel,reviewEditRevision,displayIntrinsics,extrinsicInitializationMode])}
            matrix={workingMatrix} disabled={optimizing || !groupStateReady} activeId={activeAnnotationId}
            getRequest={()=>({group_key:currentCalibrationKey,initialization_mode:extrinsicInitializationMode,base_matrix:initialMatrix,previous_matrix:extrinsicInitializationMode==='forward'?null:optimization?.matrix,pairs:buildCalibrationPairs(currentProjectionModel)})}
            current={cameraFrame && lidarFrame && dataset ? {datasetId:dataset.id,frame:frameLabel,imageUrl:cameraFrame.url,width:cameraFrame.width,height:cameraFrame.height,cloudUrl:lidarFrame.url,selectedIndices:Array.from(pointSegment),intrinsic:displayIntrinsics.intrinsic,distortion:displayIntrinsics.distortion,distortionModel:currentProjectionModel,temporalProjection} : null}
            onStatuses={setReviewStatuses} onHighlight={setReviewHighlight}
            onStructureLine={setStructureLine}
            extractionSceneKey={JSON.stringify([dataset?.id,lidar?.id,camera?.id,currentProjectionModel,displayIntrinsics.intrinsic,displayIntrinsics.distortion,camera?.frames.map(f=>f.label)])}
            getSceneFrames={()=>camera&&lidar&&dataset?camera.frames.map(f=>({datasetId:dataset.id,frame:f.label,imageUrl:f.url,width:f.width,height:f.height,cloudUrl:lidar.frames.find(l=>l.label===f.label)?.url??'',intrinsic:displayIntrinsics.intrinsic,distortion:displayIntrinsics.distortion,distortionModel:currentProjectionModel})):[]}
            onExtractionView={(view)=>{
              setExtractionView(view);
              if(!view)return;
              const index=manifest?.datasets.findIndex(d=>d.id===view.datasetId)??-1;
              if(index<0)return;
              setDatasetIndex(index);setFrameLabel(view.frame);setStageMode('annotate');
              if(view.focus)setExtractionFocus(value=>value+1);
            }}
            onStructurePreview={(item)=>{
              const index=manifest?.datasets.findIndex(d=>d.id===item.datasetId)??-1;
              if(index<0)return;
              setStructurePreview(item);setDatasetIndex(index);setFrameLabel(item.frame);setStageMode('annotate');
            }}
            onPreview={(id)=>{const a=savedAnnotations.find((item)=>item.id===id);if(a){previewAnnotation(a);setStageMode('annotate');}}}
            onApply={(matrix)=>{
              const id=`${currentCalibrationKey}:structure:${Date.now()}`;
              const version:ExtrinsicVersion={id,groupKey:currentCalibrationKey,matrix,createdAt:new Date().toISOString(),label:'物体＋棱线联合结果（人工应用·实验性）',source:'calculated',isInitial:false,distortionModel:currentProjectionModel};
              setImportedExtrinsics((all)=>({...all,[currentCalibrationKey]:[...(all[currentCalibrationKey]??[]),version]}));
              setComparisonLeftVersionId(exportVersionId);setComparisonRightVersionId(id);setExportVersionId(id);setStageMode('compare');
            }}
          />}
          {stageMode === "annotate" ? <div className="two-view-grid">
            <article className="work-panel cloud-panel">
              <header className="panel-head annotation-panel-head">
                <div className="panel-title-block"><strong>三维点云</strong></div>
                <div className="inline-annotation-tools" role="group" aria-label="点云标注工具">
                  <button className={!projectionPicking && pointInteractionMode === "navigate" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("navigate"); }}>旋转视角</button>
                  <button className={!projectionPicking && pointInteractionMode === "box" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("box"); }}>框选聚类</button>
                  <button className={!projectionPicking && pointInteractionMode === "brush-add" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("brush-add"); }}>画笔添加</button>
                  <button className={!projectionPicking && pointInteractionMode === "brush-remove" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("brush-remove"); }}>画笔删除</button>
                  <button className={!projectionPicking && pointInteractionMode === "point-add" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("point-add"); }}>添加单点</button>
                  <button className={!projectionPicking && pointInteractionMode === "point-remove" ? "active" : ""} onClick={() => { setProjectionPicking(false); setPointInteractionMode("point-remove"); }}>删除单点</button>
                </div>
                <div className="point-cluster-mode-tools" role="group" aria-label="投影图点云操作">
                  <button disabled={!canPickProjectedPoint} className={projectionPicking && pointMode === "replace" ? "active" : ""} onClick={() => { setPointMode("replace"); setProjectionPicking(true); }}>单点聚类替换</button>
                  <button disabled={!canPickProjectedPoint} className={projectionPicking && pointMode === "add" ? "active" : ""} onClick={() => { setPointMode("add"); setProjectionPicking(true); }}>单点聚类合并</button>
                  {!canPickProjectedPoint && <span className="tool-disabled-hint">暂无可操作的投影图</span>}
                </div>
                <div className="panel-actions compact-panel-actions">
                  <details className="panel-settings">
                    <summary>点云设置</summary>
                    <div className="panel-settings-popover point-annotation-popover">
                      <label><span>三维点大小 <b>{pointSize.toFixed(3)}</b></span><input aria-label="三维点大小" type="range" min="0.005" max="0.24" step="0.005" value={pointSize} onChange={(event) => setPointSize(Number(event.target.value))} /></label>
                      <label><span>画笔直径 <b>{pointBrushSize}px</b></span><input type="range" min="4" max="120" step="2" value={pointBrushSize} onChange={(event) => setPointBrushSize(Number(event.target.value))} /></label>
                      <label><span>画笔深度容差 <b>{pointBrushDepthTolerance.toFixed(1)}m</b></span><input type="range" min="0.1" max="3" step="0.1" value={pointBrushDepthTolerance} onChange={(event) => setPointBrushDepthTolerance(Number(event.target.value))} /></label>
                      <label className="popover-toggle"><span>抑制地面 <small>聚类时排除低矮地面点</small></span><input type="checkbox" checked={removeGround} onChange={(event) => setRemoveGround(event.target.checked)} /></label>
                      <label><span>聚类补全尺度 <b>{clusterVoxelSize.toFixed(2)}m</b></span><input type="range" min="0.10" max="0.60" step="0.05" value={clusterVoxelSize} onChange={(event) => setClusterVoxelSize(Number(event.target.value))} /></label>
                      <label><span>最大法向夹角 <b>{surfaceAngle}°</b></span><input type="range" min="10" max="60" step="5" value={surfaceAngle} onChange={(event) => setSurfaceAngle(Number(event.target.value))} /></label>
                      <label><span>最大面间跳变 <b>{surfaceThickness.toFixed(2)}m</b></span><input type="range" min="0.03" max="0.30" step="0.01" value={surfaceThickness} onChange={(event) => setSurfaceThickness(Number(event.target.value))} /></label>
                      <div className={pointGrowFeedback.kind === "error" ? "point-grow-status error" : "point-grow-status"} aria-live="polite"><strong>聚类状态</strong><span>{pointGrowFeedback.text}</span></div>
                    </div>
                  </details>
                  <button onClick={() => setViewRequest((request) => ({ mode: "near", token: request.token + 1 }))}>近景</button>
                  <button onClick={() => setViewRequest((request) => ({ mode: "all", token: request.token + 1 }))}>全景</button>
                </div>
              </header>
              <div className="panel-body cloud-body">
                <PointCloudViewport
                  preserveViewOnCloudChange
                  clouds={clouds}
                  frameIndex={frameIndex}
                  stableOnly={stableOnly}
                  hideGround={hideGround}
                  pointSize={pointSize}
                  brushSize={pointBrushSize}
                  brushDepthTolerance={pointBrushDepthTolerance}
                  interactionMode={activeExtraction?"navigate":pointInteractionMode}
                  selectedIndices={activeExtraction?new Uint32Array():pointSegment}
                  structureLine={activeExtraction?null:structureLine}
                  inspectionLines={activeExtraction?.lines}
                  inspectionFocus={activeExtraction?extractionFocus:0}
                  selectedOnly={activeExtraction?activeExtraction.isolate:selectedOnly}
                  overlayClouds={activeExtraction?extractionOverlays:reviewOverlayClouds}
                  overlayOpacity={1}
                  clearSelectionToken={clearSelectionToken}
                  viewRequest={viewRequest}
                  onSelection={(indices) => { if (!activeExtraction&&!selectedOnly) handlePointCandidate(indices); }}
                  onPointEdit={(indices,mode)=>{if(!activeExtraction)handlePointEdit(indices,mode);}}
                  onInteractionModeChange={setPointInteractionMode}
                />
                {loading && <div className="empty-state"><span className="loader" /><strong>正在载入 {lidarDisplayLabel(lidar?.topic, lidar?.name, lidarTopicRemarks)}</strong></div>}
                {error && <div className="empty-state error"><strong>{error}</strong></div>}
                <div className="axis-note">Z ↑ · X 前 · Y 左</div>
                {pointInteractionMode !== "navigate" && <div className="selection-hint"><b>{pointInteractionMode === "box" ? "框选聚类" : pointInteractionMode === "point-add" ? "添加单点" : pointInteractionMode === "point-remove" ? "删除单点" : pointInteractionMode === "brush-add" ? "画笔添加" : "画笔删除"}</b><span>{pointInteractionMode === "box" ? "框选后按 Enter 执行" : pointInteractionMode.startsWith("brush") ? "按住拖动直接精修可见点" : "点击离鼠标最近的可见点"}</span></div>}
              </div>
            </article>

            <article className="work-panel unified-image-panel">
              <header className="panel-head annotation-panel-head">
                <div className="panel-title-block"><strong>图像与投影</strong></div>
                <div className="inline-annotation-tools" role="group" aria-label="图像标注工具">
                  {([[
                    "pan", "平移图像"], ["box-replace", "SAM 替换"], ["box-add", "SAM 追加"],
                    ["brush-add", "画笔添加"], ["brush-remove", "画笔擦除"],
                  ] as Array<[ImageSegmentationTool, string]>).map(([value, label]) => (
                    <button key={value} className={imageTool === value && !projectionPicking ? "active" : ""} onClick={() => { setProjectionPicking(false); setImageTool(value); }}>{label}</button>
                  ))}
                  <label className="inline-brush-size">
                    <span>画笔直径</span>
                    <input aria-label="图像画笔直径" type="range" min="1" max="80" step="1" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
                    <b>{brushSize}px</b>
                  </label>
                </div>
                <div className="panel-actions compact-panel-actions">
                  <details className="panel-settings">
                    <summary>图像设置</summary>
                    <div className="panel-settings-popover projection-settings-popover">
                      <label><span>Mask 透明度 <b>{Math.round(maskOpacity * 100)}%</b></span><input aria-label="Mask 透明度" type="range" min="0.05" max="1" step="0.05" value={maskOpacity} onChange={(event) => setMaskOpacity(Number(event.target.value))} /></label>
                      <label><span>投影点大小 <b>{projectionSize}px</b></span><input aria-label="投影点大小" type="range" min="1" max="10" step="1" value={projectionSize} onChange={(event) => setProjectionSize(Number(event.target.value))} /></label>
                      <label><span>点云透明度 <b>{Math.round(projectionOpacity * 100)}%</b></span><input aria-label="点云投影透明度" type="range" min="0" max="1" step="0.01" value={projectionOpacity} onChange={(event) => setProjectionOpacity(Number(event.target.value))} /></label>
                      <label><span>背景图透明度 <b>{Math.round(projectionBackgroundOpacity * 100)}%</b></span><input aria-label="投影背景图透明度" type="range" min="0" max="1" step="0.01" value={projectionBackgroundOpacity} onChange={(event) => setProjectionBackgroundOpacity(Number(event.target.value))} /></label>
                    </div>
                  </details>
                </div>
                <div className="projection-visibility-controls" role="group" aria-label="点云投影显示控制">
                  <label><input type="checkbox" checked={showSelectedPointProjection} onChange={(event) => setShowSelectedPointProjection(event.target.checked)} /><span>显示选中点云投影</span></label>
                  <label><input type="checkbox" checked={showFullPointProjection} onChange={(event) => setShowFullPointProjection(event.target.checked)} /><span>显示点云完整投影</span></label>
                  <label><input type="checkbox" checked={filterProjectionOcclusion} disabled={!showFullPointProjection || fullProjectionScope !== "all"} onChange={(event) => setFilterProjectionOcclusion(event.target.checked)} /><span>全部点云遮挡过滤</span></label>
                  <label className="projection-scope-select"><span>完整投影范围</span><select aria-label="选择完整点云投影范围" value={fullProjectionScope} disabled={!showFullPointProjection} onChange={(event) => setFullProjectionScope(event.target.value as FullProjectionScope)}>
                    <option value="all">全部点云</option>
                    <option value="frame-annotations">当前图所有配对</option>
                    <option value="single-annotation">单个配对</option>
                  </select></label>
                  {fullProjectionScope === "single-annotation" && <label className="projection-pair-select"><span>选择配对</span><select aria-label="选择单个投影配对" value={singleProjectionAnnotationId} disabled={!showFullPointProjection || currentFrameAnnotations.length === 0} onChange={(event) => setSingleProjectionAnnotationId(event.target.value)}>
                    {currentFrameAnnotations.length === 0 && <option value="">当前图无配对</option>}
                    {currentFrameAnnotations.map((annotation) => <option key={annotation.id} value={annotation.id}>#{visibleAnnotations.findIndex((item) => item.id === annotation.id) + 1}</option>)}
                  </select></label>}
                </div>
              </header>
              <div className="panel-body image-body unified-image-body">
                {displayedCameraFrame ? <ImageCanvas
                  frame={displayedCameraFrame}
                  cloud={workingMatrix ? currentCloud : null}
                  matrix={workingMatrix ?? FORWARD_FACING_LIDAR_TO_CAMERA}
                  intrinsic={displayIntrinsics.intrinsic}
                  distortion={displayIntrinsics.distortion}
                  distortionModel={displayIntrinsics.distortionModel}
                  temporalProjection={temporalProjection}
                  showProjection={Boolean(!activeExtraction && workingMatrix && currentCloud && showFullPointProjection)}
                  stableOnly={stableOnly}
                  hideGround={hideGround}
                  projectionSize={projectionSize}
                  projectionOpacity={projectionOpacity}
                  backgroundOpacity={projectionBackgroundOpacity}
                  maskOpacity={maskOpacity}
                  mask={!activeExtraction && imageAnnotationEnabled ? canvasMaskView(selectedOnly ? imageMask : displayedPrimaryMask) : null}
                  secondaryMask={!activeExtraction && !selectedOnly && imageAnnotationEnabled ? canvasMaskView(displayedSecondaryMask) : null}
                  selectedPointIndices={pointSegment}
                  selectedOnly={!activeExtraction&&selectedOnly}
                  isolationMask={canvasMaskView(imageMask)}
                  diagnosticHighlight={activeExtraction?null:visibleReviewHighlight}
                  projectionPointIndices={fullProjectionPointIndices}
                  occlusionFiltering={filterProjectionOcclusion}
                  projectionScope={fullProjectionScope === "all" ? "all" : "selected"}
                  highlightSelectedProjection={!activeExtraction&&showSelectedPointProjection}
                  selectable={!activeExtraction&&canEditDisplayedMask && !projectionPicking}
                  segmentationTool={activeExtraction?"pan":imageTool}
                  onSegmentationToolChange={(tool) => { setProjectionPicking(false); setImageTool(tool); }}
                  brushSize={brushSize}
                  zoom={1}
                  localZoom
                  zoomResetToken={imageZoomResetToken}
                  granularity={granularity}
                  preserveMaskLayers
                  projectionPicking={!activeExtraction&&projectionPicking && canPickProjectedPoint}
                  onProjectedPointPick={handleProjectedPointPick}
                  onProjectedCount={setProjectedCount}
                  onMaskEditStart={imageAnnotationEnabled ? beginImageMaskEdit : undefined}
                  onMaskChange={imageAnnotationEnabled ? updateImageMask : undefined}
                /> : <div className="empty-state"><strong>当前没有可显示的相机帧</strong></div>}
              </div>
            </article>
          </div> : <div className="comparison-workspace">
            <div className="comparison-toolbar">
              <div className="comparison-scope" role="group" aria-label="对比投影范围">
                <span>投影范围</span>
                <button
                  className={comparisonProjectionScope === "all" ? "active" : ""}
                  aria-pressed={comparisonProjectionScope === "all"}
                  onClick={() => setComparisonProjectionScope("all")}
                >全部点云</button>
                <button
                  className={comparisonProjectionScope === "selected" ? "active" : ""}
                  aria-pressed={comparisonProjectionScope === "selected"}
                  onClick={() => setComparisonProjectionScope("selected")}
                >仅选中点云</button>
                <button
                  className={comparisonProjectionScope === "frame-annotations" ? "active" : ""}
                  aria-pressed={comparisonProjectionScope === "frame-annotations"}
                  onClick={() => setComparisonProjectionScope("frame-annotations")}
                >当前图全部配对</button>
              </div>
              <div className="comparison-view-controls">
                <span>两图同步 · {comparisonView.zoom.toFixed(2)}×</span>
                <details className="comparison-settings">
                  <summary>显示设置</summary>
                  <div className="comparison-settings-popover">
                    <label><span>Mask 清晰度 <b>{Math.round(maskOpacity * 100)}%</b></span><input aria-label="对比 Mask 清晰度" type="range" min="0.05" max="1" step="0.05" value={maskOpacity} onChange={(event) => setMaskOpacity(Number(event.target.value))} /></label>
                    <label><span>点云透明度 <b>{Math.round(projectionOpacity * 100)}%</b></span><input aria-label="对比点云透明度" type="range" min="0" max="1" step="0.01" value={projectionOpacity} onChange={(event) => setProjectionOpacity(Number(event.target.value))} /></label>
                    <label><span>投影点大小 <b>{projectionSize}px</b></span><input aria-label="对比投影点大小" type="range" min="1" max="10" step="1" value={projectionSize} onChange={(event) => setProjectionSize(Number(event.target.value))} /></label>
                    <label><span>背景图透明度 <b>{Math.round(projectionBackgroundOpacity * 100)}%</b></span><input aria-label="对比背景图透明度" type="range" min="0.1" max="1" step="0.05" value={projectionBackgroundOpacity} onChange={(event) => setProjectionBackgroundOpacity(Number(event.target.value))} /></label>
                  </div>
                </details>
                <button onClick={() => setComparisonView({ zoom: 1, centerX: 0.5, centerY: 0.5 })}>复位视图</button>
              </div>
            </div>
            <div className="comparison-images">
              {cameraFrame && comparisonIntrinsics && selectedLeftVersion && <article className="comparison-card">
                <header className="comparison-version-header">
                  <label><span>左侧外参</span><select aria-label="选择左侧对比外参" value={comparisonLeftVersionId} onChange={(event) => setComparisonLeftVersionId(event.target.value)}>
                    {comparisonVersions.map((item) => <option key={item.id} value={item.id}>{item.isInitial ? "当前初值 · " : ""}{item.label}</option>)}
                  </select></label>
                </header>
                <div className="comparison-canvas"><ImageCanvas frame={cameraFrame} cloud={currentCloud} matrix={selectedLeftVersion.matrix} intrinsic={comparisonIntrinsics.intrinsic} distortion={comparisonIntrinsics.distortion} distortionModel={comparisonIntrinsics.distortionModel} temporalProjection={temporalProjection} showProjection stableOnly={stableOnly} hideGround={hideGround} projectionSize={projectionSize} projectionOpacity={projectionOpacity} maskOpacity={maskOpacity} backgroundOpacity={projectionBackgroundOpacity} mask={canvasMaskView(comparisonMask)} selectedPointIndices={comparisonPointIndices} projectionScope={comparisonProjectionScope} selectable={false} brushSize={brushSize} zoom={comparisonView.zoom} synchronizedView={comparisonView} onSynchronizedViewChange={setComparisonView} granularity={granularity} /></div>
              </article>}
              {cameraFrame && comparisonIntrinsics && selectedRightVersion && <article className="comparison-card optimized">
                <header className="comparison-version-header">
                  <label><span>右侧外参</span><select aria-label="选择右侧对比外参" value={comparisonRightVersionId} onChange={(event) => setComparisonRightVersionId(event.target.value)}>
                    {comparisonVersions.map((item) => <option key={item.id} value={item.id}>{item.isInitial ? "当前初值 · " : ""}{item.label}</option>)}
                  </select></label>
                </header>
                <div className="comparison-canvas"><ImageCanvas frame={cameraFrame} cloud={currentCloud} matrix={selectedRightVersion.matrix} intrinsic={comparisonIntrinsics.intrinsic} distortion={comparisonIntrinsics.distortion} distortionModel={comparisonIntrinsics.distortionModel} temporalProjection={temporalProjection} showProjection stableOnly={stableOnly} hideGround={hideGround} projectionSize={projectionSize} projectionOpacity={projectionOpacity} maskOpacity={maskOpacity} backgroundOpacity={projectionBackgroundOpacity} mask={canvasMaskView(comparisonMask)} selectedPointIndices={comparisonPointIndices} projectionScope={comparisonProjectionScope} selectable={false} brushSize={brushSize} zoom={comparisonView.zoom} synchronizedView={comparisonView} onSynchronizedViewChange={setComparisonView} granularity={granularity} /></div>
              </article>}
            </div>
                {comparisonOptimization && <div className="optimization-report">
                  <div><span>结果诊断</span><b>{comparisonOptimization.qualityStatus === "good" ? "质量良好" : comparisonOptimization.qualityStatus === "warning" ? "建议补充配对" : comparisonOptimization.qualityStatus === "poor" ? "建议复核" : "请检查投影"}</b></div>
                  <div><span>投影模型</span><b>{comparisonOptimization.distortionModel === "fisheye" ? "鱼眼" : comparisonOptimization.distortionModel === "radtan" ? "针孔 RadTan" : `针孔 Rational${(comparisonOptimization.distortionParameterCount ?? 8) < 8 ? "（缺失项补0）" : ""}`}</b></div>
              <div><span>联合配对</span><b>{comparisonOptimization.pairCount} 对</b></div>
              <div><span>自动迭代</span><b>{comparisonOptimization.iterationCount ?? 1} 轮 · {comparisonOptimization.converged ? "已收敛" : "达到上限"}</b></div>
              <div><span>初始入 Mask</span><b>{Math.round((comparisonOptimization.originalInsideRatio ?? 0) * 100)}%</b></div>
              <div><span>最终入 Mask</span><b>{Math.round((comparisonOptimization.optimizedInsideRatio ?? 0) * 100)}%</b></div>
              <div><span>平均越界距离</span><b>{(comparisonOptimization.optimizedContainmentError ?? 0).toFixed(2)} px</b></div>
              {['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z'].map((label, index) => <div key={label}><span>Δ{label}</span><b>{index < 3 ? `${(comparisonOptimization.delta[index] * 180 / Math.PI).toFixed(3)}°` : `${comparisonOptimization.delta[index].toFixed(3)} m`}</b></div>)}
            </div>}
          </div>}
          </>}
        </section>
      </section>}
    </main>
    {bagBrowserOpen && <div className="bag-browser-backdrop" role="presentation" onMouseDown={() => { if (!bagPreparation.active && !localBagUpload.active) setBagBrowserOpen(false); }}>
      <section className={`bag-browser-dialog ${manualTimeline || manualTimelineLoading ? "manual-frame-open" : ""} ${manualTimeline && calibrationWorkspaceMode === "lidar-lidar" ? "lidar-manual-frame-open" : ""}`} role="dialog" aria-modal="true" aria-label="选择数据源" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><span>{editingDatasetId ? "FRAME SELECTION" : "DATA IMPORT"}</span><h2>{editingDatasetId ? "选帧 / 改帧" : "添加数据源"}</h2></div>
          <button aria-label="关闭" disabled={bagPreparation.active || localBagUpload.active} onClick={() => setBagBrowserOpen(false)}>×</button>
        </header>
        <div className="bag-calibration-mode" role="group" aria-label="本任务组标定类型">
          <span>这批数据用于</span>
          <button className={calibrationWorkspaceMode === "lidar-camera" ? "active" : ""} disabled={bagPreparation.active || localBagUpload.active} onClick={() => setCalibrationWorkspaceMode("lidar-camera")}><b>雷达 × 相机外参</b><small>点云目标与图像 Mask 配对</small></button>
          {lidarCalibrationAvailable && <button className={calibrationWorkspaceMode === "lidar-lidar" ? "active" : ""} disabled={bagPreparation.active || localBagUpload.active} onClick={() => setCalibrationWorkspaceMode("lidar-lidar")}><b>雷达 × 雷达外参</b><small>双点云参照物配对</small></button>}
        </div>
        {!editingDatasetId && <div className="bag-import-source-tabs" role="tablist" aria-label="数据源类型">
          <button role="tab" aria-selected={bagImportSource === "local"} className={bagImportSource === "local" ? "active" : ""} disabled={bagPreparation.active || localBagUpload.active} onClick={() => switchBagImportSource("local")}><b>上传 ROS bag</b><small>从 Windows 直接选择</small></button>
          <button role="tab" aria-selected={bagImportSource === "server"} className={bagImportSource === "server" ? "active" : ""} disabled={bagPreparation.active || localBagUpload.active} onClick={() => switchBagImportSource("server")}><b>服务器 ROS bag</b><small>使用已经上传的数据包</small></button>
          <button role="tab" aria-selected={bagImportSource === "paired"} className={bagImportSource === "paired" ? "active" : ""} disabled={bagPreparation.active || localBagUpload.active} onClick={() => switchBagImportSource("paired")}><b>成对数据集</b><small>图像与点云显式配对</small></button>
        </div>}
        {!editingDatasetId && (bagImportSource === "server" ? <>
          <div className="bag-browser-location">
            <button disabled={bagBrowser?.parent === null || bagBrowserLoading} onClick={() => void browseBagDirectory(bagBrowser?.parent ?? "")}>← 上一级</button>
            <code>{bagBrowser?.root ?? "/bag-source"}{bagBrowser?.path ? `/${bagBrowser.path}` : ""}</code>
          </div>
          <div className="bag-browser-list">
            {bagBrowserLoading ? <div className="bag-browser-message">正在读取目录…</div> : !bagBrowser?.entries.length ? <div className="bag-browser-message">这个目录中没有文件夹或 rosbag</div> : bagBrowser.entries.map((entry) => entry.kind === "directory" ?
              <button className="bag-directory-row" disabled={bagPreparation.active} key={entry.path} onClick={() => void browseBagDirectory(entry.path)}>
                <span className="bag-icon">▸</span><strong>{entry.name}</strong><small>文件夹</small>
              </button> :
              <label className={`bag-file-row ${entry.prepared ? "prepared" : "unprepared"}`} key={entry.path}>
                <input type="radio" name="manual-bag" checked={Boolean(selectedBags[entry.path])} disabled={bagPreparation.active || manualTimelineLoading} onChange={() => void toggleSelectedBag(entry)} />
                <span className="bag-icon">▣</span>
                <strong>{entry.name}</strong>
                <small>{entry.prepared ? `已有处理记录 · 手选组合可继续缓存 · ${(entry.sizeBytes / 1024 / 1024 / 1024).toFixed(2)} GB` : `选择后手动选帧 · ${(entry.sizeBytes / 1024 / 1024 / 1024).toFixed(2)} GB`}</small>
              </label>
            )}
          </div>
        </> : bagImportSource === "paired" ? <>
          <div className="bag-browser-location">
            <button disabled={pairedBrowser?.parent === null || pairedBrowserLoading} onClick={() => void browsePairedDirectory(pairedBrowser?.parent ?? "")}>← 上一级</button>
            <code>{pairedBrowser?.root ?? "/data/paired"}{pairedBrowser?.path ? `/${pairedBrowser.path}` : ""}</code>
          </div>
          <div className="bag-browser-list">
            {pairedBrowserLoading ? <div className="bag-browser-message">正在读取成对数据目录…</div> : !pairedBrowser?.entries.length ? <div className="bag-browser-message">这个目录中没有包含 autocalib.json 的数据集</div> : pairedBrowser.entries.map((entry) => {
              if (entry.kind === "directory") return <button className="bag-directory-row" disabled={bagPreparation.active} key={entry.path} onClick={() => void browsePairedDirectory(entry.path)}>
                <span className="bag-icon">▸</span><strong>{entry.name}</strong><small>文件夹</small>
              </button>;
              const compatible = calibrationWorkspaceMode === "lidar-lidar"
                ? (entry.lidarCount ?? 0) >= 2
                : (entry.lidarCount ?? 0) >= 1 && (entry.cameraCount ?? 0) >= 1;
              const selectable = Boolean(entry.valid && compatible);
              return <label className={`bag-file-row ${selectable ? "prepared" : "unprepared"}`} key={entry.path} title={entry.error ?? undefined}>
                <input type="checkbox" checked={Boolean(selectedPairedDatasets[entry.path])} disabled={bagPreparation.active || !selectable} onChange={() => toggleSelectedPairedDataset(entry)} />
                <span className="bag-icon">⇄</span>
                <strong>{entry.name}</strong>
                <small>{entry.error ?? (!compatible ? "不包含当前标定模式所需的传感器" : `${entry.frameCount ?? 0} 对 · ${entry.lidarCount ?? 0} 雷达 · ${entry.cameraCount ?? 0} 相机 · rig ${entry.rigId ?? "未声明"}`)}</small>
              </label>;
            })}
          </div>
        </> : <section className="local-bag-upload" aria-label="从本机上传 rosbag">
          <input ref={localBagFileInputRef} className="visually-hidden-input" type="file" multiple accept=".db3,.mcap,.yaml,.yml,.json" onChange={(event) => selectLocalBagFiles(event.target.files)} />
          <input ref={localBagFolderInputRef} className="visually-hidden-input" type="file" multiple {...{ webkitdirectory: "" }} onChange={(event) => selectLocalBagFiles(event.target.files)} />
          {!localBagFiles.length ? <div className="local-upload-empty">
            <span className="local-upload-icon">⇧</span>
            <h3>从 Windows 电脑选择 ROS bag</h3>
            <p>推荐直接选择包含 <code>metadata.yaml</code> 和 <code>.db3</code> 的整个文件夹。大文件会自动分片，中断后重新选择同一个包即可续传。</p>
            <div><button className="primary" onClick={() => localBagFolderInputRef.current?.click()}>选择 rosbag 文件夹</button><button onClick={() => localBagFileInputRef.current?.click()}>选择单个文件</button></div>
          </div> : <div className="local-upload-selected">
            <div className="local-upload-summary"><span>ROS</span><div><strong>{localBagPackageName}</strong><small>{localBagFiles.length} 个文件 · {formatFileSize(localBagUpload.totalBytes)}</small></div><button disabled={localBagUpload.active} onClick={() => localBagFolderInputRef.current?.click()}>重新选择</button></div>
            <div className="local-upload-file-list">{localBagFiles.map((item) => <div key={item.relativePath}><span>{item.relativePath}</span><small>{formatFileSize(item.file.size)}</small></div>)}</div>
            <div className="local-upload-help"><b>可断点续传</b><span>上传期间可以暂停；网络断开后点击继续，不会从头开始。</span></div>
          </div>}
          {localBagUploadError && <div className="local-upload-error" role="alert">{localBagUploadError}</div>}
        </section>)}
        {(manualTimelineLoading || manualTimeline) && <section className="manual-frame-picker" aria-label="手动选择数据帧">
          {manualTimelineLoading ? <div className="manual-frame-loading">正在读取 1 秒帧列表…</div> : manualTimeline && currentManualFrame && <>
            <nav className="manual-camera-tabs" aria-label={calibrationWorkspaceMode === "lidar-lidar" ? "选择雷达预览" : "选择相机视角"}>
              {(calibrationWorkspaceMode === "lidar-lidar" ? manualTimeline.lidars : manualTimeline.cameras).map((view) => {
                const count = selectedManualFrames[view.topic]?.length ?? 0;
                return <button key={view.topic} className={activeManualTopic === view.topic ? "active" : ""} onClick={() => calibrationWorkspaceMode === "lidar-lidar" ? setManualLidarTopic(view.topic) : setManualCameraTopic(view.topic)}>
                  <span>{calibrationWorkspaceMode === "lidar-lidar" ? lidarDisplayLabel(view.topic, view.name, lidarTopicRemarks) : view.name}</span><small>{count ? `已选 ${count} 帧` : "未选择"}</small>
                </button>;
              })}
            </nav>
            <div className="manual-frame-preview">
              {calibrationWorkspaceMode === "lidar-lidar" ? <ManualLidarCloudPreview
                key={activeManualTopic}
                url={`/api/calibration/bags/manual-lidar-cloud?path=${encodeURIComponent(manualTimeline.path)}&timestamp_ns=${currentManualFrame.timestampNs}&lidar_topic=${encodeURIComponent(activeManualTopic)}`}
                displayTransform={manualLidarDisplayTransform}
                onLoadingChange={setManualPreviewLoading}
                onError={setAnnotationFeedback}
              /> : <img
                key={`${activeManualTopic}:${currentManualFrame.timestampNs}`}
                src={`/api/calibration/bags/manual-preview?path=${encodeURIComponent(manualTimeline.path)}&timestamp_ns=${currentManualFrame.timestampNs}&camera_topic=${encodeURIComponent(activeManualTopic)}`}
                alt={`${manualTimeline.name} 第 ${currentManualFrame.offsetSeconds} 秒预览`}
                onLoad={() => setManualImagePreviewLoadedKey(manualImagePreviewKey)}
                onError={() => { setManualImagePreviewLoadedKey(manualImagePreviewKey); setAnnotationFeedback("当前预览解码失败，请切换一帧后重试"); }}
              />}
              {visibleManualPreviewLoading && <div className="manual-preview-loading"><i /><b>正在解码预览…</b><small>首次打开该视角会建立关键帧索引</small></div>}
              <span>{calibrationWorkspaceMode === "lidar-lidar"
                ? lidarDisplayLabel(activeManualTopic, manualTimeline.lidars.find((view) => view.topic === activeManualTopic)?.name, lidarTopicRemarks)
                : manualTimeline.cameras.find((view) => view.topic === activeManualTopic)?.name} · +{currentManualFrame.offsetSeconds}s · {manualFrameIndex + 1}/{manualTimeline.candidates.length}</span>
            </div>
            <div className="manual-frame-controls">
              <div className="manual-frame-toolbar">
                <button disabled={manualFrameIndex === 0} onClick={() => setManualFrameIndex((value) => Math.max(0, value - 1))}>← 前 1 秒</button>
                <strong>按 1 秒浏览，手动选择 <span className={`manual-sync-status ${manualSelectionSync}`}>{manualSelectionDirty ? "有未保存修改" : manualSelectionSync === "synced" ? "已载入" : manualSelectionSync === "error" ? "载入失败" : "正在载入"}</span></strong>
                <button disabled={manualFrameIndex >= manualTimeline.candidates.length - 1} onClick={() => setManualFrameIndex((value) => Math.min(manualTimeline.candidates.length - 1, value + 1))}>后 1 秒 →</button>
              </div>
              <input
                className="manual-frame-slider"
                aria-label="按秒浏览数据包"
                type="range"
                min={0}
                max={Math.max(0, manualTimeline.candidates.length - 1)}
                step={1}
                value={manualFrameIndex}
                onChange={(event) => setManualFrameIndex(Number(event.target.value))}
              />
              <div className="manual-frame-selection">
                <button className={currentManualSelection.includes(currentManualFrame.timestampNs) ? "remove" : "add"} disabled={currentManualSelection.includes(currentManualFrame.timestampNs) && currentManualFrameProtected} title={currentManualFrameProtected ? "该帧已有标注，需先删除对应标注" : undefined} onClick={() => void toggleCurrentManualFrame()}>
                  {currentManualSelection.includes(currentManualFrame.timestampNs) && currentManualFrameProtected ? "已有标注，不能取消" : currentManualSelection.includes(currentManualFrame.timestampNs) ? "取消当前帧" : "选择当前帧"}
                </button>
                <div className="manual-frame-chips">
                  {currentManualSelection.length ? currentManualSelection.map((timestamp) => {
                    const index = manualTimeline.candidates.findIndex((item) => item.timestampNs === timestamp);
                    return <button key={timestamp} onClick={() => setManualFrameIndex(Math.max(0, index))}>+{manualTimeline.candidates[index]?.offsetSeconds ?? index}s</button>;
                  }) : <span>{calibrationWorkspaceMode === "lidar-lidar" ? "当前雷达尚未选帧；只会提取各雷达自己选中的帧" : "这个视角尚未选择；不选择就不会导入"}</span>}
                </div>
                <b>{currentManualSelection.length} 帧</b>
              </div>
            </div>
          </>}
        </section>}
        {bagImportSource === "local" && localBagFiles.length > 0 ? <div className="bag-preparation-progress local-progress" aria-live="polite">
          <div><span>{localBagUpload.paused ? "已暂停" : localBagUpload.active ? "上传" : "待上传"}</span><b>{localBagUpload.message}</b><em>{localBagUpload.percent}%</em></div>
          <progress max={100} value={localBagUpload.percent} />
          <small>{formatFileSize(localBagUpload.uploadedBytes)} / {formatFileSize(localBagUpload.totalBytes)}</small>
        </div> : bagPreparation.active && <div className="bag-preparation-progress" aria-live="polite">
          <div><span>{bagPreparation.bagCount ? `${bagPreparation.bagIndex ?? 1}/${bagPreparation.bagCount}` : "准备"}</span><b>{bagPreparation.message}</b><em>{bagPreparation.percent}%</em></div>
          <progress max={100} value={bagPreparation.percent} />
        </div>}
        <footer>
          <span>{bagImportSource === "local"
            ? "上传完成后会自动进入选帧步骤"
            : bagImportSource === "paired"
              ? <>已选择 <b>{Object.keys(selectedPairedDatasets).length}</b> 个成对数据集；将导入全部显式配对帧</>
              : manualTimeline
                ? calibrationWorkspaceMode === "lidar-lidar"
                  ? <>已从 <b>{selectedManualCameraCount}</b> 个雷达预览选择，共 <b>{selectedManualFrameCount}</b> 个记录</>
                  : <>已选择 <b>{selectedManualCameraCount}</b> 个视角，共 <b>{selectedManualFrameCount}</b> 帧</>
                : "一次选择一个数据包"}</span>
          <div>
            <button disabled={bagPreparation.active || localBagUpload.active} onClick={() => setBagBrowserOpen(false)}>取消</button>
            {bagImportSource === "local"
              ? localBagUpload.active
                ? <button className="pause" onClick={pauseLocalBagUpload}>暂停上传</button>
                : <button className="primary" disabled={!localBagFiles.length} onClick={() => void uploadLocalBag()}>{localBagUpload.paused ? "继续上传" : "开始上传"}</button>
              : bagImportSource === "paired"
                ? <button className="primary" disabled={!Object.keys(selectedPairedDatasets).length || bagPreparation.active || pairedBrowserLoading} onClick={importPairedDatasetsIntoGroup}>{bagPreparation.active ? "处理中…" : "导入全部配对帧"}</button>
                : <button className="primary" disabled={!Object.keys(selectedBags).length || !(editingDatasetId ? allManualFrameCount : selectedManualFrameCount) || bagPreparation.active || manualTimelineLoading} onClick={importDatasetIntoGroup}>{bagPreparation.active ? "处理中…" : editingDatasetId ? "保存选帧" : calibrationWorkspaceMode === "lidar-lidar" ? "处理所选场景并导入" : "处理所选视角并导入"}</button>}
          </div>
        </footer>
      </section>
    </div>}
    {intrinsicsBrowserOpen && <div className="bag-browser-backdrop" role="presentation" onMouseDown={() => { if (!intrinsicsImporting) setIntrinsicsBrowserOpen(false); }}>
      <section className="bag-browser-dialog intrinsics-browser-dialog" role="dialog" aria-modal="true" aria-label="选择内参文件" onMouseDown={(event) => event.stopPropagation()}>
        <header>
          <div><span>Intrinsics Directory</span><h2>选择服务器内参文件</h2></div>
          <button aria-label="关闭" disabled={intrinsicsImporting} onClick={() => setIntrinsicsBrowserOpen(false)}>×</button>
        </header>
        <div className="bag-browser-location">
          <button disabled={intrinsicsBrowser?.parent === null || intrinsicsBrowserLoading || intrinsicsImporting} onClick={() => void browseIntrinsicsDirectory(intrinsicsBrowser?.parent ?? "")}>← 上一级</button>
          <code>{intrinsicsBrowser?.root ?? "/calibration-groups/intrinsics"}{intrinsicsBrowser?.path ? `/${intrinsicsBrowser.path}` : ""}</code>
        </div>
        <div className="bag-browser-list">
          {intrinsicsBrowserLoading ? <div className="bag-browser-message">正在读取目录…</div> : !intrinsicsBrowser?.entries.length ? <div className="bag-browser-message">这个目录中没有 YAML 内参文件</div> : intrinsicsBrowser.entries.map((entry) => entry.kind === "directory" ?
            <button className="bag-directory-row" disabled={intrinsicsImporting} key={entry.path} onClick={() => void browseIntrinsicsDirectory(entry.path)}>
              <span className="bag-icon">▸</span><strong>{entry.name}</strong><small>文件夹</small>
            </button> :
            <label className="bag-file-row prepared" key={entry.path}>
              <input type="radio" name="intrinsics-file" checked={selectedIntrinsicsPath === entry.path} disabled={intrinsicsImporting} onChange={() => { setSelectedIntrinsicsPath(entry.path); setIntrinsicsImportError(""); }} />
              <span className="bag-icon">Y</span>
              <strong>{entry.name}</strong>
              <small>{(entry.sizeBytes / 1024).toFixed(1)} KB</small>
            </label>
          )}
        </div>
        <div className="intrinsics-import-error" role="alert">{intrinsicsImportError}</div>
        <footer>
          <span>{selectedIntrinsicsPath ? <>已选择 <b>{selectedIntrinsicsPath.split("/").at(-1)}</b></> : "请选择一个 YAML 文件"}</span>
          <div><button disabled={intrinsicsImporting} onClick={() => setIntrinsicsBrowserOpen(false)}>取消</button><button className="primary" disabled={!selectedIntrinsicsPath || intrinsicsImporting} onClick={importCameraIntrinsics}>{intrinsicsImporting ? "正在载入原始投影参数…" : "导入当前组"}</button></div>
        </footer>
      </section>
    </div>}
    </>
  );
}
