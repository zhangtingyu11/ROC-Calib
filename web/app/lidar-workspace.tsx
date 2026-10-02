"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { clusterPointSelection, combinePointIndices } from "./segmentation";
import { type Manifest, type DatasetInfo, type LidarPairAnnotation, type LidarPairCalibration, type LidarFramePairSet, type LidarExtrinsicGraph, type FrameLabel, type CloudData, type PointInteractionMode, type ViewRequest, type LidarFramePairDiagnostic, lidarDisplayLabel, type LidarInfo, lidarPairKey, type LidarExtrinsicGraphVersion, multiply4, invertRigid4, LIDAR_CALIBRATION_ALGORITHM_VERSION, LIDAR_CALIBRATION_COMPATIBLE_VERSIONS, versionDisplayLabel, transformCloudData, timestampDeltaMs, subsetCloud, mergeAlignedClouds, correspondingRegionIndices, transformedSubsetCloud, LIDAR_SAVED_PAIR_PAGE_SIZE, readCloud, isGroundPoint, parseRigidMatrix, formatTimestamp, formatPoints, LIDAR_SELECTION_COLOR, OrthographicCloudViews } from './workspace-core';
import { PointCloudViewport } from './point-cloud-view';

export function LidarLidarWorkspace({
  manifest,
  dataset,
  datasetIds,
  sourcePaths,
  taskGroupId,
  annotations,
  calibrations,
  framePairSets,
  noAnnotationPairKeys,
  topicRemarks,
  graph,
  onDatasetChange,
  onEditDatasetFrames,
  onEditSourceFrames,
  onAnnotationsChange,
  onCalibrationsChange,
  onFramePairSetsChange,
  onNoAnnotationPairKeysChange,
  onTopicRemarkChange,
  onGraphChange,
}: {
  manifest: Manifest;
  dataset: DatasetInfo;
  datasetIds: string[];
  sourcePaths: string[];
  taskGroupId: string;
  annotations: LidarPairAnnotation[];
  calibrations: LidarPairCalibration[];
  framePairSets: LidarFramePairSet[];
  noAnnotationPairKeys: string[];
  topicRemarks: Record<string, string>;
  graph: LidarExtrinsicGraph;
  onDatasetChange: (datasetId: string) => void;
  onEditDatasetFrames: (dataset: DatasetInfo) => void;
  onEditSourceFrames: (sourcePath: string) => void;
  onAnnotationsChange: (value: LidarPairAnnotation[]) => void;
  onCalibrationsChange: (value: LidarPairCalibration[]) => void;
  onFramePairSetsChange: (value: LidarFramePairSet[]) => void;
  onNoAnnotationPairKeysChange: (value: string[]) => void;
  onTopicRemarkChange: (topic: string, remark: string) => void;
  onGraphChange: (value: LidarExtrinsicGraph) => void;
}) {
  const firstLidar = dataset.lidars[0];
  const secondLidar = dataset.lidars[1] ?? dataset.lidars[0];
  const [sourceLidarId, setSourceLidarId] = useState(firstLidar?.id ?? "");
  const [targetLidarId, setTargetLidarId] = useState(secondLidar?.id ?? "");
  const [sourceFrame, setSourceFrame] = useState<FrameLabel>(firstLidar?.frames[0]?.label ?? "1");
  const [targetFrame, setTargetFrame] = useState<FrameLabel>(secondLidar?.frames[0]?.label ?? "1");
  const [sourceCloud, setSourceCloud] = useState<CloudData | null>(null);
  const [targetCloud, setTargetCloud] = useState<CloudData | null>(null);
  const [sourceSelection, setSourceSelection] = useState<Uint32Array>(() => new Uint32Array());
  const [targetSelection, setTargetSelection] = useState<Uint32Array>(() => new Uint32Array());
  const [sourceMode, setSourceMode] = useState<PointInteractionMode>("navigate");
  const [targetMode, setTargetMode] = useState<PointInteractionMode>("navigate");
  const [stableOnly, setStableOnly] = useState(false);
  const [hideGround, setHideGround] = useState(false);
  const [pointSize, setPointSize] = useState(.05);
  const [brushSize, setBrushSize] = useState(28);
  const foregroundDepthTolerance = .3;
  const correspondenceRegionPadding = .05;
  const [clearToken, setClearToken] = useState(0);
  const [sourceView, setSourceView] = useState<ViewRequest>({ mode: "near", token: 0 });
  const [targetView, setTargetView] = useState<ViewRequest>({ mode: "near", token: 0 });
  const [alignmentView, setAlignmentView] = useState<ViewRequest>({ mode: "near", token: 0 });
  const [alignmentInspectView, setAlignmentInspectView] = useState<ViewRequest>({ mode: "near", token: 0 });
  const [alignmentInspectLidar, setAlignmentInspectLidar] = useState<"source" | "target">("source");
  const [alignmentInspectMode, setAlignmentInspectMode] = useState<PointInteractionMode>("navigate");
  const [alignmentPreview, setAlignmentPreview] = useState(false);
  const [alignmentPointSize, setAlignmentPointSize] = useState(.05);
  const [alignmentHighlightOpacity, setAlignmentHighlightOpacity] = useState(.45);
  const [alignmentDensity, setAlignmentDensity] = useState(100);
  const [alignmentLayer, setAlignmentLayer] = useState<"both" | "source" | "target">("both");
  const [alignmentBlink, setAlignmentBlink] = useState(false);
  const [alignmentBlinkLayer, setAlignmentBlinkLayer] = useState<"source" | "target">("source");
  const [alignmentSelectionOnly, setAlignmentSelectionOnly] = useState(false);
  const [alignmentDisplayMode, setAlignmentDisplayMode] = useState<"full" | "three">("full");
  const [selectedCalibrationVersionId, setSelectedCalibrationVersionId] = useState("");
  const [activeAlignmentAnnotationId, setActiveAlignmentAnnotationId] = useState<string | null>(null);
  const [savedPairPage, setSavedPairPage] = useState(0);
  const [synchronizedPairIndex, setSynchronizedPairIndex] = useState(0);
  const [synchronizedFramePairs, setSynchronizedFramePairs] = useState<LidarFramePairDiagnostic[]>([]);
  const [pairingLoading, setPairingLoading] = useState(false);
  const [sourceLoading, setSourceLoading] = useState(false);
  const [targetLoading, setTargetLoading] = useState(false);
  const [optimizing, setOptimizing] = useState(false);
  const [graphSolving, setGraphSolving] = useState(false);
  const [progress, setProgress] = useState(0);
  const [feedback, setFeedback] = useState("选择两部雷达后先生成最近帧配对，再由 GICP 使用全部配对求初值");
  const [solveFailureByPair, setSolveFailureByPair] = useState<Record<string, { title: string; message: string }>>({});
  const pendingPairPreviewRef = useRef<LidarPairAnnotation | null>(null);
  const alignmentPreviewRef = useRef<HTMLDivElement | null>(null);
  const lidarExtrinsicImportRef = useRef<HTMLInputElement | null>(null);

  const sourceLidar = dataset.lidars.find((item) => item.id === sourceLidarId) ?? firstLidar;
  const targetLidar = dataset.lidars.find((item) => item.id === targetLidarId) ?? secondLidar;
  const sourceLidarLabel = lidarDisplayLabel(sourceLidar?.topic, sourceLidar?.name ?? "源雷达", topicRemarks);
  const targetLidarLabel = lidarDisplayLabel(targetLidar?.topic, targetLidar?.name ?? "目标雷达", topicRemarks);
  const editTopicRemark = (lidar: LidarInfo | undefined) => {
    if (!lidar?.topic) return;
    const value = window.prompt(`设置 ${lidar.topic} 的备注（留空则删除）`, topicRemarks[lidar.topic] ?? "");
    if (value === null) return;
    onTopicRemarkChange(lidar.topic, value.trim());
  };
  const lidarPairOptions = useMemo(() => dataset.lidars.flatMap((source, sourceIndex) =>
    dataset.lidars.slice(sourceIndex + 1).map((target) => ({
      key: lidarPairKey(source.id, target.id),
      source,
      target,
    })),
  ), [dataset.lidars]);
  const annotatedPairKeys = useMemo(() => new Set(annotations
    .filter((item) => item.taskGroupId === taskGroupId)
    .map((item) => lidarPairKey(item.sourceLidarId, item.targetLidarId))), [annotations, taskGroupId]);
  const completedPairKeys = useMemo(() => new Set([...annotatedPairKeys, ...noAnnotationPairKeys]), [annotatedPairKeys, noAnnotationPairKeys]);
  const selectedLidarPairKey = sourceLidar && targetLidar ? lidarPairKey(sourceLidar.id, targetLidar.id) : "";
  const currentSolveFailure = solveFailureByPair[selectedLidarPairKey] ?? "";
  const completedLidarPairCount = lidarPairOptions.filter((item) => completedPairKeys.has(item.key)).length;
  const selectLidarPair = (key: string) => {
    const option = lidarPairOptions.find((item) => item.key === key);
    if (!option) return;
    const existingDirection = annotations.find((item) => item.taskGroupId === taskGroupId &&
      lidarPairKey(item.sourceLidarId, item.targetLidarId) === key) ?? calibrations.find((item) =>
      lidarPairKey(item.sourceLidarId, item.targetLidarId) === key);
    setSourceLidarId(existingDirection?.sourceLidarId ?? option.source.id);
    setTargetLidarId(existingDirection?.targetLidarId ?? option.target.id);
  };
  const sourceFrameInfo = sourceLidar?.frames.find((item) => item.label === sourceFrame) ?? sourceLidar?.frames[0];
  const targetFrameInfo = targetLidar?.frames.find((item) => item.label === targetFrame) ?? targetLidar?.frames[0];
  const sourceSelectedTimestampKey = sourceLidar?.frames.map((frame) => frame.timestampNs).join(",") ?? "";
  const targetSelectedTimestampKey = targetLidar?.frames.map((frame) => frame.timestampNs).join(",") ?? "";
  const rigId = dataset.rigId ?? "unknown-rig";
  const groupKey = `lidar:${taskGroupId}:${rigId}:${sourceLidar?.id ?? ""}:${targetLidar?.id ?? ""}`;
  const framePairSetKey = `${taskGroupId}:${dataset.id}:${sourceLidar?.id ?? ""}:${targetLidar?.id ?? ""}:${sourceSelectedTimestampKey}:${targetSelectedTimestampKey}`;
  const graphVersions: LidarExtrinsicGraphVersion[] = graph.versions?.length ? graph.versions : graph.anchorLidarId && graph.poses ? [{
    versionId: graph.versionId ?? "legacy-global-extrinsic",
    versionLabel: graph.versionLabel ?? "历史全局外参",
    generatedAt: graph.generatedAt ?? "",
    anchorLidarId: graph.anchorLidarId,
    poses: graph.poses,
    edgeDiagnostics: graph.edgeDiagnostics,
    converged: graph.converged,
    algorithmVersion: graph.algorithmVersion,
    finalPairVersionIds: graph.finalPairVersionIds,
  }] : [];
  const graphPairCalibrationVersions: LidarPairCalibration[] = sourceLidar && targetLidar ? graphVersions.flatMap((version) => {
    const sourcePose = version.poses[sourceLidar.id];
    const targetPose = version.poses[targetLidar.id];
    if (!sourcePose || !targetPose) return [];
    const suspected = version.edgeDiagnostics?.some((item) => item.suspected) ?? false;
    return [{
      versionId: `${version.versionId}:${sourceLidar.id}:${targetLidar.id}`,
      graphVersionId: version.versionId,
      versionLabel: version.versionLabel || "全局优化",
      source: "global" as const,
      groupKey,
      sourceLidarId: sourceLidar.id,
      targetLidarId: targetLidar.id,
      transform: "target_from_source" as const,
      matrix: multiply4(invertRigid4(targetPose), sourcePose),
      delta: [0, 0, 0, 0, 0, 0],
      pairCount: 0,
      initial_rmse: 0,
      final_rmse: 0,
      correspondence_count: 0,
      overlap_ratio: 0,
      pair_consistency_ratio: 1,
      converged: version.converged ?? true,
      iteration_count: 0,
      algorithmVersion: LIDAR_CALIBRATION_ALGORITHM_VERSION,
      qualityStatus: suspected ? "warning" as const : "good" as const,
      generatedAt: version.generatedAt,
      datasetId: dataset.id,
    }];
  }) : [];
  const storedPairCalibrationVersions = calibrations.filter((item) =>
    item.groupKey === groupKey && LIDAR_CALIBRATION_COMPATIBLE_VERSIONS.has(item.algorithmVersion) &&
    (!item.datasetId || item.datasetId === dataset.id) && (!item.framePairSetKey || item.framePairSetKey === framePairSetKey),
  );
  const visibleGraphPairVersions = graphPairCalibrationVersions.filter((graphVersion) =>
    !storedPairCalibrationVersions.some((storedVersion) => storedVersion.versionId === graphVersion.versionId),
  );
  const storedGicpVersion = storedPairCalibrationVersions.find((item) =>
    item.source === "gicp" || item.versionLabel?.startsWith("GICP 初值") || (item.pairCount === 0 && item.source !== "imported" && item.source !== "global"));
  const retainedGicpVersion = !storedGicpVersion
    ? [...storedPairCalibrationVersions].reverse().find((item) => item.gicpBaseMatrix)
    : null;
  const pairCalibrationVersions = retainedGicpVersion?.gicpBaseMatrix ? [{
    ...retainedGicpVersion,
    versionId: `${retainedGicpVersion.versionId ?? retainedGicpVersion.generatedAt ?? "legacy"}:gicp-base`,
    versionLabel: "GICP 初值",
    source: "gicp" as const,
    matrix: retainedGicpVersion.gicpBaseMatrix,
    pairCount: 0,
  }, ...storedPairCalibrationVersions, ...visibleGraphPairVersions] : [...storedPairCalibrationVersions, ...visibleGraphPairVersions];
  const calibrationVersionId = (item: LidarPairCalibration, index = pairCalibrationVersions.indexOf(item)) =>
    item.versionId ?? `${item.algorithmVersion}:${item.generatedAt ?? index}`;
  const calibrationVersionName = (item: LidarPairCalibration) => item.versionLabel ?? (
    item.source === "imported" ? "导入外参"
      : item.source === "global" ? "全局优化"
      : item.pairCount > 0 ? item.gicpBaseMatrix ? "基于 GICP 的人工精调" : "纯人工无初值"
        : "GICP 初值"
  );
  const calibrationVersionLabel = (item: LidarPairCalibration) => versionDisplayLabel(calibrationVersionName(item), item.generatedAt);
  const currentCalibration = pairCalibrationVersions.find((item, index) =>
    calibrationVersionId(item, index) === selectedCalibrationVersionId) ?? pairCalibrationVersions.at(-1) ?? null;
  const finalCalibration = pairCalibrationVersions.find((item) => item.isFinal) ?? null;
  const selectedGraphVersion = graphVersions.find((item) => item.versionId === graph.selectedVersionId) ?? graphVersions.at(-1) ?? null;
  const graphVersionLabel = (item: LidarExtrinsicGraphVersion) => versionDisplayLabel(item.versionLabel, item.generatedAt);
  // Keep the clouds used by calibration in their native sensor frames, but
  // render both annotation panes in the main-LiDAR frame. Point order is
  // unchanged by transformCloudData, so every picked display index still maps
  // directly to the original cloud sent to GICP and manual refinement.
  const annotationDisplayTransforms = useMemo(() => {
    if (!sourceLidar || !targetLidar) return null;
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    const carFromMain = manifest.calibration.lidars.main?.carFromLidar;
    const mainFromCar = carFromMain ? invertRigid4(carFromMain) : null;
    const mainFromLidar = (lidarId: string) => {
      if (lidarId === "main") return identity;
      const carFromLidar = manifest.calibration.lidars[lidarId]?.carFromLidar;
      return mainFromCar && carFromLidar ? multiply4(mainFromCar, carFromLidar) : null;
    };
    const mainFromSource = mainFromLidar(sourceLidar.id);
    const mainFromTarget = mainFromLidar(targetLidar.id);
    return mainFromSource && mainFromTarget ? { mainFromSource, mainFromTarget } : null;
  }, [manifest.calibration.lidars, sourceLidar, targetLidar]);
  const sourceDisplayCloud = useMemo(() => sourceCloud && annotationDisplayTransforms
    ? transformCloudData(sourceCloud, annotationDisplayTransforms.mainFromSource)
    : sourceCloud, [annotationDisplayTransforms, sourceCloud]);
  const targetDisplayCloud = useMemo(() => targetCloud && annotationDisplayTransforms
    ? transformCloudData(targetCloud, annotationDisplayTransforms.mainFromTarget)
    : targetCloud, [annotationDisplayTransforms, targetCloud]);
  const sourceCloudList = useMemo(() => [sourceDisplayCloud], [sourceDisplayCloud]);
  const targetCloudList = useMemo(() => [targetDisplayCloud], [targetDisplayCloud]);
  const alignmentPairMatrix = currentCalibration?.matrix ?? null;
  const synchronizedFramePair = synchronizedFramePairs[synchronizedPairIndex] ?? synchronizedFramePairs[0] ?? null;
  const effectiveSourceCloudUrl = synchronizedFramePair?.sourceCloudUrl ?? sourceFrameInfo?.url;
  const effectiveTargetCloudUrl = synchronizedFramePair?.targetCloudUrl ?? targetFrameInfo?.url;
  const selectedFrameDeltaMs = synchronizedFramePair?.deltaMs ?? (sourceFrameInfo && targetFrameInfo
    ? timestampDeltaMs(sourceFrameInfo.timestampNs, targetFrameInfo.timestampNs)
    : Number.POSITIVE_INFINITY);
  const mainReferenceTransforms = useMemo(() => {
    if (!sourceLidar || !targetLidar || !alignmentPairMatrix) return null;
    const identity = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];
    if (sourceLidar.id === "main") return {
      mainFromSource: identity,
      mainFromTarget: invertRigid4(alignmentPairMatrix),
    };
    if (targetLidar.id === "main") return {
      mainFromSource: alignmentPairMatrix,
      mainFromTarget: identity,
    };
    const graphTarget = graph.anchorLidarId === "main" ? graph.poses?.[targetLidar.id] : null;
    const carFromMain = manifest.calibration.lidars.main?.carFromLidar;
    const carFromTarget = manifest.calibration.lidars[targetLidar.id]?.carFromLidar;
    const mainFromTarget = graphTarget ?? (carFromMain && carFromTarget
      ? multiply4(invertRigid4(carFromMain), carFromTarget)
      : identity);
    return {
      mainFromSource: multiply4(mainFromTarget, alignmentPairMatrix),
      mainFromTarget,
    };
  }, [alignmentPairMatrix, graph.anchorLidarId, graph.poses, manifest.calibration.lidars, sourceLidar, targetLidar]);
  const activeAlignmentAnnotation = annotations.find((item) =>
    item.id === activeAlignmentAnnotationId &&
    item.sourceCloudUrl === effectiveSourceCloudUrl && item.targetCloudUrl === effectiveTargetCloudUrl,
  ) ?? null;
  const canFocusAlignmentSelection = Boolean(
    activeAlignmentAnnotation && sourceCloud && targetCloud &&
    activeAlignmentAnnotation.sourcePointIndices.length >= 6 && activeAlignmentAnnotation.targetPointIndices.length >= 6,
  );
  // Focus from the immutable saved annotation, not the editable on-screen
  // selection. Otherwise a filter, a partial reload, or a later brush edit can
  // silently remove points from “只看当前参照物”.
  const alignmentSourceCloud = useMemo(() => sourceCloud && alignmentSelectionOnly && activeAlignmentAnnotation
    ? subsetCloud(sourceCloud, Uint32Array.from(activeAlignmentAnnotation.sourcePointIndices))
    : sourceCloud, [activeAlignmentAnnotation, alignmentSelectionOnly, sourceCloud]);
  const alignmentTargetCloud = useMemo(() => targetCloud && alignmentSelectionOnly && activeAlignmentAnnotation
    ? subsetCloud(targetCloud, Uint32Array.from(activeAlignmentAnnotation.targetPointIndices))
    : targetCloud, [activeAlignmentAnnotation, alignmentSelectionOnly, targetCloud]);
  const effectiveAlignmentDensity = alignmentSelectionOnly ? 100 : alignmentDensity;
  const alignmentCloud = useMemo(() => alignmentSourceCloud && alignmentTargetCloud && mainReferenceTransforms
    ? mergeAlignedClouds(
        alignmentSourceCloud,
        alignmentTargetCloud,
        mainReferenceTransforms.mainFromSource,
        mainReferenceTransforms.mainFromTarget,
        effectiveAlignmentDensity,
        "both",
      )
    : null, [alignmentSourceCloud, alignmentTargetCloud, effectiveAlignmentDensity, mainReferenceTransforms]);
  const effectiveAlignmentLayer = alignmentBlink ? alignmentBlinkLayer : alignmentLayer;
  const alignmentCloudList = useMemo(() => [alignmentCloud], [alignmentCloud]);
  const alignmentInspectRawCloud = alignmentInspectLidar === "source" ? sourceCloud : targetCloud;
  const alignmentInspectMatrix = alignmentInspectLidar === "source"
    ? mainReferenceTransforms?.mainFromSource
    : mainReferenceTransforms?.mainFromTarget;
  const alignmentInspectCloud = useMemo(() => alignmentInspectRawCloud && alignmentInspectMatrix
    ? transformCloudData(alignmentInspectRawCloud, alignmentInspectMatrix)
    : null, [alignmentInspectMatrix, alignmentInspectRawCloud]);
  const alignmentInspectCloudList = useMemo(() => [alignmentInspectCloud], [alignmentInspectCloud]);
  const alignmentInspectSelection = alignmentInspectLidar === "source" ? sourceSelection : targetSelection;
  const alignmentCounterpartIndices = useMemo(() => alignmentInspectLidar === "source"
    ? correspondingRegionIndices(sourceCloud, sourceSelection, targetCloud, alignmentPairMatrix, correspondenceRegionPadding)
    : correspondingRegionIndices(targetCloud, targetSelection, sourceCloud, alignmentPairMatrix ? invertRigid4(alignmentPairMatrix) : null, correspondenceRegionPadding),
  [alignmentInspectLidar, alignmentPairMatrix, correspondenceRegionPadding, sourceCloud, sourceSelection, targetCloud, targetSelection]);
  const alignmentHighlightClouds = useMemo(() => {
    if (!mainReferenceTransforms || !sourceCloud || !targetCloud || !alignmentInspectSelection.length) return [];
    if (alignmentInspectLidar === "source") return [
      { cloud: transformedSubsetCloud(sourceCloud, sourceSelection, mainReferenceTransforms.mainFromSource), color: 0xff3cac },
      { cloud: transformedSubsetCloud(targetCloud, alignmentCounterpartIndices, mainReferenceTransforms.mainFromTarget), color: 0x00ffff },
    ];
    return [
      { cloud: transformedSubsetCloud(targetCloud, targetSelection, mainReferenceTransforms.mainFromTarget), color: 0xff3cac },
      { cloud: transformedSubsetCloud(sourceCloud, alignmentCounterpartIndices, mainReferenceTransforms.mainFromSource), color: 0x00ffff },
    ];
  }, [alignmentCounterpartIndices, alignmentInspectLidar, alignmentInspectSelection.length, mainReferenceTransforms, sourceCloud, sourceSelection, targetCloud, targetSelection]);
  const alignmentLocalClouds = useMemo(() => {
    if (!mainReferenceTransforms || !sourceCloud || !targetCloud || !alignmentInspectSelection.length) return null;
    const sourceIndices = alignmentInspectLidar === "source" ? sourceSelection : alignmentCounterpartIndices;
    const targetIndices = alignmentInspectLidar === "target" ? targetSelection : alignmentCounterpartIndices;
    if (!sourceIndices.length || !targetIndices.length) return null;
    return {
      source: transformedSubsetCloud(sourceCloud, sourceIndices, mainReferenceTransforms.mainFromSource),
      target: transformedSubsetCloud(targetCloud, targetIndices, mainReferenceTransforms.mainFromTarget),
    };
  }, [alignmentCounterpartIndices, alignmentInspectLidar, alignmentInspectSelection.length, mainReferenceTransforms, sourceCloud, sourceSelection, targetCloud, targetSelection]);
  const emptyAlignmentSelection = useMemo(() => new Uint32Array(), []);
  const visibleAnnotations = annotations.filter((item) => item.taskGroupId === taskGroupId && item.datasetId === dataset.id &&
    item.sourceLidarId === sourceLidar?.id && item.targetLidarId === targetLidar?.id);
  const annotatedFramePairKeys = new Set(visibleAnnotations
    .filter((item) => item.sourceCloudUrl && item.targetCloudUrl)
    .map((item) => `${item.sourceCloudUrl}\n${item.targetCloudUrl}`));
  const savedPairPageCount = Math.max(1, Math.ceil(visibleAnnotations.length / LIDAR_SAVED_PAIR_PAGE_SIZE));
  const effectiveSavedPairPage = Math.min(savedPairPage, savedPairPageCount - 1);
  const pagedVisibleAnnotations = visibleAnnotations.slice(
    effectiveSavedPairPage * LIDAR_SAVED_PAIR_PAGE_SIZE,
    (effectiveSavedPairPage + 1) * LIDAR_SAVED_PAIR_PAGE_SIZE,
  );
  const annotationSyncDeltaMs = (item: LidarPairAnnotation) => item.syncOffsetMs ??
    timestampDeltaMs(item.sourceTimestampNs, item.targetTimestampNs);
  const eligibleAnnotations = annotations.filter((item) => item.taskGroupId === taskGroupId && item.datasetId === dataset.id &&
    item.useForOptimization !== false && item.sourceLidarId === sourceLidar?.id && item.targetLidarId === targetLidar?.id &&
    Boolean(item.sourceCloudUrl && item.targetCloudUrl) && annotationSyncDeltaMs(item) <= 100);
  useEffect(() => {
    if (!alignmentBlink || !alignmentPreview) return;
    setAlignmentBlinkLayer("source");
    const interval = window.setInterval(() => setAlignmentBlinkLayer((layer) => layer === "source" ? "target" : "source"), 550);
    return () => window.clearInterval(interval);
  }, [alignmentBlink, alignmentPreview]);

  useEffect(() => {
    if (!alignmentPreview || !currentCalibration) return;
    const frame = window.requestAnimationFrame(() => {
      alignmentPreviewRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [alignmentPreview, currentCalibration]);

  useEffect(() => {
    if (!canFocusAlignmentSelection && alignmentSelectionOnly) setAlignmentSelectionOnly(false);
  }, [alignmentSelectionOnly, canFocusAlignmentSelection]);

  useEffect(() => {
    const sourceAvailable = dataset.lidars.some((item) => item.id === sourceLidarId);
    const targetAvailable = dataset.lidars.some((item) => item.id === targetLidarId);
    if (!sourceAvailable) setSourceLidarId(firstLidar?.id ?? "");
    if (!targetAvailable || targetLidarId === sourceLidarId) {
      setTargetLidarId(dataset.lidars.find((item) => item.id !== (sourceAvailable ? sourceLidarId : firstLidar?.id))?.id ?? "");
    }
  }, [dataset.id, dataset.lidars, firstLidar?.id, sourceLidarId, targetLidarId]);

  useEffect(() => {
    if (sourceLidar && !sourceLidar.frames.some((item) => item.label === sourceFrame)) setSourceFrame(sourceLidar.frames[0]?.label ?? "1");
  }, [sourceFrame, sourceLidar]);
  useEffect(() => {
    if (targetLidar && !targetLidar.frames.some((item) => item.label === targetFrame)) setTargetFrame(targetLidar.frames[0]?.label ?? "1");
  }, [targetFrame, targetLidar]);

  useEffect(() => {
    if (!sourceLidar || !targetLidar || !sourceSelectedTimestampKey || !targetSelectedTimestampKey) {
      setSynchronizedFramePairs([]);
      return;
    }
    const stored = framePairSets.find((item) => item.key === framePairSetKey);
    if (stored?.pairs.length) {
      setSynchronizedFramePairs(stored.pairs);
      setSynchronizedPairIndex(0);
      setPairingLoading(false);
      setFeedback(`已恢复 ${stored.pairs.length} 个已保存的最近帧配对`);
      return;
    }
    if (currentCalibration?.framePairDiagnostics?.length) {
      const migrated: LidarFramePairSet = {
        key: framePairSetKey,
        datasetId: dataset.id,
        sourceLidarId: sourceLidar.id,
        targetLidarId: targetLidar.id,
        pairs: currentCalibration.framePairDiagnostics,
        generatedAt: currentCalibration.generatedAt ?? new Date().toISOString(),
      };
      setSynchronizedFramePairs(migrated.pairs);
      setSynchronizedPairIndex(0);
      setPairingLoading(false);
      onFramePairSetsChange([...framePairSets, migrated]);
      setFeedback(`已从保存的 GICP 结果恢复 ${migrated.pairs.length} 个最近帧配对`);
      return;
    }
    let active = true;
    setPairingLoading(true);
    setSynchronizedFramePairs([]);
    setSynchronizedPairIndex(0);
    setFeedback("正在从两路完整时间轴生成最近帧配对…");
    fetch("/api/calibration/lidar-pairs/frame-pairs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_bag_path: dataset.sourcePath,
        source_lidar_topic: sourceLidar.topic,
        target_lidar_topic: targetLidar.topic,
        source_selected_timestamps: sourceLidar.frames.map((frame) => frame.timestampNs),
        target_selected_timestamps: targetLidar.frames.map((frame) => frame.timestampNs),
      }),
    }).then(async (response) => {
      const payload = await response.json() as { pairs?: LidarFramePairDiagnostic[]; detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? `帧配对服务返回 ${response.status}`);
      return payload.pairs ?? [];
    }).then((pairs) => {
      if (!active) return;
      setSynchronizedFramePairs(pairs);
      onFramePairSetsChange([
        ...framePairSets.filter((item) => item.key !== framePairSetKey),
        {
          key: framePairSetKey,
          datasetId: dataset.id,
          sourceLidarId: sourceLidar.id,
          targetLidarId: targetLidar.id,
          pairs,
          generatedAt: new Date().toISOString(),
        },
      ]);
      setFeedback(`已生成 ${pairs.length} 个去重最近帧配对；GICP 将严格使用这些配对`);
    }).catch((reason) => {
      if (!active) return;
      setSynchronizedFramePairs([]);
      setFeedback(reason instanceof Error ? reason.message : "最近帧配对生成失败");
    }).finally(() => { if (active) setPairingLoading(false); });
    return () => { active = false; };
  }, [currentCalibration, dataset.id, dataset.sourcePath, framePairSetKey, framePairSets, onFramePairSetsChange, sourceLidarId, sourceSelectedTimestampKey, targetLidarId, targetSelectedTimestampKey]);

  useEffect(() => {
    setSynchronizedPairIndex(0);
    pendingPairPreviewRef.current = null;
    setActiveAlignmentAnnotationId(null);
    setAlignmentSelectionOnly(false);
    setSavedPairPage(0);
  }, [dataset.id, sourceLidarId, targetLidarId]);

  useEffect(() => {
    if (savedPairPage >= savedPairPageCount) setSavedPairPage(savedPairPageCount - 1);
  }, [savedPairPage, savedPairPageCount]);

  useEffect(() => {
    if (synchronizedPairIndex >= synchronizedFramePairs.length) setSynchronizedPairIndex(0);
  }, [synchronizedFramePairs.length, synchronizedPairIndex]);

  useEffect(() => {
    if (!effectiveSourceCloudUrl) return;
    let active = true;
    setSourceLoading(true);
    readCloud(effectiveSourceCloudUrl)
      .then((source) => {
        if (!active) return;
        setSourceCloud(source);
        const pending = pendingPairPreviewRef.current;
        if (pending && (pending.sourceCloudUrl ? pending.sourceCloudUrl === effectiveSourceCloudUrl : pending.sourceFrame === sourceFrameInfo?.label)) {
          setSourceSelection(Uint32Array.from(pending.sourcePointIndices));
        } else setSourceSelection(new Uint32Array());
      })
      .catch((reason) => { if (active) setFeedback(reason instanceof Error ? `左侧：${reason.message}` : "左侧点云载入失败"); })
      .finally(() => { if (active) setSourceLoading(false); });
    return () => { active = false; };
  }, [effectiveSourceCloudUrl]);

  useEffect(() => {
    if (!effectiveTargetCloudUrl) return;
    let active = true;
    setTargetLoading(true);
    readCloud(effectiveTargetCloudUrl)
      .then((target) => {
        if (!active) return;
        setTargetCloud(target);
        const pending = pendingPairPreviewRef.current;
        if (pending && (pending.targetCloudUrl ? pending.targetCloudUrl === effectiveTargetCloudUrl : pending.targetFrame === targetFrameInfo?.label)) {
          setTargetSelection(Uint32Array.from(pending.targetPointIndices));
        } else setTargetSelection(new Uint32Array());
      })
      .catch((reason) => { if (active) setFeedback(reason instanceof Error ? `右侧：${reason.message}` : "右侧点云载入失败"); })
      .finally(() => { if (active) setTargetLoading(false); });
    return () => { active = false; };
  }, [effectiveTargetCloudUrl]);

  const selectRegion = (cloud: CloudData | null, indices: Uint32Array, update: (value: Uint32Array) => void) => {
    if (!cloud) return;
    setAlignmentSelectionOnly(false);
    const usable = Uint32Array.from(indices.filter((index) => !hideGround || !isGroundPoint(cloud, index, stableOnly)));
    update(clusterPointSelection(cloud.positions, usable, .18, false));
  };
  const editRegion = (current: Uint32Array, indices: Uint32Array, mode: "add" | "remove", update: (value: Uint32Array) => void) => {
    setAlignmentSelectionOnly(false);
    update(combinePointIndices(current, indices, mode));
  };
  const clearDraft = () => {
    pendingPairPreviewRef.current = null;
    setActiveAlignmentAnnotationId(null);
    setAlignmentSelectionOnly(false);
    setSourceSelection(new Uint32Array());
    setTargetSelection(new Uint32Array());
    setClearToken((value) => value + 1);
  };
  const savePair = () => {
    if (!sourceLidar || !targetLidar || !sourceFrameInfo || !targetFrameInfo || sourceSelection.length < 6 || targetSelection.length < 6) return;
    if (!synchronizedFramePair) {
      setFeedback("请先计算 GICP；人工标注只能绑定系统从两路完整时间轴找到的最近帧对");
      return;
    }
    if (synchronizedFramePair.deltaMs > 100) {
      setFeedback(`当前帧对相差 ${synchronizedFramePair.deltaMs.toFixed(1)} ms，拒绝保存：请重新计算最近帧对`);
      return;
    }
    const editingAnnotation = annotations.find((item) => item.id === activeAlignmentAnnotationId) ?? null;
    const annotation: LidarPairAnnotation = {
      id: editingAnnotation?.id ?? `lidar-pair-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
      taskGroupId,
      datasetId: dataset.id,
      sourceLidarId: sourceLidar.id,
      targetLidarId: targetLidar.id,
      sourceFrame: sourceFrameInfo.label,
      targetFrame: targetFrameInfo.label,
      sourceTimestampNs: synchronizedFramePair.sourceTimestampNs,
      targetTimestampNs: synchronizedFramePair.targetTimestampNs,
      sourceCloudUrl: synchronizedFramePair.sourceCloudUrl,
      targetCloudUrl: synchronizedFramePair.targetCloudUrl,
      syncOffsetMs: synchronizedFramePair.deltaMs,
      sourcePointIndices: Array.from(sourceSelection),
      targetPointIndices: Array.from(targetSelection),
      useForOptimization: editingAnnotation?.useForOptimization !== false,
      savedAt: new Date().toISOString(),
    };
    onAnnotationsChange(editingAnnotation
      ? annotations.map((item) => item.id === editingAnnotation.id ? annotation : item)
      : [...annotations, annotation]);
    if (editingAnnotation) {
      onCalibrationsChange(calibrations.filter((item) =>
        item.groupKey !== groupKey || item.source === "imported" || item.pairCount === 0,
      ));
      onGraphChange({});
    }
    onNoAnnotationPairKeysChange(noAnnotationPairKeys.filter((key) => key !== lidarPairKey(sourceLidar.id, targetLidar.id)));
    clearDraft();
    setFeedback(`${editingAnnotation ? "已更新" : "已保存"}最近帧配对（Δ ${synchronizedFramePair.deltaMs.toFixed(3)} ms）：${annotation.sourcePointIndices.length} × ${annotation.targetPointIndices.length} 点${editingAnnotation ? "；相关人工精调结果已作废" : ""}`);
  };

  const runPairOptimization = async (useAnnotations: boolean, ignoreExistingBase = false) => {
    if (!sourceLidar || !targetLidar || !sourceFrameInfo || !targetFrameInfo || optimizing) return;
    if (useAnnotations && !eligibleAnnotations.length) {
      setFeedback("请先保存至少一个完整参照物配对");
      return;
    }
    if (!useAnnotations && !synchronizedFramePairs.length) {
      setFeedback("最近帧配对尚未生成，暂时不能运行 GICP");
      return;
    }
    const pairs = useAnnotations ? eligibleAnnotations.map((annotation) => {
      const sourceDataset = manifest.datasets.find((item) => item.id === annotation.datasetId);
      const annotationSource = sourceDataset?.lidars.find((item) => item.id === annotation.sourceLidarId);
      const annotationTarget = sourceDataset?.lidars.find((item) => item.id === annotation.targetLidarId);
      const sourceInfo = annotationSource?.frames.find((item) => item.label === annotation.sourceFrame);
      const targetInfo = annotationTarget?.frames.find((item) => item.label === annotation.targetFrame);
      const sourceCloudUrl = annotation.sourceCloudUrl ?? sourceInfo?.url;
      const targetCloudUrl = annotation.targetCloudUrl ?? targetInfo?.url;
      if (!sourceCloudUrl || !targetCloudUrl) throw new Error(`配对 ${annotation.id} 的点云帧已不可用`);
      return {
        annotation_id: annotation.id,
        source_cloud_url: sourceCloudUrl,
        target_cloud_url: targetCloudUrl,
        source_indices: annotation.sourcePointIndices,
        target_indices: annotation.targetPointIndices,
      };
    }) : [];
    setOptimizing(true);
    setProgress(1);
    setSolveFailureByPair((current) => {
      if (!current[selectedLidarPairKey]) return current;
      const next = { ...current };
      delete next[selectedLidarPairKey];
      return next;
    });
    setFeedback(useAnnotations
      ? ignoreExistingBase ? `正在用 ${pairs.length} 个人工配对无初值求解…` : `正在基于选中外参用 ${pairs.length} 个人工配对精调…`
      : "正在用完整点云计算 GICP 初值…");
    try {
      const readStreamingCalibration = async (response: Response) => {
        if (!response.ok || !response.body) throw new Error((await response.text()) || `求解服务返回 ${response.status}`);
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let streamedResult: LidarPairCalibration | null = null;
        const consume = (line: string) => {
          if (!line.trim()) return;
          const event = JSON.parse(line) as { type: "progress" | "result" | "error"; percent?: number; message?: string; result?: LidarPairCalibration };
          if (event.type === "error") throw new Error(event.message ?? "雷达外参优化失败");
          setProgress(event.percent ?? 0);
          if (event.message) setFeedback(event.message);
          if (event.type === "result" && event.result) streamedResult = event.result;
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
        if (!streamedResult) throw new Error("GICP 求解器没有返回外参结果");
        return streamedResult as LidarPairCalibration;
      };
      const refinementBaseMatrix = useAnnotations && !ignoreExistingBase ? currentCalibration?.matrix ?? null : null;
      const gicpBaseMatrix = useAnnotations && !ignoreExistingBase
        ? currentCalibration?.gicpBaseMatrix ?? currentCalibration?.matrix ?? null
        : null;
      const requestPayload = {
        group_key: groupKey,
        source_lidar_id: sourceLidar.id,
        target_lidar_id: targetLidar.id,
        source_cloud_url: useAnnotations ? pairs[0]?.source_cloud_url ?? sourceFrameInfo.url : synchronizedFramePairs[0]?.sourceCloudUrl ?? sourceFrameInfo.url,
        target_cloud_url: useAnnotations ? pairs[0]?.target_cloud_url ?? targetFrameInfo.url : synchronizedFramePairs[0]?.targetCloudUrl ?? targetFrameInfo.url,
        // This action explicitly recomputes the automatic seed. Supplying an
        // existing matrix would turn it back into local-only GICP and bypass
        // FPFH-RANSAC, so automatic initialization must start without a base.
        base_matrix: useAnnotations ? refinementBaseMatrix : null,
        pairs,
        frame_pairs: useAnnotations ? [] : synchronizedFramePairs.map((pair) => ({
          source_cloud_url: pair.sourceCloudUrl,
          target_cloud_url: pair.targetCloudUrl,
          source_timestamp_ns: pair.sourceTimestampNs,
          target_timestamp_ns: pair.targetTimestampNs,
          delta_ms: pair.deltaMs,
          origins: pair.origins,
        })),
        stable_only: stableOnly,
        source_bag_path: null,
        source_lidar_topic: null,
        target_lidar_topic: null,
        source_selected_timestamps: [],
        target_selected_timestamps: [],
      };
      const response = await fetch(useAnnotations ? "/api/calibration/lidar-pairs/refine" : "/api/calibration/lidar-pairs/optimize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });
      let result: LidarPairCalibration | null = null;
      let automaticResult: LidarPairCalibration | null = null;
      if (useAnnotations) {
        const payload = await response.json() as LidarPairCalibration & { detail?: string };
        if (!response.ok) throw new Error(payload.detail ?? `精调服务返回 ${response.status}`);
        result = {
          ...payload,
          datasetId: dataset.id,
          framePairSetKey,
          gicpBaseMatrix: gicpBaseMatrix ?? undefined,
          generatedAt: new Date().toISOString(),
        };
      } else {
        const payload = await readStreamingCalibration(response);
        automaticResult = {
          ...payload,
          datasetId: dataset.id,
          framePairSetKey,
          gicpBaseMatrix: payload.matrix,
          generatedAt: new Date().toISOString(),
        };
        // Keep the automatic result as its own selectable version. Existing
        // manual refinements are already retained in the version history.
        result = automaticResult;
      }
      if (!result) throw new Error("求解器没有返回外参结果");
      if (useAnnotations && currentCalibration) {
        result = {
          ...result,
          framePairDiagnostics: currentCalibration.framePairDiagnostics ?? [],
          framePairCount: currentCalibration.framePairCount ?? result.framePairCount,
        };
      } else if (!useAnnotations && result.framePairDiagnostics?.length) {
        setSynchronizedFramePairs(result.framePairDiagnostics);
      }
      const versionId = `lidar-version-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      result = {
        ...result,
        versionId,
        versionLabel: useAnnotations
          ? `${ignoreExistingBase ? "纯人工无初值" : "基于选中初值人工精调"} · ${new Date().toLocaleString("zh-CN")}`
          : `GICP 初值 · ${new Date().toLocaleString("zh-CN")}`,
        source: useAnnotations ? "manual" : "gicp",
      };
      onCalibrationsChange([...calibrations, result]);
      setSolveFailureByPair((current) => {
        if (!current[selectedLidarPairKey]) return current;
        const next = { ...current };
        delete next[selectedLidarPairKey];
        return next;
      });
      setSelectedCalibrationVersionId(versionId);
      setAlignmentPreview(true);
      setAlignmentView((value) => ({ mode: "near", token: value.token + 1 }));
      const reportedResult = !useAnnotations && automaticResult ? automaticResult : result;
      setFeedback(useAnnotations && reportedResult.contour_error != null
        ? `${ignoreExistingBase ? "纯人工无初值求解" : "基于选中初值人工精调"}完成 · 轮廓分布差 ${(reportedResult.contour_error * 100).toFixed(1)} cm · 轮廓中心 ${((reportedResult.contour_center_error ?? 0) * 100).toFixed(1)} cm · 形状差 ${((reportedResult.contour_shape_error ?? 0) * 100).toFixed(1)} cm · 配对一致 ${Math.round((reportedResult.pair_consistency_ratio ?? 1) * 100)}%`
        : `${useAnnotations ? "参照物精调" : `GICP 初值 · ${reportedResult.framePairCount ?? 1} 个去重帧对`}完成 · 地面 ${(reportedResult.ground_angle_deg ?? 0).toFixed(2)}° / ${((reportedResult.ground_height_error ?? 0) * 100).toFixed(1)} cm · 重叠 ${Math.round(reportedResult.overlap_ratio * 100)}% · 帧对一致 ${Math.round((reportedResult.pair_consistency_ratio ?? 1) * 100)}%`);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "雷达外参优化失败";
      setFeedback(`${useAnnotations ? "精调失败" : "GICP 失败"}：${message}`);
      setSolveFailureByPair((current) => ({
        ...current,
        [selectedLidarPairKey]: {
          title: useAnnotations ? "人工精调失败" : "GICP 求解失败",
          message,
        },
      }));
    } finally {
      setOptimizing(false);
      setProgress(0);
    }
  };

  const setCurrentCalibrationAsFinal = () => {
    if (!currentCalibration) return;
    let matchedStoredVersion = false;
    const next = calibrations.map((item) => {
      if (item.groupKey !== groupKey) return item;
      const matches = item === currentCalibration || Boolean(item.versionId && item.versionId === currentCalibration.versionId);
      if (matches) matchedStoredVersion = true;
      return { ...item, isFinal: matches };
    });
    if (!matchedStoredVersion) next.push({ ...currentCalibration, isFinal: true });
    onCalibrationsChange(next);
    setFeedback(`最终结果：${calibrationVersionLabel(currentCalibration)}`);
  };

  const renameCurrentCalibration = () => {
    if (!currentCalibration) return;
    const name = window.prompt("外参版本名称", calibrationVersionName(currentCalibration))?.trim();
    if (!name) return;
    if (currentCalibration.source === "global" && currentCalibration.graphVersionId) {
      const versions = graphVersions.map((item) => item.versionId === currentCalibration.graphVersionId
        ? { ...item, versionLabel: name }
        : item);
      const selected = versions.find((item) => item.versionId === currentCalibration.graphVersionId);
      if (selected) onGraphChange({ ...selected, selectedVersionId: selected.versionId, versions });
      setFeedback(`已重命名为：${name}`);
      return;
    }
    let matchedStoredVersion = false;
    const next = calibrations.map((item) => {
      const matches = item === currentCalibration || Boolean(item.versionId && item.versionId === currentCalibration.versionId);
      if (matches) matchedStoredVersion = true;
      return matches ? { ...item, versionLabel: name } : item;
    });
    if (!matchedStoredVersion) next.push({ ...currentCalibration, versionLabel: name });
    onCalibrationsChange(next);
    setFeedback(`已重命名为：${name}`);
  };

  const solveGraph = async () => {
    const latestByPair = new Map<string, LidarPairCalibration>();
    calibrations.filter((item) =>
      item.isFinal && item.groupKey.startsWith(`lidar:${taskGroupId}:${rigId}:`) && LIDAR_CALIBRATION_COMPATIBLE_VERSIONS.has(item.algorithmVersion),
    ).forEach((item) => latestByPair.set(item.groupKey, item));
    const edges = [...latestByPair.values()].map((item) => ({
      edge_id: item.groupKey,
      source_lidar_id: item.sourceLidarId,
      target_lidar_id: item.targetLidarId,
      matrix: item.matrix,
      weight: Math.max(.1, (item.pair_consistency_ratio ?? 1) / Math.max(item.contour_error ?? item.median_error ?? item.final_rmse, .03)),
    }));
    const lidarIds = [...new Set(edges.flatMap((edge) => [edge.source_lidar_id, edge.target_lidar_id]))];
    const anchor = lidarIds.includes("main") ? "main" : lidarIds[0];
    if (!anchor || !edges.length) { setFeedback("请先为雷达对设置最终结果"); return; }
    setGraphSolving(true);
    setFeedback(`正在统一求解 ${lidarIds.length} 个雷达的外参图…`);
    try {
      const response = await fetch("/api/calibration/lidar-graph/solve", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ anchor_lidar_id: anchor, lidar_ids: lidarIds, edges }),
      });
      const payload = await response.json() as LidarExtrinsicGraph & { detail?: string };
      if (!response.ok) throw new Error(payload.detail ?? `外参图求解返回 ${response.status}`);
      if (!payload.anchorLidarId || !payload.poses) throw new Error("外参图求解没有返回完整结果");
      const generatedAt = new Date().toISOString();
      const versionId = `lidar-graph-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const version: LidarExtrinsicGraphVersion = {
        versionId,
        versionLabel: `全局优化（${lidarIds.length}雷达）`,
        generatedAt,
        anchorLidarId: payload.anchorLidarId,
        poses: payload.poses,
        edgeDiagnostics: payload.edgeDiagnostics,
        converged: payload.converged,
        algorithmVersion: payload.algorithmVersion,
        finalPairVersionIds: Object.fromEntries([...latestByPair.entries()].map(([key, item]) => [key, calibrationVersionId(item)])),
      };
      onGraphChange({ ...version, selectedVersionId: versionId, versions: [...graphVersions, version] });
      if (sourceLidar && targetLidar && payload.poses[sourceLidar.id] && payload.poses[targetLidar.id]) {
        setSelectedCalibrationVersionId(`${versionId}:${sourceLidar.id}:${targetLidar.id}`);
        setAlignmentPreview(true);
      }
      const suspected = payload.edgeDiagnostics?.filter((item) => item.suspected).length ?? 0;
      setFeedback(`已生成全局优化版本 · ${lidarIds.length} 个雷达 · 已加入上方外参版本${suspected ? ` · ${suspected} 条边建议复查` : " · 闭环一致"}`);
    } catch (reason) {
      setFeedback(reason instanceof Error ? reason.message : "外参图求解失败");
    } finally {
      setGraphSolving(false);
    }
  };

  const exportGraph = (graphVersion: LidarExtrinsicGraphVersion | null = selectedGraphVersion) => {
    if (!graphVersion) return;
    const blob = new Blob([JSON.stringify({
      version: 1,
      type: "autocalib-lidar-extrinsic-graph",
      taskGroupId,
      rigId,
      versionId: graphVersion.versionId,
      versionLabel: graphVersion.versionLabel,
      transform: `${graphVersion.anchorLidarId}_from_lidar`,
      anchorLidarId: graphVersion.anchorLidarId,
      poses: graphVersion.poses,
      edgeDiagnostics: graphVersion.edgeDiagnostics ?? [],
      finalPairVersionIds: graphVersion.finalPairVersionIds ?? {},
      exportedAt: new Date().toISOString(),
    }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `autocalib-lidar-graph-${taskGroupId}-${new Date().toISOString().replaceAll(":", "-")}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setFeedback("已导出全局一致的雷达外参图");
  };

  const renameSelectedGraphVersion = () => {
    if (!selectedGraphVersion) return;
    const name = window.prompt("全局外参版本名称", selectedGraphVersion.versionLabel)?.trim();
    if (!name) return;
    const versions = graphVersions.map((item) => item.versionId === selectedGraphVersion.versionId ? { ...item, versionLabel: name } : item);
    const selected = versions.find((item) => item.versionId === selectedGraphVersion.versionId)!;
    onGraphChange({ ...selected, selectedVersionId: selected.versionId, versions });
    setFeedback(`全局外参已重命名为：${name}`);
  };

  const importPairExtrinsic = async (file: File | null) => {
    if (!file || !sourceLidar || !targetLidar) return;
    try {
      const payload = JSON.parse(await file.text()) as Record<string, unknown> | unknown[];
      let matrix = parseRigidMatrix(Array.isArray(payload) ? payload : payload.matrix);
      let importedSourceId = Array.isArray(payload) ? sourceLidar.id : String(payload.sourceLidarId ?? sourceLidar.id);
      let importedTargetId = Array.isArray(payload) ? targetLidar.id : String(payload.targetLidarId ?? targetLidar.id);
      if (!Array.isArray(payload) && payload.poses && typeof payload.poses === "object") {
        const poses = payload.poses as Record<string, unknown>;
        const sourcePose = parseRigidMatrix(poses[sourceLidar.id]);
        const targetPose = parseRigidMatrix(poses[targetLidar.id]);
        if (sourcePose && targetPose) {
          matrix = multiply4(invertRigid4(targetPose), sourcePose);
          importedSourceId = sourceLidar.id;
          importedTargetId = targetLidar.id;
        }
      }
      if (!matrix && !Array.isArray(payload) && Array.isArray(payload.extrinsics)) {
        const entry = payload.extrinsics.find((candidate) => {
          if (!candidate || typeof candidate !== "object") return false;
          const item = candidate as Record<string, unknown>;
          return lidarPairKey(String(item.sourceLidarId ?? ""), String(item.targetLidarId ?? "")) === selectedLidarPairKey;
        }) as Record<string, unknown> | undefined;
        if (entry) {
          matrix = parseRigidMatrix(entry.matrix);
          importedSourceId = String(entry.sourceLidarId);
          importedTargetId = String(entry.targetLidarId);
        }
      }
      if (!matrix) throw new Error("文件中没有当前雷达对可用的 4×4 外参矩阵");
      if (lidarPairKey(importedSourceId, importedTargetId) !== selectedLidarPairKey) {
        throw new Error("文件中的雷达对与当前选择不一致");
      }
      if (importedSourceId === targetLidar.id && importedTargetId === sourceLidar.id) matrix = invertRigid4(matrix);
      const versionId = `lidar-import-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const imported: LidarPairCalibration = {
        versionId,
        versionLabel: `导入 · ${file.name}`,
        source: "imported",
        groupKey,
        sourceLidarId: sourceLidar.id,
        targetLidarId: targetLidar.id,
        transform: "target_from_source",
        matrix,
        delta: [0, 0, 0, 0, 0, 0],
        pairCount: 0,
        initial_rmse: 0,
        final_rmse: 0,
        correspondence_count: 0,
        overlap_ratio: 0,
        pair_consistency_ratio: 1,
        converged: true,
        iteration_count: 0,
        algorithmVersion: "imported-lidar-extrinsic-v1",
        qualityStatus: "good",
        generatedAt: new Date().toISOString(),
        datasetId: dataset.id,
        framePairSetKey,
      };
      onCalibrationsChange([...calibrations, imported]);
      setSelectedCalibrationVersionId(versionId);
      setAlignmentPreview(true);
      setAlignmentView((value) => ({ mode: "near", token: value.token + 1 }));
      setFeedback(`已导入 ${file.name}，正在按该外参查看对齐`);
    } catch (reason) {
      setFeedback(reason instanceof Error ? `导入失败：${reason.message}` : "外参导入失败");
    }
  };

  const exportSelectedPairExtrinsic = () => {
    if (!currentCalibration || !sourceLidar || !targetLidar) return;
    if (currentCalibration.source === "global" && currentCalibration.graphVersionId) {
      const graphVersion = graphVersions.find((item) => item.versionId === currentCalibration.graphVersionId) ?? null;
      if (!graphVersion) {
        setFeedback("当前全局优化版本的数据不存在，无法导出");
        return;
      }
      exportGraph(graphVersion);
      return;
    }
    const blob = new Blob([JSON.stringify({
      version: 1,
      type: "autocalib-lidar-pair-extrinsic",
      sourceLidarId: sourceLidar.id,
      targetLidarId: targetLidar.id,
      transform: "target_from_source",
      versionId: calibrationVersionId(currentCalibration),
      versionLabel: currentCalibration.versionLabel,
      matrix: Array.from({ length: 4 }, (_, row) => currentCalibration.matrix.slice(row * 4, row * 4 + 4)),
      exportedAt: new Date().toISOString(),
    }, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `lidar-${sourceLidar.id}-to-${targetLidar.id}-${Date.now()}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setFeedback(`已导出：${currentCalibration.versionLabel ?? "选中版本"}`);
  };

  const deleteSavedPair = (annotation: LidarPairAnnotation, displayIndex: number) => {
    if (!window.confirm(`确定删除参照物配对 #${displayIndex + 1} 吗？\n\n只会删除这条左右雷达点云配对。`)) return;
    onAnnotationsChange(annotations.filter((item) => item.id !== annotation.id));
    clearDraft();
    if (currentCalibration?.pairCount) {
      onCalibrationsChange(calibrations.filter((item) => item.groupKey !== groupKey || item.source === "imported" || item.pairCount === 0));
      onGraphChange({});
      setAlignmentPreview(false);
      setFeedback(`已删除配对 #${displayIndex + 1}；使用过该配对的精调结果已作废，请重新计算`);
    } else {
      setFeedback(`已删除参照物配对 #${displayIndex + 1}`);
    }
  };

  const pointTools = (mode: PointInteractionMode, setMode: (mode: PointInteractionMode) => void) => (
    <div className="inline-annotation-tools" role="group" aria-label="点云标注工具">
      {([[
        "navigate", "旋转视角"], ["box", "框选聚类"], ["brush-add", "画笔添加"], ["brush-remove", "画笔删除"], ["point-add", "添加单点"], ["point-remove", "删除单点"],
      ] as Array<[PointInteractionMode, string]>).map(([value, label]) => <button key={value} className={mode === value ? "active" : ""} onClick={() => setMode(value)}>{label}</button>)}
    </div>
  );

  return <section className="lidar-pair-workspace">
    <aside className="lidar-pair-sidebar">
      <div className="section-heading"><span>01</span><div><strong>雷达标定数据</strong></div></div>
      <label className="select-field"><span>数据集</span><select value={dataset.id} onChange={(event) => onDatasetChange(event.target.value)}>
        {manifest.datasets.filter((item) => datasetIds.includes(item.id)).map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select></label>
      {dataset.sourceKind === "nuscenes-keyframe-lazy" ? <button className="dataset-frame-edit-button" disabled>关键帧 {dataset.anchorFrames.length} 张 · 点云最多 7 帧融合 · 按需加载</button> : dataset.sourceKind === "paired-directory" ? <button className="dataset-frame-edit-button" disabled>已导入全部 {dataset.anchorFrames.length} 组显式帧对</button> : sourcePaths.length > 0 ? <div className="group-source-frame-list">{sourcePaths.map((sourcePath) => <button key={sourcePath} title={sourcePath} onClick={() => onEditSourceFrames(sourcePath)}><span>{sourcePath.split("/").at(-1)}</span><b>选帧</b></button>)}</div> : <button className="dataset-frame-edit-button" disabled={!dataset.sourcePath} onClick={() => onEditDatasetFrames(dataset)}>选帧 / 改帧</button>}
      <div className="section-heading lidar-pair-heading"><span>02</span><div><strong>雷达组合与最近帧对</strong></div></div>
      <label className="select-field lidar-pair-combination-field"><span>雷达配对 · 已完成 {completedLidarPairCount}/{lidarPairOptions.length}</span><select aria-label="选择雷达配对" value={selectedLidarPairKey} onChange={(event) => selectLidarPair(event.target.value)}>
        {lidarPairOptions.map((item) => <option key={item.key} value={item.key}>{completedPairKeys.has(item.key) ? "🟢" : "🔴"} {lidarDisplayLabel(item.source.topic, item.source.name, topicRemarks)} ↔ {lidarDisplayLabel(item.target.topic, item.target.name, topicRemarks)}</option>)}
      </select></label>
      <div className="lidar-topic-remark-actions"><button onClick={() => editTopicRemark(sourceLidar)}>备注左侧</button><button onClick={() => editTopicRemark(targetLidar)}>备注右侧</button></div>
      <button className={`confirm-no-lidar-annotation${noAnnotationPairKeys.includes(selectedLidarPairKey) ? " confirmed" : ""}`} disabled={!selectedLidarPairKey || annotatedPairKeys.has(selectedLidarPairKey)} onClick={() => {
        if (noAnnotationPairKeys.includes(selectedLidarPairKey)) {
          onNoAnnotationPairKeysChange(noAnnotationPairKeys.filter((key) => key !== selectedLidarPairKey));
          return;
        }
        if (!window.confirm("确认当前雷达对没有可标注的共同参照物吗？确认后该配对会标为绿灯。")) return;
        onNoAnnotationPairKeysChange([...noAnnotationPairKeys, selectedLidarPairKey]);
      }}>{annotatedPairKeys.has(selectedLidarPairKey) ? "已有标注配对" : noAnnotationPairKeys.includes(selectedLidarPairKey) ? "撤销无标注确认" : "确认无可标注参照物"}</button>
      {synchronizedFramePairs.length > 0 ? <label className="select-field synchronized-frame-pair-field"><span>视角配对</span><select value={Math.min(synchronizedPairIndex, synchronizedFramePairs.length - 1)} onChange={(event) => {
        pendingPairPreviewRef.current = null;
        setActiveAlignmentAnnotationId(null);
        setAlignmentSelectionOnly(false);
        setSynchronizedPairIndex(Number(event.target.value));
      }}>
        {synchronizedFramePairs.map((pair, index) => <option key={`${pair.sourceTimestampNs}:${pair.targetTimestampNs}`} value={index}>{annotatedFramePairKeys.has(`${pair.sourceCloudUrl}\n${pair.targetCloudUrl}`) ? "🟢" : "🔴"} 配对 {index + 1}</option>)}
      </select>{synchronizedFramePair && <small>源 {formatTimestamp(synchronizedFramePair.sourceTimestampNs)} ↔ 目标 {formatTimestamp(synchronizedFramePair.targetTimestampNs)} · Δ {synchronizedFramePair.deltaMs.toFixed(3)} ms</small>}</label> : <div className="synchronized-frame-pair-empty">{pairingLoading ? "正在生成最近帧配对……" : "当前雷达组合没有可用配对，请检查两侧是否都已选帧。"}</div>}
      <div className="section-heading lidar-pair-heading"><span>03</span><div><strong>过滤与显示</strong></div></div>
      <label className="toggle-row"><span><strong>仅显示静止点</strong></span><input type="checkbox" checked={stableOnly} onChange={(event) => setStableOnly(event.target.checked)} /></label>
      <label className="toggle-row"><span><strong>隐藏地面点</strong></span><input type="checkbox" checked={hideGround} onChange={(event) => setHideGround(event.target.checked)} /></label>
    </aside>
    <main className="lidar-pair-stage">
      <header className="lidar-pair-stage-header">
        <div><strong>{sourceLidarLabel} → {targetLidarLabel}</strong></div>
        <div className="lidar-pair-draft-summary"><span>左 {formatPoints(sourceSelection.length)} 点</span><b>↔</b><span>右 {formatPoints(targetSelection.length)} 点</span><label className="lidar-annotation-point-size"><span>点大小</span><input aria-label="标注点云点大小" type="range" min=".01" max=".12" step=".01" value={pointSize} onChange={(event) => setPointSize(Number(event.target.value))} /><b>{pointSize.toFixed(2)}</b></label><label className="lidar-annotation-point-size lidar-annotation-brush-size"><span>画笔</span><input aria-label="标注画笔直径" type="range" min="8" max="80" step="2" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} /><b>{brushSize}px</b></label><button onClick={clearDraft}>清空</button></div>
        <div className="lidar-saved-pairs"><b>{visibleAnnotations.length}</b><span>个参照物配对</span>{pagedVisibleAnnotations.map((item, pageIndex) => {
          const index = effectiveSavedPairPage * LIDAR_SAVED_PAIR_PAGE_SIZE + pageIndex;
          const deltaMs = annotationSyncDeltaMs(item);
          const synchronized = Boolean(item.sourceCloudUrl && item.targetCloudUrl) && deltaMs <= 100;
          const optimizedPairIndex = eligibleAnnotations.findIndex((annotation) => annotation.id === item.id);
          const contourMetric = optimizedPairIndex >= 0 ? currentCalibration?.frame_pair_metrics?.[optimizedPairIndex] : undefined;
          return <div className={`lidar-saved-pair${synchronized ? "" : " unsynchronized"}${activeAlignmentAnnotationId === item.id ? " active" : ""}`} key={item.id}>
          <button className={`load${contourMetric?.contourError != null ? " with-metric" : ""}${contourMetric && !contourMetric.reliable ? " contour-warning" : ""}`} title={synchronized ? `载入此最近帧配对 · Δ ${deltaMs.toFixed(3)} ms${contourMetric?.contourError != null ? ` · 轮廓分布差 ${(contourMetric.contourError * 100).toFixed(1)} cm · 中心 ${((contourMetric.contourCenterError ?? 0) * 100).toFixed(1)} cm · 形状 ${((contourMetric.contourShapeError ?? 0) * 100).toFixed(1)} cm` : ""}` : `旧配对不同步 · Δ ${deltaMs.toFixed(1)} ms · 不参与精调`} onClick={() => {
            if (activeAlignmentAnnotationId === item.id) {
              clearDraft();
              return;
            }
            if (!synchronized) {
              setFeedback(`配对 #${index + 1} 相差 ${deltaMs.toFixed(1)} ms，不是最近帧，已禁止参与精调；可用右侧 × 删除`);
              return;
            }
            const pairIndex = synchronizedFramePairs.findIndex((pair) => pair.sourceCloudUrl === item.sourceCloudUrl && pair.targetCloudUrl === item.targetCloudUrl);
            if (pairIndex < 0) {
              setFeedback(`配对 #${index + 1} 不属于当前 GICP 最近帧集合，请重新计算 GICP`);
              return;
            }
            pendingPairPreviewRef.current = item;
            setActiveAlignmentAnnotationId(item.id);
            if (pairIndex === synchronizedPairIndex && effectiveSourceCloudUrl === item.sourceCloudUrl && effectiveTargetCloudUrl === item.targetCloudUrl) {
              setSourceSelection(Uint32Array.from(item.sourcePointIndices)); setTargetSelection(Uint32Array.from(item.targetPointIndices));
              pendingPairPreviewRef.current = null;
            } else setSynchronizedPairIndex(pairIndex);
          }}>#{index + 1}{contourMetric?.contourError != null ? ` ${(contourMetric.contourError * 100).toFixed(1)}cm` : ""}{synchronized ? "" : "!"}</button>
          <label className="participate" title="参与本次外参计算">
            <input type="checkbox" aria-label={`配对 #${index + 1} 参与本次外参计算`} disabled={!synchronized} checked={synchronized && item.useForOptimization !== false} onChange={(event) => {
              const checked = event.currentTarget.checked;
              onAnnotationsChange(annotations.map((annotation) => annotation.id === item.id
                ? { ...annotation, useForOptimization: checked }
                : annotation));
            }} />
          </label>
          <button className="delete" title={`删除配对 #${index + 1}`} aria-label={`删除配对 #${index + 1}`} onClick={() => deleteSavedPair(item, index)}>×</button>
        </div>;})}{savedPairPageCount > 1 && <nav className="lidar-saved-pair-pagination" aria-label="参照物配对翻页"><button aria-label="上一页" disabled={effectiveSavedPairPage === 0} onClick={() => setSavedPairPage((value) => Math.max(0, value - 1))}>‹</button><span>{effectiveSavedPairPage + 1}/{savedPairPageCount}</span><button aria-label="下一页" disabled={effectiveSavedPairPage + 1 >= savedPairPageCount} onClick={() => setSavedPairPage((value) => Math.min(savedPairPageCount - 1, value + 1))}>›</button></nav>}</div>
      </header>
      <section className="calibration-workflow-bar lidar-pair-workflow-bar" aria-label="雷达外参标定工作流">
        {currentSolveFailure && <div className="lidar-solve-failure" role="alert">
          <b>{currentSolveFailure.title}</b>
          <span>{currentSolveFailure.message}</span>
          <button aria-label="关闭求解失败提示" onClick={() => setSolveFailureByPair((current) => {
            const next = { ...current };
            delete next[selectedLidarPairKey];
            return next;
          })}>×</button>
        </div>}
        <div className="lidar-pair-workflow-status">
          <strong>{feedback}</strong>
        </div>
        <div className="calibration-workflow-actions lidar-pair-workflow-actions">
          <input ref={lidarExtrinsicImportRef} className="extrinsic-file-input" type="file" accept="application/json,.json" aria-label="选择雷达外参文件" onChange={(event) => { void importPairExtrinsic(event.currentTarget.files?.[0] ?? null); event.currentTarget.value = ""; }} />
          <div className="lidar-workflow-row lidar-version-row">
            <b>版本</b>
            <button onClick={() => lidarExtrinsicImportRef.current?.click()}>导入外参</button>
            <label className="lidar-version-select"><select aria-label="选择雷达外参版本" value={currentCalibration ? calibrationVersionId(currentCalibration) : ""} disabled={!pairCalibrationVersions.length} onChange={(event) => { setSelectedCalibrationVersionId(event.target.value); setAlignmentPreview(true); }}><option value="">暂无版本</option>{pairCalibrationVersions.map((item, index) => <option key={calibrationVersionId(item, index)} value={calibrationVersionId(item, index)}>{item.isFinal ? `最终 · ${calibrationVersionLabel(item)}` : calibrationVersionLabel(item)}</option>)}</select></label>
            <button onClick={renameCurrentCalibration} disabled={!currentCalibration}>重命名</button>
            <button onClick={setCurrentCalibrationAsFinal} disabled={!currentCalibration || currentCalibration.isFinal}>{currentCalibration?.isFinal ? "已是最终结果" : "设为最终结果"}</button>
            <span className="lidar-final-result">最终：{finalCalibration ? calibrationVersionLabel(finalCalibration) : "未设置"}</span>
            <button onClick={exportSelectedPairExtrinsic} disabled={!currentCalibration}>{currentCalibration?.source === "global" ? "导出全局外参" : "导出选中外参"}</button>
            <button className={alignmentPreview ? "alignment-preview-button active" : "alignment-preview-button"} onClick={() => setAlignmentPreview((value) => !value)} disabled={!currentCalibration}>{alignmentPreview ? "返回配对" : "查看对齐"}</button>
          </div>
          <div className="lidar-workflow-row lidar-solve-row">
            <b>求解</b>
            <button className="generate-calibration-button" onClick={() => void runPairOptimization(false)} disabled={optimizing || pairingLoading || !synchronizedFramePairs.length || sourceLoading || targetLoading}>GICP 求初值（{synchronizedFramePairs.length} 帧对）</button>
            <button onClick={() => void runPairOptimization(true, false)} disabled={optimizing || !currentCalibration || !eligibleAnnotations.length}>基于上方选中外参精调（{eligibleAnnotations.length} 对）</button>
            <button onClick={() => void runPairOptimization(true, true)} disabled={optimizing || !eligibleAnnotations.length}>纯人工无初值求解（{eligibleAnnotations.length} 对）</button>
            <button className="save-lidar-pair-button" onClick={savePair} disabled={!synchronizedFramePair || selectedFrameDeltaMs > 100 || sourceSelection.length < 6 || targetSelection.length < 6}>{activeAlignmentAnnotationId ? "更新参照物配对" : "保存参照物配对"}</button>
            <button onClick={() => void solveGraph()} disabled={optimizing || graphSolving || !calibrations.some((item) => item.isFinal && LIDAR_CALIBRATION_COMPATIBLE_VERSIONS.has(item.algorithmVersion))}>{graphSolving ? "正在统一求解…" : "统一求解外参图"}</button>
          </div>
        </div>
        {optimizing && <div className="workflow-progress" role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><i style={{ width: `${progress}%` }} /></div>}
      </section>
      {alignmentPreview && alignmentCloud && currentCalibration ? <div ref={alignmentPreviewRef} className="lidar-alignment-split">
        <article className="lidar-pair-panel lidar-alignment-inspector"><header className="panel-head annotation-panel-head lidar-pair-panel-head"><div className="alignment-inspect-switch" role="group" aria-label="选择单点云"><button className={alignmentInspectLidar === "source" ? "active" : ""} onClick={() => setAlignmentInspectLidar("source")}>{sourceLidarLabel}</button><button className={alignmentInspectLidar === "target" ? "active" : ""} onClick={() => setAlignmentInspectLidar("target")}>{targetLidarLabel}</button></div>{pointTools(alignmentInspectMode, setAlignmentInspectMode)}<div className="panel-actions compact-panel-actions"><button onClick={() => alignmentInspectLidar === "source" ? setSourceSelection(new Uint32Array()) : setTargetSelection(new Uint32Array())}>清空</button></div></header><div className="lidar-pair-cloud">
          {alignmentInspectCloud && <PointCloudViewport clouds={alignmentInspectCloudList} frameIndex={0} stableOnly={stableOnly} hideGround={hideGround} pointSize={pointSize} brushSize={brushSize} brushDepthTolerance={foregroundDepthTolerance} interactionMode={alignmentInspectMode} selectedIndices={alignmentInspectSelection} selectedPointColor={LIDAR_SELECTION_COLOR} clearSelectionToken={clearToken} viewRequest={alignmentInspectView} onSelection={(indices) => { if (alignmentInspectLidar === "source") selectRegion(sourceCloud, indices, setSourceSelection); else selectRegion(targetCloud, indices, setTargetSelection); if (indices.length) setAlignmentDisplayMode("three"); }} onPointEdit={(indices, mode) => alignmentInspectLidar === "source" ? editRegion(sourceSelection, indices, mode, setSourceSelection) : editRegion(targetSelection, indices, mode, setTargetSelection)} onInteractionModeChange={setAlignmentInspectMode} preserveViewOnCloudChange />}
        </div></article>
        <article className="lidar-alignment-fusion">
          {alignmentDisplayMode === "three" && alignmentLocalClouds
            ? <OrthographicCloudViews source={alignmentLocalClouds.source} target={alignmentLocalClouds.target} />
            : <PointCloudViewport clouds={alignmentCloudList} frameIndex={0} stableOnly={false} hideGround={false} pointSize={alignmentPointSize} brushSize={brushSize} brushDepthTolerance={.8} interactionMode="navigate" selectedIndices={emptyAlignmentSelection} overlayClouds={alignmentHighlightClouds} overlayOpacity={alignmentHighlightOpacity} clearSelectionToken={0} viewRequest={alignmentView} onSelection={() => undefined} onPointEdit={() => undefined} onInteractionModeChange={() => undefined} preserveViewOnCloudChange gridReferenceCloud={alignmentCloud} comparisonMode comparisonVisibleLayer={effectiveAlignmentLayer} />}
          <div className="lidar-alignment-legend">
          <b>外参对齐</b><span className="source">● {sourceLidarLabel}</span><span className="target">● {targetLidarLabel}</span>
          <div className="alignment-layer-switch" role="group" aria-label="选择对齐视图"><button className={alignmentDisplayMode === "full" ? "active" : ""} onClick={() => setAlignmentDisplayMode("full")}>全图</button><button className={alignmentDisplayMode === "three" ? "active" : ""} disabled={!alignmentLocalClouds} onClick={() => setAlignmentDisplayMode("three")}>三视图</button></div>
          {alignmentDisplayMode === "full" && <>
          <label><span>点大小 {alignmentPointSize.toFixed(3)}</span><input aria-label="融合点大小" type="range" min=".005" max=".12" step=".005" value={alignmentPointSize} onChange={(event) => setAlignmentPointSize(Number(event.target.value))} /></label>
          <label><span>高亮透明度 {Math.round(alignmentHighlightOpacity * 100)}%</span><input aria-label="选中区域高亮透明度" type="range" min=".1" max="1" step=".05" value={alignmentHighlightOpacity} onChange={(event) => setAlignmentHighlightOpacity(Number(event.target.value))} /></label>
          <label title={alignmentSelectionOnly ? "当前参照物固定显示全部已标点" : undefined}><span>显示密度 {effectiveAlignmentDensity}%</span><input aria-label="融合显示密度" type="range" min="10" max="100" step="10" value={effectiveAlignmentDensity} disabled={alignmentSelectionOnly} onChange={(event) => setAlignmentDensity(Number(event.target.value))} /></label>
          <select className="alignment-version-select" aria-label="选择对比外参版本" value={calibrationVersionId(currentCalibration)} onChange={(event) => setSelectedCalibrationVersionId(event.target.value)}>{pairCalibrationVersions.map((item, index) => <option key={calibrationVersionId(item, index)} value={calibrationVersionId(item, index)}>{calibrationVersionLabel(item)}</option>)}</select>
          <div className="alignment-layer-switch" role="group" aria-label="融合点云图层"><button className={!alignmentBlink && alignmentLayer === "both" ? "active" : ""} onClick={() => { setAlignmentBlink(false); setAlignmentLayer("both"); }}>双层</button><button className={!alignmentBlink && alignmentLayer === "source" ? "active" : ""} onClick={() => { setAlignmentBlink(false); setAlignmentLayer("source"); }}>只看 {sourceLidarLabel}</button><button className={!alignmentBlink && alignmentLayer === "target" ? "active" : ""} onClick={() => { setAlignmentBlink(false); setAlignmentLayer("target"); }}>只看 {targetLidarLabel}</button><button className={alignmentBlink ? "active blink" : ""} onClick={() => setAlignmentBlink((value) => !value)}>交替闪烁</button><button className={alignmentSelectionOnly ? "active" : ""} disabled={!canFocusAlignmentSelection} title={canFocusAlignmentSelection ? "只显示当前载入配对所选的参照物，并保持当前视角" : "请先载入一个已保存的参照物配对"} onClick={() => setAlignmentSelectionOnly((value) => !value)}>只看当前参照物</button></div>
          </>}
          {alignmentDisplayMode === "three" && <select className="alignment-version-select" aria-label="选择三视图外参版本" value={calibrationVersionId(currentCalibration)} onChange={(event) => setSelectedCalibrationVersionId(event.target.value)}>{pairCalibrationVersions.map((item, index) => <option key={calibrationVersionId(item, index)} value={calibrationVersionId(item, index)}>{calibrationVersionLabel(item)}</option>)}</select>}
          </div>
        </article>
      </div> : <div className="lidar-pair-views">
        <article className="lidar-pair-panel"><header className="panel-head annotation-panel-head lidar-pair-panel-head"><div className="panel-title-block"><strong>{sourceLidarLabel}</strong></div>{pointTools(sourceMode, setSourceMode)}</header><div className="lidar-pair-cloud">
          {sourceCloud && <PointCloudViewport key={sourceLidar?.id} clouds={sourceCloudList} frameIndex={0} stableOnly={stableOnly} hideGround={hideGround} pointSize={pointSize} brushSize={brushSize} brushDepthTolerance={foregroundDepthTolerance} interactionMode={sourceMode} selectedIndices={sourceSelection} selectedPointColor={LIDAR_SELECTION_COLOR} clearSelectionToken={clearToken} viewRequest={sourceView} onSelection={(indices) => selectRegion(sourceCloud, indices, setSourceSelection)} onPointEdit={(indices, mode) => editRegion(sourceSelection, indices, mode, setSourceSelection)} onInteractionModeChange={setSourceMode} preserveViewOnCloudChange />}
        </div></article>
        <article className="lidar-pair-panel"><header className="panel-head annotation-panel-head lidar-pair-panel-head"><div className="panel-title-block"><strong>{targetLidarLabel}</strong></div>{pointTools(targetMode, setTargetMode)}</header><div className="lidar-pair-cloud">
          {targetCloud && <PointCloudViewport key={targetLidar?.id} clouds={targetCloudList} frameIndex={0} stableOnly={stableOnly} hideGround={hideGround} pointSize={pointSize} brushSize={brushSize} brushDepthTolerance={foregroundDepthTolerance} interactionMode={targetMode} selectedIndices={targetSelection} selectedPointColor={LIDAR_SELECTION_COLOR} clearSelectionToken={clearToken} viewRequest={targetView} onSelection={(indices) => selectRegion(targetCloud, indices, setTargetSelection)} onPointEdit={(indices, mode) => editRegion(targetSelection, indices, mode, setTargetSelection)} onInteractionModeChange={setTargetMode} preserveViewOnCloudChange />}
        </div></article>
      </div>}
    </main>
  </section>;
}
