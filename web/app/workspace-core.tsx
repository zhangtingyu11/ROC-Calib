"use client";

import { useEffect, useMemo, useState } from "react";
import { countMask, ImageMask, MaskDirtyRect, mergeImageMasks } from "./segmentation";
import { WaymoTemporalProjection } from "./waymo-camera-model";
import { OrthographicCloudView } from './orthographic-view';
import { PointCloudViewport } from './point-cloud-view';

export type FrameLabel = string;

export type FitMode = "near" | "all";

export type ViewRequest = { mode: FitMode; token: number };

export type ImageSegmentationTool = "pan" | "box-replace" | "box-add" | "brush-add" | "brush-remove";

export type BrushMaskChange = { dirtyRects: MaskDirtyRect[]; pixelDelta: number };

export type PointSegmentationMode = "replace" | "add" | "remove";

export type SegmentationGranularity = "fine" | "balanced" | "coarse";

export type ImageMaskLayer = {
  id: string;
  mask: ImageMask;
  granularity: SegmentationGranularity;
  brushModified: boolean;
};

export type ImageMaskSnapshot = {
  layers: ImageMaskLayer[];
  activeLayerId: string | null;
  manualAddedMask: ImageMask | null;
  manualRemovedMask: ImageMask | null;
};

export type PointInteractionMode = "navigate" | "pan" | "box" | "point-add" | "point-remove" | "brush-add" | "brush-remove";

export type CalibrationWorkspaceMode = "lidar-camera" | "lidar-lidar";

export type StageMode = "annotate" | "compare";

export type ProjectionScope = "all" | "selected" | "frame-annotations";

export type FullProjectionScope = "all" | "frame-annotations" | "single-annotation";

export type ExtrinsicInitializationMode = "existing" | "forward";

export type ProjectionModel = "fisheye" | "radtan" | "rational";

export type ProjectionModelChoice = "auto" | ProjectionModel;

export type SynchronizedImageView = { zoom: number; centerX: number; centerY: number };

export type PointGrowFeedback = { kind: "success" | "error"; text: string };

export type ProjectedPoint = { index: number; x: number; y: number; depth: number };

export type RectDragMode = "new" | "move" | "left" | "right" | "top" | "bottom" | "top-left" | "top-right" | "bottom-left" | "bottom-right";

export function cursorForRectDragMode(mode: RectDragMode): React.CSSProperties["cursor"] {
  if (mode === "move") return "move";
  if (mode === "left" || mode === "right") return "ew-resize";
  if (mode === "top" || mode === "bottom") return "ns-resize";
  if (mode === "top-left" || mode === "bottom-right") return "nwse-resize";
  if (mode === "top-right" || mode === "bottom-left") return "nesw-resize";
  return "crosshair";
}

export type CloudFrameInfo = {
  label: FrameLabel;
  timestampNs: string;
  pointCount: number;
  stableRatio: number;
  url: string;
};

export type LidarInfo = {
  id: string;
  name: string;
  topic: string;
  frames: CloudFrameInfo[];
};

export type CameraFrameInfo = {
  label: FrameLabel;
  timestampNs: string;
  syncOffsetMs: number;
  width: number;
  height: number;
  url: string;
};

export type CameraInfo = {
  id: string;
  name: string;
  topic: string;
  frames: CameraFrameInfo[];
};

export type WaymoTemporalProjectionFrame = Pick<WaymoTemporalProjection,
  "referenceWorldFromVehicle" | "cameraWorldFromVehicle" |
  "linearVelocityWorld" | "angularVelocityVehicle" | "poseTimestamp" |
  "shutter" | "triggerTime" | "readoutDoneTime">;

export type WaymoTemporalProjectionModel = Pick<WaymoTemporalProjection,
  "model" | "vehicleFromLidar" | "rollingShutterDirection" |
  "imageWidth" | "imageHeight"> & {
    source: string;
    frames: Record<string, WaymoTemporalProjectionFrame>;
  };

export type DatasetInfo = {
  id: string;
  rigId?: string;
  name: string;
  sourceFile: string;
  sourcePath?: string;
  sourceKind?: "rosbag" | "paired-directory" | "nuscenes-keyframe-lazy";
  stationaryIntervalCount: number;
  calibrationCompatibility: "after-camera-adjustment" | "reference-only" | "unknown-extrinsic";
  selectedInterval: { startNs: string; endNs: string; durationSeconds: number };
  anchorFrames: Array<{ label: FrameLabel; timestampNs: string }>;
  lidars: LidarInfo[];
  cameras: CameraInfo[];
  cameraCalibration?: Record<string, { intrinsic: number[]; distortion: number[]; distortionModel?: string; rawIntrinsic?: number[]; rawDistortion?: number[]; rawDistortionModel?: string; coordinateSystem?: string; serial?: string; model?: string }>;
  temporalProjectionModels?: Record<string, WaymoTemporalProjectionModel>;
  extrinsics?: null;
};

export function temporalProjectionFor(
  dataset: DatasetInfo | undefined,
  lidarId: string | undefined,
  cameraId: string | undefined,
  frame: FrameLabel,
): WaymoTemporalProjection | null {
  if (!dataset || !lidarId || !cameraId) return null;
  const model = dataset.temporalProjectionModels?.[`${lidarId}:${cameraId}`];
  const frameMetadata = model?.frames[String(frame)];
  return model && frameMetadata ? { ...model, ...frameMetadata } : null;
}

export type CalibrationManifest = {
  source: string;
  note: string;
  lidars: Record<string, { carFromLidar: number[] }>;
  cameras: Record<string, { cameraFromCar: number[]; intrinsic: number[]; distortion: number[]; distortionModel?: string }>;
};

export type Manifest = { datasets: DatasetInfo[]; calibration: CalibrationManifest };

export type CalibrationTaskGroup = { id: string; name: string; rigId: string; datasetIds: string[]; sourcePaths?: string[]; pairedSourcePaths?: string[]; status: string; createdAt: string; updatedAt: string; storagePath: string };

export type ImportedCameraIntrinsics = {
  topic: string;
  cameraType: "pinhole" | "fisheye";
  intrinsic: number[];
  distortion: number[];
  distortionModel: string;
  rawIntrinsic: number[];
  rawDistortion: number[];
  rawDistortionModel: string;
  coordinateSystem: string;
  frames: Record<string, string>;
};

export type IntrinsicsProfile = {
  version: number;
  sourceFile: string | null;
  importedAt: string | null;
  cameras: Array<{ topic: string; cameraType: string }>;
  datasets: Record<string, { cameras: Record<string, ImportedCameraIntrinsics> }>;
};

export type BagBrowserEntry = { name: string; path: string; kind: "directory" | "bag"; sizeBytes: number; datasetIds: string[]; rigIds: string[]; prepared: boolean };

export type BagBrowserResult = { root: string; path: string; parent: string | null; entries: BagBrowserEntry[] };

export type PairedDatasetBrowserEntry = { name: string; path: string; kind: "directory" | "dataset"; valid?: boolean; error?: string | null; rigId?: string; cameraCount?: number; lidarCount?: number; frameCount?: number };

export type PairedDatasetBrowserResult = { root: string; path: string; parent: string | null; entries: PairedDatasetBrowserEntry[] };

export type ManualFrameCandidate = { timestampNs: string; offsetSeconds: number };

export type ManualCameraView = { id: string; name: string; topic: string };

export type ManualFrameTimeline = { path: string; name: string; startNs: string; endNs: string; durationSeconds: number; stepSeconds: number; previewTopic: string; lidars: ManualCameraView[]; cameras: ManualCameraView[]; candidates: ManualFrameCandidate[] };

export type ManualSelectionState = { version: number; sourcePath: string; selections: Record<string, string[]>; updatedAt: string | null; updatedBy: string | null };

export type IntrinsicsBrowserEntry = { name: string; path: string; kind: "directory" | "yaml"; sizeBytes: number };

export type IntrinsicsBrowserResult = { root: string; path: string; parent: string | null; entries: IntrinsicsBrowserEntry[] };

export type BagPreparationProgress = { active: boolean; percent: number; message: string; bagIndex?: number; bagCount?: number };

export type LocalBagUploadFile = { file: File; relativePath: string };

export type LocalBagUploadProgress = { active: boolean; paused: boolean; percent: number; message: string; uploadedBytes: number; totalBytes: number };

export type BagUploadSession = {
  uploadId: string;
  packageName: string;
  sourcePath: string | null;
  complete: boolean;
  chunkSize: number;
  files: Array<{ index: number; relativePath: string; size: number; offset: number; complete: boolean }>;
  entry?: BagBrowserEntry;
};

export type CloudData = { positions: Float32Array; stable: Uint8Array; ground: Uint8Array; hasGroundLabels: boolean; timestampNs: bigint };

export const EMPTY_CLOUD_OVERLAYS: Array<{ cloud: CloudData; color: number }> = [];

export type SavedAnnotation = {
  id: string;
  datasetId: string;
  frame: FrameLabel;
  lidarId: string;
  cameraId: string;
  lidarTimestampNs: string;
  cameraTimestampNs: string;
  calibrationSource: string;
  status: "image-draft" | "point-draft" | "paired";
  pointCount: number;
  maskPixelCount: number;
  pointIndices: number[];
  imageMask: { width: number; height: number; rle: number[] } | null;
  imageMaskLayers?: Array<{
    id: string;
    width: number;
    height: number;
    rle: number[];
    granularity: SegmentationGranularity;
    brushModified: boolean;
  }>;
  manualAddedMask?: { width: number; height: number; rle: number[] } | null;
  manualRemovedMask?: { width: number; height: number; rle: number[] } | null;
  useForOptimization: boolean;
  taskGroupId?: string;
  imageCoordinateSystem?: string;
  savedAt: string;
};

export type LidarPairAnnotation = {
  id: string;
  taskGroupId: string;
  datasetId: string;
  sourceLidarId: string;
  targetLidarId: string;
  sourceFrame: FrameLabel;
  targetFrame: FrameLabel;
  sourceTimestampNs: string;
  targetTimestampNs: string;
  sourceCloudUrl?: string;
  targetCloudUrl?: string;
  syncOffsetMs?: number;
  sourcePointIndices: number[];
  targetPointIndices: number[];
  useForOptimization: boolean;
  savedAt: string;
};

export type LidarFramePairDiagnostic = {
  index: number;
  sourceTimestampNs: string;
  targetTimestampNs: string;
  deltaMs: number;
  origins: string[];
  sourceCloudUrl: string;
  targetCloudUrl: string;
};

export type LidarFramePairSet = {
  key: string;
  datasetId: string;
  sourceLidarId: string;
  targetLidarId: string;
  pairs: LidarFramePairDiagnostic[];
  generatedAt: string;
};

export function lidarPairKey(firstLidarId: string, secondLidarId: string) {
  return [firstLidarId, secondLidarId].sort().join(":");
}

export function lidarDisplayLabel(topic: string | undefined, fallback: string | undefined, remarks: Record<string, string>) {
  const remark = topic ? remarks[topic]?.trim() : "";
  return remark ? `${remark}（${topic}）` : topic ?? fallback ?? "雷达";
}

export function versionDisplayLabel(name: string, generatedAt?: string) {
  if (/\d{4}[/-]\d{1,2}[/-]\d{1,2}\s+\d{1,2}:\d{2}/.test(name)) return name;
  if (!generatedAt) return `${name} · 时间未知`;
  const timestamp = new Date(generatedAt);
  return `${name} · ${Number.isNaN(timestamp.getTime()) ? "时间未知" : timestamp.toLocaleString("zh-CN")}`;
}

export function parseRigidMatrix(value: unknown): number[] | null {
  if (!Array.isArray(value)) return null;
  const flat = value.flatMap((row) => Array.isArray(row) ? row : [row]).map(Number);
  return flat.length === 16 && flat.every(Number.isFinite) ? flat : null;
}

export type LidarPairCalibration = {
  versionId?: string;
  versionLabel?: string;
  isFinal?: boolean;
  source?: "gicp" | "manual" | "imported" | "global";
  graphVersionId?: string;
  groupKey: string;
  sourceLidarId: string;
  targetLidarId: string;
  transform: "target_from_source";
  matrix: number[];
  delta: number[];
  pairCount: number;
  framePairCount?: number;
  framePairDiagnostics?: LidarFramePairDiagnostic[];
  initial_rmse: number;
  final_rmse: number;
  median_error?: number;
  p90_error?: number;
  contour_error?: number;
  contour_p90_error?: number;
  contour_center_error?: number;
  contour_shape_error?: number;
  contour_error_spread?: number;
  frame_pair_metrics?: Array<{
    index: number;
    contourError?: number;
    contourP90Error?: number;
    contourCenterError?: number;
    contourShapeError?: number;
    rangeM?: number;
    reliable: boolean;
    medianError: number;
    p90Error: number;
    overlapRatio: number;
  }>;
  pair_consistency_ratio?: number;
  pair_error_spread?: number;
  ground_angle_deg?: number | null;
  ground_height_error?: number | null;
  correspondence_count: number;
  overlap_ratio: number;
  converged: boolean;
  iteration_count: number;
  algorithmVersion: string;
  projectionModel?: "waymo-camera-model-v1" | "opencv-static-v1";
  qualityStatus: "good" | "warning" | "poor" | "unverified";
  generatedAt?: string;
  datasetId?: string;
  framePairSetKey?: string;
  gicpBaseMatrix?: number[];
};

export type LidarExtrinsicGraph = {
  versionId?: string;
  versionLabel?: string;
  generatedAt?: string;
  anchorLidarId?: string;
  poses?: Record<string, number[]>;
  edgeDiagnostics?: Array<{ edgeId: string; rotationResidualDeg: number; translationResidualM: number; suspected: boolean }>;
  converged?: boolean;
  algorithmVersion?: string;
  finalPairVersionIds?: Record<string, string>;
  selectedVersionId?: string;
  versions?: LidarExtrinsicGraphVersion[];
};

export type LidarExtrinsicGraphVersion = Required<Pick<LidarExtrinsicGraph,
  "versionId" | "versionLabel" | "generatedAt" | "anchorLidarId" | "poses">> &
  Pick<LidarExtrinsicGraph, "edgeDiagnostics" | "converged" | "algorithmVersion" | "finalPairVersionIds">;

export function frameIndexByLabel(frames: Array<{ label: FrameLabel }> | undefined, label: FrameLabel) {
  const index = frames?.findIndex((frame) => frame.label === label) ?? -1;
  return Math.max(0, index);
}

export function validManualSelections(state: ManualSelectionState, timeline: ManualFrameTimeline, mode: CalibrationWorkspaceMode) {
  const validTimestamps = new Set(timeline.candidates.map((candidate) => candidate.timestampNs));
  const sensors = mode === "lidar-lidar" ? timeline.lidars : timeline.cameras;
  const topicByNormalized = new Map(sensors.map((sensor) => [sensor.topic.replace(/^\/+/, ""), sensor.topic]));
  return Object.fromEntries(Object.entries(state.selections).flatMap(([storedTopic, timestamps]) => {
    const topic = topicByNormalized.get(storedTopic.replace(/^\/+/, ""));
    if (!topic) return [];
    const filtered = [...new Set(timestamps.filter((timestamp) => validTimestamps.has(timestamp)))]
      .sort((left, right) => Number(BigInt(left) - BigInt(right)));
    return filtered.length ? [[topic, filtered]] : [];
  }));
}

export function normalizeManualSelections(selections: Record<string, string[]>, timeline: ManualFrameTimeline) {
  const sensors = [...timeline.lidars, ...timeline.cameras];
  const topicByNormalized = new Map(sensors.map((sensor) => [sensor.topic.replace(/^\/+/, ""), sensor.topic]));
  const normalized: Record<string, string[]> = {};
  Object.entries(selections).forEach(([storedTopic, timestamps]) => {
    const topic = topicByNormalized.get(storedTopic.replace(/^\/+/, ""));
    if (topic) normalized[topic] = [...new Set([...(normalized[topic] ?? []), ...timestamps])].sort((left, right) => Number(BigInt(left) - BigInt(right)));
  });
  return normalized;
}

export function copyMask(mask: ImageMask): ImageMask {
  return { width: mask.width, height: mask.height, data: mask.data.slice() };
}

export const canvasMaskViewCache = new WeakMap<ImageMask, ImageMask>();

export function canvasMaskView(mask: ImageMask | null): ImageMask | null {
  if (!mask) return null;
  const cached = canvasMaskViewCache.get(mask);
  if (cached) return cached;
  const view = { width: mask.width, height: mask.height } as ImageMask;
  // React 19 development diagnostics recursively enumerate changed props.
  // Hide the multi-million-pixel buffer so a commit only compares dimensions.
  Object.defineProperty(view, "data", { value: mask.data, enumerable: false });
  canvasMaskViewCache.set(mask, view);
  return view;
}

export function copyMaskLayers(layers: ImageMaskLayer[]): ImageMaskLayer[] {
  return layers.map((layer) => ({ ...layer, mask: copyMask(layer.mask) }));
}

export function combineMaskLayers(layers: ImageMaskLayer[]): ImageMask | null {
  let combined: ImageMask | null = null;
  for (const layer of layers) combined = mergeImageMasks(combined, layer.mask, "add");
  return combined;
}

export function applyManualMaskOverrides(base: ImageMask | null, added: ImageMask | null, removed: ImageMask | null): ImageMask | null {
  const template = base ?? added ?? removed;
  if (!template) return null;
  const result: ImageMask = {
    width: template.width,
    height: template.height,
    data: base?.data.slice() ?? new Uint8Array(template.width * template.height),
  };
  if (added) for (let index = 0; index < result.data.length; index += 1) {
    if (added.data[index]) result.data[index] = 1;
  }
  if (removed) for (let index = 0; index < result.data.length; index += 1) {
    if (removed.data[index]) result.data[index] = 0;
  }
  return countMask(result) ? result : null;
}

export function updateManualMaskOverridesRegion(
  previous: ImageMask | null,
  edited: ImageMask,
  added: ImageMask | null,
  removed: ImageMask | null,
  dirtyRects: MaskDirtyRect[],
) {
  const nextAdded: ImageMask = added
    ? added
    : { width: edited.width, height: edited.height, data: new Uint8Array(edited.width * edited.height) };
  const nextRemoved: ImageMask = removed
    ? removed
    : { width: edited.width, height: edited.height, data: new Uint8Array(edited.width * edited.height) };
  const previousData = previous?.data;
  for (const dirty of dirtyRects) {
    const left = Math.max(0, dirty.left);
    const right = Math.min(edited.width - 1, dirty.right);
    const top = Math.max(0, dirty.top);
    const bottom = Math.min(edited.height - 1, dirty.bottom);
    for (let y = top; y <= bottom; y += 1) for (let x = left; x <= right; x += 1) {
      const index = y * edited.width + x;
      const before = previousData?.[index] ?? 0;
      const after = edited.data[index];
      if (!before && after) {
        nextAdded.data[index] = 1;
        nextRemoved.data[index] = 0;
      } else if (before && !after) {
        nextRemoved.data[index] = 1;
        nextAdded.data[index] = 0;
      }
    }
  }
  return { added: nextAdded, removed: nextRemoved };
}

export type OptimizationResult = {
  storageKey?: string;
  matrix: number[];
  delta: number[];
  originalError: number;
  optimizedError: number;
  fullMatchMatrix?: number[];
  fullMatchError?: number;
  originalIoU?: number;
  fullMatchIoU?: number;
  optimizedIoU?: number;
  silhouetteMetric?: string;
  envelopePrecision?: number;
  maskCoverage?: number;
  originalInsideRatio?: number;
  fullMatchInsideRatio?: number;
  optimizedInsideRatio?: number;
  originalContainmentError?: number;
  fullMatchContainmentError?: number;
  optimizedContainmentError?: number;
  backgroundIntrusionRatio?: number;
  backgroundIntrusionCount?: number;
  occludedBackgroundCount?: number;
  medianEdgeError?: number;
  p90EdgeError?: number;
  partialMatchRatio?: number;
  partialRefinementAccepted?: boolean;
  fullMatchIterationCount?: number;
  pairCount: number;
  annotationIds: string[];
  annotationSavedAts: string[];
  groupKey: string;
  generatedAt: string;
  initializationMode?: ExtrinsicInitializationMode;
  iterationCount?: number;
  converged?: boolean;
  pnpInlierCount?: number;
  pnpReprojectionError?: number;
  algorithmVersion?: string;
  distortionModel?: ProjectionModel;
  distortionParameterCount?: number;
  qualityStatus?: "good" | "warning" | "poor" | "unverified";
  qualityWarnings?: string[];
  pairDiagnostics?: Array<{
    annotationId: string;
    insideRatio: number;
    containmentError: number;
    medianEdgeError: number;
    p90EdgeError: number;
    iou: number;
    envelopePrecision: number;
    maskCoverage: number;
    suspected: boolean;
  }>;
  suggestedReviewAnnotationIds?: string[];
};

export type ExtrinsicVersion = {
  id: string;
  groupKey: string;
  matrix: number[];
  createdAt: string;
  label: string;
  source: "imported" | "calculated" | "builtin";
  sourceName?: string;
  isInitial: boolean;
  optimization?: OptimizationResult;
  distortionModel?: ProjectionModel;
};

export type StoredExtrinsicVersion = Partial<ExtrinsicVersion> & Pick<ExtrinsicVersion, "groupKey" | "matrix"> & {
  importedAt?: string;
};

export type CalibrationObservation = {
  annotationId: string;
  cloud: CloudData;
  indices: Uint32Array;
  mask: ImageMask;
  intrinsic: number[];
  distortion: number[];
  distortionModel?: string;
};

export const ANNOTATION_DATABASE = "roc-calib-annotations";

export const ANNOTATION_STORE = "pairs";

export const CALIBRATION_STORE = "calibrations-by-model";

export const PAIRS_PER_PAGE = 6;

export const CALIBRATION_ALGORITHM_VERSION = "roc-calib-region-v1";

export const LIDAR_CALIBRATION_ALGORITHM_VERSION = "fpfh-ransac-multiseed-small-gicp-contour-v12";

export const LIDAR_CALIBRATION_COMPATIBLE_VERSIONS = new Set([
  LIDAR_CALIBRATION_ALGORITHM_VERSION,
  "fpfh-ransac-multiseed-small-gicp-contour-v11",
  "fpfh-ransac-multiseed-small-gicp-contour-v10",
  "fpfh-ransac-multiseed-small-gicp-contour-v9",
  "fpfh-ransac-small-gicp-contour-v8",
  "fpfh-ransac-small-gicp-v7",
  "imported-lidar-extrinsic-v1",
]);

export const LIDAR_SAVED_PAIR_PAGE_SIZE = 4;

export const RAW_COORDINATE_SYSTEM = "raw-distorted-v1";

export function defaultTaskGroupId(rigId: string) {
  return rigId === "e005-20260813" ? "parameter-group-2" : "parameter-group-1";
}

export function calibrationGroupKey(taskGroupId: string, rigId: string, lidarId: string, cameraId: string) {
  return `shared:${taskGroupId}:${rigId}:${lidarId}:${cameraId}`;
}

export function normalizedCalibrationKey(groupKey: string) {
  const fields = groupKey.split(":");
  if (fields.length >= 5) return calibrationGroupKey(fields.at(-4)!, fields.at(-3)!, fields.at(-2)!, fields.at(-1)!);
  if (fields.length >= 4) {
    const rigId = fields.at(-3)!;
    return calibrationGroupKey(defaultTaskGroupId(rigId), rigId, fields.at(-2)!, fields.at(-1)!);
  }
  return fields.length >= 2 ? calibrationGroupKey("parameter-group-1", "truck5-legacy", fields.at(-2)!, fields.at(-1)!) : groupKey;
}

export function calibrationModelKey(groupKey: string, model: string) {
  return `${normalizedCalibrationKey(groupKey)}|model=${model}`;
}

export function normalizeStoredExtrinsic(value: StoredExtrinsicVersion, index: number): ExtrinsicVersion {
  const groupKey = normalizedCalibrationKey(value.groupKey);
  const createdAt = value.createdAt ?? value.importedAt ?? new Date(index).toISOString();
  const source = value.source === "calculated" ? "calculated" : "imported";
  return {
    id: value.id ?? [groupKey, source, createdAt, index].join(":"),
    groupKey,
    matrix: value.matrix,
    createdAt,
    label: value.label?.replace("可见边缘精调", "最终结果") ?? (source === "calculated" ? "计算结果 · " + new Date(createdAt).toLocaleString("zh-CN") : "导入 · " + (value.sourceName ?? "已有外参")),
    source,
    sourceName: value.sourceName,
    isInitial: value.isInitial ?? false,
    optimization: value.optimization,
    distortionModel: value.distortionModel ?? value.optimization?.distortionModel,
  };
}

export function openAnnotationDatabase() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(ANNOTATION_DATABASE, 3);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(ANNOTATION_STORE)) {
        request.result.createObjectStore(ANNOTATION_STORE, { keyPath: "id" });
      }
      if (!request.result.objectStoreNames.contains(CALIBRATION_STORE)) {
        request.result.createObjectStore(CALIBRATION_STORE, { keyPath: "storageKey" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("无法打开浏览器标注库"));
  });
}

export async function loadStoredCalibrations() {
  const database = await openAnnotationDatabase();
  try {
    return await new Promise<OptimizationResult[]>((resolve, reject) => {
      const request = database.transaction(CALIBRATION_STORE, "readonly").objectStore(CALIBRATION_STORE).getAll();
      request.onsuccess = () => resolve(request.result as OptimizationResult[]);
      request.onerror = () => reject(request.error ?? new Error("无法读取共享外参"));
    });
  } finally {
    database.close();
  }
}

export async function storeCalibration(calibration: OptimizationResult) {
  const database = await openAnnotationDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(CALIBRATION_STORE, "readwrite");
      transaction.objectStore(CALIBRATION_STORE).put(calibration);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("无法持久保存共享外参"));
      transaction.onabort = () => reject(transaction.error ?? new Error("保存共享外参已中止"));
    });
  } finally {
    database.close();
  }
}

export async function loadStoredAnnotations() {
  const database = await openAnnotationDatabase();
  try {
    const annotations = await new Promise<SavedAnnotation[]>((resolve, reject) => {
      const request = database.transaction(ANNOTATION_STORE, "readonly").objectStore(ANNOTATION_STORE).getAll();
      request.onsuccess = () => resolve(request.result as SavedAnnotation[]);
      request.onerror = () => reject(request.error ?? new Error("无法读取已保存标注"));
    });
    return annotations
      .map((annotation, index) => ({
        ...annotation,
        useForOptimization: annotation.useForOptimization ?? true,
        savedAt: annotation.savedAt ?? new Date(index).toISOString(),
      }))
      .sort((left, right) => left.savedAt.localeCompare(right.savedAt));
  } finally {
    database.close();
  }
}

export async function storeAnnotation(annotation: SavedAnnotation) {
  const database = await openAnnotationDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(ANNOTATION_STORE, "readwrite");
      transaction.objectStore(ANNOTATION_STORE).put(annotation);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("无法持久保存标注"));
      transaction.onabort = () => reject(transaction.error ?? new Error("保存标注已中止"));
    });
  } finally {
    database.close();
  }
}

export async function deleteStoredAnnotation(annotationId: string) {
  const database = await openAnnotationDatabase();
  try {
    await new Promise<void>((resolve, reject) => {
      const transaction = database.transaction(ANNOTATION_STORE, "readwrite");
      transaction.objectStore(ANNOTATION_STORE).delete(annotationId);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error ?? new Error("无法删除标注"));
      transaction.onabort = () => reject(transaction.error ?? new Error("删除标注已中止"));
    });
  } finally {
    database.close();
  }
}

export function formatPoints(value: number) {
  return new Intl.NumberFormat("zh-CN").format(value);
}

export function formatFileSize(value: number) {
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MB`;
  return `${Math.max(0, value / 1024).toFixed(1)} KB`;
}

export function formatTimestamp(timestampNs: string) {
  const date = new Date(Number(BigInt(timestampNs) / 1_000_000n));
  return new Intl.DateTimeFormat("zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    fractionalSecondDigits: 3,
    hour12: false,
  }).format(date);
}

export function timestampDeltaMs(sourceTimestampNs: string, targetTimestampNs: string) {
  try {
    const source = BigInt(sourceTimestampNs);
    const target = BigInt(targetTimestampNs);
    return Number(source >= target ? source - target : target - source) / 1e6;
  } catch {
    return Number.POSITIVE_INFINITY;
  }
}

export async function readCloud(url: string, signal?: AbortSignal): Promise<CloudData> {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`点云读取失败：${response.status}`);
  const buffer = await response.arrayBuffer();
  const view = new DataView(buffer);
  const magic = String.fromCharCode(...new Uint8Array(buffer, 0, 4));
  if (magic !== "ACP1") throw new Error("无法识别点云数据格式");
  const count = view.getUint32(4, true);
  const timestampNs = view.getBigInt64(8, true);
  if (buffer.byteLength !== 16 + count * 16) throw new Error("点云数据不完整");
  const positions = new Float32Array(count * 3);
  const stable = new Uint8Array(count);
  const ground = new Uint8Array(count);
  let hasGroundLabels = false;
  for (let index = 0; index < count; index += 1) {
    const source = 16 + index * 16;
    positions[index * 3] = view.getFloat32(source, true);
    positions[index * 3 + 1] = view.getFloat32(source + 4, true);
    positions[index * 3 + 2] = view.getFloat32(source + 8, true);
    stable[index] = view.getUint8(source + 12);
    ground[index] = view.getUint8(source + 13);
    hasGroundLabels ||= view.getUint8(source + 14) === 1;
  }
  return { positions, stable, ground, hasGroundLabels, timestampNs };
}

export type DisplayGroundPlane = { normal: [number, number, number]; offset: number };

export function estimateDisplayGroundPlane(cloud: CloudData | null): DisplayGroundPlane | null {
  if (!cloud || cloud.positions.length < 240) return null;
  const pointCount = cloud.positions.length / 3;
  const stride = Math.max(1, Math.ceil(pointCount / 6500));
  const points: Array<[number, number, number]> = [];
  for (let index = 0; index < pointCount; index += stride) {
    if (cloud.stable[index] === 2) continue;
    const x = cloud.positions[index * 3], y = cloud.positions[index * 3 + 1], z = cloud.positions[index * 3 + 2];
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && Math.hypot(x, y) <= 45) points.push([x, y, z]);
  }
  if (points.length < 80) return null;
  let seed = 0x6d2b79f5;
  const randomIndex = () => {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed % points.length;
  };
  let bestNormal: [number, number, number] | null = null;
  let bestOffset = 0;
  let bestCount = 0;
  for (let attempt = 0; attempt < 360; attempt += 1) {
    const a = points[randomIndex()], b = points[randomIndex()], c = points[randomIndex()];
    const abx = b[0] - a[0], aby = b[1] - a[1], abz = b[2] - a[2];
    const acx = c[0] - a[0], acy = c[1] - a[1], acz = c[2] - a[2];
    let nx = aby * acz - abz * acy, ny = abz * acx - abx * acz, nz = abx * acy - aby * acx;
    const length = Math.hypot(nx, ny, nz);
    if (length < 1e-7) continue;
    nx /= length; ny /= length; nz /= length;
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    if (nz < .55) continue;
    const offset = nx * a[0] + ny * a[1] + nz * a[2];
    let count = 0;
    for (const point of points) {
      if (Math.abs(nx * point[0] + ny * point[1] + nz * point[2] - offset) <= .10) count += 1;
    }
    if (count > bestCount) { bestCount = count; bestNormal = [nx, ny, nz]; bestOffset = offset; }
  }
  if (!bestNormal || bestCount < Math.max(45, points.length * .06)) return null;
  const offsets = points
    .map((point) => bestNormal![0] * point[0] + bestNormal![1] * point[1] + bestNormal![2] * point[2])
    .filter((offset) => Math.abs(offset - bestOffset) <= .12)
    .sort((left, right) => left - right);
  return { normal: bestNormal, offset: offsets[Math.floor(offsets.length / 2)] ?? bestOffset };
}

export function mergeAlignedClouds(
  source: CloudData,
  target: CloudData,
  referenceFromSource: number[],
  referenceFromTarget: number[],
  densityPercent = 100,
  visibleLayer: "both" | "source" | "target" = "both",
): CloudData {
  const keep = (index: number) => densityPercent >= 100 || ((Math.imul(index + 1, 2654435761) >>> 0) % 100) < densityPercent;
  const sourceIndices = visibleLayer === "target" ? [] : Array.from({ length: source.positions.length / 3 }, (_, index) => index).filter(keep);
  const targetIndices = visibleLayer === "source" ? [] : Array.from({ length: target.positions.length / 3 }, (_, index) => index).filter(keep);
  const positions = new Float32Array((sourceIndices.length + targetIndices.length) * 3);
  const append = (cloud: CloudData, indices: number[], matrix: number[], outputStart: number) => {
    indices.forEach((index, outputIndex) => {
      const input = index * 3, output = (outputStart + outputIndex) * 3;
      const x = cloud.positions[input], y = cloud.positions[input + 1], z = cloud.positions[input + 2];
      positions[output] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
      positions[output + 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
      positions[output + 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
    });
  };
  append(source, sourceIndices, referenceFromSource, 0);
  append(target, targetIndices, referenceFromTarget, sourceIndices.length);
  const stable = new Uint8Array(sourceIndices.length + targetIndices.length);
  stable.fill(1, 0, sourceIndices.length);
  return {
    positions,
    stable,
    ground: new Uint8Array(sourceIndices.length + targetIndices.length),
    hasGroundLabels: false,
    timestampNs: target.timestampNs,
  };
}

export function subsetCloud(cloud: CloudData, selectedIndices: Uint32Array): CloudData {
  const valid = Array.from(selectedIndices).filter((index) => index >= 0 && index < cloud.positions.length / 3 && cloud.stable[index] !== 2);
  const positions = new Float32Array(valid.length * 3);
  const stable = new Uint8Array(valid.length);
  const ground = new Uint8Array(valid.length);
  valid.forEach((index, outputIndex) => {
    positions.set(cloud.positions.subarray(index * 3, index * 3 + 3), outputIndex * 3);
    stable[outputIndex] = cloud.stable[index];
    ground[outputIndex] = cloud.ground[index];
  });
  return { positions, stable, ground, hasGroundLabels: cloud.hasGroundLabels, timestampNs: cloud.timestampNs };
}

export function transformedSubsetCloud(cloud: CloudData, selectedIndices: Uint32Array, matrix: number[]): CloudData {
  const subset = subsetCloud(cloud, selectedIndices);
  return transformCloudData(subset, matrix);
}

export function transformCloudData(cloud: CloudData, matrix: number[]): CloudData {
  const positions = new Float32Array(cloud.positions.length);
  for (let offset = 0; offset < cloud.positions.length; offset += 3) {
    const x = cloud.positions[offset], y = cloud.positions[offset + 1], z = cloud.positions[offset + 2];
    positions[offset] = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
    positions[offset + 1] = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
    positions[offset + 2] = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
  }
  return { ...cloud, positions };
}

export function correspondingRegionIndices(
  selectedCloud: CloudData | null,
  selectedIndices: Uint32Array,
  candidateCloud: CloudData | null,
  candidateFromSelected: number[] | null,
  padding: number,
) {
  if (!selectedCloud || !candidateCloud || !candidateFromSelected || selectedIndices.length < 3) return new Uint32Array();
  const axes: number[][] = [[], [], []];
  selectedIndices.forEach((index) => {
    if (index < 0 || index >= selectedCloud.positions.length / 3) return;
    const offset = index * 3;
    const x = selectedCloud.positions[offset], y = selectedCloud.positions[offset + 1], z = selectedCloud.positions[offset + 2];
    axes[0].push(candidateFromSelected[0] * x + candidateFromSelected[1] * y + candidateFromSelected[2] * z + candidateFromSelected[3]);
    axes[1].push(candidateFromSelected[4] * x + candidateFromSelected[5] * y + candidateFromSelected[6] * z + candidateFromSelected[7]);
    axes[2].push(candidateFromSelected[8] * x + candidateFromSelected[9] * y + candidateFromSelected[10] * z + candidateFromSelected[11]);
  });
  if (axes[0].length < 3) return new Uint32Array();
  axes.forEach((axis) => axis.sort((left, right) => left - right));
  const lowerIndex = Math.floor((axes[0].length - 1) * .02);
  const upperIndex = Math.ceil((axes[0].length - 1) * .98);
  const lower = axes.map((axis) => axis[lowerIndex] - padding);
  const upper = axes.map((axis) => axis[upperIndex] + padding);
  const matches: number[] = [];
  for (let index = 0; index < candidateCloud.positions.length / 3; index += 1) {
    const offset = index * 3;
    const x = candidateCloud.positions[offset], y = candidateCloud.positions[offset + 1], z = candidateCloud.positions[offset + 2];
    if (x >= lower[0] && x <= upper[0] && y >= lower[1] && y <= upper[1] && z >= lower[2] && z <= upper[2]) matches.push(index);
  }
  return Uint32Array.from(matches);
}

export function multiply4(left: number[], right: number[]) {
  const output = new Array<number>(16).fill(0);
  for (let row = 0; row < 4; row += 1) {
    for (let column = 0; column < 4; column += 1) {
      for (let index = 0; index < 4; index += 1) {
        output[row * 4 + column] += left[row * 4 + index] * right[index * 4 + column];
      }
    }
  }
  return output;
}

export const FORWARD_FACING_LIDAR_TO_CAMERA = [
  0, -1, 0, 0,
  0, 0, -1, 0,
  1, 0, 0, 0,
  0, 0, 0, 1,
];

export function deltaMatrix(delta: number[]) {
  const [rx, ry, rz, tx, ty, tz] = delta;
  const cx = Math.cos(rx), sx = Math.sin(rx), cy = Math.cos(ry), sy = Math.sin(ry), cz = Math.cos(rz), sz = Math.sin(rz);
  return [
    cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx, tx,
    sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx, ty,
    -sy, cy * sx, cy * cx, tz,
    0, 0, 0, 1,
  ];
}

export function invertRigid4(matrix: number[]) {
  const tx = matrix[3], ty = matrix[7], tz = matrix[11];
  return [
    matrix[0], matrix[4], matrix[8], -(matrix[0] * tx + matrix[4] * ty + matrix[8] * tz),
    matrix[1], matrix[5], matrix[9], -(matrix[1] * tx + matrix[5] * ty + matrix[9] * tz),
    matrix[2], matrix[6], matrix[10], -(matrix[2] * tx + matrix[6] * ty + matrix[10] * tz),
    0, 0, 0, 1,
  ];
}

export function matrixDeltaFromBase(matrix: number[], base: number[]) {
  const relative = multiply4(matrix, invertRigid4(base));
  const ry = Math.asin(Math.max(-1, Math.min(1, -relative[8])));
  const cosineY = Math.cos(ry);
  const rx = Math.abs(cosineY) > 1e-6 ? Math.atan2(relative[9], relative[10]) : 0;
  const rz = Math.abs(cosineY) > 1e-6 ? Math.atan2(relative[4], relative[0]) : Math.atan2(-relative[1], relative[5]);
  return [rx, ry, rz, relative[3], relative[7], relative[11]];
}

export function projectPixel(pointIndex: number, cloud: CloudData, matrix: number[], intrinsic: number[], distortion: number[], distortionModel = "rational") {
  const x = cloud.positions[pointIndex * 3], y = cloud.positions[pointIndex * 3 + 1], z = cloud.positions[pointIndex * 3 + 2];
  const cx = matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3];
  const cy = matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7];
  const cz = matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11];
  if (cz <= 0.1) return null;
  const nx = cx / cz, ny = cy / cz, r2 = nx * nx + ny * ny, r4 = r2 * r2, r6 = r4 * r2;
  if (distortionModel === "fisheye") {
    const radius = Math.sqrt(r2);
    const theta = Math.atan(radius), theta2 = theta * theta;
    const [k1 = 0, k2 = 0, k3 = 0, k4 = 0] = distortion;
    const thetaDistorted = theta * (1 + k1 * theta2 + k2 * theta2 ** 2 + k3 * theta2 ** 3 + k4 * theta2 ** 4);
    const scale = radius > 1e-12 ? thetaDistorted / radius : 1;
    return [intrinsic[0] * nx * scale + intrinsic[2], intrinsic[4] * ny * scale + intrinsic[5]];
  }
  const [k1 = 0, k2 = 0, p1 = 0, p2 = 0, k3 = 0, k4 = 0, k5 = 0, k6 = 0] = distortion;
  const denominator = distortionModel === "radtan" ? 1 : 1 + k4 * r2 + k5 * r4 + k6 * r6;
  if (Math.abs(denominator) < 1e-8) return null;
  const radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / denominator;
  const dx = nx * radial + 2 * p1 * nx * ny + p2 * (r2 + 2 * nx * nx);
  const dy = ny * radial + p1 * (r2 + 2 * ny * ny) + 2 * p2 * nx * ny;
  return [intrinsic[0] * dx + intrinsic[2], intrinsic[4] * dy + intrinsic[5]];
}

export function rawProjectionCalibration<T extends {
  intrinsic: number[];
  distortion: number[];
  distortionModel?: string;
  rawIntrinsic?: number[];
  rawDistortion?: number[];
  rawDistortionModel?: string;
  coordinateSystem?: string;
}>(calibration: T, modelOverride?: ProjectionModel): T {
  const rawDistortion = calibration.rawDistortion ?? calibration.distortion;
  const distortion = modelOverride === "fisheye"
    ? rawDistortion.slice(0, 4)
    : modelOverride === "radtan"
      ? rawDistortion.slice(0, rawDistortion.length >= 5 ? 5 : 4)
      : modelOverride === "rational"
        ? rawDistortion.slice(0, 8)
        : rawDistortion;
  return {
    ...calibration,
    intrinsic: calibration.rawIntrinsic ?? calibration.intrinsic,
    distortion,
    distortionModel: modelOverride ?? calibration.rawDistortionModel ?? calibration.distortionModel ?? "rational",
    coordinateSystem: RAW_COORDINATE_SYSTEM,
  };
}

export type GroundPlane = { a: number; b: number; c: number; lowerMargin: number; upperMargin: number };

export const groundPlaneCache = new WeakMap<CloudData, Map<boolean, GroundPlane>>();

export function solveGroundPlane(points: Array<{ x: number; y: number; z: number }>, fallback: GroundPlane) {
  if (points.length < 6) return fallback;
  let sxx = 0, syy = 0, sxy = 0, sx = 0, sy = 0, sxz = 0, syz = 0, sz = 0;
  for (const point of points) {
    sxx += point.x * point.x; syy += point.y * point.y; sxy += point.x * point.y;
    sx += point.x; sy += point.y; sxz += point.x * point.z; syz += point.y * point.z; sz += point.z;
  }
  const matrix = [sxx, sxy, sx, sxy, syy, sy, sx, sy, points.length];
  const vector = [sxz, syz, sz];
  for (let pivot = 0; pivot < 3; pivot += 1) {
    let best = pivot;
    for (let row = pivot + 1; row < 3; row += 1) if (Math.abs(matrix[row * 3 + pivot]) > Math.abs(matrix[best * 3 + pivot])) best = row;
    if (Math.abs(matrix[best * 3 + pivot]) < 1e-8) return fallback;
    for (let column = pivot; column < 3; column += 1) [matrix[pivot * 3 + column], matrix[best * 3 + column]] = [matrix[best * 3 + column], matrix[pivot * 3 + column]];
    [vector[pivot], vector[best]] = [vector[best], vector[pivot]];
    const scale = matrix[pivot * 3 + pivot];
    for (let column = pivot; column < 3; column += 1) matrix[pivot * 3 + column] /= scale;
    vector[pivot] /= scale;
    for (let row = 0; row < 3; row += 1) {
      if (row === pivot) continue;
      const factor = matrix[row * 3 + pivot];
      for (let column = pivot; column < 3; column += 1) matrix[row * 3 + column] -= factor * matrix[pivot * 3 + column];
      vector[row] -= factor * vector[pivot];
    }
  }
  // Reject implausible slopes: a failed fit must never eat elevated objects.
  if (Math.hypot(vector[0], vector[1]) > 0.22) return fallback;
  return {
    a: vector[0], b: vector[1], c: vector[2],
    lowerMargin: fallback.lowerMargin, upperMargin: fallback.upperMargin,
  };
}

export function planeFromThreePoints(
  first: { x: number; y: number; z: number },
  second: { x: number; y: number; z: number },
  third: { x: number; y: number; z: number },
  fallback: GroundPlane,
) {
  const denominator = first.x * (second.y - third.y) + second.x * (third.y - first.y) + third.x * (first.y - second.y);
  if (Math.abs(denominator) < 1e-6) return null;
  const a = (first.z * (second.y - third.y) + second.z * (third.y - first.y) + third.z * (first.y - second.y)) / denominator;
  const b = (first.x * (second.z - third.z) + second.x * (third.z - first.z) + third.x * (first.z - second.z)) / denominator;
  const c = (first.x * (third.y * second.z - second.y * third.z) + second.x * (first.y * third.z - third.y * first.z) + third.x * (second.y * first.z - first.y * second.z)) / denominator;
  if (!Number.isFinite(a + b + c) || Math.hypot(a, b) > 0.18) return null;
  return { a, b, c, lowerMargin: fallback.lowerMargin, upperMargin: fallback.upperMargin };
}

export function estimateGroundPlane(cloud: CloudData, stableOnly: boolean): GroundPlane {
  const cached = groundPlaneCache.get(cloud)?.get(stableOnly);
  if (cached) return cached;
  const heights: number[] = [];
  const cells = new Map<string, { x: number; y: number; z: number }>();
  for (let index = 0; index < cloud.stable.length; index += 1) {
      if (cloud.stable[index] === 2) continue;
    if (stableOnly && cloud.stable[index] !== 1) continue;
    const offset = index * 3;
    const x = cloud.positions[offset], y = cloud.positions[offset + 1], z = cloud.positions[offset + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    heights.push(z);
    if (Math.hypot(x, y) < 3 || Math.hypot(x, y) > 55) continue;
    const key = `${Math.floor(x / 2)},${Math.floor(y / 2)}`;
    const current = cells.get(key);
    if (!current || z < current.z) cells.set(key, { x, y, z });
  }
  heights.sort((left, right) => left - right);
  const fallback = {
    a: 0, b: 0,
    c: heights[Math.floor(Math.max(0, heights.length - 1) * 0.03)] ?? -1.2,
    lowerMargin: 0.08, upperMargin: 0.07,
  };
  const candidates = [...cells.values()];
  let plane = fallback;
  let bestInliers: typeof candidates = [];
  // Deterministic RANSAC over spatial-cell minima. A wall foot or ditch can no
  // longer tilt the entire ground model unless it is supported across the scene.
  for (let iteration = 0; iteration < Math.min(320, candidates.length * 5); iteration += 1) {
    if (candidates.length < 3) break;
    const first = candidates[(iteration * 37 + 3) % candidates.length];
    const second = candidates[(iteration * 73 + 17) % candidates.length];
    const third = candidates[(iteration * 109 + 41) % candidates.length];
    const candidatePlane = planeFromThreePoints(first, second, third, fallback);
    if (!candidatePlane) continue;
    const inliers = candidates.filter((point) =>
      Math.abs(point.z - (candidatePlane.a * point.x + candidatePlane.b * point.y + candidatePlane.c)) <= 0.10,
    );
    if (inliers.length > bestInliers.length) { bestInliers = inliers; plane = candidatePlane; }
  }
  if (bestInliers.length >= 6) {
    plane = solveGroundPlane(bestInliers, plane);
    const tightInliers = bestInliers.filter((point) =>
      Math.abs(point.z - (plane.a * point.x + plane.b * point.y + plane.c)) <= 0.08,
    );
    if (tightInliers.length >= 6) plane = solveGroundPlane(tightInliers, plane);
  }
  const cache = groundPlaneCache.get(cloud) ?? new Map<boolean, GroundPlane>();
  cache.set(stableOnly, plane);
  groundPlaneCache.set(cloud, cache);
  return plane;
}

export function isGroundPoint(cloud: CloudData, index: number, stableOnly: boolean) {
  if (cloud.hasGroundLabels) return cloud.ground[index] === 1;
  const offset = index * 3;
  const x = cloud.positions[offset], y = cloud.positions[offset + 1], z = cloud.positions[offset + 2];
  const plane = estimateGroundPlane(cloud, stableOnly);
  const residual = z - (plane.a * x + plane.b * y + plane.c);
  return residual >= -plane.lowerMargin && residual <= plane.upperMargin;
}

export function maskEdgeDistance(mask: ImageMask) {
  const distance = new Float32Array(mask.width * mask.height);
  distance.fill(1e4);
  for (let y = 1; y + 1 < mask.height; y += 1) for (let x = 1; x + 1 < mask.width; x += 1) {
    const i = y * mask.width + x;
    if (mask.data[i] && (!mask.data[i - 1] || !mask.data[i + 1] || !mask.data[i - mask.width] || !mask.data[i + mask.width])) distance[i] = 0;
  }
  const diagonal = Math.SQRT2;
  for (let y = 1; y < mask.height; y += 1) for (let x = 1; x < mask.width; x += 1) {
    const i = y * mask.width + x;
    distance[i] = Math.min(distance[i], distance[i - 1] + 1, distance[i - mask.width] + 1, distance[i - mask.width - 1] + diagonal);
  }
  for (let y = mask.height - 2; y >= 0; y -= 1) for (let x = mask.width - 2; x >= 0; x -= 1) {
    const i = y * mask.width + x;
    distance[i] = Math.min(distance[i], distance[i + 1] + 1, distance[i + mask.width] + 1, distance[i + mask.width + 1] + diagonal);
  }
  return distance;
}

export function convexHull(points: Array<[number, number]>) {
  const sorted = [...points].sort((left, right) => left[0] - right[0] || left[1] - right[1]);
  if (sorted.length <= 2) return sorted;
  const cross = (origin: [number, number], first: [number, number], second: [number, number]) =>
    (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (second[0] - origin[0]);
  const lower: Array<[number, number]> = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower.at(-2)!, lower.at(-1)!, point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper: Array<[number, number]> = [];
  for (const point of sorted.reverse()) {
    while (upper.length >= 2 && cross(upper.at(-2)!, upper.at(-1)!, point) <= 0) upper.pop();
    upper.push(point);
  }
  lower.pop(); upper.pop();
  return lower.concat(upper);
}

export function projectionColor(depth: number) {
  if (depth < 10) return "#fde725";
  if (depth < 20) return "#7ad151";
  if (depth < 35) return "#22a884";
  if (depth < 55) return "#2a788e";
  return "#414487";
}

export const activeChordPanViewport: { current: HTMLDivElement | null } = { current: null };

export function OrthographicCloudViews({ source, target }: { source: CloudData; target: CloudData }) {
  const [rotation, setRotation] = useState({ yaw: 0, pitch: 0 });
  const [zoom, setZoom] = useState(1);
  const bounds = useMemo(() => {
    const lower = [Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY, Number.POSITIVE_INFINITY];
    const upper = [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
    [source, target].forEach((cloud) => {
      for (let offset = 0; offset < cloud.positions.length; offset += 3) {
        for (let axis = 0; axis < 3; axis += 1) {
          lower[axis] = Math.min(lower[axis], cloud.positions[offset + axis]);
          upper[axis] = Math.max(upper[axis], cloud.positions[offset + axis]);
        }
      }
    });
    const center = lower.map((value, axis) => Number.isFinite(value) ? (value + upper[axis]) / 2 : 0) as [number, number, number];
    const span = Math.max(...lower.map((value, axis) => Number.isFinite(value) ? upper[axis] - value : 0), .5) * 1.12;
    return { center, span };
  }, [source, target]);
  const rotate = (deltaX: number, deltaY: number) => setRotation((value) => ({
    yaw: value.yaw + deltaX * .008,
    pitch: Math.max(-Math.PI / 2, Math.min(Math.PI / 2, value.pitch + deltaY * .008)),
  }));
  const changeZoom = (delta: number) => setZoom((value) => Math.max(.35, Math.min(8, value * Math.exp(-delta * .0015))));
  const reset = () => { setRotation({ yaw: 0, pitch: 0 }); setZoom(1); };
  const interaction = { rotation, zoom, onRotate: rotate, onZoom: changeZoom, onReset: reset };
  return <div className="orthographic-cloud-views">
    <OrthographicCloudView source={source} target={target} plane="xy" label="俯视" {...bounds} {...interaction} />
    <OrthographicCloudView source={source} target={target} plane="xz" label="正视" {...bounds} {...interaction} />
    <OrthographicCloudView source={source} target={target} plane="yz" label="侧视" {...bounds} {...interaction} />
  </div>;
}

export function ManualLidarCloudPreview({
  url,
  displayTransform,
  onLoadingChange,
  onError,
}: {
  url: string;
  displayTransform: number[] | null;
  onLoadingChange: (loading: boolean) => void;
  onError: (message: string) => void;
}) {
  const [cloud, setCloud] = useState<CloudData | null>(null);
  const [viewRequest, setViewRequest] = useState<ViewRequest>({ mode: "near", token: 0 });
  const displayCloud = useMemo(() => cloud && displayTransform
    ? transformCloudData(cloud, displayTransform)
    : cloud, [cloud, displayTransform]);
  const cloudList = useMemo(() => [displayCloud], [displayCloud]);

  useEffect(() => {
    let active = true;
    onLoadingChange(true);
    readCloud(url)
      .then((value) => { if (active) setCloud(value); })
      .catch((reason) => { if (active) onError(reason instanceof Error ? reason.message : "点云预览加载失败"); })
      .finally(() => { if (active) onLoadingChange(false); });
    return () => { active = false; };
  }, [onError, onLoadingChange, url]);

  return <>
    {cloud && <PointCloudViewport
      clouds={cloudList}
      frameIndex={0}
      stableOnly={false}
      hideGround={false}
      pointSize={0.065}
      brushSize={24}
      brushDepthTolerance={0.8}
      interactionMode="navigate"
      selectedIndices={new Uint32Array()}
      clearSelectionToken={0}
      viewRequest={viewRequest}
      onSelection={() => undefined}
      onPointEdit={() => undefined}
      onInteractionModeChange={() => undefined}
      preserveViewOnCloudChange
      uniformPointColor={0xffd84d}
    />}
    <div className="manual-lidar-navigation">
      <span>左键旋转 · Shift+左键或右键平移 · 滚轮缩放</span>
      <button onClick={() => setViewRequest((value) => ({ mode: "near", token: value.token + 1 }))}>复位视角</button>
    </div>
  </>;
}

export const MASK_OVERLAY_RGB = [255, 0, 128] as const;

export const LIDAR_SELECTION_COLOR = 0x2457ff;

export const MASK_OUTLINE_RADIUS = 1;

export const MASK_OUTLINE_ALPHA = 0.55;

export function isMaskOutline(mask: ImageMask, x: number, y: number) {
  const index = y * mask.width + x;
  if (!mask.data[index]) return false;
  for (let offset = 1; offset <= MASK_OUTLINE_RADIUS; offset += 1) {
    if (x - offset < 0 || x + offset >= mask.width || y - offset < 0 || y + offset >= mask.height ||
        !mask.data[index - offset] || !mask.data[index + offset] ||
        !mask.data[index - offset * mask.width] || !mask.data[index + offset * mask.width]) return true;
  }
  return false;
}

export function renderMaskCanvas(
  canvas: HTMLCanvasElement,
  visibleMask: ImageMask | null,
  opacity: number,
  secondaryMask: ImageMask | null = null,
) {
  const width = visibleMask?.width ?? secondaryMask?.width ?? canvas.width;
  const height = visibleMask?.height ?? secondaryMask?.height ?? canvas.height;
  if (canvas.width !== width) canvas.width = width;
  if (canvas.height !== height) canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return;
  context.clearRect(0, 0, width, height);
  if (!visibleMask && !secondaryMask) return;
  const imageData = context.createImageData(width, height);
  if (secondaryMask) for (let index = 0; index < secondaryMask.data.length; index += 1) {
    if (!secondaryMask.data[index]) continue;
    const offset = index * 4;
    imageData.data[offset] = MASK_OVERLAY_RGB[0];
    imageData.data[offset + 1] = MASK_OVERLAY_RGB[1];
    imageData.data[offset + 2] = MASK_OVERLAY_RGB[2];
    imageData.data[offset + 3] = Math.round(255 * Math.min(opacity * 0.24, 0.2));
  }
  if (visibleMask) for (let index = 0; index < visibleMask.data.length; index += 1) {
    if (!visibleMask.data[index]) continue;
    const offset = index * 4;
    imageData.data[offset] = MASK_OVERLAY_RGB[0];
    imageData.data[offset + 1] = MASK_OVERLAY_RGB[1];
    imageData.data[offset + 2] = MASK_OVERLAY_RGB[2];
    imageData.data[offset + 3] = Math.round(255 * opacity);
  }
  if (visibleMask) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) {
    const index = y * width + x;
    if (!visibleMask.data[index]) continue;
    if (!isMaskOutline(visibleMask, x, y)) continue;
    const offset = index * 4;
    imageData.data[offset] = 255;
    imageData.data[offset + 1] = 255;
    imageData.data[offset + 2] = 255;
    imageData.data[offset + 3] = Math.round(255 * MASK_OUTLINE_ALPHA);
  }
  // One replacement write is essential: a second transparent ImageData write
  // would erase the previously drawn fill instead of alpha-compositing it.
  context.putImageData(imageData, 0, 0);
}

export function renderMaskCanvasRegion(
  canvas: HTMLCanvasElement,
  visibleMask: ImageMask,
  dirty: { left: number; top: number; right: number; bottom: number },
  opacity: number,
) {
  if (canvas.width !== visibleMask.width || canvas.height !== visibleMask.height) {
    renderMaskCanvas(canvas, visibleMask, opacity);
    return;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  const left = Math.max(0, dirty.left - MASK_OUTLINE_RADIUS);
  const top = Math.max(0, dirty.top - MASK_OUTLINE_RADIUS);
  const right = Math.min(visibleMask.width - 1, dirty.right + MASK_OUTLINE_RADIUS);
  const bottom = Math.min(visibleMask.height - 1, dirty.bottom + MASK_OUTLINE_RADIUS);
  const width = right - left + 1;
  const height = bottom - top + 1;
  if (width <= 0 || height <= 0) return;
  const imageData = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) {
    const sourceRow = (top + y) * visibleMask.width + left;
    for (let x = 0; x < width; x += 1) {
      if (!visibleMask.data[sourceRow + x]) continue;
      const offset = (y * width + x) * 4;
      imageData.data[offset] = MASK_OVERLAY_RGB[0];
      imageData.data[offset + 1] = MASK_OVERLAY_RGB[1];
      imageData.data[offset + 2] = MASK_OVERLAY_RGB[2];
      imageData.data[offset + 3] = Math.round(255 * opacity);
      if (isMaskOutline(visibleMask, left + x, top + y)) {
        imageData.data[offset] = 255;
        imageData.data[offset + 1] = 255;
        imageData.data[offset + 2] = 255;
        imageData.data[offset + 3] = Math.round(255 * MASK_OUTLINE_ALPHA);
      }
    }
  }
  context.clearRect(left, top, width, height);
  context.putImageData(imageData, left, top);
}
