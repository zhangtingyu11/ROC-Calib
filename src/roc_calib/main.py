from __future__ import annotations

import asyncio
from contextlib import nullcontext
import json
import os
import queue
import sqlite3
import threading
import tomllib
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field
try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    torch = None
    build_sam2 = None
    SAM2ImagePredictor = None
from .calibration import CalibrationRequest, release_preparation_batch
from .calibration_worker import optimize_isolated as optimize_calibration
from .groups import CameraGroupState, GROUP_ROOT, LidarGroupState, PREPARED_ROOT, GroupCreate, GroupImport, GroupState, UploadCreate, _cached_dataset, append_upload_chunk, create_group, delete_group, existing_camera_selections, finalize_upload, import_datasets, initialize as initialize_groups, initialize_upload, list_bag_directory, list_groups, merge_manual_selection, protected_frame_selections, read_camera_state, read_domain_state_status, read_lidar_state, read_manual_selection, read_state, read_state_status, replace_manual_selection, toggle_manual_selection, write_camera_state, write_lidar_state, write_state
from .intrinsics import import_intrinsics_path, list_intrinsics_directory, read_intrinsics, refresh_imported_intrinsics, resolve_undistorted_frame
from .lidar_calibration import LidarFramePairRequest, LidarGraphRequest, LidarPairRequest, build_lidar_frame_pairs, optimize_lidar_pair, solve_lidar_graph
from .paired_datasets import list_paired_directory, prepare_paired_dataset
from .preparation import manual_frame_candidates, manual_frame_preview, manual_lidar_cloud, manual_lidar_preview, prepare_bag
from .storage import STORAGE


PUBLIC_ROOT = Path(os.environ.get("AUTOCALIB_PUBLIC_ROOT", "/workspace/web/public")).resolve()
CHECKPOINT = os.environ.get("SAM2_CHECKPOINT", "/models/sam2.1_hiera_base_plus.pt")
MODEL_CONFIG = os.environ.get("SAM2_MODEL_CONFIG", "configs/sam2.1/sam2.1_hiera_b+.yaml")
DEVICE = os.environ.get("SAM2_DEVICE", "cuda:0")
MAX_CACHE_ITEMS = int(os.environ.get("SAM2_IMAGE_CACHE", "4"))
LIDAR_CALIBRATION_ENABLED = os.environ.get("AUTOCALIB_ENABLE_LIDAR_CALIBRATION", "1").lower() in {"1", "true", "yes", "on"}
LIDAR_EXTRINSICS_TOML = Path(os.environ.get("ROC_CALIB_LIDAR_CONFIG", "/data/intrinsics/lidar.toml"))
LIDAR_EXTRINSIC_SECTIONS = {
    "back": "BackLidarToCar",
    "front": "FrontLidarToCar",
    "left": "LeftLidarToCar",
    "main": "MianLidarToCar",
    "right": "RightLidarToCar",
}
_PREPARE_LOCKS: dict[str, threading.Lock] = {}
_PREPARE_LOCKS_GUARD = threading.Lock()


def _prepare_lock(group_id: str) -> threading.Lock:
    with _PREPARE_LOCKS_GUARD:
        return _PREPARE_LOCKS.setdefault(group_id, threading.Lock())


def _imported_lidar_extrinsics() -> dict[str, dict[str, list[float]]]:
    try:
        values = tomllib.loads(LIDAR_EXTRINSICS_TOML.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    imported = {}
    for lidar_id, section_name in LIDAR_EXTRINSIC_SECTIONS.items():
        section = values.get(section_name, {})
        rotation = section.get("rotation")
        translation = section.get("transform")
        if not isinstance(rotation, list) or len(rotation) != 9 or not isinstance(translation, list) or len(translation) != 3:
            continue
        imported[lidar_id] = {"carFromLidar": [
            float(rotation[0]), float(rotation[1]), float(rotation[2]), float(translation[0]),
            float(rotation[3]), float(rotation[4]), float(rotation[5]), float(translation[1]),
            float(rotation[6]), float(rotation[7]), float(rotation[8]), float(translation[2]),
            0.0, 0.0, 0.0, 1.0,
        ]}
    return imported


def _merge_imported_lidar_extrinsics(manifest: dict) -> dict:
    calibration = manifest.setdefault("calibration", {})
    calibration["lidars"] = {**calibration.get("lidars", {}), **_imported_lidar_extrinsics()}
    calibration.setdefault("cameras", {})
    return manifest


class Rect(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class SegmentRequest(BaseModel):
    image_url: str
    box: Rect
    granularity: Literal["fine", "balanced", "coarse"] = "fine"


class MaskResult(BaseModel):
    width: int
    height: int
    area: int
    score: float
    rle: list[int]


class SegmentResponse(BaseModel):
    model: str
    cached_embedding: bool
    candidates: list[MaskResult]


class IntrinsicsImportRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)


class ManualSelectionMergeRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)
    selections: dict[str, list[str]] = Field(default_factory=dict)
    client_id: str = Field(default="anonymous", max_length=120)


class ManualSelectionToggleRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)
    camera_topic: str = Field(min_length=1, max_length=500)
    timestamp_ns: str = Field(min_length=1, max_length=24)
    selected: bool
    client_id: str = Field(default="anonymous", max_length=120)


class ManualSelectionReplaceRequest(BaseModel):
    source_path: str = Field(min_length=1, max_length=1000)
    selections: dict[str, list[str]] = Field(default_factory=dict)
    expected_version: int = Field(ge=0)
    client_id: str = Field(default="anonymous", max_length=120)


def encode_rle(mask: np.ndarray) -> list[int]:
    flat = mask.astype(np.uint8, copy=False).reshape(-1)
    if flat.size == 0:
        return [0]
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(boundaries).astype(np.int64).tolist()
    if int(flat[0]) != 0:
        runs.insert(0, 0)
    return runs


def resolve_image(image_url: str) -> Path:
    relative = image_url.split("?", 1)[0].lstrip("/")
    parts = relative.split("/")
    if len(parts) == 8 and parts[:3] == ["api", "calibration", "groups"] and parts[4:6] == ["intrinsics", "frames"]:
        return resolve_undistorted_frame(unquote(parts[3]), unquote(parts[6]), unquote(parts[7]))
    if relative.startswith("data/cache/"):
        root = PREPARED_ROOT
        relative = relative.removeprefix("data/cache/")
    else:
        root = PUBLIC_ROOT
    path = (root / relative).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(status_code=404, detail="Image is outside the prepared data directory")
    return path


def scaled_box(rect: Rect, width: int, height: int, granularity: str) -> np.ndarray:
    expansion = {"fine": 0.0, "balanced": 0.08, "coarse": 0.2}[granularity]
    x1, y1, x2, y2 = rect.x1 * width, rect.y1 * height, rect.x2 * width, rect.y2 * height
    dx, dy = (x2 - x1) * expansion, (y2 - y1) * expansion
    return np.asarray([
        max(0, x1 - dx), max(0, y1 - dy), min(width - 1, x2 + dx), min(height - 1, y2 + dy),
    ], dtype=np.float32)


class Sam2Runtime:
    def __init__(self) -> None:
        self.predictor: SAM2ImagePredictor | None = None
        self.cache: OrderedDict[str, tuple[np.ndarray, dict]] = OrderedDict()
        self.active_key: str | None = None
        self.lock = asyncio.Lock()

    def load(self) -> None:
        if self.predictor is not None:
            return
        if torch is None or build_sam2 is None or SAM2ImagePredictor is None:
            raise RuntimeError("SAM2 is unavailable in this local runtime")
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable inside the SAM2 service")
        model = build_sam2(MODEL_CONFIG, CHECKPOINT, device=DEVICE)
        self.predictor = SAM2ImagePredictor(model)

    def set_image(self, path: Path) -> tuple[np.ndarray, bool]:
        self.load()
        assert self.predictor is not None
        key = f"{path}:{path.stat().st_mtime_ns}"
        image = np.asarray(Image.open(path).convert("RGB")).copy()
        cached = key in self.cache
        if cached:
            features = self.cache.pop(key)[1]
            self.cache[key] = (image, features)
            self.predictor._features = features
            self.predictor._is_image_set = True
            self.predictor._orig_hw = [image.shape[:2]]
        else:
            self.predictor.set_image(image)
            self.cache[key] = (image, self.predictor._features)
            while len(self.cache) > MAX_CACHE_ITEMS:
                self.cache.popitem(last=False)
        self.active_key = key
        return image, cached

    def predict(self, image: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.predictor is not None
        assert torch is not None
        precision = torch.autocast("cuda", dtype=torch.bfloat16) if DEVICE.startswith("cuda") else nullcontext()
        with torch.inference_mode(), precision:
            masks, scores, _ = self.predictor.predict(box=box, multimask_output=True)
        order = np.argsort(np.asarray([mask.sum() for mask in masks]))
        return masks[order], scores[order]


runtime = Sam2Runtime()
app = FastAPI(title="ROC-Calib", version="0.1.0")

from .groups import StateConflict

@app.exception_handler(StateConflict)
async def state_conflict(_request: Request, error: StateConflict):
    return JSONResponse(status_code=409, content={"detail": str(error)})

@app.middleware("http")
async def read_only_mode(request: Request, call_next):
    if os.environ.get("ROC_CALIB_READ_ONLY") == "1" and request.method not in {"GET", "HEAD", "OPTIONS"}:
        return JSONResponse(status_code=403, content={"detail": "This demonstration is read-only."})
    return await call_next(request)

from .calibration_review_api import router as calibration_review_router
app.include_router(calibration_review_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x for x in os.environ.get("ROC_CALIB_ALLOWED_ORIGINS", "").split(",") if x],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)
initialize_groups()
from .nuscenes_fusion import PreparedFiles
app.mount("/v1/prepared", PreparedFiles(directory=PREPARED_ROOT), name="prepared-data")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda": bool(torch is not None and torch.cuda.is_available()),
        "model_loaded": runtime.predictor is not None,
        "cached_images": len(runtime.cache),
        "calibration_group_root": str(GROUP_ROOT),
        "data_root": str(STORAGE.data_root) if STORAGE.data_root else None,
        "storage_layout": STORAGE.layout,
        "bag_root": str(STORAGE.bag_root),
        "paired_root": str(STORAGE.paired_root),
        "prepared_root": str(PREPARED_ROOT),
        "lidar_calibration_enabled": LIDAR_CALIBRATION_ENABLED,
    }


@app.post("/v1/calibration/demo")
def import_demo():
    from .demo import install
    try:
        return install()
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Cannot load example: {error}") from error


@app.get("/v1/calibration/groups")
def calibration_groups() -> dict:
    return {"groups": list_groups()}


@app.get("/v1/calibration/bags")
def calibration_bags(path: str = "") -> dict:
    try:
        return list_bag_directory(path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Bag directory not found") from error


@app.get("/v1/calibration/bags/manual-frames")
def calibration_bag_manual_frames(path: str) -> dict:
    try:
        return manual_frame_candidates(path)
    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/paired")
def calibration_paired_datasets(path: str = "") -> dict:
    try:
        return list_paired_directory(path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Paired dataset directory not found") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/bags/manual-preview")
def calibration_bag_manual_preview(path: str, timestamp_ns: int, camera_topic: str = "") -> FileResponse:
    try:
        return FileResponse(manual_frame_preview(path, timestamp_ns, camera_topic), media_type="image/jpeg")
    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/bags/manual-lidar-preview")
def calibration_bag_manual_lidar_preview(path: str, timestamp_ns: int, lidar_topic: str = "") -> FileResponse:
    try:
        return FileResponse(manual_lidar_preview(path, timestamp_ns, lidar_topic), media_type="image/jpeg")
    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/bags/manual-lidar-cloud")
def calibration_bag_manual_lidar_cloud(path: str, timestamp_ns: int, lidar_topic: str = "") -> FileResponse:
    try:
        return FileResponse(manual_lidar_cloud(path, timestamp_ns, lidar_topic), media_type="application/octet-stream")
    except (FileNotFoundError, ValueError, OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/groups/{group_id}/manual-selection")
def calibration_group_manual_selection(group_id: str, source_path: str) -> dict:
    try:
        return read_manual_selection(group_id, source_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/manual-selection/merge")
def calibration_group_manual_selection_merge(group_id: str, request: ManualSelectionMergeRequest) -> dict:
    try:
        return merge_manual_selection(group_id, request.source_path, request.selections, request.client_id)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/manual-selection/toggle")
def calibration_group_manual_selection_toggle(group_id: str, request: ManualSelectionToggleRequest) -> dict:
    try:
        return toggle_manual_selection(
            group_id,
            request.source_path,
            request.camera_topic,
            request.timestamp_ns,
            request.selected,
            request.client_id,
        )
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.put("/v1/calibration/groups/{group_id}/manual-selection")
def calibration_group_manual_selection_replace(group_id: str, request: ManualSelectionReplaceRequest) -> dict:
    try:
        return replace_manual_selection(
            group_id,
            request.source_path,
            request.selections,
            request.client_id,
            request.expected_version,
        )
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/groups/{group_id}/manual-selection/events")
async def calibration_group_manual_selection_events(group_id: str, source_path: str, request: Request) -> StreamingResponse:
    try:
        read_manual_selection(group_id, source_path)
    except (FileNotFoundError, ValueError, OSError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    async def stream():
        last_version = -1
        idle_ticks = 0
        while not await request.is_disconnected():
            state = read_manual_selection(group_id, source_path)
            version = int(state.get("version", 0))
            if version != last_version:
                yield "data: " + json.dumps(state, ensure_ascii=False) + "\n\n"
                last_version = version
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks >= 30:
                    yield ": keep-alive\n\n"
                    idle_ticks = 0
            # Collaboration updates should feel immediate. Reading the tiny
            # settings cursor is cheap, while the previous 500 ms interval was
            # the dominant part of the end-to-end sync delay.
            await asyncio.sleep(0.1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/v1/calibration/manifest")
def calibration_manifest() -> dict:
    path = PREPARED_ROOT / "manifest.json"
    if not path.is_file():
        return _merge_imported_lidar_extrinsics({"generatedAt": None, "datasets": [], "calibration": {"source": "user-import", "note": "相机内参尚未导入", "lidars": {}, "cameras": {}}})
    try:
        manifest = json.loads(path.read_text())
        datasets = list(manifest.get("datasets", []))
        known_ids = {str(item.get("id")) for item in datasets}
        # A preparation attempt can create a new cache and then be rejected by
        # annotation validation. Always expose caches still referenced by a
        # group so its existing workspace never disappears from the UI.
        for group in list_groups():
            for dataset_id in group.get("datasetIds", []):
                dataset_id = str(dataset_id)
                if dataset_id in known_ids:
                    continue
                cached = _cached_dataset(dataset_id)
                if cached is not None:
                    datasets.append(cached)
                    known_ids.add(dataset_id)
        manifest["datasets"] = sorted(datasets, key=lambda item: item.get("name", ""))
        return _merge_imported_lidar_extrinsics(manifest)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=500, detail="Prepared manifest is invalid") from error


@app.post("/v1/calibration/groups")
def calibration_group_create(request: GroupCreate) -> dict:
    try:
        return create_group(request)
    except FileExistsError as error:
        raise HTTPException(status_code=409, detail="Calibration group already exists") from error


@app.delete("/v1/calibration/groups/{group_id}")
def calibration_group_delete(group_id: str) -> dict:
    try:
        return delete_group(group_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/imports")
def calibration_group_import(group_id: str, request: GroupImport) -> dict:
    try:
        return import_datasets(group_id, request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/uploads")
def calibration_group_upload_create(group_id: str, request: UploadCreate) -> dict:
    try:
        return initialize_upload(group_id, request)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.put("/v1/calibration/groups/{group_id}/uploads/{upload_id}/files/{file_index}")
async def calibration_group_upload_chunk(group_id: str, upload_id: str, file_index: int, request: Request) -> dict:
    try:
        offset = int(request.headers.get("x-upload-offset", "0"))
        result = append_upload_chunk(group_id, upload_id, file_index, offset, await request.body())
        if result.get("mismatch"):
            raise HTTPException(status_code=409, detail={"message": "上传位置已变化", **result})
        return result
    except HTTPException:
        raise
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Upload not found") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/uploads/{upload_id}/complete")
def calibration_group_upload_complete(group_id: str, upload_id: str) -> dict:
    try:
        return finalize_upload(group_id, upload_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Upload not found") from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@app.post("/v1/calibration/groups/{group_id}/prepare")
def calibration_group_prepare(group_id: str, request: GroupImport) -> StreamingResponse:
    """Prepare raw bags, stream progress, then attach the cached datasets to a group."""
    events: queue.Queue[dict | None] = queue.Queue()

    def emit(event: dict) -> None:
        events.put(event)

    def run_locked() -> None:
        try:
            if not request.source_paths:
                raise ValueError("请至少选择一个 rosbag")
            datasets: list[dict] = []
            cached_count = 0
            total = len(request.source_paths)
            for index, source_path in enumerate(request.source_paths):
                if request.replace_frame_selection:
                    selected_by_topic = {
                        **request.selected_camera_frames.get(source_path, {}),
                        **request.selected_lidar_frames.get(source_path, {}),
                    }
                    for topic, protected_timestamps in protected_frame_selections(group_id, source_path).items():
                        selected = set(selected_by_topic.get(topic, selected_by_topic.get(topic.lstrip("/"), [])))
                        if any(timestamp not in selected for timestamp in protected_timestamps):
                            raise ValueError("已有标注使用的帧不能移除，请先删除对应标注")
                def progress(percent: int, message: str, details: dict | None = None) -> None:
                    overall = int((index * 100 + percent) / total)
                    emit({"type": "progress", "percent": overall, "message": message,
                          "bagIndex": index + 1, "bagCount": total, **(details or {})})

                selected = request.selected_frames.get(source_path, [])
                selected_cameras = {
                    topic: [int(value) for value in values]
                    for topic, values in request.selected_camera_frames.get(source_path, {}).items()
                }
                # Selecting more frames/views for an imported bag must extend
                # the current dataset, not replace it. This also guarantees
                # that annotations attached to older views remain addressable.
                if not request.replace_frame_selection:
                    for topic, values in existing_camera_selections(group_id, source_path).items():
                        selected_cameras.setdefault(topic, []).extend(int(value) for value in values)
                selected_cameras = {
                    topic: sorted(set(values)) for topic, values in selected_cameras.items() if values
                }
                selected_lidars = {
                    topic: [int(value) for value in values]
                    for topic, values in request.selected_lidar_frames.get(source_path, {}).items()
                }
                dataset, cached = prepare_bag(
                    source_path,
                    progress,
                    [int(value) for value in selected],
                    selected_cameras,
                    selected_lidars,
                )
                datasets.append(dataset)
                cached_count += int(cached)
                if selected_cameras:
                    merge_manual_selection(
                        group_id,
                        source_path,
                        {topic: [str(value) for value in values] for topic, values in selected_cameras.items()},
                        "prepared-dataset-sync",
                    )
            rig_ids = {str(dataset["rigId"]) for dataset in datasets}
            if len(rig_ids) != 1:
                raise ValueError("传感器 topic 组合不同的数据包不能导入同一个标定组")
            group = import_datasets(group_id, GroupImport(
                dataset_ids=[str(dataset["id"]) for dataset in datasets],
                source_paths=request.source_paths,
                rig_id=next(iter(rig_ids)),
            ))
            intrinsics_warning = ""
            try:
                refresh_imported_intrinsics(group_id)
            except (OSError, ValueError) as error:
                intrinsics_warning = f"；已有内参未覆盖新增视角，请重新导入内参（{error}）"
            emit({"type": "result", "percent": 100, "message": "数据包处理并导入完成" + intrinsics_warning, "group": group,
                  "datasetIds": [dataset["id"] for dataset in datasets], "cachedCount": cached_count})
        except Exception as error:
            emit({"type": "error", "message": str(error)})
        finally:
            events.put(None)

    def run() -> None:
        lock = _prepare_lock(group_id)
        if lock.locked():
            emit({"type": "progress", "percent": 0, "message": "同一标定组正在导入，当前任务排队等待"})
        with lock:
            run_locked()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/v1/calibration/groups/{group_id}/paired-import")
def calibration_group_paired_import(group_id: str, request: GroupImport) -> StreamingResponse:
    """Validate, cache, and attach explicit image/point-cloud frame pairs."""
    events: queue.Queue[dict | None] = queue.Queue()

    def emit(event: dict) -> None:
        events.put(event)

    def run_locked() -> None:
        try:
            if not request.paired_source_paths:
                raise ValueError("请至少选择一个成对数据集")
            datasets: list[dict] = []
            cached_count = 0
            total = len(request.paired_source_paths)
            for index, source_path in enumerate(request.paired_source_paths):
                def progress(percent: int, message: str, details: dict | None = None) -> None:
                    overall = int((index * 100 + percent) / total)
                    emit({
                        "type": "progress",
                        "percent": overall,
                        "message": message,
                        "datasetIndex": index + 1,
                        "datasetCount": total,
                        **(details or {}),
                    })

                dataset, cached = prepare_paired_dataset(source_path, progress)
                datasets.append(dataset)
                cached_count += int(cached)
            rig_ids = {str(dataset["rigId"]) for dataset in datasets}
            if len(rig_ids) != 1:
                raise ValueError("不同 rigId 的成对数据集不能导入同一个标定组")
            group = import_datasets(group_id, GroupImport(
                dataset_ids=[str(dataset["id"]) for dataset in datasets],
                paired_source_paths=request.paired_source_paths,
                rig_id=next(iter(rig_ids)),
            ))
            intrinsics_warning = ""
            try:
                refresh_imported_intrinsics(group_id)
            except (OSError, ValueError) as error:
                intrinsics_warning = f"；部分相机缺少可用内参，请为当前组导入内参（{error}）"
            emit({
                "type": "result",
                "percent": 100,
                "message": "成对数据集导入完成" + intrinsics_warning,
                "group": group,
                "datasetIds": [dataset["id"] for dataset in datasets],
                "cachedCount": cached_count,
            })
        except Exception as error:
            emit({"type": "error", "message": str(error)})
        finally:
            events.put(None)

    def run() -> None:
        lock = _prepare_lock(group_id)
        if lock.locked():
            emit({"type": "progress", "percent": 0, "message": "同一标定组正在导入，当前任务排队等待"})
        with lock:
            run_locked()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/v1/calibration/groups/{group_id}/state/camera", response_model=CameraGroupState)
def calibration_group_camera_state(group_id: str) -> CameraGroupState:
    try:
        return read_camera_state(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state/lidar", response_model=LidarGroupState)
def calibration_group_lidar_state(group_id: str) -> LidarGroupState:
    try:
        return read_lidar_state(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state/{domain}/status")
def calibration_group_domain_state_status(group_id: str, domain: Literal["camera", "lidar"]) -> dict:
    try:
        return read_domain_state_status(group_id, domain)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state/{domain}/events")
async def calibration_group_domain_state_events(group_id: str, domain: Literal["camera", "lidar"], request: Request) -> StreamingResponse:
    try:
        read_domain_state_status(group_id, domain)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error

    async def stream():
        last_revision = -1
        while not await request.is_disconnected():
            status = read_domain_state_status(group_id, domain)
            revision = int(status["revision"])
            if revision != last_revision:
                yield "data: " + json.dumps(status, ensure_ascii=False) + "\n\n"
                last_revision = revision
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.put("/v1/calibration/groups/{group_id}/state/camera")
def calibration_group_camera_state_write(group_id: str, state: CameraGroupState) -> dict:
    try:
        saved = write_camera_state(group_id, state)
        return {"revision": saved.revision, "updatedBy": saved.updatedBy}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.put("/v1/calibration/groups/{group_id}/state/lidar")
def calibration_group_lidar_state_write(group_id: str, state: LidarGroupState) -> dict:
    try:
        saved = write_lidar_state(group_id, state)
        return {"revision": saved.revision, "updatedBy": saved.updatedBy}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state", response_model=GroupState)
def calibration_group_state(group_id: str) -> GroupState:
    try:
        return read_state(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state/status")
def calibration_group_state_status(group_id: str) -> dict:
    """Return the small collaboration cursor without downloading the full state."""
    try:
        return read_state_status(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/state/events")
async def calibration_group_state_events(group_id: str, request: Request) -> StreamingResponse:
    try:
        read_state_status(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error

    async def stream():
        last_revision = -1
        idle_ticks = 0
        while not await request.is_disconnected():
            status = read_state_status(group_id)
            revision = int(status["revision"])
            if revision != last_revision:
                yield "data: " + json.dumps(status, ensure_ascii=False) + "\n\n"
                last_revision = revision
                idle_ticks = 0
            else:
                idle_ticks += 1
                if idle_ticks >= 30:
                    yield ": keepalive\n\n"
                    idle_ticks = 0
            # The state document itself is fetched only after the cursor moves;
            # checking this tiny cursor frequently keeps collaboration snappy.
            await asyncio.sleep(0.1)

    return StreamingResponse(stream(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.put("/v1/calibration/groups/{group_id}/state")
def calibration_group_state_write(group_id: str, state: GroupState) -> dict:
    try:
        saved = write_state(group_id, state)
        # The browser only needs the collaboration cursor here. Returning the
        # entire multi-megabyte state delayed the saving browser unnecessarily.
        return {"revision": saved.revision, "updatedBy": saved.updatedBy}
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/groups/{group_id}/intrinsics")
def calibration_group_intrinsics(group_id: str) -> dict:
    try:
        return read_intrinsics(group_id)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Calibration group not found") from error


@app.get("/v1/calibration/intrinsics")
def calibration_intrinsics_directory(path: str = "") -> dict:
    try:
        return list_intrinsics_directory(path)
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Intrinsics directory not found") from error


@app.post("/v1/calibration/groups/{group_id}/intrinsics")
def calibration_group_intrinsics_import(group_id: str, request: IntrinsicsImportRequest) -> dict:
    try:
        return import_intrinsics_path(group_id, request.source_path)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="内参文件或标定组不存在") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/v1/calibration/groups/{group_id}/intrinsics/frames/{dataset_id}/{filename}")
def calibration_group_undistorted_frame(group_id: str, dataset_id: str, filename: str) -> FileResponse:
    try:
        return FileResponse(resolve_undistorted_frame(group_id, dataset_id, filename), media_type="image/jpeg")
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=404, detail="Undistorted frame not found") from error


@app.post("/v1/image/segment", response_model=SegmentResponse)
async def segment_image(request: SegmentRequest) -> SegmentResponse:
    path = resolve_image(request.image_url)
    async with runtime.lock:
        try:
            image, cached = await asyncio.to_thread(runtime.set_image, path)
            height, width = image.shape[:2]
            box = scaled_box(request.box, width, height, request.granularity)
            masks, scores = await asyncio.to_thread(runtime.predict, image, box)
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
    candidates = [
        MaskResult(
            width=width,
            height=height,
            area=int(mask.sum()),
            score=float(score),
            rle=encode_rle(mask),
        )
        for mask, score in zip(masks, scores, strict=True)
    ]
    return SegmentResponse(model="sam2.1-hiera-base-plus", cached_embedding=cached, candidates=candidates)


_OPTIMIZE_QUEUE = threading.BoundedSemaphore(8)

@app.post("/v1/calibration/optimize")
async def calibration_optimize(request: CalibrationRequest, http_request: Request) -> StreamingResponse:
    if not _OPTIMIZE_QUEUE.acquire(blocking=False):
        raise HTTPException(status_code=429, detail="Solver queue is full. Try again shortly.")
    loop = asyncio.get_running_loop()
    events = asyncio.Queue()
    cancelled = threading.Event()

    def put(event):
        loop.call_soon_threadsafe(events.put_nowait, event)

    def run():
        try:
            result = optimize_calibration(request, lambda percent, message, details=None: put({
                "type": "progress", "percent": percent, "message": message, **(details or {}),
            }), cancelled=cancelled)
            put({"type": "result", "percent": 100, "message": "Calibration complete", "result": result})
        except Exception as error:
            put({"type": "error", "message": str(error)})
        finally:
            release_preparation_batch(request.optimization_batch_id)
            _OPTIMIZE_QUEUE.release()
            put(None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        try:
            while not await http_request.is_disconnected():
                try:
                    event = await asyncio.wait_for(events.get(), timeout=1)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False) + "\n"
        finally:
            cancelled.set()

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/v1/calibration/lidar-pairs/optimize")
def calibration_lidar_pair_optimize(request: LidarPairRequest) -> StreamingResponse:
    if not LIDAR_CALIBRATION_ENABLED:
        raise HTTPException(status_code=404, detail="Lidar calibration preview is disabled")
    events: queue.Queue[dict | None] = queue.Queue()

    def progress(percent: int, message: str, details: dict | None = None) -> None:
        events.put({"type": "progress", "percent": percent, "message": message, **(details or {})})

    def run() -> None:
        try:
            result = optimize_lidar_pair(request, progress)
            events.put({"type": "result", "percent": 100, "message": "雷达外参优化完成", "result": result})
        except Exception as error:
            events.put({"type": "error", "message": str(error)})
        finally:
            events.put(None)

    threading.Thread(target=run, daemon=True).start()

    async def stream():
        while True:
            item = await asyncio.to_thread(events.get)
            if item is None:
                break
            yield json.dumps(item, ensure_ascii=False) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/v1/calibration/lidar-pairs/frame-pairs")
def calibration_lidar_frame_pairs(request: LidarFramePairRequest) -> dict:
    if not LIDAR_CALIBRATION_ENABLED:
        raise HTTPException(status_code=404, detail="Lidar calibration is disabled")
    try:
        return {"pairs": build_lidar_frame_pairs(request)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/calibration/lidar-pairs/refine")
def calibration_lidar_pair_refine(request: LidarPairRequest) -> dict:
    if not LIDAR_CALIBRATION_ENABLED:
        raise HTTPException(status_code=404, detail="Lidar calibration is disabled")
    if not request.pairs:
        raise HTTPException(status_code=400, detail="人工精调至少需要一个参照物配对")
    try:
        return optimize_lidar_pair(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/v1/calibration/lidar-graph/solve")
def calibration_lidar_graph_solve(request: LidarGraphRequest) -> dict:
    if not LIDAR_CALIBRATION_ENABLED:
        raise HTTPException(status_code=404, detail="Lidar calibration preview is disabled")
    try:
        return solve_lidar_graph(request)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
