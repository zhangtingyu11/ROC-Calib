from __future__ import annotations

import json
import os
import shutil
import threading
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from .storage import STORAGE

GROUP_ROOT = STORAGE.group_root
BAG_ROOT = STORAGE.bag_root
PAIRED_ROOT = STORAGE.paired_root
PUBLIC_ROOT = Path(os.environ.get("AUTOCALIB_PUBLIC_ROOT", "/workspace/web/public")).resolve()
PREPARED_ROOT = STORAGE.prepared_root
EXPORT_ROOT = STORAGE.export_root
_LOCK = threading.RLock()


class GroupCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    rig_id: str = Field(min_length=1, max_length=120)
    dataset_ids: list[str] = Field(default_factory=list)
    description: str = Field(default="", max_length=1000)


class GroupState(BaseModel):
    revision: int = Field(default=0, ge=0)
    updatedBy: str | None = Field(default=None, max_length=120)
    annotations: list[dict] = Field(default_factory=list)
    calibrations: list[dict] = Field(default_factory=list)
    initialExtrinsics: list[dict] = Field(default_factory=list)
    lidarPairAnnotations: list[dict] = Field(default_factory=list)
    lidarPairCalibrations: list[dict] = Field(default_factory=list)
    lidarFramePairSets: list[dict] = Field(default_factory=list)
    lidarPairNoAnnotationKeys: list[str] = Field(default_factory=list)
    lidarTopicRemarks: dict[str, str] = Field(default_factory=dict)
    extrinsicGraph: dict = Field(default_factory=dict)
    calibrationMode: Literal["lidar-camera", "lidar-lidar"] = "lidar-camera"
    projectionModelOverrides: dict[str, Literal["fisheye", "radtan", "rational"]] = Field(default_factory=dict)


class CameraGroupState(BaseModel):
    revision: int = Field(default=0, ge=0)
    updatedBy: str | None = Field(default=None, max_length=120)
    annotations: list[dict] = Field(default_factory=list)
    calibrations: list[dict] = Field(default_factory=list)
    initialExtrinsics: list[dict] = Field(default_factory=list)
    projectionModelOverrides: dict[str, Literal["fisheye", "radtan", "rational"]] = Field(default_factory=dict)


class LidarGroupState(BaseModel):
    revision: int = Field(default=0, ge=0)
    updatedBy: str | None = Field(default=None, max_length=120)
    lidarPairAnnotations: list[dict] = Field(default_factory=list)
    lidarPairCalibrations: list[dict] = Field(default_factory=list)
    lidarFramePairSets: list[dict] = Field(default_factory=list)
    lidarPairNoAnnotationKeys: list[str] = Field(default_factory=list)
    lidarTopicRemarks: dict[str, str] = Field(default_factory=dict)
    extrinsicGraph: dict = Field(default_factory=dict)


class GroupImport(BaseModel):
    dataset_ids: list[str] = Field(default_factory=list)
    source_paths: list[str] = Field(default_factory=list)
    paired_source_paths: list[str] = Field(default_factory=list)
    selected_frames: dict[str, list[str]] = Field(default_factory=dict)
    selected_camera_frames: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    selected_lidar_frames: dict[str, dict[str, list[str]]] = Field(default_factory=dict)
    replace_frame_selection: bool = False
    rig_id: str = Field(default="", max_length=120)


class UploadFile(BaseModel):
    relative_path: str = Field(min_length=1, max_length=500)
    size: int = Field(ge=0, le=2_000_000_000_000)
    last_modified: int = Field(default=0, ge=0)


class UploadCreate(BaseModel):
    package_name: str = Field(min_length=1, max_length=180)
    files: list[UploadFile] = Field(min_length=1, max_length=128)


UPLOAD_CHUNK_LIMIT = 16 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _path(group_id: str) -> Path:
    if not group_id or len(group_id) > 120 or group_id in {".", ".."} or "/" in group_id or "\\" in group_id or any(ord(char) < 32 for char in group_id):
        raise ValueError("invalid calibration group id")
    path = (GROUP_ROOT / group_id).resolve()
    if path.parent != GROUP_ROOT:
        raise ValueError("calibration group escapes root")
    return path


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _manual_selection_key(source_path: str) -> tuple[str, str]:
    normalized = source_path.strip().replace("\\", "/")
    if not normalized or len(normalized) > 1000 or normalized.startswith("/") or "\x00" in normalized:
        raise ValueError("invalid bag path")
    key = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return normalized, key


def _manual_selection_path(group_id: str, source_path: str) -> Path:
    """Return the bag-scoped selection shared by every calibration group."""
    group_path = _path(group_id)
    if not (group_path / "group.json").is_file():
        raise FileNotFoundError(group_id)
    _, key = _manual_selection_key(source_path)
    return GROUP_ROOT / "manual-selections" / f"{key}.json"


def _legacy_manual_selection_paths(source_path: str) -> list[Path]:
    """Find group-scoped selections written by versions before bag-level storage."""
    _, key = _manual_selection_key(source_path)
    candidates: list[Path] = []
    try:
        group_paths = list(GROUP_ROOT.iterdir())
    except OSError:
        return candidates
    for group_path in group_paths:
        if group_path.is_dir() and (group_path / "group.json").is_file():
            candidate = group_path / "source" / "manual-selections" / f"{key}.json"
            if candidate.is_file():
                candidates.append(candidate)
    return candidates


def _empty_manual_selection(source_path: str) -> dict:
    return {
        "version": 0,
        "sourcePath": source_path,
        "selections": {},
        "updatedAt": None,
        "updatedBy": None,
    }


def read_manual_selection(group_id: str, source_path: str) -> dict:
    path = _manual_selection_path(group_id, source_path)
    if not path.is_file():
        legacy_values: list[dict] = []
        for legacy_path in _legacy_manual_selection_paths(source_path):
            try:
                legacy = json.loads(legacy_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(legacy, dict) and legacy.get("sourcePath") == source_path:
                legacy_values.append(legacy)
        if legacy_values:
            value = max(legacy_values, key=lambda item: (int(item.get("version", 0)), str(item.get("updatedAt") or "")))
            _write_json(path, value)
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return _empty_manual_selection(source_path)
    if not isinstance(value, dict) or value.get("sourcePath") != source_path:
        return _empty_manual_selection(source_path)
    value.setdefault("version", 0)
    value.setdefault("selections", {})
    value.setdefault("updatedAt", None)
    value.setdefault("updatedBy", None)
    return value


def _clean_manual_selections(value: dict[str, list[str]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for topic, timestamps in value.items():
        normalized_topic = str(topic).strip()
        if not normalized_topic or len(normalized_topic) > 500 or not isinstance(timestamps, list):
            continue
        cleaned = sorted({str(item) for item in timestamps if str(item).isdigit() and len(str(item)) <= 24}, key=int)
        if cleaned:
            result[normalized_topic] = cleaned
    return result


def merge_manual_selection(group_id: str, source_path: str, selections: dict[str, list[str]], client_id: str) -> dict:
    path = _manual_selection_path(group_id, source_path)
    incoming = _clean_manual_selections(selections)
    with _LOCK:
        current = read_manual_selection(group_id, source_path)
        merged = {topic: set(values) for topic, values in current.get("selections", {}).items()}
        for topic, timestamps in incoming.items():
            merged.setdefault(topic, set()).update(timestamps)
        next_selections = {
            topic: sorted(values, key=int)
            for topic, values in merged.items() if values
        }
        changed = next_selections != current.get("selections", {})
        if changed:
            current.update({
                "version": int(current.get("version", 0)) + 1,
                "selections": next_selections,
                "updatedAt": _now(),
                "updatedBy": client_id[:120],
            })
            _write_json(path, current)
        return current


def toggle_manual_selection(
    group_id: str,
    source_path: str,
    camera_topic: str,
    timestamp_ns: str,
    selected: bool,
    client_id: str,
) -> dict:
    topic = camera_topic.strip()
    timestamp = str(timestamp_ns)
    if not topic or len(topic) > 500 or not timestamp.isdigit() or len(timestamp) > 24:
        raise ValueError("invalid manual frame selection")
    path = _manual_selection_path(group_id, source_path)
    with _LOCK:
        current = read_manual_selection(group_id, source_path)
        selections = {
            key: set(values) for key, values in _clean_manual_selections(current.get("selections", {})).items()
        }
        values = selections.setdefault(topic, set())
        if selected:
            values.add(timestamp)
        else:
            values.discard(timestamp)
        if not values:
            selections.pop(topic, None)
        next_selections = {
            key: sorted(values, key=int)
            for key, values in selections.items() if values
        }
        if next_selections != current.get("selections", {}):
            current.update({
                "version": int(current.get("version", 0)) + 1,
                "selections": next_selections,
                "updatedAt": _now(),
                "updatedBy": client_id[:120],
            })
            _write_json(path, current)
        return current


def replace_manual_selection(
    group_id: str,
    source_path: str,
    selections: dict[str, list[str]],
    client_id: str,
    expected_version: int,
) -> dict:
    path = _manual_selection_path(group_id, source_path)
    next_selections = _clean_manual_selections(selections)
    with _LOCK:
        current = read_manual_selection(group_id, source_path)
        if int(current.get("version", 0)) != expected_version:
            raise RuntimeError("选帧已被其他用户修改，请重新打开后再保存")
        if next_selections != current.get("selections", {}):
            current.update({
                "version": expected_version + 1,
                "selections": next_selections,
                "updatedAt": _now(),
                "updatedBy": client_id[:120],
            })
            _write_json(path, current)
        return current


def _migrate_shared_manual_selections() -> None:
    shared_root = GROUP_ROOT / "manual-selections"
    marker = shared_root / ".bag-scoped-v1"
    if marker.is_file():
        return
    with _LOCK:
        if marker.is_file():
            return
        candidates: dict[str, dict] = {}
        for group_path in GROUP_ROOT.iterdir():
            if not group_path.is_dir() or not (group_path / "group.json").is_file():
                continue
            for source in (group_path / "source" / "manual-selections").glob("*.json"):
                try:
                    value = json.loads(source.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                source_path = str(value.get("sourcePath") or "") if isinstance(value, dict) else ""
                if not source_path:
                    continue
                current = candidates.get(source_path)
                if current is None or (int(value.get("version", 0)), str(value.get("updatedAt") or "")) > (int(current.get("version", 0)), str(current.get("updatedAt") or "")):
                    candidates[source_path] = value
        for source_path, value in candidates.items():
            _, key = _manual_selection_key(source_path)
            destination = shared_root / f"{key}.json"
            if not destination.is_file():
                _write_json(destination, value)
        _write_json(marker, {"migratedAt": _now()})


def initialize() -> None:
    GROUP_ROOT.mkdir(parents=True, exist_ok=True)
    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    _migrate_shared_manual_selections()


def _prepared_datasets() -> list[dict]:
    manifest_path = PREPARED_ROOT / "manifest.json"
    try:
        value = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return value.get("datasets", []) if isinstance(value, dict) else []


def _bag_path(relative_path: str) -> Path:
    if relative_path.startswith("/") or "\x00" in relative_path:
        raise ValueError("invalid bag path")
    path = (BAG_ROOT / relative_path).resolve()
    if path != BAG_ROOT and BAG_ROOT not in path.parents:
        raise ValueError("bag path escapes source root")
    return path


def _bag_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".db3", ".mcap"}:
        return [path]
    if not path.is_dir():
        return []
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in {".db3", ".mcap"})


def _safe_upload_component(value: str, fallback: str = "rosbag") -> str:
    """Return a Windows/Linux-safe single path component for uploaded bags."""
    cleaned = "".join(character if character.isalnum() or character in "-_." else "-" for character in value.strip())
    cleaned = cleaned.strip(" .-")[:120]
    return cleaned or fallback


def _safe_upload_relative_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    relative = Path(normalized)
    if (
        not normalized or normalized.startswith("/") or "\x00" in normalized
        or relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"invalid upload file path: {value}")
    if relative.suffix.lower() not in {".db3", ".mcap", ".yaml", ".yml", ".json"}:
        raise ValueError(f"unsupported rosbag file: {value}")
    return relative


def _upload_root() -> Path:
    return BAG_ROOT / ".uploads"


def _upload_manifest_path(upload_id: str) -> Path:
    if len(upload_id) != 32 or any(character not in "0123456789abcdef" for character in upload_id):
        raise ValueError("invalid upload id")
    return _upload_root() / upload_id / "upload.json"


def _read_upload(group_id: str, upload_id: str) -> tuple[Path, dict]:
    _path(group_id)
    manifest_path = _upload_manifest_path(upload_id)
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise FileNotFoundError(upload_id) from error
    if manifest.get("groupId") != group_id:
        raise FileNotFoundError(upload_id)
    return manifest_path.parent, manifest


def _upload_status(upload_id: str, upload_path: Path, manifest: dict) -> dict:
    files = []
    for index, item in enumerate(manifest["files"]):
        part_path = upload_path / "parts" / f"{index:04d}.part"
        offset = min(part_path.stat().st_size if part_path.is_file() else 0, int(item["size"]))
        files.append({**item, "index": index, "offset": offset, "complete": offset == int(item["size"])})
    source_path = str(manifest.get("sourcePath", ""))
    complete = bool(source_path and _bag_files(_bag_path(source_path)))
    entry = None
    if complete:
        parent = str(_bag_path(source_path).parent.relative_to(BAG_ROOT))
        entry = next((item for item in list_bag_directory(parent)["entries"] if item["path"] == source_path), None)
    return {
        "uploadId": upload_id,
        "packageName": manifest["packageName"],
        "sourcePath": source_path if complete else None,
        "files": files,
        "complete": complete,
        "chunkSize": 8 * 1024 * 1024,
        "entry": entry,
    }


def initialize_upload(group_id: str, request: UploadCreate) -> dict:
    maximum = int(os.environ.get("ROC_CALIB_UPLOAD_GB", "20")) * 1024 ** 3
    if sum(item.size for item in request.files) > maximum:
        raise ValueError("Upload exceeds the configured size limit")
    if sum(item.size for item in request.files) + 1024 ** 3 > shutil.disk_usage(GROUP_ROOT).free:
        raise ValueError("Not enough disk space for this upload")
    if not (_path(group_id) / "group.json").is_file():
        raise FileNotFoundError(group_id)
    files = [{
        "relativePath": str(_safe_upload_relative_path(item.relative_path)),
        "size": item.size,
        "lastModified": item.last_modified,
    } for item in request.files]
    if len({item["relativePath"].lower() for item in files}) != len(files):
        raise ValueError("上传包中存在重名文件")
    if not any(Path(item["relativePath"]).suffix.lower() in {".db3", ".mcap"} for item in files):
        raise ValueError("请选择包含 .db3 或 .mcap 的 ROS bag")
    signature_payload = json.dumps({"groupId": group_id, "files": files}, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(signature_payload.encode()).hexdigest()
    root = _upload_root()
    root.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        for candidate in root.iterdir():
            manifest_path = candidate / "upload.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if manifest.get("fingerprint") == fingerprint and manifest.get("groupId") == group_id:
                return _upload_status(candidate.name, candidate, manifest)
        upload_id = uuid4().hex
        upload_path = root / upload_id
        (upload_path / "parts").mkdir(parents=True)
        package_name = _safe_upload_component(request.package_name)
        destination_name = f"{package_name}-{upload_id[:8]}"
        source_path = str(Path("uploads") / _safe_upload_component(group_id, "group") / destination_name)
        manifest = {
            "version": 1,
            "groupId": group_id,
            "packageName": package_name,
            "fingerprint": fingerprint,
            "sourcePath": source_path,
            "createdAt": _now(),
            "files": files,
        }
        _write_json(upload_path / "upload.json", manifest)
    return _upload_status(upload_id, upload_path, manifest)


def append_upload_chunk(group_id: str, upload_id: str, file_index: int, offset: int, contents: bytes) -> dict:
    if len(contents) > UPLOAD_CHUNK_LIMIT:
        raise ValueError("上传分片不能超过 16 MB")
    upload_path, manifest = _read_upload(group_id, upload_id)
    if file_index < 0 or file_index >= len(manifest["files"]):
        raise ValueError("invalid upload file index")
    expected_size = int(manifest["files"][file_index]["size"])
    part_path = upload_path / "parts" / f"{file_index:04d}.part"
    with _LOCK:
        current_size = part_path.stat().st_size if part_path.is_file() else 0
        if offset != current_size:
            return {"offset": current_size, "complete": current_size == expected_size, "mismatch": True}
        if current_size + len(contents) > expected_size:
            raise ValueError("上传内容超过文件声明大小")
        with part_path.open("ab") as output:
            output.write(contents)
            output.flush()
            os.fsync(output.fileno())
        current_size += len(contents)
    return {"offset": current_size, "complete": current_size == expected_size, "mismatch": False}


def finalize_upload(group_id: str, upload_id: str) -> dict:
    upload_path, manifest = _read_upload(group_id, upload_id)
    destination = _bag_path(str(manifest["sourcePath"]))
    with _LOCK:
        if destination.is_dir() and _bag_files(destination):
            return _upload_status(upload_id, upload_path, manifest)
        if destination.exists():
            raise ValueError("上传目标已存在但不是有效 rosbag")
        for index, item in enumerate(manifest["files"]):
            part_path = upload_path / "parts" / f"{index:04d}.part"
            if int(item["size"]) == 0 and not part_path.exists():
                part_path.touch()
            if not part_path.is_file() or part_path.stat().st_size != int(item["size"]):
                raise ValueError(f"文件尚未上传完整：{item['relativePath']}")
        temporary = destination.with_name(f".{destination.name}.{upload_id}.tmp")
        temporary.mkdir(parents=True, exist_ok=False)
        try:
            for index, item in enumerate(manifest["files"]):
                relative = _safe_upload_relative_path(str(item["relativePath"]))
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(upload_path / "parts" / f"{index:04d}.part", target)
            if not _bag_files(temporary):
                raise ValueError("上传目录顶层没有 .db3 或 .mcap 文件")
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temporary, destination)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise
    entry = next(item for item in list_bag_directory(str(destination.parent.relative_to(BAG_ROOT)))["entries"] if item["path"] == manifest["sourcePath"])
    return {**_upload_status(upload_id, upload_path, manifest), "complete": True, "sourcePath": manifest["sourcePath"], "entry": entry}


def _matches_for_bag(path: Path, prepared: list[dict]) -> list[dict]:
    names = {item.name for item in _bag_files(path)}
    return [item for item in prepared if Path(str(item.get("sourceFile", ""))).name in names]


def _cached_dataset(dataset_id: str) -> dict | None:
    prefix = next((candidate for candidate in ("bag-", "paired-") if dataset_id.startswith(candidate)), None)
    if prefix is None:
        return None
    # Bag IDs are exactly bag-<fingerprint>. Paired IDs also contain a source
    # slug, so use the manifest first and fall back to the cache scan needed
    # when a group still references an entry omitted from manifest.json.
    if prefix == "bag-":
        fingerprint = dataset_id[4:]
        candidates = [PREPARED_ROOT / fingerprint / "dataset.json"] if fingerprint and all(character in "0123456789abcdef" for character in fingerprint) else []
    else:
        candidates = PREPARED_ROOT.glob("*/dataset.json") if PREPARED_ROOT.is_dir() else []
    for candidate in candidates:
        try:
            value = json.loads(candidate.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and str(value.get("id")) == dataset_id:
            return value
    return None


def _dataset_by_id(dataset_id: str, prepared: list[dict]) -> dict | None:
    cached = _cached_dataset(dataset_id)
    if cached is not None:
        return cached
    return next((item for item in prepared if str(item.get("id")) == dataset_id), None)


def existing_camera_selections(group_id: str, source_path: str) -> dict[str, list[str]]:
    """Recover the manual target timestamps already attached for one bag.

    Re-importing a bag is an additive operation in the UI: operators use it to
    add another view or more frames.  Recovering targets from the attached
    dataset prevents a later partial selection from silently replacing views
    that were imported and may already contain annotations.
    """
    metadata_path = _path(group_id) / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    metadata = json.loads(metadata_path.read_text())
    prepared = _prepared_datasets()
    recovered: dict[str, set[str]] = {}
    for dataset_id in metadata.get("datasetIds", []):
        dataset = _dataset_by_id(str(dataset_id), prepared)
        if dataset is None or str(dataset.get("sourcePath", "")) != source_path:
            continue
        target_by_label = {
            str(frame.get("label")): str(frame.get("timestampNs"))
            for frame in dataset.get("anchorFrames", [])
            if str(frame.get("label", "")) and str(frame.get("timestampNs", "")).isdigit()
        }
        for camera in dataset.get("cameras", []):
            topic = str(camera.get("topic", "")).strip()
            if not topic:
                continue
            targets = {
                target_by_label[label]
                for frame in camera.get("frames", [])
                if (label := str(frame.get("label", ""))) in target_by_label
            }
            if targets:
                recovered.setdefault(topic, set()).update(targets)
    return {topic: sorted(values, key=int) for topic, values in recovered.items()}


def protected_frame_selections(group_id: str, source_path: str) -> dict[str, list[str]]:
    """Return topic/timestamps that are still referenced by saved annotations."""
    group_path = _path(group_id)
    metadata = json.loads((group_path / "group.json").read_text())
    prepared = _prepared_datasets()
    try:
        camera_annotations = json.loads((group_path / "annotations" / "annotations.json").read_text())
    except (OSError, json.JSONDecodeError):
        camera_annotations = []
    try:
        lidar_annotations = json.loads((group_path / "annotations" / "lidar-pairs.json").read_text())
    except (OSError, json.JSONDecodeError):
        lidar_annotations = []
    protected: dict[str, set[str]] = {}
    for dataset_id in metadata.get("datasetIds", []):
        dataset = _dataset_by_id(str(dataset_id), prepared)
        if dataset is None or str(dataset.get("sourcePath", "")) != source_path:
            continue
        target_by_label = {str(item.get("label")): str(item.get("timestampNs")) for item in dataset.get("anchorFrames", [])}
        camera_topic_by_id = {str(item.get("id")): str(item.get("topic", "")).strip() for item in dataset.get("cameras", [])}
        lidar_topic_by_id = {str(item.get("id")): str(item.get("topic", "")).strip() for item in dataset.get("lidars", [])}
        for annotation in camera_annotations:
            if str(annotation.get("datasetId")) != str(dataset_id):
                continue
            topic = camera_topic_by_id.get(str(annotation.get("cameraId")), "")
            timestamp = target_by_label.get(str(annotation.get("frame")), "")
            if topic and timestamp:
                protected.setdefault(topic, set()).add(timestamp)
        for annotation in lidar_annotations:
            if str(annotation.get("datasetId")) != str(dataset_id):
                continue
            for id_key, frame_key in (("sourceLidarId", "sourceFrame"), ("targetLidarId", "targetFrame")):
                topic = lidar_topic_by_id.get(str(annotation.get(id_key)), "")
                timestamp = target_by_label.get(str(annotation.get(frame_key)), "")
                if topic and timestamp:
                    protected.setdefault(topic, set()).add(timestamp)
    return {topic: sorted(values, key=int) for topic, values in protected.items()}


def _migrate_annotations(annotations: list[dict], replacements: dict[str, dict]) -> list[dict]:
    """Move annotations to replacement caches using stable source timestamps."""
    migrated: list[dict] = []
    for annotation in annotations:
        old_id = str(annotation.get("datasetId", ""))
        replacement = replacements.get(old_id)
        if replacement is None:
            migrated.append(annotation)
            continue
        old_dataset = replacement["old"]
        new_dataset = replacement["new"]
        old_timestamp_by_label = {
            str(frame.get("label")): str(frame.get("timestampNs"))
            for frame in old_dataset.get("anchorFrames", [])
        }
        new_label_by_timestamp = {
            str(frame.get("timestampNs")): str(frame.get("label"))
            for frame in new_dataset.get("anchorFrames", [])
        }
        old_label = str(annotation.get("frame", ""))
        timestamp = old_timestamp_by_label.get(old_label)
        new_label = new_label_by_timestamp.get(timestamp or "")
        camera_ids = {str(item.get("id")) for item in new_dataset.get("cameras", [])}
        lidar_ids = {str(item.get("id")) for item in new_dataset.get("lidars", [])}
        if not new_label or str(annotation.get("cameraId")) not in camera_ids or str(annotation.get("lidarId")) not in lidar_ids:
            raise ValueError(
                "已有标注使用的视角或帧不能移除，请保留原选择，或先删除对应标注后再导入"
            )
        migrated.append({**annotation, "datasetId": str(new_dataset["id"]), "frame": new_label})
    return migrated


def _migrate_annotations_to_current_datasets(metadata: dict, annotations: list[dict]) -> list[dict]:
    """Normalize state written by browser tabs that still show an older cache."""
    prepared = _prepared_datasets()
    current = [
        dataset for dataset_id in metadata.get("datasetIds", [])
        if (dataset := _dataset_by_id(str(dataset_id), prepared)) is not None
    ]
    current_ids = {str(dataset.get("id")) for dataset in current}
    replacements: dict[str, dict] = {}
    for annotation in annotations:
        old_id = str(annotation.get("datasetId", ""))
        if not old_id or old_id in current_ids or old_id in replacements:
            continue
        old_dataset = _dataset_by_id(old_id, prepared)
        if old_dataset is None:
            continue
        new_dataset = next((
            dataset for dataset in current
            if dataset.get("sourcePath") == old_dataset.get("sourcePath")
        ), None)
        if new_dataset is not None:
            replacements[old_id] = {"old": old_dataset, "new": new_dataset}
    return _migrate_annotations(annotations, replacements)


def _group_bag_link_directory(group_id: str) -> Path:
    """Keep a group's bag references inside that group's source directory."""
    return _path(group_id) / "source" / "rosbags"


def _link_group_bags(group_id: str, relative_paths: list[str]) -> None:
    if not relative_paths:
        return
    link_directory = _group_bag_link_directory(group_id)
    planned: list[tuple[Path, Path, bool]] = []
    for relative_path in relative_paths:
        relative = Path(relative_path)
        link = link_directory / relative.name
        # The canonical bags live once in GROUP_ROOT/rosbags.  Use a relative
        # link so it remains valid both on the host and in the container, where
        # GROUP_ROOT is mounted at a different absolute path.
        target = Path(os.path.relpath(GROUP_ROOT / "rosbags" / relative, link_directory))
        if link.is_symlink():
            if Path(os.readlink(link)) != target:
                raise ValueError(f"group rosbag link already points elsewhere: {relative.name}")
            continue
        if link.exists():
            raise ValueError(f"group rosbag entry already exists: {relative.name}")
        planned.append((link, target, (BAG_ROOT / relative).is_dir()))
    link_directory.mkdir(parents=True, exist_ok=True)
    for link, target, is_directory in planned:
        link.symlink_to(target, target_is_directory=is_directory)


def list_bag_directory(relative_path: str = "") -> dict:
    BAG_ROOT.mkdir(parents=True, exist_ok=True)
    path = _bag_path(relative_path)
    if not path.is_dir():
        raise FileNotFoundError(relative_path)
    prepared = _prepared_datasets()
    entries: list[dict] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        try:
            resolved = child.resolve()
            if resolved != BAG_ROOT and BAG_ROOT not in resolved.parents:
                continue
            files = _bag_files(resolved)
            is_bag = bool(files)
            if not child.is_dir() and not is_bag:
                continue
            matches = _matches_for_bag(resolved, prepared) if is_bag else []
            entries.append({
                "name": child.name,
                "path": str(resolved.relative_to(BAG_ROOT)),
                "kind": "bag" if is_bag else "directory",
                "sizeBytes": sum(item.stat().st_size for item in files),
                "datasetIds": [str(item.get("id")) for item in matches],
                "rigIds": sorted({str(item.get("rigId") or "truck5-legacy") for item in matches}),
                "prepared": bool(matches),
            })
        except (OSError, ValueError):
            continue
    current = "" if path == BAG_ROOT else str(path.relative_to(BAG_ROOT))
    parent = None if not current else str(Path(current).parent)
    if parent == ".":
        parent = ""
    return {
        "root": str(BAG_ROOT),
        "path": current,
        "parent": parent,
        "entries": entries,
    }


def list_groups() -> list[dict]:
    initialize()
    groups: list[dict] = []
    for path in GROUP_ROOT.iterdir():
        metadata = path / "group.json"
        if not path.is_dir() or not metadata.is_file():
            continue
        try:
            group = json.loads(metadata.read_text())
            source_metadata_path = path / "source" / "datasets.json"
            try:
                source_metadata = json.loads(source_metadata_path.read_text())
            except (OSError, json.JSONDecodeError):
                source_metadata = {}
            group["sourcePaths"] = source_metadata.get("sourcePaths", [])
            group["pairedSourcePaths"] = source_metadata.get("pairedSourcePaths", [])
            groups.append(group)
        except (OSError, json.JSONDecodeError):
            continue
    return sorted(groups, key=lambda item: item.get("createdAt", ""), reverse=True)


def create_group(request: GroupCreate, group_id: str | None = None) -> dict:
    initialize()
    # The user-visible group name is intentionally the folder name. Production
    # operators can therefore locate a task without consulting another ID map.
    identifier = group_id or request.name.strip()
    path = _path(identifier)
    with _LOCK:
        path.mkdir(parents=False, exist_ok=False)
        for directory in ("source", "frames", "annotations", "runs", "result", "logs"):
            (path / directory).mkdir()
        now = _now()
        metadata = {
            "id": identifier,
            "name": request.name.strip(),
            "rigId": request.rig_id,
            "datasetIds": request.dataset_ids,
            "description": request.description.strip(),
            "status": "draft",
            "createdAt": now,
            "updatedAt": now,
            "storagePath": str(path),
        }
        _write_json(path / "group.json", metadata)
        _write_json(path / "annotations" / "annotations.json", [])
        _write_json(path / "result" / "calibrations.json", [])
        _write_json(path / "result" / "initial-extrinsics.json", [])
        _write_json(path / "annotations" / "lidar-pairs.json", [])
        _write_json(path / "result" / "lidar-calibrations.json", [])
        _write_json(path / "result" / "lidar-frame-pair-sets.json", [])
        _write_json(path / "result" / "extrinsic-graph.json", {})
        _write_json(path / "source" / "datasets.json", {
            "datasetIds": request.dataset_ids,
            "sourcePaths": [],
            "pairedSourcePaths": [],
        })
    return metadata


def delete_group(group_id: str) -> dict:
    """Delete one group's private workspace without touching shared sources."""
    path = _path(group_id)
    metadata_path = path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    with _LOCK:
        metadata = json.loads(metadata_path.read_text())
        trash = GROUP_ROOT.parent / "trash"
        trash.mkdir(parents=True, exist_ok=True)
        path.rename(trash / f"{group_id}-{uuid4().hex[:8]}")
    return {"id": group_id, "name": metadata.get("name", group_id), "deleted": True}


def import_datasets(group_id: str, request: GroupImport) -> dict:
    path = _path(group_id)
    metadata_path = path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    with _LOCK:
        metadata = json.loads(metadata_path.read_text())
        selected_paths: list[str] = []
        selected_paired_paths: list[str] = []
        selected_datasets: list[dict] = []
        prepared = _prepared_datasets()
        for relative_path in request.source_paths:
            source_path = _bag_path(relative_path)
            if not _bag_files(source_path):
                raise ValueError(f"not a rosbag: {relative_path}")
            matches = _matches_for_bag(source_path, prepared)
            if not matches:
                raise ValueError(f"rosbag has not been prepared yet: {relative_path}")
            selected_paths.append(relative_path)
            selected_datasets.extend(matches)
        for relative_path in request.paired_source_paths:
            normalized = str(Path(relative_path)).replace(os.sep, "/")
            if not normalized or normalized.startswith("/") or ".." in Path(normalized).parts:
                raise ValueError(f"invalid paired dataset path: {relative_path}")
            matches = [
                item for item in prepared
                if item.get("sourceKind") == "paired-directory" and item.get("sourcePath") == normalized
            ]
            if not matches:
                raise ValueError(f"paired dataset has not been prepared yet: {relative_path}")
            selected_paired_paths.append(normalized)
            selected_datasets.extend(matches)
        # The prepare endpoint supplies the exact cache ID it just produced.
        # Falling back to all source matches is kept for the direct import API.
        requested_ids = list(request.dataset_ids) or [str(item.get("id")) for item in selected_datasets]
        if not requested_ids:
            raise ValueError("select at least one prepared data source")
        requested_datasets = [
            dataset for dataset_id in requested_ids
            if (dataset := _dataset_by_id(str(dataset_id), prepared)) is not None
        ]
        if len(requested_datasets) != len(set(requested_ids)):
            raise ValueError("prepared dataset cache is missing")
        selected_rigs = {str(item.get("rigId") or "truck5-legacy") for item in requested_datasets}
        if len(selected_rigs) > 1:
            raise ValueError("datasets from different rigs cannot share one calibration group")
        import_rig = next(iter(selected_rigs), request.rig_id)
        if not import_rig:
            raise ValueError("rig id is required")
        if request.rig_id and selected_rigs and request.rig_id != import_rig:
            raise ValueError("selected data source rig does not match request")
        existing_rig = metadata.get("rigId", "unassigned")
        if existing_rig not in {"unassigned", import_rig}:
            raise ValueError("datasets from different rigs cannot share one calibration group")
        # Re-preparing a source with a newer scene-selection algorithm should
        # replace that source's stale dataset ID in the group instead of
        # showing both versions side by side.
        retained_ids = []
        replacements: dict[str, dict] = {}
        selected_source_keys = {
            *[("rosbag", source_path) for source_path in selected_paths],
            *[("paired-directory", source_path) for source_path in selected_paired_paths],
        }

        def source_key(dataset: dict) -> tuple[str, str]:
            return str(dataset.get("sourceKind") or "rosbag"), str(dataset.get("sourcePath") or "")

        for dataset_id in metadata.get("datasetIds", []):
            old_dataset = _dataset_by_id(str(dataset_id), prepared)
            if old_dataset and source_key(old_dataset) in selected_source_keys:
                new_dataset = next((
                    item for item in requested_datasets
                    if source_key(item) == source_key(old_dataset)
                ), None)
                if new_dataset is None:
                    raise ValueError("replacement dataset for selected source is missing")
                if str(new_dataset.get("id")) != str(dataset_id):
                    replacements[str(dataset_id)] = {"old": old_dataset, "new": new_dataset}
                continue
            retained_ids.append(str(dataset_id))
        dataset_ids = list(dict.fromkeys([*retained_ids, *requested_ids]))
        annotations_path = path / "annotations" / "annotations.json"
        try:
            annotations = json.loads(annotations_path.read_text())
        except (OSError, json.JSONDecodeError):
            annotations = []
        # Older versions could replace group.json without migrating its
        # annotations. Recover those orphaned references on the next import.
        for annotation in annotations:
            annotation_dataset_id = str(annotation.get("datasetId", ""))
            if not annotation_dataset_id or annotation_dataset_id in replacements or annotation_dataset_id in dataset_ids:
                continue
            old_dataset = _dataset_by_id(annotation_dataset_id, prepared)
            if old_dataset is None or source_key(old_dataset) not in selected_source_keys:
                continue
            new_dataset = next((
                item for item in requested_datasets
                if source_key(item) == source_key(old_dataset)
            ), None)
            if new_dataset is not None:
                replacements[annotation_dataset_id] = {"old": old_dataset, "new": new_dataset}
        migrated_annotations = _migrate_annotations(annotations, replacements)
        if selected_paths and STORAGE.layout != "v2":
            _link_group_bags(group_id, selected_paths)
        metadata["rigId"] = import_rig
        metadata["datasetIds"] = dataset_ids
        metadata["status"] = "data-imported"
        metadata["updatedAt"] = _now()
        _write_json(metadata_path, metadata)
        if migrated_annotations != annotations:
            _write_json(annotations_path, migrated_annotations)
        source_metadata_path = path / "source" / "datasets.json"
        try:
            existing_source_metadata = json.loads(source_metadata_path.read_text())
        except (OSError, json.JSONDecodeError):
            existing_source_metadata = {}
        existing_source_paths = existing_source_metadata.get("sourcePaths", [])
        existing_paired_paths = existing_source_metadata.get("pairedSourcePaths", [])
        _write_json(source_metadata_path, {
            "rigId": import_rig,
            "datasetIds": dataset_ids,
            "sourcePaths": list(dict.fromkeys([*existing_source_paths, *selected_paths])),
            "pairedSourcePaths": list(dict.fromkeys([*existing_paired_paths, *selected_paired_paths])),
            "mode": "symlink" if selected_paths and not selected_paired_paths and STORAGE.layout != "v2" else "shared-references",
            "bagDirectory": str(_group_bag_link_directory(group_id)),
            "pairedDirectory": str(PAIRED_ROOT),
            "note": "Raw sources are shared; groups only store references, annotations, and results.",
        })
    return metadata


def read_state(group_id: str) -> GroupState:
    path = _path(group_id)
    if not (path / "group.json").is_file():
        raise FileNotFoundError(group_id)
    annotations = json.loads((path / "annotations" / "annotations.json").read_text())
    calibrations = json.loads((path / "result" / "calibrations.json").read_text())
    initial_extrinsics_path = path / "result" / "initial-extrinsics.json"
    initial_extrinsics = json.loads(initial_extrinsics_path.read_text()) if initial_extrinsics_path.is_file() else []
    settings_path = path / "result" / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    lidar_annotations_path = path / "annotations" / "lidar-pairs.json"
    lidar_calibrations_path = path / "result" / "lidar-calibrations.json"
    lidar_frame_pair_sets_path = path / "result" / "lidar-frame-pair-sets.json"
    extrinsic_graph_path = path / "result" / "extrinsic-graph.json"
    return GroupState(
        revision=int(settings.get("revision", 0)),
        updatedBy=settings.get("updatedBy"),
        annotations=annotations,
        calibrations=calibrations,
        initialExtrinsics=initial_extrinsics,
        lidarPairAnnotations=json.loads(lidar_annotations_path.read_text()) if lidar_annotations_path.is_file() else [],
        lidarPairCalibrations=json.loads(lidar_calibrations_path.read_text()) if lidar_calibrations_path.is_file() else [],
        lidarFramePairSets=json.loads(lidar_frame_pair_sets_path.read_text()) if lidar_frame_pair_sets_path.is_file() else [],
        lidarPairNoAnnotationKeys=settings.get("lidarPairNoAnnotationKeys", []),
        lidarTopicRemarks=settings.get("lidarTopicRemarks", {}),
        extrinsicGraph=json.loads(extrinsic_graph_path.read_text()) if extrinsic_graph_path.is_file() else {},
        calibrationMode=settings.get("calibrationMode", "lidar-camera"),
        projectionModelOverrides=settings.get("projectionModelOverrides", {}),
    )


def write_state(group_id: str, state: GroupState) -> GroupState:
    path = _path(group_id)
    metadata_path = path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    with _LOCK:
        metadata = json.loads(metadata_path.read_text())
        settings_path = path / "result" / "settings.json"
        try:
            current_settings = json.loads(settings_path.read_text())
        except (OSError, json.JSONDecodeError):
            current_settings = {}
        current_revision = int(current_settings.get("revision", 0))
        stale_write = state.revision != current_revision
        if stale_write:
            raise StateConflict("This group changed in another window. Save a local backup and reload before editing.")

        def merge_records(existing_path: Path, incoming: list[dict], keys: tuple[str, ...]) -> list[dict]:
            if not stale_write:
                return incoming
            try:
                existing = json.loads(existing_path.read_text())
            except (OSError, json.JSONDecodeError):
                existing = []
            merged: dict[str, dict] = {}
            order: list[str] = []
            for item in [*existing, *incoming]:
                key = next((str(item.get(field)) for field in keys if item.get(field)), "")
                if not key:
                    key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if key not in merged:
                    order.append(key)
                merged[key] = item
            return [merged[key] for key in order]

        annotations_path = path / "annotations" / "annotations.json"
        calibrations_path = path / "result" / "calibrations.json"
        initial_extrinsics_path = path / "result" / "initial-extrinsics.json"
        lidar_annotations_path = path / "annotations" / "lidar-pairs.json"
        lidar_calibrations_path = path / "result" / "lidar-calibrations.json"
        lidar_frame_pair_sets_path = path / "result" / "lidar-frame-pair-sets.json"
        annotations = merge_records(annotations_path, state.annotations, ("id",))
        annotations = _migrate_annotations_to_current_datasets(metadata, annotations)
        calibrations = merge_records(calibrations_path, state.calibrations, ("storageKey", "id", "groupKey"))
        initial_extrinsics = merge_records(initial_extrinsics_path, state.initialExtrinsics, ("id",))
        lidar_annotations = merge_records(lidar_annotations_path, state.lidarPairAnnotations, ("id",))
        lidar_calibrations = merge_records(lidar_calibrations_path, state.lidarPairCalibrations, ("id", "groupKey"))
        lidar_frame_pair_sets = merge_records(lidar_frame_pair_sets_path, state.lidarFramePairSets, ("key",))
        next_revision = current_revision + 1
        _write_json(path / "annotations" / "annotations.json", annotations)
        _write_json(path / "result" / "calibrations.json", calibrations)
        _write_json(path / "result" / "initial-extrinsics.json", initial_extrinsics)
        _write_json(path / "annotations" / "lidar-pairs.json", lidar_annotations)
        _write_json(path / "result" / "lidar-calibrations.json", lidar_calibrations)
        _write_json(path / "result" / "lidar-frame-pair-sets.json", lidar_frame_pair_sets)
        _write_json(path / "result" / "extrinsic-graph.json", state.extrinsicGraph)
        _write_json(path / "result" / "settings.json", {
            "revision": next_revision,
            "updatedBy": state.updatedBy,
            "calibrationMode": state.calibrationMode,
            "projectionModelOverrides": state.projectionModelOverrides,
            "lidarPairNoAnnotationKeys": state.lidarPairNoAnnotationKeys,
            "lidarTopicRemarks": state.lidarTopicRemarks,
        })
        metadata["updatedAt"] = _now()
        _write_json(metadata_path, metadata)
    return state.model_copy(update={
        "revision": next_revision,
        "updatedBy": state.updatedBy,
        "annotations": annotations,
        "calibrations": calibrations,
        "initialExtrinsics": initial_extrinsics,
        "lidarPairAnnotations": lidar_annotations,
        "lidarPairCalibrations": lidar_calibrations,
        "lidarFramePairSets": lidar_frame_pair_sets,
    })


def read_state_status(group_id: str) -> dict:
    path = _path(group_id)
    if not (path / "group.json").is_file():
        raise FileNotFoundError(group_id)
    settings_path = path / "result" / "settings.json"
    try:
        settings = json.loads(settings_path.read_text())
    except (OSError, json.JSONDecodeError):
        settings = {}
    return {
        "revision": int(settings.get("revision", 0)),
        "updatedBy": settings.get("updatedBy"),
    }


def _read_settings(path: Path) -> dict:
    try:
        value = json.loads((path / "result" / "settings.json").read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


class StateConflict(RuntimeError):
    pass


def _merge_domain_records(existing_path: Path, incoming: list[dict], keys: tuple[str, ...], stale: bool) -> list[dict]:
    if not stale:
        return incoming
    try:
        existing = json.loads(existing_path.read_text())
    except (OSError, json.JSONDecodeError):
        existing = []
    merged: dict[str, dict] = {}
    order: list[str] = []
    for item in [*existing, *incoming]:
        key = next((str(item.get(field)) for field in keys if item.get(field)), "")
        if not key:
            key = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if key not in merged:
            order.append(key)
        merged[key] = item
    return [merged[key] for key in order]


def read_camera_state(group_id: str) -> CameraGroupState:
    state = read_state(group_id)
    settings = _read_settings(_path(group_id))
    return CameraGroupState(
        revision=int(settings.get("cameraRevision", settings.get("revision", 0))),
        updatedBy=settings.get("cameraUpdatedBy", settings.get("updatedBy")),
        annotations=state.annotations,
        calibrations=state.calibrations,
        initialExtrinsics=state.initialExtrinsics,
        projectionModelOverrides=state.projectionModelOverrides,
    )


def read_lidar_state(group_id: str) -> LidarGroupState:
    state = read_state(group_id)
    settings = _read_settings(_path(group_id))
    return LidarGroupState(
        revision=int(settings.get("lidarRevision", 0)),
        updatedBy=settings.get("lidarUpdatedBy"),
        lidarPairAnnotations=state.lidarPairAnnotations,
        lidarPairCalibrations=state.lidarPairCalibrations,
        lidarFramePairSets=state.lidarFramePairSets,
        lidarPairNoAnnotationKeys=state.lidarPairNoAnnotationKeys,
        lidarTopicRemarks=state.lidarTopicRemarks,
        extrinsicGraph=state.extrinsicGraph,
    )


def read_domain_state_status(group_id: str, domain: Literal["camera", "lidar"]) -> dict:
    path = _path(group_id)
    if not (path / "group.json").is_file():
        raise FileNotFoundError(group_id)
    settings = _read_settings(path)
    fallback_revision = settings.get("revision", 0) if domain == "camera" else 0
    return {
        "revision": int(settings.get(f"{domain}Revision", fallback_revision)),
        "updatedBy": settings.get(f"{domain}UpdatedBy", settings.get("updatedBy") if domain == "camera" else None),
    }


def write_camera_state(group_id: str, state: CameraGroupState) -> CameraGroupState:
    path = _path(group_id)
    metadata_path = path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    with _LOCK:
        metadata = json.loads(metadata_path.read_text())
        settings = _read_settings(path)
        current_revision = int(settings.get("cameraRevision", settings.get("revision", 0)))
        stale = state.revision != current_revision
        if stale:
            raise StateConflict("This group changed in another window. Save a local backup and reload before editing.")
        annotations_path = path / "annotations" / "annotations.json"
        calibrations_path = path / "result" / "calibrations.json"
        initial_extrinsics_path = path / "result" / "initial-extrinsics.json"
        annotations = _merge_domain_records(annotations_path, state.annotations, ("id",), stale)
        annotations = _migrate_annotations_to_current_datasets(metadata, annotations)
        calibrations = _merge_domain_records(calibrations_path, state.calibrations, ("storageKey", "id", "groupKey"), stale)
        initial_extrinsics = _merge_domain_records(initial_extrinsics_path, state.initialExtrinsics, ("id",), stale)
        next_revision = current_revision + 1
        _write_json(annotations_path, annotations)
        _write_json(calibrations_path, calibrations)
        _write_json(initial_extrinsics_path, initial_extrinsics)
        settings.update({
            "cameraRevision": next_revision,
            "cameraUpdatedBy": state.updatedBy,
            "projectionModelOverrides": state.projectionModelOverrides,
        })
        _write_json(path / "result" / "settings.json", settings)
        metadata["updatedAt"] = _now()
        _write_json(metadata_path, metadata)
    return state.model_copy(update={
        "revision": next_revision,
        "annotations": annotations,
        "calibrations": calibrations,
        "initialExtrinsics": initial_extrinsics,
    })


def write_lidar_state(group_id: str, state: LidarGroupState) -> LidarGroupState:
    path = _path(group_id)
    metadata_path = path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    with _LOCK:
        metadata = json.loads(metadata_path.read_text())
        settings = _read_settings(path)
        current_revision = int(settings.get("lidarRevision", 0))
        stale = state.revision != current_revision
        if stale:
            raise StateConflict("This group changed in another window. Save a local backup and reload before editing.")
        annotations_path = path / "annotations" / "lidar-pairs.json"
        calibrations_path = path / "result" / "lidar-calibrations.json"
        frame_pairs_path = path / "result" / "lidar-frame-pair-sets.json"
        annotations = _merge_domain_records(annotations_path, state.lidarPairAnnotations, ("id",), stale)
        calibrations = _merge_domain_records(calibrations_path, state.lidarPairCalibrations, ("versionId", "id", "groupKey"), stale)
        frame_pairs = _merge_domain_records(frame_pairs_path, state.lidarFramePairSets, ("key",), stale)
        next_revision = current_revision + 1
        _write_json(annotations_path, annotations)
        _write_json(calibrations_path, calibrations)
        _write_json(frame_pairs_path, frame_pairs)
        _write_json(path / "result" / "extrinsic-graph.json", state.extrinsicGraph)
        settings.update({
            "lidarRevision": next_revision,
            "lidarUpdatedBy": state.updatedBy,
            "lidarPairNoAnnotationKeys": state.lidarPairNoAnnotationKeys,
            "lidarTopicRemarks": state.lidarTopicRemarks,
        })
        _write_json(path / "result" / "settings.json", settings)
        metadata["updatedAt"] = _now()
        _write_json(metadata_path, metadata)
    return state.model_copy(update={
        "revision": next_revision,
        "lidarPairAnnotations": annotations,
        "lidarPairCalibrations": calibrations,
        "lidarFramePairSets": frame_pairs,
    })
