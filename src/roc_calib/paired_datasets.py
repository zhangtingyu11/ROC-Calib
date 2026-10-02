from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .groups import PAIRED_ROOT, PREPARED_ROOT
from .preparation import _install_dataset


FORMAT_VERSION = 1
PREPARATION_ALGORITHM = "paired-manifest-v1-full-density"
MANIFEST_NAME = "autocalib.json"
Progress = Callable[[int, str, dict | None], None]
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _worker_count(job_count: int) -> int:
    try:
        configured = int(os.environ.get("AUTOCALIB_DATA_WORKERS", "4"))
    except ValueError:
        configured = 4
    return max(1, min(job_count, configured))


def _inside(root: Path, value: str | Path, description: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or "\x00" in str(relative):
        raise ValueError(f"{description}必须是数据集内的相对路径")
    resolved = (root / relative).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{description}超出允许的数据目录")
    return resolved


def _source_manifest(relative_path: str) -> tuple[Path, Path, str]:
    if not relative_path or relative_path.startswith("/") or "\x00" in relative_path:
        raise ValueError("成对数据集路径无效")
    selected = _inside(PAIRED_ROOT, relative_path, "成对数据集路径")
    manifest_path = selected if selected.is_file() else selected / MANIFEST_NAME
    if manifest_path.name != MANIFEST_NAME or not manifest_path.is_file():
        raise ValueError(f"数据集目录缺少 {MANIFEST_NAME}")
    dataset_root = manifest_path.parent.resolve()
    if PAIRED_ROOT not in dataset_root.parents:
        raise ValueError("成对数据集超出允许的数据目录")
    normalized = str(dataset_root.relative_to(PAIRED_ROOT)).replace(os.sep, "/")
    return dataset_root, manifest_path, normalized


def _read_manifest(relative_path: str) -> tuple[Path, Path, str, dict]:
    dataset_root, manifest_path, normalized = _source_manifest(relative_path)
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{MANIFEST_NAME} 不是有效 JSON：{error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{MANIFEST_NAME} 顶层必须是对象")
    return dataset_root, manifest_path, normalized, value


def _identifier(value: object, description: str) -> str:
    result = str(value or "").strip()
    if not _IDENTIFIER.fullmatch(result):
        raise ValueError(f"{description}只能包含字母、数字、点、下划线和连字符")
    return result


def _timestamp(value: object, fallback: int, description: str) -> int:
    if value is None or value == "":
        return fallback
    try:
        result = int(str(value))
    except ValueError as error:
        raise ValueError(f"{description}不是整数纳秒时间戳") from error
    if result < -(2**63) or result >= 2**63:
        raise ValueError(f"{description}超出 int64 范围")
    return result


def _point_format(value: object) -> tuple[str, list[str], str]:
    presets = {
        "xyz-f32": ["x", "y", "z"],
        "xyzi-f32": ["x", "y", "z", "intensity"],
        "xyzring-f32": ["x", "y", "z", "ring"],
        "xyzir-f32": ["x", "y", "z", "intensity", "ring"],
        "autocalib-acp1": [],
    }
    if isinstance(value, str):
        if value not in presets:
            raise ValueError(f"不支持的点云格式：{value}")
        return value, presets[value], "little"
    if not isinstance(value, dict):
        raise ValueError("雷达 pointFormat 必须是格式名或对象")
    encoding = str(value.get("encoding", "float32")).lower()
    if encoding not in {"float32", "npy", "autocalib-acp1"}:
        raise ValueError(f"不支持的点云编码：{encoding}")
    fields = [str(item).strip().lower() for item in value.get("fields", [])]
    if encoding != "autocalib-acp1" and (not fields or not {"x", "y", "z"}.issubset(fields)):
        raise ValueError("点云 fields 必须包含 x、y、z")
    if len(fields) != len(set(fields)) or len(fields) > 32:
        raise ValueError("点云 fields 包含重复项或字段过多")
    byte_order = str(value.get("byteOrder", "little")).lower()
    if byte_order not in {"little", "big"}:
        raise ValueError("pointFormat.byteOrder 必须是 little 或 big")
    return encoding, fields, byte_order


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: object) -> int:
    if expected is None or expected == "":
        return 0
    normalized = str(expected).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError(f"{path.name} 的 sha256 格式无效")
    if _sha256(path) != normalized:
        raise ValueError(f"{path.name} 的 sha256 校验失败")
    return 1


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _acp1_header(path: Path) -> tuple[int, int] | None:
    with path.open("rb") as stream:
        header = stream.read(16)
    if len(header) != 16 or header[:4] != b"ACP1":
        return None
    count, timestamp_ns = struct.unpack_from("<Iq", header, 4)
    if path.stat().st_size != 16 + count * 16:
        raise ValueError(f"点云文件不完整：{path.name}")
    return count, timestamp_ns


def _load_float_cloud(path: Path, encoding: str, fields: list[str], byte_order: str) -> np.ndarray:
    if encoding == "npy":
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.ndim != 2 or values.shape[1] != len(fields):
            raise ValueError(f"{path.name} 的 NPY 形状与 fields 不一致")
        return np.asarray(values)
    dtype = np.dtype("<f4" if byte_order == "little" else ">f4")
    values = np.fromfile(path, dtype=dtype)
    if not fields or len(values) % len(fields):
        raise ValueError(f"{path.name} 的大小不是点字段步长的整数倍")
    return values.reshape(-1, len(fields))


def _write_acp1(
    destination: Path,
    source: Path,
    timestamp_ns: int,
    point_format: object,
) -> tuple[int, float, int]:
    encoding, fields, byte_order = _point_format(point_format)
    header = _acp1_header(source)
    if encoding == "autocalib-acp1" or header is not None:
        if header is None:
            raise ValueError(f"{source.name} 不是有效的 ACP1 点云")
        if header[1] == timestamp_ns:
            _link_or_copy(source, destination)
        else:
            # A manifest timestamp is authoritative.  Copy before patching so
            # a hard link can never mutate the user's source point cloud.
            shutil.copy2(source, destination)
            with destination.open("r+b") as output:
                output.seek(8)
                output.write(struct.pack("<q", timestamp_ns))
        count = int(header[0])
        if not count:
            return count, 0.0, 0
        records = np.memmap(source, dtype=np.uint8, mode="r", offset=16, shape=(count, 16))
        stable_ratio = float(np.asarray(records[:, 12], dtype=np.float64).mean())
        return count, stable_ratio, 0

    values = _load_float_cloud(source, encoding, fields, byte_order)
    source_point_count = len(values)
    indexes = {name: fields.index(name) for name in fields}
    xyz = np.column_stack((values[:, indexes["x"]], values[:, indexes["y"]], values[:, indexes["z"]]))
    finite = np.isfinite(xyz).all(axis=1)
    xyz = np.asarray(xyz[finite], dtype="<f4")
    if not len(xyz):
        raise ValueError(f"{source.name} 没有有效 XYZ 点")

    stable = np.ones(len(xyz), dtype=np.uint8)
    if "stable" in indexes:
        stable = (np.asarray(values[finite, indexes["stable"]]) > 0).astype(np.uint8)
    ground = np.zeros(len(xyz), dtype=np.uint8)
    ground_field = next((name for name in ("ground", "is_ground") if name in indexes), None)
    if ground_field:
        ground = (np.asarray(values[finite, indexes[ground_field]]) > 0).astype(np.uint8)
    scanline = np.zeros(len(xyz), dtype=np.uint8)
    ring_field = next((name for name in ("ring", "scanline") if name in indexes), None)
    if ring_field:
        raw_rings = np.asarray(values[finite, indexes[ring_field]], dtype=np.float64)
        valid_rings = np.isfinite(raw_rings)
        rings = np.rint(raw_rings[valid_rings]).astype(np.int64)
        unique = np.unique(rings)
        if len(unique) > 255:
            raise ValueError(f"{source.name} 包含超过 255 条 ring")
        scanline[valid_rings] = (np.searchsorted(unique, rings) + 1).astype(np.uint8)

    records = np.zeros(len(xyz), dtype={
        "names": ["x", "y", "z", "stable", "ground", "has_ground", "scanline"],
        "formats": ["<f4", "<f4", "<f4", "u1", "u1", "u1", "u1"],
        "offsets": [0, 4, 8, 12, 13, 14, 15],
        "itemsize": 16,
    })
    records["x"], records["y"], records["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    records["stable"] = stable
    records["ground"] = ground
    records["has_ground"] = 1 if ground_field else 0
    records["scanline"] = scanline
    with destination.open("wb") as output:
        output.write(b"ACP1")
        output.write(struct.pack("<Iq", len(records), timestamp_ns))
        records.tofile(output)
    return len(records), float(stable.mean()), source_point_count - len(records)


def _manifest_summary(value: dict, fallback_name: str) -> dict:
    cameras = value.get("cameras", [])
    lidars = value.get("lidars", [])
    frame_labels = {
        str(frame.get("label", ""))
        for sensor in [*cameras, *lidars]
        if isinstance(sensor, dict)
        for frame in sensor.get("frames", [])
        if isinstance(frame, dict) and frame.get("label") not in {None, ""}
    }
    return {
        "name": str(value.get("name") or fallback_name),
        "rigId": str(value.get("rigId") or ""),
        "cameraCount": len(cameras) if isinstance(cameras, list) else 0,
        "lidarCount": len(lidars) if isinstance(lidars, list) else 0,
        "frameCount": len(frame_labels),
        "formatVersion": value.get("formatVersion"),
    }


def list_paired_directory(relative_path: str = "") -> dict:
    root = PAIRED_ROOT
    root.mkdir(parents=True, exist_ok=True)
    path = _inside(root, relative_path, "成对数据集目录")
    if not path.is_dir():
        raise FileNotFoundError(relative_path)
    entries: list[dict] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        try:
            resolved_child = child.resolve()
        except OSError:
            continue
        if resolved_child != root and root not in resolved_child.parents:
            continue
        manifest = resolved_child / MANIFEST_NAME
        item = {
            "name": child.name,
            "path": str(child.relative_to(root)).replace(os.sep, "/"),
            "kind": "dataset" if manifest.is_file() else "directory",
        }
        if manifest.is_file():
            try:
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("顶层必须是对象")
                item.update(_manifest_summary(value, child.name))
                item["valid"] = value.get("formatVersion") == FORMAT_VERSION
                item["error"] = None if item["valid"] else f"formatVersion 必须是 {FORMAT_VERSION}"
            except (OSError, json.JSONDecodeError, ValueError) as error:
                item.update({"valid": False, "error": str(error), "cameraCount": 0, "lidarCount": 0, "frameCount": 0})
        entries.append(item)
    current = "" if path == root else str(path.relative_to(root)).replace(os.sep, "/")
    parent = None if not current else str(Path(current).parent).replace(os.sep, "/")
    if parent == ".":
        parent = ""
    return {"root": str(root), "path": current, "parent": parent, "entries": entries}


def _validated_sensors(dataset_root: Path, value: dict) -> tuple[list[dict], list[dict], list[str]]:
    if value.get("formatVersion") != FORMAT_VERSION:
        raise ValueError(f"formatVersion 必须是 {FORMAT_VERSION}")
    _identifier(value.get("rigId"), "rigId")
    cameras = value.get("cameras", [])
    lidars = value.get("lidars", [])
    if not isinstance(cameras, list) or not isinstance(lidars, list):
        raise ValueError("cameras 和 lidars 必须是数组")
    if not lidars or (not cameras and len(lidars) < 2):
        raise ValueError("数据集必须包含相机与雷达，或至少两路雷达")

    sensor_ids: set[tuple[str, str]] = set()
    reference_labels: list[str] | None = None
    for kind, sensors in (("camera", cameras), ("lidar", lidars)):
        for sensor_index, sensor in enumerate(sensors):
            if not isinstance(sensor, dict):
                raise ValueError(f"{kind} #{sensor_index + 1} 必须是对象")
            sensor_id = _identifier(sensor.get("id"), f"{kind} id")
            if (kind, sensor_id) in sensor_ids:
                raise ValueError(f"重复的 {kind} id：{sensor_id}")
            sensor_ids.add((kind, sensor_id))
            frames = sensor.get("frames", [])
            if not isinstance(frames, list) or not frames:
                raise ValueError(f"{kind} {sensor_id} 没有 frames")
            labels: list[str] = []
            seen_labels: set[str] = set()
            for frame_index, frame in enumerate(frames):
                if not isinstance(frame, dict):
                    raise ValueError(f"{kind} {sensor_id} 的 frame 必须是对象")
                label = str(frame.get("label", "")).strip()
                if not label or label in seen_labels:
                    raise ValueError(f"{kind} {sensor_id} 的 frame label 为空或重复")
                labels.append(label)
                seen_labels.add(label)
                source = _inside(dataset_root, str(frame.get("path", "")), f"{kind} {sensor_id} frame.path")
                if not source.is_file():
                    raise ValueError(f"找不到文件：{source.relative_to(dataset_root)}")
                if kind == "camera" and source.suffix.lower() not in _IMAGE_SUFFIXES:
                    raise ValueError(f"不支持的图像格式：{source.name}")
                if kind == "lidar":
                    _point_format(frame.get("pointFormat", sensor.get("pointFormat", "autocalib-acp1")))
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise ValueError("所有传感器必须按相同顺序提供完全一致的 frame label")
            if kind == "lidar":
                _point_format(sensor.get("pointFormat", "autocalib-acp1"))
            else:
                intrinsic = sensor.get("intrinsic")
                if intrinsic is not None:
                    if not isinstance(intrinsic, list) or len(intrinsic) != 9:
                        raise ValueError(f"camera {sensor_id} intrinsic 必须包含 9 个数")
                    try:
                        [float(item) for item in intrinsic]
                        [float(item) for item in sensor.get("distortion", [])]
                    except (TypeError, ValueError) as error:
                        raise ValueError(f"camera {sensor_id} 内参和畸变必须是数值") from error
    return cameras, lidars, reference_labels or []


def _fingerprint(
    manifest_path: Path,
    dataset_root: Path,
    source_path: str,
    cameras: list[dict],
    lidars: list[dict],
) -> str:
    digest = hashlib.sha256(PREPARATION_ALGORITHM.encode())
    digest.update(source_path.encode())
    digest.update(manifest_path.read_bytes())
    references: dict[str, tuple[Path, object]] = {}
    for sensor in [*cameras, *lidars]:
        for frame in sensor["frames"]:
            relative = str(frame["path"]).replace("\\", "/")
            source = _inside(dataset_root, relative, "frame.path")
            expected_hash = frame.get("sha256")
            previous = references.get(relative)
            previous_hash = previous[1] if previous is not None else None
            has_expected_hash = expected_hash is not None and expected_hash != ""
            has_previous_hash = previous_hash is not None and previous_hash != ""
            if has_previous_hash and has_expected_hash and str(previous_hash).lower() != str(expected_hash).lower():
                raise ValueError(f"{relative} 声明了不一致的 sha256")
            references[relative] = (source, expected_hash if has_expected_hash else previous_hash)
    for relative, (source, frame_hash) in sorted(references.items()):
        stat = source.stat()
        digest.update(relative.encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
        if frame_hash:
            digest.update(str(frame_hash).lower().encode())
    return digest.hexdigest()[:20]


def _safe_slug(value: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "dataset")).strip("-.")
    return (slug or "dataset")[:60]


def prepare_paired_dataset(relative_path: str, progress: Progress) -> tuple[dict, bool]:
    dataset_root, manifest_path, normalized_path, value = _read_manifest(relative_path)
    cameras, lidars, labels = _validated_sensors(dataset_root, value)
    fingerprint = _fingerprint(manifest_path, dataset_root, normalized_path, cameras, lidars)
    destination = PREPARED_ROOT / fingerprint
    cached_metadata = destination / "dataset.json"
    if cached_metadata.is_file():
        dataset = json.loads(cached_metadata.read_text(encoding="utf-8"))
        _install_dataset(dataset)
        progress(100, f"已命中缓存：{value.get('name') or dataset_root.name}", {"cached": True})
        return dataset, True

    PREPARED_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint}-", dir=PREPARED_ROOT))
    total_frames = len(labels) * (len(cameras) + len(lidars))
    completed = 0
    verified_hashes = 0
    anchor_timestamps: dict[str, int] = {}
    try:
        output_lidars: list[dict] = []
        for sensor_index, sensor in enumerate(lidars):
            sensor_id = _identifier(sensor.get("id"), "lidar id")
            jobs = []
            for frame_index, frame in enumerate(sensor["frames"]):
                label = labels[frame_index]
                source = _inside(dataset_root, str(frame["path"]), "lidar frame.path")
                source_header = _acp1_header(source)
                fallback = source_header[1] if source_header is not None else frame_index * 100_000_000
                timestamp_ns = _timestamp(frame.get("timestampNs"), fallback, f"lidar {sensor_id} timestampNs")
                anchor_timestamps.setdefault(label, timestamp_ns)
                filename = f"lidar-{sensor_index}-{frame_index}.bin"
                jobs.append((label, source, frame.get("sha256"), filename, timestamp_ns,
                             frame.get("pointFormat", sensor.get("pointFormat", "autocalib-acp1"))))

            def prepare_lidar(job: tuple) -> tuple[dict, int]:
                label, source, expected_hash, filename, timestamp_ns, point_format = job
                verified = _verify_hash(source, expected_hash)
                point_count, stable_ratio, invalid_point_count = _write_acp1(
                    staging / filename,
                    source,
                    timestamp_ns,
                    point_format,
                )
                return ({
                    "label": label,
                    "timestampNs": str(timestamp_ns),
                    "pointCount": point_count,
                    "sourcePointCount": point_count + invalid_point_count,
                    "invalidPointCount": invalid_point_count,
                    "stableRatio": round(stable_ratio, 6),
                    "url": f"/data/cache/{fingerprint}/{filename}",
                }, verified)

            output_frames = []
            with ThreadPoolExecutor(max_workers=_worker_count(len(jobs))) as executor:
                results = executor.map(prepare_lidar, jobs)
                for (output_frame, verified), job in zip(results, jobs):
                    output_frames.append(output_frame)
                    verified_hashes += verified
                    label = job[0]
                    completed += 1
                    progress(5 + round(90 * completed / total_frames), f"准备点云 {sensor_id} · 帧 {label}", None)
            output_lidars.append({
                "id": sensor_id,
                "name": str(sensor.get("name") or sensor_id),
                "topic": str(sensor.get("topic") or f"paired/lidar/{sensor_id}"),
                "frames": output_frames,
            })

        output_cameras: list[dict] = []
        camera_calibration: dict[str, dict] = {}
        for sensor_index, sensor in enumerate(cameras):
            sensor_id = _identifier(sensor.get("id"), "camera id")
            jobs = []
            for frame_index, frame in enumerate(sensor["frames"]):
                label = labels[frame_index]
                fallback = anchor_timestamps.get(label, frame_index * 100_000_000)
                timestamp_ns = _timestamp(frame.get("timestampNs"), fallback, f"camera {sensor_id} timestampNs")
                source = _inside(dataset_root, str(frame["path"]), "camera frame.path")
                suffix = source.suffix.lower()
                filename = f"camera-{sensor_index}-{frame_index}{suffix}"
                anchor = anchor_timestamps.get(label, timestamp_ns)
                jobs.append((label, source, frame.get("sha256"), filename, timestamp_ns, anchor))

            def prepare_camera(job: tuple) -> tuple[dict, int]:
                label, source, expected_hash, filename, timestamp_ns, anchor = job
                verified = _verify_hash(source, expected_hash)
                with Image.open(source) as image:
                    width, height = image.size
                _link_or_copy(source, staging / filename)
                return ({
                    "label": label,
                    "timestampNs": str(timestamp_ns),
                    "syncOffsetMs": round((timestamp_ns - anchor) / 1e6, 6),
                    "width": width,
                    "height": height,
                    "url": f"/data/cache/{fingerprint}/{filename}",
                }, verified)

            output_frames = []
            with ThreadPoolExecutor(max_workers=_worker_count(len(jobs))) as executor:
                results = executor.map(prepare_camera, jobs)
                for (output_frame, verified), job in zip(results, jobs):
                    output_frames.append(output_frame)
                    verified_hashes += verified
                    label = job[0]
                    completed += 1
                    progress(5 + round(90 * completed / total_frames), f"准备图像 {sensor_id} · 帧 {label}", None)
            output_cameras.append({
                "id": sensor_id,
                "name": str(sensor.get("name") or sensor_id),
                "topic": str(sensor.get("topic") or f"paired/camera/{sensor_id}"),
                "frames": output_frames,
            })
            intrinsic = sensor.get("intrinsic")
            if intrinsic is not None:
                if not isinstance(intrinsic, list) or len(intrinsic) != 9:
                    raise ValueError(f"camera {sensor_id} intrinsic 必须包含 9 个数")
                camera_calibration[sensor_id] = {
                    "intrinsic": [float(item) for item in intrinsic],
                    "distortion": [float(item) for item in sensor.get("distortion", [])],
                    "distortionModel": str(sensor.get("distortionModel") or "rational"),
                    "coordinateSystem": str(sensor.get("coordinateSystem") or "raw"),
                    **({"serial": str(sensor["serial"])} if sensor.get("serial") else {}),
                    **({"model": str(sensor["model"])} if sensor.get("model") else {}),
                }

        timestamps = [anchor_timestamps[label] for label in labels]
        dataset = {
            "id": f"paired-{_safe_slug(value.get('id') or dataset_root.name)}-{fingerprint[:12]}",
            "rigId": _identifier(value.get("rigId"), "rigId"),
            "name": str(value.get("name") or dataset_root.name),
            "sourceFile": MANIFEST_NAME,
            "sourcePath": normalized_path,
            "sourceKind": "paired-directory",
            "fingerprint": fingerprint,
            "preparationAlgorithm": PREPARATION_ALGORITHM,
            "sourceManifestVersion": FORMAT_VERSION,
            "verifiedFileHashCount": verified_hashes,
            "stationaryIntervalCount": len(labels),
            "stationaryDetectionMethod": "explicit-paired-frames",
            "selectedInterval": {
                "startNs": str(min(timestamps)),
                "endNs": str(max(timestamps)),
                "durationSeconds": round((max(timestamps) - min(timestamps)) / 1e9, 6),
            },
            "stationaryIntervals": [
                {"startNs": str(anchor_timestamps[label]), "endNs": str(anchor_timestamps[label]), "durationSeconds": 0.0}
                for label in labels
            ],
            "anchorFrames": [{"label": label, "timestampNs": str(anchor_timestamps[label])} for label in labels],
            "lidars": output_lidars,
            "cameras": output_cameras,
            "cameraCalibration": camera_calibration,
            "extrinsics": None,
            "calibrationCompatibility": "unknown-extrinsic",
            "topicSummary": [],
        }
        (staging / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        # Files may be hard links to the user's full-density source data.
        # Never chmod a linked file because that would also change the source.
        for generated in staging.rglob("*"):
            if generated.is_dir():
                generated.chmod(0o755)
        staging.chmod(0o755)
        if destination.exists():
            shutil.rmtree(staging)
            dataset = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
        else:
            try:
                os.replace(staging, destination)
            except OSError:
                # Another group may have prepared the same immutable source in
                # parallel. Reuse its completed cache instead of failing.
                completed_cache = destination / "dataset.json"
                if not completed_cache.is_file():
                    raise
                shutil.rmtree(staging, ignore_errors=True)
                dataset = json.loads(completed_cache.read_text(encoding="utf-8"))
        _install_dataset(dataset)
        progress(100, f"成对数据集准备完成：{dataset['name']}", {"cached": False, "frameCount": len(labels)})
        return dataset, False
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
