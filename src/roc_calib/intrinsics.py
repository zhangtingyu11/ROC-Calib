from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import cv2
import numpy as np
import yaml

from .groups import GROUP_ROOT, PREPARED_ROOT, _path, _prepared_datasets, _write_json


PROFILE_NAME = "camera-intrinsics.json"
COORDINATE_SYSTEM = "raw-distorted-v1"
INTRINSICS_ROOT = Path(os.environ.get("AUTOCALIB_INTRINSICS_ROOT", str(GROUP_ROOT / "intrinsics"))).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _numbers(value: object, name: str, length: int | None = None) -> list[float]:
    if not isinstance(value, list) or not all(isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{name} 必须是单层数值数组")
    result = [float(item) for item in value]
    if length is not None and len(result) != length:
        raise ValueError(f"{name} 必须包含 {length} 个数值")
    return result


def _normalize_topic(topic: str) -> str:
    return topic.strip().lstrip("/")


def _parse(contents: bytes) -> tuple[dict, str]:
    if len(contents) > 2 * 1024 * 1024:
        raise ValueError("内参 YAML 不能超过 2 MB")
    try:
        text = contents.decode("utf-8")
        value = yaml.safe_load(text)
    except (UnicodeDecodeError, yaml.YAMLError) as error:
        raise ValueError("内参文件不是有效的 UTF-8 YAML") from error
    if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("cameras"), list):
        raise ValueError("YAML 必须包含 version: 1 和 cameras 列表")
    cameras: list[dict] = []
    topics: set[str] = set()
    for index, item in enumerate(value["cameras"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"cameras[{index}] 必须是对象")
        topic = str(item.get("topic", "")).strip()
        normalized_topic = _normalize_topic(topic)
        if not normalized_topic or normalized_topic in topics:
            raise ValueError(f"cameras[{index}] 的 topic 为空或重复")
        camera_type = str(item.get("camera_type", "")).strip().lower()
        if camera_type not in {"pinhole", "fisheye"}:
            raise ValueError(f"{topic} 的 camera_type 必须是 pinhole 或 fisheye")
        matrix = _numbers(item.get("camera_matrix"), f"{topic}.camera_matrix", 9)
        distortion = _numbers(item.get("distortion_coefficients"), f"{topic}.distortion_coefficients")
        if camera_type == "fisheye" and len(distortion) != 4:
            raise ValueError(f"{topic} 的 fisheye 畸变参数必须为 4 项")
        if camera_type == "pinhole" and len(distortion) not in {4, 5, 8, 12, 14}:
            raise ValueError(f"{topic} 的 pinhole 畸变参数必须为 4、5、8、12 或 14 项")
        width, height = item.get("image_width"), item.get("image_height")
        if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
            raise ValueError(f"{topic} 必须提供有效的 image_width 和 image_height")
        topics.add(normalized_topic)
        cameras.append({
            "topic": topic,
            "normalizedTopic": normalized_topic,
            "cameraType": camera_type,
            "imageWidth": width,
            "imageHeight": height,
            "cameraMatrix": matrix,
            "distortionCoefficients": distortion,
        })
    return {"version": 1, "cameras": cameras}, text


def _source_path(url: str) -> Path:
    prefix = "/data/cache/"
    if not url.startswith(prefix):
        raise ValueError(f"不支持的原始图像路径：{url}")
    path = (PREPARED_ROOT / url.removeprefix(prefix)).resolve()
    if PREPARED_ROOT not in path.parents or not path.is_file():
        raise ValueError(f"原始图像不存在：{url}")
    return path


def _intrinsics_path(relative_path: str) -> Path:
    if relative_path.startswith("/") or "\x00" in relative_path:
        raise ValueError("invalid intrinsics path")
    path = (INTRINSICS_ROOT / relative_path).resolve()
    if path != INTRINSICS_ROOT and INTRINSICS_ROOT not in path.parents:
        raise ValueError("intrinsics path escapes public root")
    return path


def list_intrinsics_directory(relative_path: str = "") -> dict:
    INTRINSICS_ROOT.mkdir(parents=True, exist_ok=True)
    path = _intrinsics_path(relative_path)
    if not path.is_dir():
        raise FileNotFoundError(relative_path)
    entries: list[dict] = []
    for child in sorted(path.iterdir(), key=lambda item: item.name.lower()):
        try:
            resolved = child.resolve()
            if resolved != INTRINSICS_ROOT and INTRINSICS_ROOT not in resolved.parents:
                continue
            if child.is_dir():
                kind = "directory"
            elif child.is_file() and child.suffix.lower() in {".yaml", ".yml"}:
                kind = "yaml"
            else:
                continue
            entries.append({
                "name": child.name,
                "path": str(resolved.relative_to(INTRINSICS_ROOT)),
                "kind": kind,
                "sizeBytes": child.stat().st_size if kind == "yaml" else 0,
            })
        except (OSError, ValueError):
            continue
    current = "" if path == INTRINSICS_ROOT else str(path.relative_to(INTRINSICS_ROOT))
    parent = None if not current else str(Path(current).parent)
    if parent == ".":
        parent = ""
    return {"root": str(INTRINSICS_ROOT), "path": current, "parent": parent, "entries": entries}


def _undistort(image: np.ndarray, camera: dict) -> tuple[np.ndarray, list[float]]:
    height, width = image.shape[:2]
    if (width, height) != (camera["imageWidth"], camera["imageHeight"]):
        raise ValueError(
            f'{camera["topic"]} 图像尺寸为 {width}x{height}，YAML 声明为 '
            f'{camera["imageWidth"]}x{camera["imageHeight"]}'
        )
    size = (width, height)
    matrix = np.asarray(camera["cameraMatrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(camera["distortionCoefficients"], dtype=np.float64)
    if camera["cameraType"] == "fisheye":
        output_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            matrix, distortion.reshape(4, 1), size, np.eye(3), balance=0.0, new_size=size,
        )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            matrix, distortion.reshape(4, 1), np.eye(3), output_matrix, size, cv2.CV_32FC1,
        )
        output = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    else:
        output_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 0.0, size)
        output = cv2.undistort(image, matrix, distortion, None, output_matrix)
    return output, output_matrix.reshape(-1).tolist()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def import_intrinsics(group_id: str, filename: str, contents: bytes, source_path: Path | None = None) -> dict:
    group_path = _path(group_id)
    metadata_path = group_path / "group.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(group_id)
    parsed, source_text = _parse(contents)
    group = json.loads(metadata_path.read_text())
    dataset_ids = set(group.get("datasetIds", []))
    datasets = [item for item in _prepared_datasets() if item.get("id") in dataset_ids]
    if not datasets:
        raise ValueError("当前标定组还没有可匹配的数据包")
    config_by_topic = {item["normalizedTopic"]: item for item in parsed["cameras"]}
    required_topics = {
        _normalize_topic(str(camera.get("topic", "")))
        for dataset in datasets for camera in dataset.get("cameras", [])
    }
    missing = sorted(required_topics - set(config_by_topic))
    unknown = sorted(set(config_by_topic) - required_topics)
    if missing:
        raise ValueError("YAML 缺少当前组相机 topic：" + "、".join(missing))

    dataset_profiles: dict[str, dict] = {}
    for dataset in datasets:
        dataset_id = str(dataset["id"])
        camera_profiles: dict[str, dict] = {}
        for camera in dataset.get("cameras", []):
            camera_id = str(camera["id"])
            config = config_by_topic[_normalize_topic(str(camera["topic"]))]
            frames = {
                str(frame["label"]): str(frame["url"])
                for frame in camera.get("frames", [])
            }
            distortion_model = (
                "fisheye" if config["cameraType"] == "fisheye"
                else "radtan" if len(config["distortionCoefficients"]) <= 5
                else "rational"
            )
            # Calibration, annotation, projection, and evaluation all stay in
            # the original distorted-image coordinate system. No rectified
            # frame, replacement K, balance/crop, or image resampling exists.
            camera_profiles[camera_id] = {
                "topic": camera["topic"], "cameraType": config["cameraType"],
                "intrinsic": config["cameraMatrix"],
                "distortion": config["distortionCoefficients"],
                "distortionModel": distortion_model,
                "rawIntrinsic": config["cameraMatrix"],
                "rawDistortion": config["distortionCoefficients"],
                "rawDistortionModel": distortion_model,
                "coordinateSystem": COORDINATE_SYSTEM, "frames": frames,
            }
        dataset_profiles[dataset_id] = {"cameras": camera_profiles}

    profile = {
        "version": 1, "sourceFile": filename, "importedAt": _now(),
        "cameras": [camera for camera in parsed["cameras"] if camera["normalizedTopic"] in required_topics],
        "ignoredTopics": unknown,
        "datasets": dataset_profiles,
    }
    group_source = group_path / "source" / "camera_intrinsics.yaml"
    if source_path is None:
        _atomic_text(group_source, source_text)
    else:
        temporary_link = group_source.with_name(f".{group_source.name}.{uuid4().hex}.tmp")
        temporary_link.symlink_to(Path(os.path.relpath(source_path, group_source.parent)))
        os.replace(temporary_link, group_source)
    _write_json(group_path / "frames" / PROFILE_NAME, profile)
    group["intrinsicsImportedAt"] = profile["importedAt"]
    group["intrinsicsSourceFile"] = filename
    group["updatedAt"] = profile["importedAt"]
    _write_json(metadata_path, group)
    return profile


def import_intrinsics_path(group_id: str, relative_path: str) -> dict:
    source = _intrinsics_path(relative_path)
    if not source.is_file() or source.suffix.lower() not in {".yaml", ".yml"}:
        raise FileNotFoundError(relative_path)
    return import_intrinsics(group_id, source.name, source.read_bytes(), source)


def refresh_imported_intrinsics(group_id: str) -> dict | None:
    """Reapply the group's YAML after its selected dataset cache changes."""
    group_source = _path(group_id) / "source" / "camera_intrinsics.yaml"
    if not group_source.is_file():
        return None
    metadata = json.loads((_path(group_id) / "group.json").read_text())
    filename = str(metadata.get("intrinsicsSourceFile") or group_source.resolve().name)
    source_path = group_source.resolve() if group_source.is_symlink() else None
    return import_intrinsics(group_id, filename, group_source.read_bytes(), source_path)


def read_intrinsics(group_id: str) -> dict:
    path = _path(group_id) / "frames" / PROFILE_NAME
    if not path.is_file():
        return {"version": 1, "sourceFile": None, "importedAt": None, "cameras": [], "datasets": {}}
    return json.loads(path.read_text())


def resolve_undistorted_frame(group_id: str, dataset_id: str, filename: str) -> Path:
    if not dataset_id or "/" in dataset_id or "\\" in dataset_id or not filename or "/" in filename or "\\" in filename:
        raise ValueError("invalid frame path")
    root = (_path(group_id) / "frames" / "undistorted" / dataset_id).resolve()
    path = (root / filename).resolve()
    if path.parent != root or not path.is_file():
        raise FileNotFoundError(filename)
    return path
