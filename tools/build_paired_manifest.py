#!/usr/bin/env python3
"""Build an AutoCalib manifest for already paired image and point-cloud files.

The tool writes only ``autocalib.json``.  Raw files are neither copied nor
rewritten, and point clouds are never sampled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


MANIFEST_NAME = "autocalib.json"
FORMAT_VERSION = 1
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
CLOUD_SUFFIXES = (".bin", ".npy")
POINT_FORMAT_PRESETS = {"xyz-f32", "xyzi-f32", "xyzring-f32", "xyzir-f32", "autocalib-acp1"}


@dataclass(frozen=True)
class SensorInput:
    kind: str
    identifier: str
    directory: Path

    @property
    def key(self) -> str:
        return f"{self.kind}.{self.identifier}"


def _assignment(value: str, description: str) -> tuple[str, str]:
    key, separator, raw_value = value.partition("=")
    key = key.strip()
    raw_value = raw_value.strip()
    if not separator or not key or not raw_value:
        raise argparse.ArgumentTypeError(f"{description} must use NAME=VALUE")
    return key, raw_value


def _assignment_map(values: Iterable[str], description: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, raw_value = _assignment(value, description)
        if key in result:
            raise ValueError(f"{description} repeats {key!r}")
        result[key] = raw_value
    return result


def _identifier(value: str, description: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"{description} may contain only letters, numbers, dots, underscores, and hyphens")
    return value


def _inside(root: Path, value: Path, description: str) -> Path:
    candidate = value if value.is_absolute() else root / value
    resolved = candidate.expanduser().resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{description} must be inside the dataset root")
    return resolved


def _suffixes(values: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple((value if value.startswith(".") else f".{value}").lower() for value in values)
    if not normalized or any(value == "." for value in normalized):
        raise ValueError("file suffix list cannot be empty")
    return normalized


def _collect_files(root: Path, sensor: SensorInput, suffixes: tuple[str, ...]) -> dict[str, Path]:
    directory = _inside(root, sensor.directory, f"{sensor.key} directory")
    if not directory.is_dir():
        raise ValueError(f"{sensor.key} directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        resolved = _inside(root, path, f"{sensor.key} frame")
        label = path.relative_to(directory).with_suffix("").as_posix()
        if label in result:
            raise ValueError(f"{sensor.key} has duplicate frame label {label!r}")
        result[label] = resolved
    if not result:
        raise ValueError(f"{sensor.key} has no files with suffixes {', '.join(suffixes)}")
    return result


def _point_format(value: str) -> str | dict:
    normalized = value.strip().lower()
    if normalized in POINT_FORMAT_PRESETS:
        return normalized
    encoding, separator, raw_fields = normalized.partition(":")
    fields = [field.strip() for field in raw_fields.split(",") if field.strip()]
    if not separator or encoding not in {"float32", "npy"} or not {"x", "y", "z"}.issubset(fields):
        raise ValueError(
            "point format must be a preset or float32:x,y,z,... / npy:x,y,z,..."
        )
    if len(fields) != len(set(fields)):
        raise ValueError("point format fields must be unique")
    return {"encoding": encoding, "fields": fields, "byteOrder": "little"}


def _timestamp_map(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"timestamp file does not exist: {path}")
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            rows = value.items()
        elif isinstance(value, list):
            rows = ((item.get("label"), item.get("timestampNs")) for item in value if isinstance(item, dict))
        else:
            raise ValueError(f"timestamp JSON must be an object or list: {path}")
    else:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = [(row.get("label"), row.get("timestampNs")) for row in csv.DictReader(stream)]
    result: dict[str, str] = {}
    for raw_label, raw_timestamp in rows:
        label = str(raw_label or "").strip()
        if not label or raw_timestamp is None or str(raw_timestamp).strip() == "":
            raise ValueError(f"timestamp rows require label and timestampNs: {path}")
        try:
            timestamp = int(str(raw_timestamp))
        except ValueError as error:
            raise ValueError(f"invalid timestampNs for {label!r}: {raw_timestamp}") from error
        if not -(2**63) <= timestamp < 2**63:
            raise ValueError(f"timestampNs is outside int64 for {label!r}")
        if label in result:
            raise ValueError(f"duplicate timestamp label {label!r}: {path}")
        result[label] = str(timestamp)
    return result


def _camera_calibration(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"camera calibration must be a JSON object: {path}")
    intrinsic = value.get("intrinsic")
    if not isinstance(intrinsic, list) or len(intrinsic) != 9:
        raise ValueError(f"camera calibration intrinsic must contain 9 values: {path}")
    result = {
        "intrinsic": [float(item) for item in intrinsic],
        "distortion": [float(item) for item in value.get("distortion", [])],
        "distortionModel": str(value.get("distortionModel") or "rational"),
        "coordinateSystem": str(value.get("coordinateSystem") or "raw"),
    }
    for key in ("name", "serial", "model"):
        if value.get(key) is not None and value.get(key) != "":
            result[key] = str(value[key])
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    root: Path,
    dataset_id: str,
    rig_id: str,
    name: str,
    cameras: list[SensorInput],
    lidars: list[SensorInput],
    point_formats: dict[str, str | dict],
    timestamp_files: dict[str, Path] | None = None,
    camera_calibrations: dict[str, Path] | None = None,
    image_suffixes: tuple[str, ...] = IMAGE_SUFFIXES,
    cloud_suffixes: tuple[str, ...] = CLOUD_SUFFIXES,
    allow_unmatched: bool = False,
    include_sha256: bool = False,
    hash_workers: int = 4,
) -> tuple[dict, dict[str, int]]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root does not exist: {root}")
    dataset_id = _identifier(dataset_id, "dataset id")
    rig_id = _identifier(rig_id, "rig id")
    if not lidars or (not cameras and len(lidars) < 2):
        raise ValueError("provide at least one camera and one lidar, or at least two lidars")

    sensors = [*cameras, *lidars]
    if len({sensor.key for sensor in sensors}) != len(sensors):
        raise ValueError("sensor IDs must be unique within each sensor kind")
    expected_formats = {sensor.identifier for sensor in lidars}
    if set(point_formats) != expected_formats:
        missing = sorted(expected_formats - set(point_formats))
        extra = sorted(set(point_formats) - expected_formats)
        raise ValueError(f"point formats do not match lidars; missing={missing}, extra={extra}")

    files: dict[str, dict[str, Path]] = {}
    normalized_image_suffixes = _suffixes(image_suffixes)
    normalized_cloud_suffixes = _suffixes(cloud_suffixes)
    for sensor in cameras:
        files[sensor.key] = _collect_files(root, sensor, normalized_image_suffixes)
    for sensor in lidars:
        files[sensor.key] = _collect_files(root, sensor, normalized_cloud_suffixes)

    label_sets = [set(items) for items in files.values()]
    common_labels = set.intersection(*label_sets)
    union_labels = set.union(*label_sets)
    dropped = {key: len(items) - len(common_labels) for key, items in files.items()}
    if not common_labels:
        raise ValueError("sensors do not share any frame labels")
    if common_labels != union_labels and not allow_unmatched:
        details = ", ".join(f"{key}={len(items)}" for key, items in files.items())
        raise ValueError(
            f"sensor frame labels differ ({details}, common={len(common_labels)}); "
            "use --allow-unmatched to keep only the intersection"
        )
    labels = sorted(common_labels)

    timestamp_files = timestamp_files or {}
    unknown_timestamp_keys = set(timestamp_files) - {sensor.key for sensor in sensors}
    if unknown_timestamp_keys:
        raise ValueError(f"timestamp files reference unknown sensors: {sorted(unknown_timestamp_keys)}")
    timestamps: dict[str, dict[str, str]] = {}
    for sensor_key, path in timestamp_files.items():
        values = _timestamp_map(path.expanduser().resolve())
        missing = set(labels) - set(values)
        if missing:
            raise ValueError(f"{sensor_key} timestamps are missing {len(missing)} selected frame labels")
        timestamps[sensor_key] = values

    camera_calibrations = camera_calibrations or {}
    unknown_calibrations = set(camera_calibrations) - {sensor.identifier for sensor in cameras}
    if unknown_calibrations:
        raise ValueError(f"camera calibrations reference unknown cameras: {sorted(unknown_calibrations)}")

    file_hashes: dict[Path, str] = {}
    if include_sha256:
        unique_paths = sorted({files[sensor.key][label] for sensor in sensors for label in labels})
        with ThreadPoolExecutor(max_workers=max(1, min(len(unique_paths), hash_workers))) as executor:
            file_hashes = dict(zip(unique_paths, executor.map(_sha256, unique_paths)))

    def frames_for(sensor: SensorInput) -> list[dict]:
        frames = []
        for label in labels:
            path = files[sensor.key][label]
            frame = {"label": label, "path": path.relative_to(root).as_posix()}
            if sensor.key in timestamps:
                frame["timestampNs"] = timestamps[sensor.key][label]
            if include_sha256:
                frame["sha256"] = file_hashes[path]
            frames.append(frame)
        return frames

    output_cameras = []
    for sensor in cameras:
        item = {"id": sensor.identifier, "name": sensor.identifier, "frames": frames_for(sensor)}
        if sensor.identifier in camera_calibrations:
            item.update(_camera_calibration(camera_calibrations[sensor.identifier].expanduser().resolve()))
            item["id"] = sensor.identifier
        output_cameras.append(item)
    output_lidars = [
        {
            "id": sensor.identifier,
            "name": sensor.identifier,
            "pointFormat": point_formats[sensor.identifier],
            "frames": frames_for(sensor),
        }
        for sensor in lidars
    ]
    manifest = {
        "formatVersion": FORMAT_VERSION,
        "id": dataset_id,
        "rigId": rig_id,
        "name": name,
        "cameras": output_cameras,
        "lidars": output_lidars,
    }
    return manifest, dropped


def _write_manifest(path: Path, manifest: dict, force: bool) -> None:
    if path.exists() and not force:
        raise ValueError(f"manifest already exists: {path}; pass --force to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = ""
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary_name = stream.name
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dataset directory that will contain autocalib.json")
    parser.add_argument("--id", required=True, dest="dataset_id")
    parser.add_argument("--rig-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--camera", action="append", default=[], metavar="ID=DIRECTORY")
    parser.add_argument("--lidar", action="append", default=[], metavar="ID=DIRECTORY")
    parser.add_argument("--point-format", action="append", default=[], metavar="LIDAR_ID=FORMAT")
    parser.add_argument("--timestamps", action="append", default=[], metavar="camera.ID|lidar.ID=JSON_OR_CSV")
    parser.add_argument("--camera-calibration", action="append", default=[], metavar="CAMERA_ID=JSON")
    parser.add_argument("--image-suffix", action="append", default=[])
    parser.add_argument("--cloud-suffix", action="append", default=[])
    parser.add_argument("--allow-unmatched", action="store_true")
    parser.add_argument("--sha256", action="store_true", help="record a SHA-256 digest for every referenced file")
    parser.add_argument("--hash-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    camera_inputs = _assignment_map(args.camera, "--camera")
    lidar_inputs = _assignment_map(args.lidar, "--lidar")
    point_format_inputs = _assignment_map(args.point_format, "--point-format")
    timestamp_inputs = _assignment_map(args.timestamps, "--timestamps")
    calibration_inputs = _assignment_map(args.camera_calibration, "--camera-calibration")
    cameras = [
        SensorInput("camera", _identifier(identifier, "camera id"), Path(directory))
        for identifier, directory in camera_inputs.items()
    ]
    lidars = [
        SensorInput("lidar", _identifier(identifier, "lidar id"), Path(directory))
        for identifier, directory in lidar_inputs.items()
    ]
    point_formats = {
        identifier: _point_format(value)
        for identifier, value in point_format_inputs.items()
    }
    timestamp_files = {
        key: Path(value)
        for key, value in timestamp_inputs.items()
    }
    camera_calibrations = {
        key: Path(value)
        for key, value in calibration_inputs.items()
    }
    manifest, dropped = build_manifest(
        root=root,
        dataset_id=args.dataset_id,
        rig_id=args.rig_id,
        name=args.name,
        cameras=cameras,
        lidars=lidars,
        point_formats=point_formats,
        timestamp_files=timestamp_files,
        camera_calibrations=camera_calibrations,
        image_suffixes=tuple(args.image_suffix) or IMAGE_SUFFIXES,
        cloud_suffixes=tuple(args.cloud_suffix) or CLOUD_SUFFIXES,
        allow_unmatched=args.allow_unmatched,
        include_sha256=args.sha256,
        hash_workers=args.hash_workers,
    )
    output = root / MANIFEST_NAME
    _write_manifest(output, manifest, args.force)
    dropped_total = sum(dropped.values())
    first_sensor = (manifest["cameras"] or manifest["lidars"])[0]
    print(f"Wrote {output} with {len(first_sensor['frames'])} paired frames")
    if dropped_total:
        print("Unmatched files excluded: " + ", ".join(f"{key}={count}" for key, count in dropped.items() if count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
