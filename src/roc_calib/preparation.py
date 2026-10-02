from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import shutil
import sqlite3
import struct
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .rosbag import (
    Rosbag2Reader,
    deserialize_compressed_image,
    deserialize_gnss_info,
    deserialize_pointcloud2,
    structured_to_matrix,
)

from .groups import BAG_ROOT, PREPARED_ROOT, _bag_files, _bag_path


ALGORITHM_VERSION = "manual-scenes-v2-per-lidar-frames"
MAX_STATIC_SCENES = 10
MAX_NEAR_FIELD_SAMPLES = 300
Progress = Callable[[int, str, dict | None], None]
_MANIFEST_LOCK = threading.RLock()
_CODEC_HEADERS: dict[str, bytes] = {}
_CAMERA_INDEX_LOCK = threading.RLock()
_CAMERA_INDICES: dict[tuple[str, tuple[tuple[str, int, int], ...]], list[tuple[Path, list[int], list[int]]]] = {}
_POINTCLOUD_INDEX_LOCK = threading.RLock()
_POINTCLOUD_INDICES: dict[tuple[str, tuple[tuple[str, int, int], ...]], list[tuple[int, Path, int]]] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _manifest_path() -> Path:
    return PREPARED_ROOT / "manifest.json"


def _empty_manifest() -> dict:
    return {
        "generatedAt": _now(),
        "datasets": [],
        "calibration": {"source": "user-import", "note": "相机内参尚未导入", "lidars": {}, "cameras": {}},
    }


def _load_manifest() -> dict:
    try:
        value = json.loads(_manifest_path().read_text())
        return value if isinstance(value, dict) else _empty_manifest()
    except (OSError, json.JSONDecodeError):
        return _empty_manifest()


def _database_paths(relative_path: str) -> tuple[Path, list[Path]]:
    bag_path = _bag_path(relative_path)
    databases = _bag_files(bag_path)
    if not databases:
        raise ValueError(f"不是可读取的 ROS2 bag：{relative_path}")
    if any(path.suffix.lower() != ".db3" for path in databases):
        raise ValueError("当前处理器暂只支持 ROS2 sqlite3 (.db3) 数据包")
    return bag_path, databases


def _fingerprint(databases: list[Path], targets: list[int] | None = None, selection_key: str = "") -> str:
    digest = hashlib.sha256(ALGORITHM_VERSION.encode())
    for database in databases:
        stat = database.stat()
        digest.update(database.name.encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    if targets is not None:
        digest.update("|".join(str(target) for target in targets).encode())
    digest.update(selection_key.encode())
    return digest.hexdigest()[:20]


def _topic_time_bounds(databases: list[Path], topic: str) -> tuple[int, int]:
    bounds: list[tuple[int, int]] = []
    for database in databases:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
            if row is None:
                continue
            first = connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp ASC LIMIT 1", (row[0],)
            ).fetchone()
            last = connection.execute(
                "SELECT timestamp FROM messages WHERE topic_id=? ORDER BY timestamp DESC LIMIT 1", (row[0],)
            ).fetchone()
            if first and last:
                bounds.append((int(first[0]), int(last[0])))
    if not bounds:
        raise ValueError(f"topic 没有消息：{topic}")
    return min(item[0] for item in bounds), max(item[1] for item in bounds)


def manual_frame_candidates(relative_path: str) -> dict:
    """Return a one-second manual timeline without decoding the complete bag."""
    bag_path, databases = _database_paths(relative_path)
    timeline_cache = PREPARED_ROOT / "manual-timelines" / f"{_fingerprint(databases)}.json"
    try:
        cached = json.loads(timeline_cache.read_text())
        if isinstance(cached, dict) and cached.get("candidates"):
            return {**cached, "path": relative_path, "name": bag_path.name}
    except (OSError, json.JSONDecodeError):
        pass
    topics: dict[str, dict[str, object]] = {}
    for database in databases:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            for name, message_type in connection.execute("SELECT name,type FROM topics"):
                topics.setdefault(str(name), {"name": str(name), "type": str(message_type)})
    lidar_topics = sorted(
        name for name, item in topics.items()
        if item["type"] == "sensor_msgs/msg/PointCloud2"
    )
    camera_topics = sorted(
        name for name, item in topics.items()
        if item["type"] == "sensor_msgs/msg/CompressedImage"
    )
    if not lidar_topics:
        raise ValueError(f"{bag_path.name} 未发现 PointCloud2 topic")
    start, end = _topic_time_bounds(databases, lidar_topics[0])
    targets = list(range(start, end + 1, 1_000_000_000))
    if not targets:
        targets = [start]
    used_cameras: set[str] = set()
    used_lidars: set[str] = set()
    lidars = []
    for topic in lidar_topics:
        identifier = _sensor_id(topic, "lidar", used_lidars)
        lidars.append({"id": identifier, "name": _sensor_name(identifier, "雷达"), "topic": topic})
    cameras = []
    for topic in camera_topics:
        identifier = _sensor_id(topic, "camera", used_cameras)
        cameras.append({"id": identifier, "name": _sensor_name(identifier, "相机"), "topic": topic})
    preview_topic = next((item["topic"] for item in cameras if item["id"] == "front120"), cameras[0]["topic"] if cameras else "")
    timeline = {
        "path": relative_path,
        "name": bag_path.name,
        "startNs": str(start),
        "endNs": str(end),
        "durationSeconds": round((end - start) / 1e9, 3),
        "stepSeconds": 1,
        "previewTopic": preview_topic,
        "lidars": lidars,
        "cameras": cameras,
        "candidates": [
            {"timestampNs": str(target), "offsetSeconds": index}
            for index, target in enumerate(targets)
        ],
    }
    try:
        _atomic_json(timeline_cache, timeline)
    except OSError:
        # Timeline caching is an optimization. Read-only source inspection must
        # still work when the cache mount is absent or temporarily unavailable.
        pass
    return timeline


def manual_lidar_preview(relative_path: str, timestamp_ns: int, lidar_topic: str = "") -> Path:
    bag_path, databases = _database_paths(relative_path)
    timeline = manual_frame_candidates(relative_path)
    start_ns, end_ns = int(timeline["startNs"]), int(timeline["endNs"])
    if timestamp_ns < start_ns or timestamp_ns > end_ns:
        raise ValueError("预览时间超出数据包范围")
    topics = {str(item["topic"]) for item in timeline["lidars"]}
    topic = lidar_topic or str(timeline["lidars"][0]["topic"])
    if topic not in topics:
        raise ValueError("雷达视角不属于当前数据包")
    fingerprint = _fingerprint(databases)
    topic_key = hashlib.sha256(topic.encode()).hexdigest()[:10]
    destination = PREPARED_ROOT / "manual-lidar-previews" / fingerprint / topic_key / f"{timestamp_ns}.jpg"
    if destination.is_file():
        return destination
    message, _reference = _nearest_pointcloud(databases, topic, timestamp_ns)
    points = structured_to_matrix(message)[:, :3].astype(np.float32)
    points = points[np.all(np.isfinite(points), axis=1)]
    distance = np.linalg.norm(points[:, :2], axis=1)
    points = points[(distance > 1.0) & (distance < 90.0)]
    if len(points) > 160_000:
        points = points[::math.ceil(len(points) / 160_000)]
    width, height = 960, 540
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (8, 16, 14)
    x = np.clip(((points[:, 1] + 45.0) / 90.0 * (width - 1)).astype(np.int32), 0, width - 1)
    y = np.clip(((85.0 - points[:, 0]) / 95.0 * (height - 1)).astype(np.int32), 0, height - 1)
    height_color = np.clip((points[:, 2] + 3.0) / 7.0, 0.0, 1.0)
    canvas[y, x, 0] = (35 + height_color * 80).astype(np.uint8)
    canvas[y, x, 1] = (120 + height_color * 120).astype(np.uint8)
    canvas[y, x, 2] = (180 - height_color * 90).astype(np.uint8)
    center_x = width // 2
    canvas[-28:-8, center_x - 2:center_x + 2] = (70, 220, 170)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.jpg")
    Image.fromarray(canvas, "RGB").save(temporary, quality=90)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def manual_lidar_cloud(relative_path: str, timestamp_ns: int, lidar_topic: str = "") -> Path:
    bag_path, databases = _database_paths(relative_path)
    timeline = manual_frame_candidates(relative_path)
    start_ns, end_ns = int(timeline["startNs"]), int(timeline["endNs"])
    if timestamp_ns < start_ns or timestamp_ns > end_ns:
        raise ValueError("预览时间超出数据包范围")
    topics = {str(item["topic"]) for item in timeline["lidars"]}
    topic = lidar_topic or str(timeline["lidars"][0]["topic"])
    if topic not in topics:
        raise ValueError("雷达视角不属于当前数据包")
    fingerprint = _fingerprint(databases)
    topic_key = hashlib.sha256(topic.encode()).hexdigest()[:10]
    destination = PREPARED_ROOT / "manual-lidar-clouds-v2-stable" / fingerprint / topic_key / f"{timestamp_ns}.bin"
    if destination.is_file():
        return destination
    message, reference = _nearest_pointcloud(databases, topic, timestamp_ns)
    points = structured_to_matrix(message)[:, :3].astype(np.float32)
    reference_points = structured_to_matrix(reference)[:, :3].astype(np.float32)
    stable = _stable_flags(points, reference_points)
    finite = np.all(np.isfinite(points), axis=1)
    points, stable = points[finite], stable[finite]
    distance = np.linalg.norm(points[:, :2], axis=1)
    usable = (distance > 1.0) & (distance < 120.0)
    points, stable = points[usable], stable[usable]
    if len(points) > 240_000:
        step = math.ceil(len(points) / 240_000)
        points, stable = points[::step], stable[::step]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    _write_cloud(temporary, int(message.timestamp_ns), points, stable)
    os.replace(temporary, destination)
    destination.chmod(0o644)
    return destination


def nearest_lidar_frame_pairs(
    relative_path: str,
    source_topic: str,
    target_topic: str,
    source_selected_timestamps: list[int],
    target_selected_timestamps: list[int],
) -> list[dict]:
    _bag_path, databases = _database_paths(relative_path)
    timeline = manual_frame_candidates(relative_path)
    topics = {str(item["topic"]) for item in timeline["lidars"]}
    if source_topic not in topics or target_topic not in topics or source_topic == target_topic:
        raise ValueError("雷达帧配对包含无效 topic")
    source_records = _pointcloud_index(databases, source_topic)
    target_records = _pointcloud_index(databases, target_topic)
    if not source_records or not target_records:
        raise ValueError("雷达完整时间轴为空")

    def closest(records: list[tuple[int, Path, int]], timestamp: int) -> int:
        index = bisect.bisect_left(records, (timestamp,))
        nearby = records[max(0, index - 1):min(len(records), index + 1)]
        return min(nearby, key=lambda item: abs(item[0] - timestamp))[0]

    pair_origins: dict[tuple[int, int], set[str]] = {}
    for selected in sorted(set(source_selected_timestamps)):
        source_timestamp = closest(source_records, selected)
        target_timestamp = closest(target_records, source_timestamp)
        pair_origins.setdefault((source_timestamp, target_timestamp), set()).add("source")
    for selected in sorted(set(target_selected_timestamps)):
        target_timestamp = closest(target_records, selected)
        source_timestamp = closest(source_records, target_timestamp)
        pair_origins.setdefault((source_timestamp, target_timestamp), set()).add("target")
    if not pair_origins:
        raise ValueError("请先为左右雷达各选择至少一帧")

    cloud_keys = {
        (source_topic, source_timestamp)
        for source_timestamp, _target_timestamp in pair_origins
    } | {
        (target_topic, target_timestamp)
        for _source_timestamp, target_timestamp in pair_origins
    }

    def materialize(key: tuple[str, int]) -> tuple[tuple[str, int], Path, int]:
        topic, timestamp = key
        path = manual_lidar_cloud(relative_path, timestamp, topic)
        with path.open("rb") as stream:
            header = stream.read(16)
        return key, path, struct.unpack_from("<q", header, 8)[0]

    materialized: dict[tuple[str, int], tuple[Path, int]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(cloud_keys))) as executor:
        for key, path, timestamp in executor.map(materialize, sorted(cloud_keys)):
            materialized[key] = (path, timestamp)

    result = []
    for (source_timestamp, target_timestamp), origins in sorted(pair_origins.items()):
        source_path, source_header = materialized[(source_topic, source_timestamp)]
        target_path, target_header = materialized[(target_topic, target_timestamp)]
        result.append({
            "sourcePath": source_path,
            "targetPath": target_path,
            "sourceTimestampNs": int(source_header),
            "targetTimestampNs": int(target_header),
            "deltaMs": round(abs(source_header - target_header) / 1e6, 3),
            "origins": sorted(origins),
        })
    return result


def manual_frame_preview(relative_path: str, timestamp_ns: int, camera_topic: str = "") -> Path:
    bag_path, databases = _database_paths(relative_path)
    timeline = manual_frame_candidates(relative_path)
    start_ns = int(timeline["startNs"])
    end_ns = int(timeline["endNs"])
    if timestamp_ns < start_ns or timestamp_ns > end_ns:
        raise ValueError("预览时间超出数据包范围")
    available_topics = {str(item["topic"]) for item in timeline["cameras"]}
    topic = camera_topic or str(timeline["previewTopic"])
    if topic not in available_topics:
        raise ValueError("相机视角不属于当前数据包")
    destination, metadata_path = _manual_preview_paths(databases, topic, timestamp_ns)
    if destination.is_file():
        if not metadata_path.is_file():
            _atomic_json(metadata_path, _manual_preview_metadata(databases, topic, timestamp_ns, destination))
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.jpg")
    try:
        metadata = _decode_camera_frame(databases, topic, timestamp_ns, temporary)
        os.replace(temporary, destination)
        destination.chmod(0o644)
        _atomic_json(metadata_path, metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _manual_preview_paths(databases: list[Path], topic: str, timestamp_ns: int) -> tuple[Path, Path]:
    preview_fingerprint = _fingerprint(databases)
    topic_key = hashlib.sha256(topic.encode()).hexdigest()[:10]
    destination = PREPARED_ROOT / "manual-previews" / preview_fingerprint / topic_key / f"{timestamp_ns}.jpg"
    return destination, destination.with_suffix(".json")


def _manual_preview_metadata(databases: list[Path], topic: str, target: int, image_path: Path) -> dict:
    timestamp = _nearest_camera_timestamp(databases, topic, target)
    with Image.open(image_path) as image:
        width, height = image.size
    return {
        "timestampNs": str(timestamp),
        "syncOffsetMs": round((timestamp - target) / 1e6, 3),
        "width": width,
        "height": height,
    }


def _reuse_manual_preview(databases: list[Path], topic: str, target: int, destination: Path) -> dict | None:
    source, metadata_path = _manual_preview_paths(databases, topic, target)
    if not source.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
        if not all(key in metadata for key in ("timestampNs", "syncOffsetMs", "width", "height")):
            raise ValueError("incomplete preview metadata")
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        metadata = _manual_preview_metadata(databases, topic, target, source)
        _atomic_json(metadata_path, metadata)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return metadata


def _nearest_camera_timestamp(databases: list[Path], topic: str, target: int) -> int:
    candidates: list[int] = []
    for database in databases:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
            if row is None:
                continue
            payloads = connection.execute(
                "SELECT data FROM messages WHERE topic_id=? AND timestamp<=? ORDER BY timestamp DESC LIMIT 4",
                (row[0], target),
            ).fetchall()
            payloads += connection.execute(
                "SELECT data FROM messages WHERE topic_id=? AND timestamp>? ORDER BY timestamp ASC LIMIT 4",
                (row[0], target),
            ).fetchall()
            for (payload,) in payloads:
                candidates.append(deserialize_compressed_image(payload).timestamp_ns)
    if not candidates:
        raise ValueError(f"图像 topic 没有消息：{topic}")
    return min(candidates, key=lambda timestamp: abs(timestamp - target))


def _topics(databases: list[Path]) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for database in databases:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            rows = connection.execute(
                "SELECT t.name,t.type,COUNT(m.id) FROM topics t LEFT JOIN messages m ON m.topic_id=t.id GROUP BY t.id"
            )
            for name, message_type, count in rows:
                item = result.setdefault(name, {"name": name, "type": message_type, "count": 0})
                item["count"] = int(item["count"]) + int(count)
    return result


def _rows(databases: list[Path], topic: str):
    for database in databases:
        with Rosbag2Reader(database) as reader:
            if topic not in reader.topics:
                continue
            yield from reader.messages(topic)


def _message_count(databases: list[Path], topic: str) -> int:
    count = 0
    for database in databases:
        with Rosbag2Reader(database) as reader:
            if topic in reader.topics:
                topic_id = reader.topics[topic]["id"]
                count += int(reader.connection.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)).fetchone()[0])
    return count


def _voxel_keys(points: np.ndarray, size: float = 0.30, maximum: int = 24_000) -> np.ndarray:
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) > maximum:
        finite = finite[:: max(1, len(finite) // maximum)]
    voxels = np.floor(finite / size).astype(np.int32)
    return np.unique(np.ascontiguousarray(voxels).view(np.dtype((np.void, 12))).reshape(-1))


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    left_keys, right_keys = _voxel_keys(left), _voxel_keys(right)
    if not len(left_keys) or not len(right_keys):
        return 0.0
    return float((np.isin(left_keys, right_keys).mean() + np.isin(right_keys, left_keys).mean()) * 0.5)


def _true_intervals(samples: list[tuple[int, bool]], minimum_seconds: float) -> list[tuple[int, int]]:
    if not samples:
        return []
    gaps = np.diff([timestamp for timestamp, _ in samples])
    maximum_gap = int(max(500_000_000, np.median(gaps) * 2.5)) if len(gaps) else 500_000_000
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    previous = samples[0][0]
    for timestamp, stationary in samples:
        if start is not None and timestamp - previous > maximum_gap:
            if previous - start >= minimum_seconds * 1e9:
                intervals.append((start, previous))
            start = None
        if stationary and start is None:
            start = timestamp
        elif not stationary and start is not None:
            if previous - start >= minimum_seconds * 1e9:
                intervals.append((start, previous))
            start = None
        previous = timestamp
    if start is not None and previous - start >= minimum_seconds * 1e9:
        intervals.append((start, previous))
    return intervals


def _gnss_intervals(databases: list[Path], topics: dict[str, dict[str, object]]) -> list[tuple[int, int]]:
    candidates = [name for name, item in topics.items() if item["type"] == "gnss_msgs/msg/GnssInfo" and item["count"]]
    for topic in candidates:
        samples: list[tuple[int, bool]] = []
        try:
            for _timestamp, payload in _rows(databases, topic):
                message = deserialize_gnss_info(payload)
                samples.append((message.timestamp_ns, abs(message.speed_kmh) <= 0.5 and abs(message.gyro_z_deg_s) <= 2.0))
        except (ValueError, struct.error):
            continue
        intervals = _true_intervals(samples, minimum_seconds=1.0)
        if intervals:
            return intervals
    return []


def _sample_clouds(databases: list[Path], topic: str, maximum: int = 120) -> list[tuple[int, np.ndarray]]:
    total = _message_count(databases, topic)
    stride = max(1, total // maximum)
    samples: list[tuple[int, np.ndarray]] = []
    index = 0
    for _bag_timestamp, payload in _rows(databases, topic):
        if index % stride == 0:
            message = deserialize_pointcloud2(payload)
            samples.append((message.timestamp_ns, structured_to_matrix(message)[:, :3].astype(np.float32)))
        index += 1
    return samples


def _lidar_intervals(databases: list[Path], topic: str) -> list[tuple[int, int]]:
    clouds = _sample_clouds(databases, topic)
    if len(clouds) < 2:
        return []
    scores = [_overlap(left[1], right[1]) for left, right in zip(clouds, clouds[1:])]
    samples = [(clouds[index + 1][0], score >= 0.72) for index, score in enumerate(scores)]
    intervals = _true_intervals(samples, minimum_seconds=0.8)
    if intervals and scores[0] >= 0.72:
        intervals[0] = (clouds[0][0], intervals[0][1])
    return intervals


def _near_field_voxel_samples(
    databases: list[Path],
    topic: str,
    intervals: list[tuple[int, int]],
    progress: Callable[[float, str], None] | None = None,
) -> list[tuple[int, np.ndarray]]:
    records: list[tuple[int, Path, int]] = []
    for database in databases:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
            if row is None:
                continue
            records.extend(
                (int(timestamp), database, int(message_id))
                for message_id, timestamp in connection.execute(
                    "SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (row[0],)
                )
            )
    records.sort(key=lambda item: item[0])
    stride = max(1, math.ceil(len(records) / MAX_NEAR_FIELD_SAMPLES))
    selected_records = [
        record
        for index, record in enumerate(records)
        if index % stride == 0 and any(start <= record[0] <= end for start, end in intervals)
    ]
    samples: list[tuple[int, np.ndarray]] = []
    connections = {database: sqlite3.connect(f"file:{database}?mode=ro", uri=True) for database in databases}
    try:
        for sampled_index, (_bag_timestamp, database, message_id) in enumerate(selected_records, 1):
            payload = connections[database].execute("SELECT data FROM messages WHERE id=?", (message_id,)).fetchone()[0]
            message = deserialize_pointcloud2(payload)
            points = structured_to_matrix(message)[:, :3]
            finite = np.isfinite(points).all(axis=1)
            radius = np.linalg.norm(points[:, :2], axis=1)
            # Exclude only the vehicle body. The requested scene signal is the
            # complete 50 m near field, independent of object class or height.
            selected = points[finite & (radius >= 3.0) & (radius <= 50.0)]
            voxels = np.floor(selected / 0.40).astype(np.int16)
            keys = np.unique(np.ascontiguousarray(voxels).view(np.dtype((np.void, 6))).reshape(-1))
            samples.append((message.timestamp_ns, keys))
            if progress and sampled_index % 15 == 0:
                progress(sampled_index / max(1, len(selected_records)) * 0.92, f"分析 50 米近场点云 {sampled_index}/{len(selected_records)}")
    finally:
        for connection in connections.values():
            connection.close()
    return samples


def _cloud_change(left: np.ndarray, right: np.ndarray) -> float:
    if not len(left) or not len(right):
        return 1.0
    shared = len(np.intersect1d(left, right, assume_unique=True))
    return float(1.0 - 0.5 * (shared / len(left) + shared / len(right)))


def _near_field_stable_scenes(
    timestamps: list[int],
    frames: list[np.ndarray],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Select centers of low-change five-second windows, at most ten."""
    if len(timestamps) < 16:
        return [], []
    cadence_seconds = float(np.median(np.diff(timestamps))) / 1e9
    comparison_steps = max(1, round(2.0 / max(cadence_seconds, 0.1)))
    changes = np.full(len(frames), np.nan, dtype=np.float32)
    for index in range(comparison_steps, len(frames) - comparison_steps):
        changes[index] = max(
            _cloud_change(frames[index], frames[index - comparison_steps]),
            _cloud_change(frames[index], frames[index + comparison_steps]),
        )

    smoothing_steps = max(1, round(1.0 / max(cadence_seconds, 0.1)))
    smooth = changes.copy()
    for index in range(comparison_steps, len(frames) - comparison_steps):
        start = max(comparison_steps, index - smoothing_steps)
        end = min(len(frames) - comparison_steps, index + smoothing_steps + 1)
        smooth[index] = np.nanmedian(changes[start:end])
    valid = smooth[np.isfinite(smooth)]
    if len(valid) < 8 or float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1)) < 0.015:
        return [], []

    # Keep prominent local low-change windows instead of blindly filling ten
    # slots. The wider baseline captures the movement on either side of a
    # five-second pause; the narrow neighborhood keeps one center per pause.
    local_steps = max(1, round(3.0 / max(cadence_seconds, 0.1)))
    baseline_steps = max(local_steps + 1, round(8.0 / max(cadence_seconds, 0.1)))
    minimum_prominence = max(0.003, float(np.quantile(valid, 0.9) - np.quantile(valid, 0.1)) * 0.03)
    candidates: list[tuple[float, int]] = []
    for index in range(comparison_steps, len(frames) - comparison_steps):
        local_start = max(comparison_steps, index - local_steps)
        local_end = min(len(frames) - comparison_steps, index + local_steps + 1)
        baseline_start = max(comparison_steps, index - baseline_steps)
        baseline_end = min(len(frames) - comparison_steps, index + baseline_steps + 1)
        if smooth[index] > np.nanmin(smooth[local_start:local_end]) + 1e-6:
            continue
        prominence = float(np.nanquantile(smooth[baseline_start:baseline_end], 0.8) - smooth[index])
        if prominence >= minimum_prominence:
            candidates.append((prominence, index))
    candidates.sort(key=lambda item: timestamps[item[1]])
    collapsed: list[tuple[float, int]] = []
    group: list[tuple[float, int]] = []
    for candidate in candidates:
        if group and timestamps[candidate[1]] - timestamps[group[0][1]] > 5_000_000_000:
            minimum = min(float(smooth[item[1]]) for item in group)
            best = [item[1] for item in group if float(smooth[item[1]]) <= minimum + 1e-6]
            collapsed.append((max(item[0] for item in group), best[0]))
            group = []
        group.append(candidate)
    if group:
        minimum = min(float(smooth[item[1]]) for item in group)
        best = [item[1] for item in group if float(smooth[item[1]]) <= minimum + 1e-6]
        collapsed.append((max(item[0] for item in group), best[0]))
    collapsed.sort(key=lambda item: (-item[0], smooth[item[1]], timestamps[item[1]]))
    selected: list[int] = []
    for _prominence, index in collapsed:
        if all(abs(timestamps[index] - timestamps[other]) >= 10_000_000_000 for other in selected):
            selected.append(index)
        if len(selected) == MAX_STATIC_SCENES:
            break
    if len(selected) < 2:
        return [], []
    selected.sort(key=lambda index: timestamps[index])
    half_window = 2_500_000_000
    targets = [timestamps[index] for index in selected]
    intervals = [(timestamps[index] - half_window, timestamps[index] + half_window) for index in selected]
    return targets, intervals


def _refine_near_field_scenes(
    databases: list[Path],
    topic: str,
    vehicle_intervals: list[tuple[int, int]],
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[int], list[tuple[int, int]]]:
    samples = _near_field_voxel_samples(databases, topic, vehicle_intervals, progress)
    targets: list[int] = []
    intervals: list[tuple[int, int]] = []
    for vehicle_start, vehicle_end in vehicle_intervals:
        selected = [(timestamp, keys) for timestamp, keys in samples if vehicle_start <= timestamp <= vehicle_end]
        if len(selected) < 16 or vehicle_end - vehicle_start < 8_000_000_000:
            targets.append((vehicle_start + vehicle_end) // 2)
            intervals.append((vehicle_start, vehicle_end))
            continue
        timestamps = [timestamp for timestamp, _keys in selected]
        frames = [keys for _timestamp, keys in selected]
        refined_targets, refined_intervals = _near_field_stable_scenes(timestamps, frames)
        if len(refined_targets) >= 2:
            targets.extend(refined_targets)
            intervals.extend(
                (max(vehicle_start, start), min(vehicle_end, end)) for start, end in refined_intervals
            )
        else:
            targets.append((vehicle_start + vehicle_end) // 2)
            intervals.append((vehicle_start, vehicle_end))
    if progress:
        progress(1.0, f"50 米近场滑窗分析完成：{len(targets)} 个位置")
    return targets, intervals


def _static_targets(
    databases: list[Path],
    topics: dict[str, dict[str, object]],
    lidar_topics: list[str],
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[int], list[tuple[int, int]], str]:
    intervals = _gnss_intervals(databases, topics)
    method = "vehicle-motion-topic"
    if intervals:
        targets, refined = _refine_near_field_scenes(databases, lidar_topics[0], intervals, progress)
        if len(refined) >= 2:
            intervals = refined
            method = "vehicle-motion+near-field-stability"
        else:
            targets = [(start + end) // 2 for start, end in intervals]
    else:
        intervals = _lidar_intervals(databases, lidar_topics[0])
        method = "pointcloud-voxel-overlap"
        targets = [(start + end) // 2 for start, end in intervals]
    if not intervals:
        raise ValueError("未检测到持续静止场景，无法生成可标注帧")
    if len(intervals) > MAX_STATIC_SCENES:
        selected = sorted(range(len(intervals)), key=lambda index: intervals[index][1] - intervals[index][0], reverse=True)[:MAX_STATIC_SCENES]
        selected.sort(key=lambda index: intervals[index])
        targets = [targets[index] for index in selected]
        intervals = [intervals[index] for index in selected]
    return targets, intervals, method


def _pointcloud_index(databases: list[Path], topic: str) -> list[tuple[int, Path, int]]:
    signature = tuple(
        (str(database), database.stat().st_size, database.stat().st_mtime_ns)
        for database in databases
    )
    key = (topic, signature)
    with _POINTCLOUD_INDEX_LOCK:
        cached = _POINTCLOUD_INDICES.get(key)
        if cached is not None:
            return cached
        records: list[tuple[int, Path, int]] = []
        for database in databases:
            with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
                row = connection.execute("SELECT id FROM topics WHERE name=?", (topic,)).fetchone()
                if row is None:
                    continue
                records.extend(
                    (int(timestamp), database, int(message_id))
                    for message_id, timestamp in connection.execute(
                        "SELECT id,timestamp FROM messages WHERE topic_id=? ORDER BY timestamp", (row[0],)
                    )
                )
        records.sort(key=lambda item: item[0])
        _POINTCLOUD_INDICES[key] = records
        return records


def _nearest_pointcloud(databases: list[Path], topic: str, target: int):
    records = _pointcloud_index(databases, topic)
    if not records:
        raise ValueError(f"点云 topic 没有消息：{topic}")
    index = bisect.bisect_left(records, (target,))
    candidates = sorted(records[max(0, index - 2):min(len(records), index + 2)], key=lambda item: abs(item[0] - target))[:2]
    messages = []
    for _timestamp, database, message_id in candidates:
        with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT data FROM messages WHERE id=?", (message_id,)).fetchone()
        if row is not None:
            messages.append(deserialize_pointcloud2(row[0]))
    if not messages:
        raise ValueError(f"点云 topic 没有消息：{topic}")
    primary = messages[0]
    reference = messages[min(1, len(messages) - 1)]
    return primary, reference


def _stable_flags(source: np.ndarray, reference: np.ndarray, threshold: float = 0.18) -> np.ndarray:
    finite_source = np.isfinite(source).all(axis=1)
    finite_reference = reference[np.isfinite(reference).all(axis=1)]
    flags = np.zeros(len(source), dtype=bool)
    if not len(finite_reference):
        return flags
    source_points = source[finite_source]
    matched = np.zeros(len(source_points), dtype=bool)
    for shift in (0.0, threshold * 0.5):
        reference_voxels = np.floor((finite_reference + shift) / threshold).astype(np.int32)
        source_voxels = np.floor((source_points + shift) / threshold).astype(np.int32)
        reference_keys = np.ascontiguousarray(reference_voxels).view(np.dtype((np.void, 12))).reshape(-1)
        source_keys = np.ascontiguousarray(source_voxels).view(np.dtype((np.void, 12))).reshape(-1)
        matched |= np.isin(source_keys, np.unique(reference_keys))
    flags[finite_source] = matched
    return flags


def _write_cloud(path: Path, timestamp: int, points: np.ndarray, stable: np.ndarray) -> None:
    records = np.zeros(len(points), dtype={
        "names": ["x", "y", "z", "stable"], "formats": ["<f4", "<f4", "<f4", "u1"],
        "offsets": [0, 4, 8, 12], "itemsize": 16,
    })
    records["x"], records["y"], records["z"], records["stable"] = points[:, 0], points[:, 1], points[:, 2], stable
    with path.open("wb") as output:
        output.write(b"ACP1")
        output.write(struct.pack("<Iq", len(points), timestamp))
        output.write(records.tobytes())


def _nal_types(data: bytes) -> set[int]:
    result: set[int] = set()
    for index in range(max(0, len(data) - 5)):
        offset = index + 4 if data[index:index + 4] == b"\0\0\0\1" else index + 3 if data[index:index + 3] == b"\0\0\1" else 0
        if offset:
            result.add((data[offset] >> 1) & 0x3F)
    return result


def _annex_b_nalus(data: bytes) -> list[tuple[int, bytes]]:
    starts: list[int] = []
    index = 0
    while index <= len(data) - 3:
        if data[index:index + 4] == b"\0\0\0\1":
            starts.append(index)
            index += 4
        elif data[index:index + 3] == b"\0\0\1":
            starts.append(index)
            index += 3
        else:
            index += 1
    result: list[tuple[int, bytes]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(data)
        prefix = 4 if data[start:start + 4] == b"\0\0\0\1" else 3
        if start + prefix < end:
            result.append(((data[start + prefix] >> 1) & 0x3F, data[start:end]))
    return result


def _codec_header(topic: str, preferred: list[Path]) -> bytes:
    if topic in _CODEC_HEADERS:
        return _CODEC_HEADERS[topic]
    databases = [*preferred, *sorted(BAG_ROOT.glob("*/*.db3"))]
    seen: set[Path] = set()
    for database in databases:
        if database in seen:
            continue
        seen.add(database)
        try:
            with Rosbag2Reader(database) as reader:
                if topic not in reader.topics:
                    continue
                for message in reader.compressed_images(topic):
                    units = _annex_b_nalus(message.data)
                    parameter_sets = b"".join(unit for nal_type, unit in units if nal_type in {32, 33, 34})
                    if {32, 33, 34}.issubset({nal_type for nal_type, _unit in units}):
                        _CODEC_HEADERS[topic] = parameter_sets
                        return parameter_sets
        except (OSError, sqlite3.Error, ValueError):
            continue
    raise ValueError(f"找不到 H.265 解码头：{topic}")


def _camera_index(databases: list[Path], topic: str) -> list[tuple[Path, list[int], list[int]]]:
    signature = tuple(
        (str(database), database.stat().st_size, database.stat().st_mtime_ns)
        for database in databases
    )
    key = (topic, signature)
    with _CAMERA_INDEX_LOCK:
        cached = _CAMERA_INDICES.get(key)
        if cached is not None:
            return cached
        result: list[tuple[Path, list[int], list[int]]] = []
        for database in databases:
            timestamps: list[int] = []
            keyframes: list[int] = []
            with Rosbag2Reader(database) as reader:
                if topic not in reader.topics:
                    continue
                for index, (_bag_timestamp, payload) in enumerate(reader.messages(topic)):
                    message = deserialize_compressed_image(payload)
                    timestamps.append(message.timestamp_ns)
                    if _nal_types(message.data).intersection({19, 20, 21}):
                        keyframes.append(index)
            if timestamps:
                result.append((database, timestamps, keyframes))
        _CAMERA_INDICES[key] = result
        return result


def _camera_source(databases: list[Path], topic: str, target: int) -> tuple[Path, list[int], list[int], int]:
    best: tuple[int, Path, list[int], list[int], int] | None = None
    for database, timestamps, keyframes in _camera_index(databases, topic):
        selected = min(range(len(timestamps)), key=lambda index: abs(timestamps[index] - target))
        candidate = (abs(timestamps[selected] - target), database, timestamps, keyframes, selected)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise ValueError(f"图像 topic 没有消息：{topic}")
    # Before the first keyframe there is no reference chain to decode. Show
    # the first decodable image instead of returning a corrupt/black preview.
    if best[3] and not any(index <= best[4] for index in best[3]):
        best = (abs(best[2][best[3][0]] - target), best[1], best[2], best[3], best[3][0])
    return best[1], best[2], best[3], best[4]


def _decode_camera_frame(databases: list[Path], topic: str, target: int, destination: Path) -> dict:
    database, timestamps, keyframes, selected = _camera_source(databases, topic, target)
    prior = [index for index in keyframes if index <= selected]
    start = prior[-1] if prior else 0
    header = _codec_header(topic, databases)
    with tempfile.TemporaryDirectory(prefix="autocalib-decode-") as temporary_name:
        temporary = Path(temporary_name)
        stream_path = temporary / "stream.h265"
        with stream_path.open("wb") as stream, Rosbag2Reader(database) as reader:
            stream.write(header)
            for _timestamp, payload in reader.messages(topic, limit=selected - start + 1, offset=start):
                stream.write(deserialize_compressed_image(payload).data)
        pattern = temporary / "frame-%05d.jpg"
        completed = subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1", "-i", str(stream_path),
            "-vsync", "0", "-q:v", "2", str(pattern),
        ], capture_output=True, text=True, check=False)
        candidates = sorted(temporary.glob("frame-*.jpg"))
        if completed.returncode or not candidates:
            raise ValueError(f"H.265 解码失败 {topic}: {completed.stderr[-300:]}")
        shutil.copy2(candidates[-1], destination)
    with Image.open(destination) as image:
        width, height = image.size
    return {
        "timestampNs": str(timestamps[selected]), "syncOffsetMs": round((timestamps[selected] - target) / 1e6, 3),
        "width": width, "height": height,
    }


def _sensor_id(topic: str, kind: str, used: set[str]) -> str:
    lowered = topic.lower()
    mappings = (
        (("front190", "fronth190"), "front190"), (("front120", "fronth120"), "front120"),
        (("front60", "fronth60"), "front60"), (("left120", "lefth120"), "left120"),
        (("right120", "righth120"), "right120"), (("back120", "backh120"), "back120"),
        (("main",), "main"), (("front",), "front"), (("left",), "left"),
        (("right",), "right"), (("back", "rear"), "back"),
    )
    identifier = next((value for needles, value in mappings if any(needle in lowered for needle in needles)), kind)
    base, suffix = identifier, 2
    while identifier in used:
        identifier, suffix = f"{base}-{suffix}", suffix + 1
    used.add(identifier)
    return identifier


def _sensor_name(identifier: str, kind: str) -> str:
    names = {"main": "主雷达", "front": "前雷达", "left": "左雷达", "right": "右雷达", "back": "后雷达",
             "front120": "正前相机", "front60": "左前相机", "front190": "右前相机", "left120": "左后相机", "right120": "右后相机", "back120": "正后相机"}
    return names.get(identifier, f"{kind} {identifier}")


def _install_dataset(dataset: dict) -> None:
    with _MANIFEST_LOCK:
        manifest = _load_manifest()
        # Keep older selections addressable: calibration groups and saved
        # annotations may still reference them until a replacement import has
        # passed validation and migrated their IDs.
        manifest["datasets"] = [
            item
            for item in manifest.get("datasets", [])
            if item.get("id") != dataset["id"]
        ]
        manifest["datasets"].append(dataset)
        manifest["datasets"].sort(key=lambda item: item.get("name", ""))
        manifest["generatedAt"] = _now()
        manifest.setdefault("calibration", _empty_manifest()["calibration"])
        _atomic_json(_manifest_path(), manifest)


def prepare_bag(
    relative_path: str,
    progress: Progress,
    selected_targets: list[int] | None = None,
    selected_camera_targets: dict[str, list[int]] | None = None,
    selected_lidar_targets: dict[str, list[int]] | None = None,
) -> tuple[dict, bool]:
    bag_path, databases = _database_paths(relative_path)
    timeline = manual_frame_candidates(relative_path)
    start_ns = int(timeline["startNs"])
    end_ns = int(timeline["endNs"])
    available_camera_topics = {str(item["topic"]) for item in timeline["cameras"]}
    available_lidar_topics = {str(item["topic"]) for item in timeline["lidars"]}
    camera_selections = {
        topic: sorted(set(values))
        for topic, values in (selected_camera_targets or {}).items()
        if values
    }
    if not camera_selections and selected_targets and timeline["previewTopic"]:
        camera_selections = {str(timeline["previewTopic"]): sorted(set(selected_targets))}
    if any(topic not in available_camera_topics for topic in camera_selections):
        raise ValueError(f"{bag_path.name} 包含未知相机视角")
    lidar_selections = {
        topic: sorted(set(values))
        for topic, values in (selected_lidar_targets or {}).items()
        if values
    }
    if any(topic not in available_lidar_topics for topic in lidar_selections):
        raise ValueError(f"{bag_path.name} 包含未知雷达视角")
    targets = sorted({
        target
        for selections in (camera_selections, lidar_selections)
        for values in selections.values()
        for target in values
    })
    if not targets:
        raise ValueError(f"{bag_path.name} 请先手动选择至少一帧")
    if any(target < start_ns or target > end_ns for target in targets):
        raise ValueError(f"{bag_path.name} 包含超出数据包范围的帧")
    selection_key = json.dumps(
        camera_selections if not lidar_selections else {"cameras": camera_selections, "lidars": lidar_selections},
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = _fingerprint(databases, targets, selection_key)
    cache_root = PREPARED_ROOT
    destination = cache_root / fingerprint
    cached_metadata = destination / "dataset.json"
    if cached_metadata.is_file():
        dataset = json.loads(cached_metadata.read_text())
        _install_dataset(dataset)
        progress(100, f"已命中缓存：{bag_path.name}", {"cached": True})
        return dataset, True

    topics = _topics(databases)
    all_lidar_topics = sorted(name for name, item in topics.items() if item["type"] == "sensor_msgs/msg/PointCloud2" and item["count"])
    lidar_topics = [topic for topic in all_lidar_topics if not lidar_selections or topic in lidar_selections]
    all_camera_topics = sorted(name for name, item in topics.items() if item["type"] == "sensor_msgs/msg/CompressedImage" and item["count"])
    camera_topics = [topic for topic in all_camera_topics if topic in camera_selections]
    if not lidar_topics:
        raise ValueError(f"{bag_path.name} 未发现 PointCloud2 topic")
    progress(8, f"识别到 {len(lidar_topics)} 路点云、已选择 {len(camera_topics)} 路图像", {"topics": topics})
    progress(25, f"已确认 {len(targets)} 个时间点、{len(camera_topics)} 个视角", {"stationaryMethod": "manual-one-second", "sceneCount": len(targets)})

    cache_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{fingerprint}-", dir=cache_root))
    try:
        labels = [str(index + 1) for index in range(len(targets))]
        label_by_target = dict(zip(targets, labels))
        lidars = []
        used_lidars: set[str] = set()
        total_steps = sum(len(lidar_selections.get(topic, targets)) for topic in lidar_topics) + sum(len(camera_selections[topic]) for topic in camera_topics)
        completed_steps = 0
        for topic in lidar_topics:
            identifier = _sensor_id(topic, "lidar", used_lidars)
            frames = []
            topic_targets = lidar_selections.get(topic, targets)
            def extract_cloud(item: tuple[str, int]) -> dict:
                label, target = item
                message, reference = _nearest_pointcloud(databases, topic, target)
                points = structured_to_matrix(message)[:, :3].astype(np.float32)
                reference_points = structured_to_matrix(reference)[:, :3].astype(np.float32)
                stable = _stable_flags(points, reference_points)
                filename = f"{identifier}-{label}.bin"
                _write_cloud(staging / filename, message.timestamp_ns, points, stable)
                return {"label": label, "timestampNs": str(message.timestamp_ns), "pointCount": len(points),
                        "stableRatio": round(float(stable.mean()), 6), "url": f"/data/cache/{fingerprint}/{filename}"}

            topic_labels = [label_by_target[target] for target in topic_targets]
            with ThreadPoolExecutor(max_workers=min(4, len(topic_targets))) as executor:
                extracted = executor.map(extract_cloud, zip(topic_labels, topic_targets))
                for frame in extracted:
                    frames.append(frame)
                    label = str(frame["label"])
                    completed_steps += 1
                    progress(25 + int(completed_steps / total_steps * 68), f"提取点云 {identifier} · 场景 {label}", None)
            lidars.append({"id": identifier, "name": _sensor_name(identifier, "雷达"), "topic": topic, "frames": frames})

        cameras = []
        used_cameras: set[str] = set()
        for topic in camera_topics:
            identifier = _sensor_id(topic, "camera", used_cameras)
            frames = []
            for target in camera_selections[topic]:
                label = label_by_target[target]
                filename = f"{identifier}-{label}.jpg"
                frame = _reuse_manual_preview(databases, topic, target, staging / filename)
                if frame is None:
                    frame = _decode_camera_frame(databases, topic, target, staging / filename)
                frames.append({"label": label, **frame, "url": f"/data/cache/{fingerprint}/{filename}"})
                completed_steps += 1
                progress(25 + int(completed_steps / total_steps * 68), f"解码图像 {identifier} · 场景 {label}", None)
            cameras.append({"id": identifier, "name": _sensor_name(identifier, "相机"), "topic": topic, "frames": frames})

        rig_id = "auto-" + hashlib.sha256(("|".join(all_lidar_topics) + "||" + "|".join(all_camera_topics)).encode()).hexdigest()[:12]
        dataset = {
            "id": f"bag-{fingerprint}", "rigId": rig_id, "name": bag_path.name,
            "sourceFile": databases[0].name, "sourcePath": relative_path,
            "fingerprint": fingerprint, "preparationAlgorithm": ALGORITHM_VERSION,
            "stationaryIntervalCount": len(targets), "stationaryDetectionMethod": "manual-one-second",
            "selectedInterval": {"startNs": str(targets[0]), "endNs": str(targets[-1]),
                                 "durationSeconds": round((targets[-1] - targets[0]) / 1e9, 3)},
            "stationaryIntervals": [{"startNs": str(target), "endNs": str(target), "durationSeconds": 0.0} for target in targets],
            "anchorFrames": [{"label": label, "timestampNs": str(target)} for label, target in zip(labels, targets)],
            "lidars": lidars, "cameras": cameras, "cameraCalibration": {}, "extrinsics": None,
            "calibrationCompatibility": "unknown-extrinsic",
            "topicSummary": list(topics.values()),
        }
        _atomic_json(staging / "dataset.json", dataset)
        for generated in staging.rglob("*"):
            generated.chmod(0o755 if generated.is_dir() else 0o644)
        staging.chmod(0o755)
        if destination.exists():
            shutil.rmtree(staging)
        else:
            os.replace(staging, destination)
        _install_dataset(dataset)
        progress(100, f"处理完成：{bag_path.name}", {"cached": False, "sceneCount": len(targets)})
        return dataset, False
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def reset_manifest() -> None:
    _atomic_json(_manifest_path(), _empty_manifest())
