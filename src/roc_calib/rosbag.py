from __future__ import annotations

import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


_POINT_DTYPES = {
    1: "i1",
    2: "u1",
    3: "<i2",
    4: "<u2",
    5: "<i4",
    6: "<u4",
    7: "<f4",
    8: "<f8",
}


class CdrReader:
    def __init__(self, data: bytes):
        if data[:4] not in (b"\x00\x01\x00\x00", b"\x00\x01\x00\x01"):
            raise ValueError("only little-endian CDR is supported")
        self.data = data
        self.offset = 4

    def align(self, size: int) -> None:
        # XCDR alignment starts after the four-byte encapsulation header.
        payload_offset = self.offset - 4
        self.offset = 4 + (payload_offset + size - 1) // size * size

    def unpack(self, code: str, alignment: int | None = None):
        size = struct.calcsize(code)
        self.align(alignment or size)
        value = struct.unpack_from("<" + code, self.data, self.offset)[0]
        self.offset += size
        return value

    def u8(self) -> int:
        value = self.data[self.offset]
        self.offset += 1
        return value

    def header(self) -> tuple[int, str]:
        sec, nanosec = self.unpack("i"), self.unpack("I")
        return sec * 1_000_000_000 + nanosec, self.string()

    def string(self) -> str:
        size = self.unpack("I")
        raw = self.data[self.offset : self.offset + size]
        self.offset += size
        return raw[:-1].decode(errors="replace") if raw.endswith(b"\0") else raw.decode(errors="replace")


class CdrWriter:
    def __init__(self):
        self.data = bytearray(b"\x00\x01\x00\x00")

    def align(self, size: int) -> None:
        payload_offset = len(self.data) - 4
        self.data.extend(b"\0" * ((-payload_offset) % size))

    def pack(self, code: str, value) -> None:
        size = struct.calcsize(code)
        self.align(size)
        self.data.extend(struct.pack("<" + code, value))

    def string(self, value: str) -> None:
        raw = value.encode() + b"\0"
        self.pack("I", len(raw))
        self.data.extend(raw)

    def header(self, timestamp_ns: int, frame_id: str) -> None:
        seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
        self.pack("i", seconds)
        self.pack("I", nanoseconds)
        self.string(frame_id)


@dataclass(frozen=True)
class PointCloudMessage:
    timestamp_ns: int
    frame_id: str
    fields: tuple[str, ...]
    points: np.ndarray


@dataclass(frozen=True)
class GnssInfoMessage:
    timestamp_ns: int
    latitude_deg: float
    longitude_deg: float
    altitude_m: float
    heading_deg: float
    speed_kmh: float
    gyro_z_deg_s: float
    pva_warning: int
    imum_warning: int
    gpsm_rlt_status: int
    imu_acc_x_mps2: float = 0.0
    imu_acc_y_mps2: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    gps_ve_kmh: float = 0.0
    gps_vn_kmh: float = 0.0


@dataclass(frozen=True)
class CompressedImageMessage:
    timestamp_ns: int
    frame_id: str
    format: str
    data: bytes


def deserialize_pointcloud2(data: bytes) -> PointCloudMessage:
    reader = CdrReader(data)
    timestamp_ns, frame_id = reader.header()
    height, width = reader.unpack("I"), reader.unpack("I")
    field_count = reader.unpack("I")
    fields: list[tuple[str, int, int, int]] = []
    for _ in range(field_count):
        fields.append((reader.string(), reader.unpack("I"), reader.u8(), reader.unpack("I")))
    big_endian = bool(reader.u8())
    point_step, row_step = reader.unpack("I"), reader.unpack("I")
    data_size = reader.unpack("I")
    raw = memoryview(data)[reader.offset : reader.offset + data_size]
    reader.offset += data_size
    _is_dense = bool(reader.u8())
    if big_endian:
        raise ValueError("big-endian PointCloud2 is unsupported")
    if row_step < width * point_step or data_size < height * row_step:
        raise ValueError("invalid PointCloud2 strides")
    dtype_fields = []
    for name, offset, datatype, count in fields:
        if datatype not in _POINT_DTYPES:
            raise ValueError(f"unsupported PointField datatype {datatype}")
        dtype_fields.append((name, _POINT_DTYPES[datatype], (count,), offset))
    dtype = np.dtype({
        "names": [x[0] for x in dtype_fields],
        "formats": [(x[1], x[2]) if x[2] != (1,) else x[1] for x in dtype_fields],
        "offsets": [x[3] for x in dtype_fields],
        "itemsize": point_step,
    })
    points = np.frombuffer(raw, dtype=dtype, count=height * width)
    return PointCloudMessage(timestamp_ns, frame_id, tuple(dtype.names or ()), points)


def serialize_pointcloud2_xyz_intensity(
    timestamp_ns: int,
    frame_id: str,
    points: np.ndarray,
) -> bytes:
    """Serialize an unorganized XYZ-intensity float32 cloud as ROS 2 CDR."""
    values = np.asarray(points, dtype="<f4")
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("point cloud must have shape (N, 4)")
    values = np.ascontiguousarray(values)
    writer = CdrWriter()
    writer.header(timestamp_ns, frame_id)
    writer.pack("I", 1)
    writer.pack("I", len(values))
    fields = (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))
    writer.pack("I", len(fields))
    for name, offset in fields:
        writer.string(name)
        writer.pack("I", offset)
        writer.pack("B", 7)  # sensor_msgs/msg/PointField.FLOAT32
        writer.pack("I", 1)
    writer.pack("B", 0)
    writer.pack("I", 16)
    writer.pack("I", len(values) * 16)
    writer.pack("I", len(values) * 16)
    writer.data.extend(values.tobytes())
    writer.pack("B", 1)
    return bytes(writer.data)


def deserialize_gnss_info(data: bytes) -> GnssInfoMessage:
    """Deserialize this fleet's fixed-layout gnss_msgs/msg/GnssInfo."""
    reader = CdrReader(data)
    timestamp_ns, _frame_id = reader.header()
    _id = reader.unpack("Q")
    _flag = reader.u8()
    latitude, longitude = reader.unpack("d"), reader.unpack("d")
    values = [reader.unpack("f") for _ in range(13)]
    # gps_alt, gps_heading, gps_ve, gps_vn, gps_speed, imu_acc_x,
    # imu_acc_y, imu_gyr_z, gps_roll, gps_pitch, gps_yaw, speed_east,
    # speed_north, followed by status ints.
    statuses = [reader.unpack("i") for _ in range(9)]
    # gnss_horizontal_speed was appended in a later message revision. Older
    # bags end after gpss_rlt_status and are still layout-compatible above.
    if reader.offset + 4 <= len(data):
        _horizontal_speed = reader.unpack("f")
    return GnssInfoMessage(
        timestamp_ns=timestamp_ns,
        latitude_deg=latitude,
        longitude_deg=longitude,
        altitude_m=values[0],
        heading_deg=values[1],
        speed_kmh=values[4],
        gyro_z_deg_s=values[7],
        pva_warning=statuses[0],
        imum_warning=statuses[3],
        gpsm_rlt_status=statuses[4],
        imu_acc_x_mps2=values[5],
        imu_acc_y_mps2=values[6],
        roll_deg=values[8],
        pitch_deg=values[9],
        yaw_deg=values[10],
        gps_ve_kmh=values[2],
        gps_vn_kmh=values[3],
    )


def deserialize_compressed_image(data: bytes) -> CompressedImageMessage:
    reader = CdrReader(data)
    timestamp_ns, frame_id = reader.header()
    image_format = reader.string()
    size = reader.unpack("I")
    payload = bytes(data[reader.offset : reader.offset + size])
    if len(payload) != size:
        raise ValueError("truncated CompressedImage payload")
    return CompressedImageMessage(timestamp_ns, frame_id, image_format, payload)


def structured_to_matrix(message: PointCloudMessage) -> np.ndarray:
    required = ("x", "y", "z")
    if not all(name in message.fields for name in required):
        raise ValueError(f"point cloud lacks XYZ fields: {message.fields}")
    columns = [message.points[name].astype(np.float64, copy=False) for name in required]
    if "intensity" in message.fields:
        columns.append(message.points["intensity"].astype(np.float64, copy=False))
    return np.column_stack(columns)


class Rosbag2Reader:
    def __init__(self, database: str | Path):
        self.path = Path(database)
        self.connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        self.topics = {
            row[1]: {"id": row[0], "type": row[2]}
            for row in self.connection.execute("SELECT id,name,type FROM topics")
        }

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Rosbag2Reader":
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def messages(
        self, topic: str, limit: int | None = None, stride: int = 1, offset: int = 0
    ) -> Iterator[tuple[int, bytes]]:
        topic_id = self.topics[topic]["id"]
        query = "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp"
        parameters: tuple[object, ...] = (topic_id,)
        if limit is not None:
            query += " LIMIT ? OFFSET ?"
            parameters += (limit * stride, offset)
        elif offset:
            query += " LIMIT -1 OFFSET ?"
            parameters += (offset,)
        for index, row in enumerate(self.connection.execute(query, parameters)):
            if index % stride == 0:
                yield int(row[0]), row[1]

    def pointclouds(
        self, topic: str, limit: int | None = None, stride: int = 1, offset: int = 0
    ) -> Iterator[PointCloudMessage]:
        if self.topics[topic]["type"] != "sensor_msgs/msg/PointCloud2":
            raise ValueError(f"{topic} is not PointCloud2")
        for _bag_time, data in self.messages(topic, limit=limit, stride=stride, offset=offset):
            yield deserialize_pointcloud2(data)

    def pointcloud_at(self, topic: str, index: int) -> PointCloudMessage:
        if index < 0:
            raise ValueError("point-cloud index must be non-negative")
        topic_id = self.topics[topic]["id"]
        row = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
            (topic_id, index),
        ).fetchone()
        if row is None:
            raise ValueError("point-cloud frame index is out of range")
        return deserialize_pointcloud2(row[0])

    def pointcloud_nearest(self, topic: str, timestamp_ns: int) -> PointCloudMessage:
        topic_id = self.topics[topic]["id"]
        row = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY ABS(timestamp-?) LIMIT 1",
            (topic_id, timestamp_ns),
        ).fetchone()
        if row is None:
            raise ValueError(f"no point cloud on {topic}")
        return deserialize_pointcloud2(row[0])

    def pointcloud_containing(self, topic: str, timestamp_ns: int, candidate_count: int = 4) -> PointCloudMessage:
        """Prefer a nearby scan whose per-point acquisition interval contains a timestamp."""
        topic_id = self.topics[topic]["id"]
        rows = self.connection.execute(
            "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY ABS(timestamp-?) LIMIT ?",
            (topic_id, timestamp_ns, candidate_count),
        ).fetchall()
        candidates = []
        for bag_timestamp, payload in rows:
            message = deserialize_pointcloud2(payload)
            if "timestamp" not in message.fields:
                continue
            point_times = np.rint(message.points["timestamp"].astype(np.float64) * 1e9).astype(np.int64)
            interval_distance = max(
                int(point_times.min()) - timestamp_ns,
                0,
                timestamp_ns - int(point_times.max()),
            )
            candidates.append((interval_distance, abs(int(bag_timestamp) - timestamp_ns), message))
        if not candidates:
            return self.pointcloud_nearest(topic, timestamp_ns)
        return min(candidates, key=lambda item: (item[0], item[1]))[2]

    def compressed_image_at(self, topic: str, index: int) -> CompressedImageMessage:
        if index < 0:
            raise ValueError("image index must be non-negative")
        topic_id = self.topics[topic]["id"]
        row = self.connection.execute(
            "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
            (topic_id, index),
        ).fetchone()
        if row is None:
            raise ValueError("compressed-image frame index is out of range")
        return deserialize_compressed_image(row[0])

    def gnss_messages(self, topic: str = "SensorGnssInfo") -> Iterator[GnssInfoMessage]:
        if self.topics[topic]["type"] != "gnss_msgs/msg/GnssInfo":
            raise ValueError(f"{topic} is not GnssInfo")
        for _bag_time, data in self.messages(topic):
            yield deserialize_gnss_info(data)

    def compressed_images(
        self, topic: str, limit: int | None = None, stride: int = 1, offset: int = 0
    ) -> Iterator[CompressedImageMessage]:
        if self.topics[topic]["type"] != "sensor_msgs/msg/CompressedImage":
            raise ValueError(f"{topic} is not CompressedImage")
        for _bag_time, data in self.messages(topic, limit=limit, stride=stride, offset=offset):
            yield deserialize_compressed_image(data)


def bag_summary(database: str | Path) -> dict[str, object]:
    with Rosbag2Reader(database) as reader:
        result: dict[str, object] = {"database": str(database), "topics": []}
        for name, metadata in sorted(reader.topics.items()):
            count = reader.connection.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?", (metadata["id"],)
            ).fetchone()[0]
            item = {"name": name, "type": metadata["type"], "count": count}
            if metadata["type"] == "sensor_msgs/msg/PointCloud2" and count:
                first = next(reader.pointclouds(name, limit=1))
                item.update({"fields": list(first.fields), "frame_id": first.frame_id, "points": len(first.points)})
                if "timestamp" in first.fields:
                    point_seconds = first.points["timestamp"].astype(np.float64)
                    offsets_ms = (point_seconds - first.timestamp_ns * 1e-9) * 1e3
                    item["point_time_offset_ms"] = {
                        "min": float(np.nanmin(offsets_ms)),
                        "max": float(np.nanmax(offsets_ms)),
                        "span": float(np.nanmax(offsets_ms) - np.nanmin(offsets_ms)),
                    }
            result["topics"].append(item)
        return result
