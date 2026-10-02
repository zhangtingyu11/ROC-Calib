"""Waymo CameraModel-compatible rolling-shutter projection.

This module follows Waymo Open Dataset's ``CameraModel::WorldToImage``
interface: points are expressed in the global frame and the camera pose,
velocity, exposure timing and candidate camera extrinsic determine the pixel.
The precomputed ``lidar_camera_projection`` component is deliberately not an
input here; it remains an evaluation-only oracle.
"""

from __future__ import annotations

import math

import numpy as np


OPENCV_FROM_WAYMO_CAMERA = np.asarray([
    [0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float64)
WAYMO_CAMERA_FROM_OPENCV = np.linalg.inv(OPENCV_FROM_WAYMO_CAMERA)
MIN_TRUSTED_RADIAL_DISTORTION = 0.8
MAX_TRUSTED_RADIAL_DISTORTION = 1.2


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([
        [0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0],
    ], dtype=np.float64)


def _distort_normalized(
    x: np.ndarray,
    y: np.ndarray,
    distortion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    k1, k2, p1, p2, k3 = np.pad(
        np.asarray(distortion, dtype=np.float64), (0, 5),
    )[:5]
    radius2 = x * x + y * y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (radius2 + 2.0 * x * x),
        y * radial + p1 * (radius2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


def _undistort_normalized(
    pixel_x: float,
    pixel_y: float,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, float]:
    target_x = (float(pixel_x) - intrinsic[0, 2]) / intrinsic[0, 0]
    target_y = (float(pixel_y) - intrinsic[1, 2]) / intrinsic[1, 1]
    x, y = target_x, target_y
    for _ in range(20):
        previous_x, previous_y = x, y
        k1, k2, p1, p2, k3 = np.pad(
            np.asarray(distortion, dtype=np.float64), (0, 5),
        )[:5]
        radius2 = previous_x * previous_x + previous_y * previous_y
        radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
        tangential_x = (
            2.0 * p1 * previous_x * previous_y
            + p2 * (radius2 + 2.0 * previous_x**2)
        )
        tangential_y = (
            2.0 * p2 * previous_x * previous_y
            + p1 * (radius2 + 2.0 * previous_y**2)
        )
        x = (target_x - tangential_x) / radial
        y = (target_y - tangential_y) / radial
    return x, y


def _pixel_timestamp(
    direction: int,
    shutter: float,
    trigger_time: float,
    readout_done_time: float,
    width: int,
    height: int,
    x: float,
    y: float,
) -> float:
    readout = float(readout_done_time) - float(trigger_time) - float(shutter)
    base = float(trigger_time) + 0.5 * float(shutter)
    if direction == 1:
        return base + readout / float(height) * float(y)
    if direction == 2:
        return base + readout / float(width) * float(x)
    if direction == 3:
        return base + readout / float(height) * (float(height) - float(y))
    if direction == 4:
        return base + readout / float(width) * (float(width) - float(x))
    if direction == 5:
        return base
    raise ValueError(f"unsupported Waymo rolling-shutter direction: {direction}")


def prepare_projection(metadata: dict, intrinsic: np.ndarray, distortion: np.ndarray) -> dict:
    """Validate and precompute candidate-independent CameraModel inputs."""
    if metadata.get("model") != "waymo-camera-model-v1":
        raise ValueError("unsupported temporal projection model")
    vehicle_from_lidar = np.asarray(
        metadata["vehicle_from_lidar"], dtype=np.float64,
    ).reshape(4, 4)
    reference_world_from_vehicle = np.asarray(
        metadata["reference_world_from_vehicle"], dtype=np.float64,
    ).reshape(4, 4)
    camera_world_from_vehicle = np.asarray(
        metadata["camera_world_from_vehicle"], dtype=np.float64,
    ).reshape(4, 4)
    direction = int(metadata["rolling_shutter_direction"])
    width, height = int(metadata["image_width"]), int(metadata["image_height"])
    shutter = float(metadata["shutter"])
    trigger_time = float(metadata["trigger_time"])
    readout_done_time = float(metadata["readout_done_time"])
    principal_time = _pixel_timestamp(
        direction, shutter, trigger_time, readout_done_time, width, height,
        float(intrinsic[0, 2]), float(intrinsic[1, 2]),
    )
    horizontal = direction in {2, 4}
    if direction == 5:
        readout_factor = 0.0
    else:
        if horizontal:
            first, _ = _undistort_normalized(0.0, 0.5 * height, intrinsic, distortion)
            last, _ = _undistort_normalized(float(width), 0.5 * height, intrinsic, distortion)
        else:
            _, first = _undistort_normalized(0.5 * width, 0.0, intrinsic, distortion)
            _, last = _undistort_normalized(0.5 * width, float(height), intrinsic, distortion)
        readout = readout_done_time - trigger_time - shutter
        readout_factor = (-1.0 if direction in {3, 4} else 1.0) * readout / (last - first)
    return {
        "model": "waymo-camera-model-v1",
        "vehicle_from_lidar": vehicle_from_lidar,
        "reference_world_from_lidar": reference_world_from_vehicle @ vehicle_from_lidar,
        "camera_world_from_vehicle": camera_world_from_vehicle,
        "linear_velocity_world": np.asarray(metadata["linear_velocity_world"], dtype=np.float64).reshape(3),
        "angular_velocity_vehicle": np.asarray(metadata["angular_velocity_vehicle"], dtype=np.float64).reshape(3),
        "pose_timestamp": float(metadata["pose_timestamp"]),
        "principal_time": principal_time,
        "readout_factor": float(readout_factor),
        "horizontal": horizontal,
        "direction": direction,
        # A calibration trial projects the same handful of point arrays many
        # times.  Their reference-time world coordinates do not depend on the
        # candidate extrinsic, so retain them instead of repeating this large
        # matrix multiply in every loss evaluation.
        "world_point_cache": {},
    }


def project(
    lidar_points: np.ndarray,
    opencv_camera_from_lidar: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    prepared: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project reference-time LiDAR points with Waymo's rolling-shutter model."""
    lidar_points = np.asarray(lidar_points, dtype=np.float64)
    candidate = np.asarray(opencv_camera_from_lidar, dtype=np.float64).reshape(4, 4)
    native_camera_from_lidar = WAYMO_CAMERA_FROM_OPENCV @ candidate
    native_rotation = native_camera_from_lidar[:3, :3]
    native_translation = native_camera_from_lidar[:3, 3]
    lidar_from_camera = np.eye(4, dtype=np.float64)
    lidar_from_camera[:3, :3] = native_rotation.T
    lidar_from_camera[:3, 3] = -(native_rotation.T @ native_translation)
    vehicle_from_camera = prepared["vehicle_from_lidar"] @ lidar_from_camera

    reference_world_from_lidar = prepared["reference_world_from_lidar"]
    cache_key = (
        id(lidar_points),
        int(lidar_points.__array_interface__["data"][0]),
        lidar_points.shape,
        lidar_points.strides,
    )
    world_point_cache = prepared["world_point_cache"]
    cached = world_point_cache.get(cache_key)
    world_points = cached[1] if cached is not None else None
    if world_points is None:
        world_points = (
            lidar_points @ reference_world_from_lidar[:3, :3].T
            + reference_world_from_lidar[:3, 3]
        )
        # Retain the source array with the result so Python cannot recycle its
        # id and data pointer while this prepared frame remains alive.
        world_point_cache[cache_key] = (lidar_points, world_points)
    world_from_vehicle = prepared["camera_world_from_vehicle"]
    world_from_camera = world_from_vehicle @ vehicle_from_camera
    camera_from_world_at_pose = world_from_camera[:3, :3].T
    omega_vehicle = prepared["angular_velocity_vehicle"]
    omega_camera = vehicle_from_camera[:3, :3].T @ omega_vehicle
    omega_skew = _skew(omega_camera)
    rotation_rate = -omega_skew @ camera_from_world_at_pose
    omega_world = world_from_vehicle[:3, :3] @ omega_vehicle
    camera_position = world_from_camera[:3, 3]
    camera_velocity = prepared["linear_velocity_world"] + _skew(omega_world) @ (
        world_from_vehicle[:3, :3] @ vehicle_from_camera[:3, 3]
    )

    def camera_points_at(time_offsets: np.ndarray) -> np.ndarray:
        relative = world_points - (
            camera_position + time_offsets[:, None] * camera_velocity
        )
        # (R + t*dR) @ p, expanded to avoid materializing one 3x3 matrix
        # per point. This is algebraically identical to Waymo's model and is
        # substantially faster for full-density TOP scans.
        return (
            relative @ camera_from_world_at_pose.T
            + time_offsets[:, None] * (relative @ rotation_rate.T)
        )

    time = np.zeros(len(world_points), dtype=np.float64)
    if prepared["direction"] != 5:
        pose_offset = prepared["pose_timestamp"] - prepared["principal_time"]
        readout_factor = prepared["readout_factor"]
        horizontal = prepared["horizontal"]
        for _ in range(4):
            camera_points = camera_points_at(time)
            normalized_x = -camera_points[:, 1] / camera_points[:, 0]
            normalized_y = -camera_points[:, 2] / camera_points[:, 0]
            spacing = normalized_x if horizontal else normalized_y
            residual = time - spacing * readout_factor + pose_offset
            landmark_to_index = (
                camera_from_world_at_pose @ -camera_velocity
                + time[:, None] * (rotation_rate @ -camera_velocity)
                - camera_points @ omega_skew.T
            )
            combined = (
                normalized_x * landmark_to_index[:, 0] - landmark_to_index[:, 1]
                if horizontal else
                normalized_y * landmark_to_index[:, 0] - landmark_to_index[:, 2]
            )
            jacobian = 1.0 - readout_factor / camera_points[:, 0] * combined
            stable = np.isfinite(jacobian) & (np.abs(jacobian) > 1e-9)
            time[stable] -= residual[stable] / jacobian[stable]

    camera_points = camera_points_at(time)
    depth_all = camera_points[:, 0]
    normalized_x = -camera_points[:, 1] / depth_all
    normalized_y = -camera_points[:, 2] / depth_all
    k1, k2, _, _, k3 = np.pad(
        np.asarray(distortion, dtype=np.float64), (0, 5),
    )[:5]
    radius2 = normalized_x * normalized_x + normalized_y * normalized_y
    radial = 1.0 + k1 * radius2 + k2 * radius2**2 + k3 * radius2**3
    distorted_x, distorted_y = _distort_normalized(normalized_x, normalized_y, distortion)
    pixel_x = intrinsic[0, 0] * distorted_x + intrinsic[0, 2]
    pixel_y = intrinsic[1, 1] * distorted_y + intrinsic[1, 2]
    valid = (
        (depth_all > 0.1) & np.isfinite(pixel_x) & np.isfinite(pixel_y)
        & np.isfinite(time)
        # Waymo CameraModel::DirectionToImage explicitly rejects directions
        # outside this range. Without it, the RadTan polynomial can flip sign
        # and fold side/rear LiDAR points back into the FRONT image as arcs.
        & (radial >= MIN_TRUSTED_RADIAL_DISTORTION)
        & (radial <= MAX_TRUSTED_RADIAL_DISTORTION)
    )
    return pixel_x[valid], pixel_y[valid], depth_all[valid], valid
