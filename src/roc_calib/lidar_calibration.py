from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

try:
    import open3d as o3d
except ImportError:  # Keep the service importable in the production image.
    o3d = None

try:
    import small_gicp
except ImportError:  # Keep the service importable in the production image.
    small_gicp = None

from .nuscenes_fusion import is_virtual_cloud, read_cloud_bytes
from .groups import PREPARED_ROOT


ALGORITHM_VERSION = "fpfh-ransac-multiseed-small-gicp-contour-v12"
MIN_MANUAL_PAIR_POINTS = 6


class LidarObjectPair(BaseModel):
    annotation_id: str
    source_cloud_url: str
    target_cloud_url: str
    source_indices: list[int] = Field(min_length=6)
    target_indices: list[int] = Field(min_length=6)


class LidarPreparedFramePair(BaseModel):
    source_cloud_url: str
    target_cloud_url: str
    source_timestamp_ns: int
    target_timestamp_ns: int
    delta_ms: float = Field(ge=0)
    origins: list[str] = Field(default_factory=list)


class LidarFramePairRequest(BaseModel):
    source_bag_path: str
    source_lidar_topic: str
    target_lidar_topic: str
    source_selected_timestamps: list[int] = Field(min_length=1)
    target_selected_timestamps: list[int] = Field(min_length=1)


class LidarPairRequest(BaseModel):
    group_key: str
    source_lidar_id: str
    target_lidar_id: str
    source_cloud_url: str
    target_cloud_url: str
    base_matrix: list[float] | None = Field(default=None, min_length=16, max_length=16)
    pairs: list[LidarObjectPair] = Field(default_factory=list)
    frame_pairs: list[LidarPreparedFramePair] = Field(default_factory=list)
    stable_only: bool = True
    max_correspondence_distance: float = Field(default=1.5, gt=0.05, le=10.0)
    source_bag_path: str | None = None
    source_lidar_topic: str | None = None
    target_lidar_topic: str | None = None
    source_selected_timestamps: list[int] = Field(default_factory=list)
    target_selected_timestamps: list[int] = Field(default_factory=list)


class LidarGraphEdge(BaseModel):
    edge_id: str
    source_lidar_id: str
    target_lidar_id: str
    matrix: list[float] = Field(min_length=16, max_length=16)
    weight: float = Field(default=1.0, gt=0.0, le=1000.0)


class LidarGraphRequest(BaseModel):
    anchor_lidar_id: str
    lidar_ids: list[str] = Field(min_length=1)
    edges: list[LidarGraphEdge] = Field(min_length=1)


def _resolve_cloud(url: str) -> Path:
    relative = url.split("?", 1)[0].lstrip("/")
    if relative.startswith("data/cache/"):
        relative = relative.removeprefix("data/cache/")
    path = (PREPARED_ROOT / relative).resolve()
    if PREPARED_ROOT not in path.parents or not (path.is_file() or is_virtual_cloud(path)):
        raise ValueError(f"点云文件不可用：{url}")
    return path


def _read_cloud_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = read_cloud_bytes(path)
    if raw[:4] != b"ACP1":
        raise ValueError("无法识别点云格式")
    count = struct.unpack_from("<I", raw, 4)[0]
    if len(raw) != 16 + count * 16:
        raise ValueError("点云数据不完整")
    records = np.ndarray((count,), dtype=np.dtype({
        "names": ["x", "y", "z", "stable"],
        "formats": ["<f4", "<f4", "<f4", "u1"],
        "offsets": [0, 4, 8, 12], "itemsize": 16,
    }), buffer=raw, offset=16)
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    points[records["stable"] == 2] = np.nan
    return points, np.asarray(records["stable"] == 1, dtype=bool)


def _read_cloud(url: str) -> tuple[np.ndarray, np.ndarray]:
    return _read_cloud_path(_resolve_cloud(url))


def _cloud_url(path: Path) -> str:
    return "/data/cache/" + str(path.resolve().relative_to(PREPARED_ROOT.resolve()))


def build_lidar_frame_pairs(request: LidarFramePairRequest) -> list[dict]:
    from .preparation import nearest_lidar_frame_pairs
    pairs = nearest_lidar_frame_pairs(
        request.source_bag_path,
        request.source_lidar_topic,
        request.target_lidar_topic,
        request.source_selected_timestamps,
        request.target_selected_timestamps,
    )
    return [{
        "index": index + 1,
        "sourceTimestampNs": str(pair["sourceTimestampNs"]),
        "targetTimestampNs": str(pair["targetTimestampNs"]),
        "deltaMs": pair["deltaMs"],
        "origins": pair["origins"],
        "sourceCloudUrl": _cloud_url(pair["sourcePath"]),
        "targetCloudUrl": _cloud_url(pair["targetPath"]),
    } for index, pair in enumerate(pairs)]


def _voxel_downsample(points: np.ndarray, voxel: float, maximum: int = 4500) -> np.ndarray:
    points = points[np.all(np.isfinite(points), axis=1)]
    if not len(points):
        return points
    keys = np.floor(points / voxel).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(indices)]
    if len(sampled) > maximum:
        sampled = sampled[np.linspace(0, len(sampled) - 1, maximum, dtype=np.int64)]
    return sampled


def _covariances(points: np.ndarray, neighbors: int = 16) -> np.ndarray:
    tree = cKDTree(points)
    _, indices = tree.query(points, k=min(neighbors, len(points)))
    if indices.ndim == 1:
        indices = indices[:, None]
    local = points[indices]
    centered = local - local.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / max(1, local.shape[1] - 1)
    values, vectors = np.linalg.eigh(covariance)
    scale = np.maximum(values[:, -1], 1e-4)
    values[:, 0] = np.maximum(values[:, 0], scale * 1e-3)
    values[:, 1] = np.maximum(values[:, 1], scale * 1e-2)
    return np.einsum("nij,nj,nkj->nik", vectors, values, vectors)


def _matrix(parameters: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_rotvec(parameters[:3]).as_matrix()
    result[:3, 3] = parameters[3:]
    return result


def _parameters(matrix: np.ndarray) -> np.ndarray:
    return np.r_[Rotation.from_matrix(matrix[:3, :3]).as_rotvec(), matrix[:3, 3]]


def _transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _registration_region(points: np.ndarray) -> np.ndarray:
    """Keep the useful geometric range without any semantic classification."""
    finite = points[np.all(np.isfinite(points), axis=1)]
    distance = np.linalg.norm(finite, axis=1)
    return finite[(distance >= 2.0) & (distance <= 80.0)]


def _fit_ground_plane(points: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Robustly estimate the road plane in a raw lidar coordinate frame."""
    usable = points[np.all(np.isfinite(points), axis=1)]
    usable = usable[np.linalg.norm(usable[:, :2], axis=1) <= 45.0]
    if len(usable) < 80:
        return None
    if len(usable) > 7000:
        usable = usable[np.linspace(0, len(usable) - 1, 7000, dtype=np.int64)]
    generator = np.random.default_rng(17)
    best_count, best_normal, best_offset = 0, None, 0.0
    for _ in range(420):
        sample = usable[generator.choice(len(usable), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        if normal[2] < 0:
            normal = -normal
        # All supported raw lidar frames keep Z broadly upward, even when the
        # sensor is mounted on a sloped body panel. This rejects vertical walls.
        if normal[2] < 0.55:
            continue
        offset = float(sample[0] @ normal)
        distances = np.abs(usable @ normal - offset)
        count = int(np.count_nonzero(distances <= .10))
        if count > best_count:
            best_count, best_normal, best_offset = count, normal.copy(), offset
    if best_normal is None or best_count < max(45, int(len(usable) * .06)):
        return None
    distances = np.abs(usable @ best_normal - best_offset)
    inliers = usable[distances <= .12]
    centered = inliers - inliers.mean(axis=0)
    _values, vectors = np.linalg.eigh(centered.T @ centered)
    normal = vectors[:, 0]
    if normal[2] < 0:
        normal = -normal
    offset = float(np.median(inliers @ normal))
    return normal, offset


def _constrain_ground_transform(
    matrix: np.ndarray,
    source_ground: tuple[np.ndarray, float],
    target_ground: tuple[np.ndarray, float],
    maximum_angle_deg: float = .35,
    maximum_height_error: float = .04,
) -> np.ndarray:
    """Clamp roll/pitch and vertical drift while leaving planar motion free."""
    source_normal, source_offset = source_ground
    target_normal, target_offset = target_ground
    constrained = matrix.copy()
    transformed_normal = constrained[:3, :3] @ source_normal
    correction, _error = Rotation.align_vectors(target_normal[None], transformed_normal[None])
    correction_vector = correction.as_rotvec()
    correction_angle = float(np.linalg.norm(correction_vector))
    allowed_angle = math.radians(maximum_angle_deg)
    if correction_angle > allowed_angle:
        applied = correction_vector * ((correction_angle - allowed_angle) / correction_angle)
        constrained[:3, :3] = Rotation.from_rotvec(applied).as_matrix() @ constrained[:3, :3]
    transformed_normal = constrained[:3, :3] @ source_normal
    transformed_offset = float(source_offset + transformed_normal @ constrained[:3, 3])
    height_error = transformed_offset - target_offset
    clamped_error = float(np.clip(height_error, -maximum_height_error, maximum_height_error))
    constrained[:3, 3] += transformed_normal * (clamped_error - height_error)
    return constrained


def _ground_diagnostics(
    matrix: np.ndarray,
    source_ground: tuple[np.ndarray, float],
    target_ground: tuple[np.ndarray, float],
) -> tuple[float, float]:
    source_normal, source_offset = source_ground
    target_normal, target_offset = target_ground
    transformed_normal = matrix[:3, :3] @ source_normal
    angle = math.degrees(math.acos(float(np.clip(transformed_normal @ target_normal, -1.0, 1.0))))
    transformed_offset = float(source_offset + transformed_normal @ matrix[:3, 3])
    return angle, abs(transformed_offset - target_offset)


def _coarse_score(
    source: np.ndarray,
    target: np.ndarray,
    source_tree: cKDTree,
    target_tree: cKDTree,
    matrix: np.ndarray,
) -> float:
    forward, _ = target_tree.query(_transform(source, matrix), k=1)
    backward, _ = source_tree.query(_transform(target, np.linalg.inv(matrix)), k=1)
    values = []
    for distances in (forward, backward):
        cutoff = min(2.5, float(np.quantile(distances, .65)))
        trimmed = float(np.mean(np.minimum(distances, cutoff) ** 2))
        overlap = float(np.mean(distances <= 1.0))
        values.append(trimmed + (1.0 - overlap) * .18)
    return float(np.mean(values))


def _partial_overlap_score(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    source_tree: cKDTree | None = None,
    target_tree: cKDTree | None = None,
) -> float:
    """Trim both directions so unobserved faces do not dominate a candidate."""
    source_tree = source_tree if source_tree is not None else cKDTree(source)
    target_tree = target_tree if target_tree is not None else cKDTree(target)
    forward = target_tree.query(_transform(source, matrix), k=1)[0]
    # Rigid transforms preserve distance, so query in the source frame and
    # reuse its tree instead of rebuilding one for every manual seed.
    backward = source_tree.query(_transform(target, np.linalg.inv(matrix)), k=1)[0]
    values = []
    for distances in (forward, backward):
        keep = max(20, int(len(distances) * .45))
        trimmed = np.partition(distances, min(keep, len(distances)) - 1)[:keep]
        robust_error = float(np.mean(np.minimum(trimmed, 1.0) ** 2))
        coverage = float(np.mean(distances <= .6))
        values.append(robust_error + max(0.0, .12 - coverage) * .45)
    return float(np.mean(values))


def _angular_contour(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Density-independent angular silhouette quantiles in one lidar frame."""
    usable = points[np.all(np.isfinite(points), axis=1)]
    ranges = np.linalg.norm(usable, axis=1)
    usable = usable[ranges > .1]
    ranges = ranges[ranges > .1]
    azimuth = np.arctan2(usable[:, 1], usable[:, 0])
    circular_center = math.atan2(float(np.mean(np.sin(azimuth))), float(np.mean(np.cos(azimuth))))
    azimuth = circular_center + np.arctan2(
        np.sin(azimuth - circular_center), np.cos(azimuth - circular_center),
    )
    elevation = np.arctan2(usable[:, 2], np.linalg.norm(usable[:, :2], axis=1))
    quantiles = (.08, .5, .92)
    return np.quantile(azimuth, quantiles), np.quantile(elevation, quantiles), float(np.median(ranges))


def _angular_contour_residual(
    transformed_source: np.ndarray,
    target_contour: tuple[np.ndarray, np.ndarray, float],
) -> np.ndarray:
    source_azimuth, source_elevation, _source_range = _angular_contour(transformed_source)
    target_azimuth, target_elevation, target_range = target_contour
    azimuth_error = np.arctan2(
        np.sin(source_azimuth - target_azimuth),
        np.cos(source_azimuth - target_azimuth),
    )
    elevation_error = source_elevation - target_elevation
    # Angular sampling noise produces lateral uncertainty proportional to
    # range. Expressing the outline directly in radians preserves the far
    # target's rotation leverage without treating sparse scan lines as exact
    # point correspondences.
    angular_sigma = math.radians(.18 + min(target_range, 100.0) * .0015)
    # Road-plane constraints already determine roll/pitch. Elevation outlines
    # vary more across different vertical beam layouts, so keep them as a weak
    # cue and let the far-range azimuth contour primarily constrain yaw.
    return np.r_[azimuth_error, elevation_error * .25] / angular_sigma


def _far_contour_weight(target_range: float) -> float:
    """Increase yaw information with range without letting sparse crops dominate."""
    # Angular leverage grows linearly with range, so its information contribution
    # grows approximately with range squared. Cap it because distant crops are
    # sparse and the two lidars may observe different parts of the outline.
    return float(np.clip((target_range / 25.0) ** 2, 1.0, 10.0))


_TANGENT_CONTOUR_QUANTILES = np.linspace(.08, .92, 9)
_TANGENT_CONTOUR_ANGLES = np.linspace(0.0, math.pi, 12, endpoint=False)
_TANGENT_CONTOUR_DIRECTIONS = np.column_stack((
    np.cos(_TANGENT_CONTOUR_ANGLES),
    np.sin(_TANGENT_CONTOUR_ANGLES),
))


def _tangent_contour_reference(points: np.ndarray) -> dict:
    """Build a target-view tangent plane and density-independent silhouette."""
    usable = points[np.all(np.isfinite(points), axis=1)]
    center = np.median(usable, axis=0)
    target_range = float(np.linalg.norm(center))
    if target_range < .1:
        raise ValueError("参照物距离雷达原点过近，无法建立局部轮廓")
    radial = center / target_range
    # Pick the least parallel Cartesian axis only to seed the plane. Metrics
    # aggregate evenly spaced directions, so the score is not tied to this
    # arbitrary in-plane orientation.
    axes = np.eye(3)
    seed = axes[int(np.argmin(np.abs(axes @ radial)))]
    tangent_u = np.cross(radial, seed)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(radial, tangent_u)
    basis = np.column_stack((tangent_u, tangent_v))
    directions_3d = basis @ _TANGENT_CONTOUR_DIRECTIONS.T
    projections = usable @ directions_3d
    return {
        "directions2d": _TANGENT_CONTOUR_DIRECTIONS,
        "directions3d": directions_3d,
        "quantiles": np.quantile(projections, _TANGENT_CONTOUR_QUANTILES, axis=0).T,
        "range": target_range,
    }


def _tangent_contour_difference(
    transformed_source: np.ndarray,
    target_contour: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compare two silhouettes with sliced quantiles, without point matches."""
    usable = transformed_source[np.all(np.isfinite(transformed_source), axis=1)]
    projections = usable @ target_contour["directions3d"]
    source_quantiles = np.quantile(projections, _TANGENT_CONTOUR_QUANTILES, axis=0).T
    differences = source_quantiles - target_contour["quantiles"]
    # Each 1-D slice estimates the same 2-D silhouette translation. Fitting all
    # slice medians separates rigid outline displacement from visibility/shape
    # differences between lidars.
    directional_centers = np.median(differences, axis=1)
    tangent_shift = np.linalg.lstsq(
        target_contour["directions2d"], directional_centers, rcond=None,
    )[0]
    shape_difference = differences - (
        target_contour["directions2d"] @ tangent_shift
    )[:, None]
    return differences, tangent_shift, shape_difference


def _tangent_contour_metrics(transformed_source: np.ndarray, target_contour: dict) -> dict:
    differences, tangent_shift, shape_difference = _tangent_contour_difference(
        transformed_source, target_contour,
    )
    absolute = np.abs(differences)
    return {
        # Mean sliced Wasserstein-1 distance: density independent, metric, and
        # does not require either direction of nearest-point correspondence.
        "contourError": float(np.mean(absolute)),
        "contourP90Error": float(np.quantile(absolute, .9)),
        "contourCenterError": float(np.linalg.norm(tangent_shift)),
        "contourShapeError": float(np.mean(np.abs(shape_difference))),
        "rangeM": float(target_contour["range"]),
    }


def _tangent_contour_residual(
    transformed_source: np.ndarray,
    target_contour: dict,
    uncertainty: float,
) -> np.ndarray:
    """Robust local-outline residual used by manual refinement."""
    differences, _shift, _shape = _tangent_contour_difference(
        transformed_source, target_contour,
    )
    # Optimize the robust displacement shared by the outline quantiles. The
    # remaining per-quantile shape term is a quality diagnostic, not a rigid
    # transform constraint: forcing it would pull one lidar toward surfaces
    # visible only to the other lidar.
    directional_centers = np.median(differences, axis=1)
    range_weight = math.sqrt(_far_contour_weight(target_contour["range"]))
    return directional_centers * .42 * range_weight / max(.04, uncertainty)


def _manual_refinement_candidates(
    paired_clouds: list[tuple[np.ndarray, np.ndarray]],
    base: np.ndarray | None,
    ground_constraint: tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]] | None,
) -> list[np.ndarray]:
    """Build manual-pair-driven seeds, with or without an existing extrinsic."""
    if base is None:
        base = np.eye(4)
        if ground_constraint:
            (source_normal, source_offset), (target_normal, target_offset) = ground_constraint
            tilt, _error = Rotation.align_vectors(target_normal[None], source_normal[None])
            base[:3, :3] = tilt.as_matrix()
            base[:3, 3] = target_normal * (target_offset - source_offset)
    if ground_constraint:
        base = _constrain_ground_transform(base, *ground_constraint)
    target_axis = ground_constraint[1][0] if ground_constraint else base[:3, :3] @ np.array([0.0, 0.0, 1.0])
    target_axis = target_axis / max(np.linalg.norm(target_axis), 1e-9)
    tangent_u = np.cross(target_axis, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(tangent_u) < .1:
        tangent_u = np.cross(target_axis, np.array([0.0, 1.0, 0.0]))
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(target_axis, tangent_u)
    source_centers = [np.median(source, axis=0) for source, _target in paired_clouds]
    target_centers = [np.median(target, axis=0) for _source, target in paired_clouds]
    candidates = [base.copy()]
    translation_offsets = tuple(
        tangent_u * offset_u + tangent_v * offset_v
        for offset_u in (-1.5, -.75, 0.0, .75, 1.5)
        for offset_v in (-1.5, -.75, 0.0, .75, 1.5)
    )
    for yaw_offset in np.linspace(-math.pi, math.pi, 36, endpoint=False):
        rotation = Rotation.from_rotvec(target_axis * yaw_offset).as_matrix() @ base[:3, :3]
        center_translations = np.asarray([
            target_center - rotation @ source_center
            for source_center, target_center in zip(source_centers, target_centers)
        ])
        translation = np.median(center_translations, axis=0)
        for offset in translation_offsets:
            candidate = np.eye(4)
            candidate[:3, :3] = rotation
            candidate[:3, 3] = translation + offset
            if ground_constraint:
                candidate = _constrain_ground_transform(candidate, *ground_constraint)
            candidates.append(candidate)
    return candidates


def _initial_candidates(source: np.ndarray, target: np.ndarray, base: np.ndarray | None) -> list[np.ndarray]:
    candidates = [base.copy()] if base is not None else [np.eye(4)]
    coarse_source = source if len(source) <= 1400 else source[np.linspace(0, len(source) - 1, 1400, dtype=np.int64)]
    coarse_target = target if len(target) <= 1400 else target[np.linspace(0, len(target) - 1, 1400, dtype=np.int64)]
    source_tree = cKDTree(coarse_source)
    target_tree = cKDTree(coarse_target)
    source_ground = _fit_ground_plane(source)
    target_ground = _fit_ground_plane(target)
    if source_ground is not None and target_ground is not None:
        source_normal, source_offset = source_ground
        target_normal, target_offset = target_ground
        tilt, _error = Rotation.align_vectors(target_normal[None], source_normal[None])
        tilt_matrix = tilt.as_matrix()
        tangent_u = np.cross(target_normal, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(tangent_u) < .1:
            tangent_u = np.cross(target_normal, np.array([0.0, 1.0, 0.0]))
        tangent_u /= np.linalg.norm(tangent_u)
        tangent_v = np.cross(target_normal, tangent_u)
        normal_translation = target_normal * (target_offset - source_offset)
        for yaw in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            yaw_matrix = Rotation.from_rotvec(target_normal * yaw).as_matrix()
            rotation = yaw_matrix @ tilt_matrix
            for offset_u in (-4.0, -2.0, 0.0, 2.0, 4.0):
                for offset_v in (-4.0, -2.0, 0.0, 2.0, 4.0):
                    candidate = np.eye(4)
                    candidate[:3, :3] = rotation
                    candidate[:3, 3] = normal_translation + tangent_u * offset_u + tangent_v * offset_v
                    candidates.append(candidate)
        candidates.sort(key=lambda item: _coarse_score(coarse_source, coarse_target, source_tree, target_tree, item))
        return candidates[:10]

    source_center = np.median(source, axis=0)
    target_center = np.median(target, axis=0)
    for yaw in np.linspace(-math.pi, math.pi, 16, endpoint=False):
        candidate = np.eye(4)
        candidate[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        candidate[:3, 3] = target_center - candidate[:3, :3] @ source_center
        candidates.append(candidate)
    candidates.sort(key=lambda item: _coarse_score(coarse_source, coarse_target, source_tree, target_tree, item))
    return candidates[:8]


def _remove_ground(points: np.ndarray, plane: tuple[np.ndarray, float] | None) -> np.ndarray:
    """Remove only the fitted road band; this is geometry, not object recognition."""
    if plane is None:
        return points
    normal, offset = plane
    distances = np.abs(points @ normal - offset)
    non_ground = points[distances >= .22]
    return non_ground if len(non_ground) >= 120 else points


def _open3d_features(points: np.ndarray, voxel: float):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    cloud = cloud.voxel_down_sample(voxel)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=40))
    features = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100),
    )
    return cloud, features


def _global_fpfh_ransac(
    source: np.ndarray,
    target: np.ndarray,
    progress: Callable[[int, str, dict | None], None] | None = None,
) -> tuple[np.ndarray, dict]:
    if o3d is None:
        raise RuntimeError("当前 3002 镜像未安装 Open3D，无法执行全局粗配准")
    voxel = .45
    source_feature_points = _remove_ground(source, _fit_ground_plane(source))
    target_feature_points = _remove_ground(target, _fit_ground_plane(target))
    source_cloud, source_feature = _open3d_features(source_feature_points, voxel)
    target_cloud, target_feature = _open3d_features(target_feature_points, voxel)
    if len(source_cloud.points) < 80 or len(target_cloud.points) < 80:
        raise ValueError("剔除地面后结构点不足，无法计算 FPFH 全局初值")
    if progress:
        progress(18, f"正在匹配 FPFH 全局特征 · {len(source_cloud.points)} × {len(target_cloud.points)}", None)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_cloud,
        target_cloud,
        source_feature,
        target_feature,
        True,
        voxel * 1.6,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(.85),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(voxel * 1.6),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(60000, .999),
    )
    matrix = np.asarray(result.transformation, dtype=np.float64)
    translation = float(np.linalg.norm(matrix[:3, 3]))
    diagnostics = {
        "globalFitness": float(result.fitness),
        "globalInlierRmse": float(result.inlier_rmse),
        "globalCorrespondenceCount": int(len(result.correspondence_set)),
        "globalVoxelSize": voxel,
    }
    if not np.all(np.isfinite(matrix)) or float(result.fitness) < .035:
        raise ValueError(
            f"FPFH-RANSAC 全局配准失败：特征内点率仅 {float(result.fitness) * 100:.1f}%"
        )
    if translation > 8.0:
        raise ValueError(f"FPFH-RANSAC 全局配准失败：平移 {translation:.2f} m 超出车载雷达合理范围")
    if progress:
        progress(38, f"全局粗配准完成 · 特征内点率 {float(result.fitness) * 100:.1f}%", diagnostics)
    return matrix, diagnostics


def _registration_metrics(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
    maximum_distance: float,
    paired_clouds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    pairs = paired_clouds or [(source, target)]
    pair_metrics = []
    all_forward, all_backward = [], []
    evaluation_distance = min(maximum_distance, .5)
    for index, (pair_source, pair_target) in enumerate(pairs):
        transformed = _transform(pair_source, matrix)
        forward = cKDTree(pair_target).query(transformed, k=1)[0]
        backward = cKDTree(transformed).query(pair_target, k=1)[0]
        all_forward.append(forward)
        all_backward.append(backward)
        valid_forward = forward <= evaluation_distance
        valid_backward = backward <= evaluation_distance
        inliers = np.r_[forward[valid_forward], backward[valid_backward]]
        overlap = float((np.mean(valid_forward) + np.mean(valid_backward)) * .5)
        pair_metrics.append({
            "index": index + 1,
            "overlapRatio": overlap,
            "medianError": float(np.median(inliers)) if len(inliers) else float("inf"),
            "p90Error": float(np.quantile(inliers, .9)) if len(inliers) else float("inf"),
            "reliable": False,
        })
    forward = np.concatenate(all_forward)
    backward = np.concatenate(all_backward)
    valid_forward = forward <= evaluation_distance
    valid_backward = backward <= evaluation_distance
    inliers = np.r_[forward[valid_forward], backward[valid_backward]]
    comparable = np.asarray([
        item["medianError"] for item in pair_metrics
        if item["overlapRatio"] >= .10 and np.isfinite(item["medianError"])
    ])
    spread = 0.0
    if len(comparable):
        center = float(np.median(comparable))
        spread = float(np.median(np.abs(comparable - center)))
        limit = max(.06, spread * 4.5)
        for item in pair_metrics:
            item["reliable"] = bool(
                item["overlapRatio"] >= .10
                and np.isfinite(item["medianError"])
                and abs(item["medianError"] - center) <= limit
            )
    consistency = float(np.mean([item["reliable"] for item in pair_metrics]))
    overlap = float((np.mean(valid_forward) + np.mean(valid_backward)) * .5)
    return {
        "final_rmse": float(np.sqrt(np.mean(inliers ** 2))) if len(inliers) else float("inf"),
        "median_error": float(np.median(inliers)) if len(inliers) else float("inf"),
        "p90_error": float(np.quantile(inliers, .9)) if len(inliers) else float("inf"),
        "overlap_ratio": overlap,
        "pair_consistency_ratio": consistency,
        "pair_error_spread": spread,
        "frame_pair_metrics": pair_metrics,
    }


def _small_gicp_registration(
    source: np.ndarray,
    target: np.ndarray,
    initial: np.ndarray,
    maximum_distance: float,
    progress: Callable[[int, str, dict | None], None] | None = None,
    paired_clouds: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> dict:
    if small_gicp is None:
        raise RuntimeError("当前 3002 镜像未安装 small_gicp，无法执行 C++ GICP")
    initial_metrics = _registration_metrics(source, target, initial, maximum_distance, paired_clouds)
    if progress:
        progress(45, "正在执行 small_gicp C++ 局部精调", None)
    result = small_gicp.align(
        np.asarray(target, dtype=np.float64),
        np.asarray(source, dtype=np.float64),
        init_T_target_source=np.asarray(initial, dtype=np.float64),
        registration_type="GICP",
        downsampling_resolution=.20,
        max_correspondence_distance=min(maximum_distance, 1.5),
        num_threads=4,
        max_iterations=64,
    )
    matrix = np.asarray(result.T_target_source, dtype=np.float64)
    metrics = _registration_metrics(source, target, matrix, maximum_distance, paired_clouds)
    source_ground = _fit_ground_plane(source)
    target_ground = _fit_ground_plane(target)
    ground_angle, ground_height = (
        _ground_diagnostics(matrix, source_ground, target_ground)
        if source_ground is not None and target_ground is not None else (None, None)
    )
    if progress:
        progress(90, f"C++ 局部精调完成 · 重叠率 {metrics['overlap_ratio'] * 100:.1f}%", None)
    return {
        "matrix": matrix,
        "initial_rmse": initial_metrics["final_rmse"],
        "correspondence_count": int(getattr(result, "num_inliers", 0)),
        "converged": bool(getattr(result, "converged", True)),
        "iteration_count": int(getattr(result, "iterations", 0)),
        "ground_angle_deg": ground_angle,
        "ground_height_error": ground_height,
        **metrics,
    }


def _gicp(
    source: np.ndarray,
    target: np.ndarray,
    base: np.ndarray | None,
    maximum_distance: float,
    progress: Callable[[int, str, dict | None], None] | None = None,
    paired_clouds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    ground_constraint: tuple[tuple[np.ndarray, float], tuple[np.ndarray, float]] | None = None,
    partial_overlap: bool = False,
) -> dict:
    if len(source) < 20 or len(target) < 20:
        raise ValueError("用于配准的有效点不足 20 个")
    source_tree = cKDTree(source)
    target_tree = cKDTree(target)
    coarse_source = source if len(source) <= 1800 else source[np.linspace(0, len(source) - 1, 1800, dtype=np.int64)]
    coarse_target = target if len(target) <= 1800 else target[np.linspace(0, len(target) - 1, 1800, dtype=np.int64)]
    coarse_source_tree = cKDTree(coarse_source)
    coarse_target_tree = cKDTree(coarse_target)
    if partial_overlap and paired_clouds:
        # Manual correspondences drive the broad yaw/translation search even
        # when there is no imported or GICP base. In that case the two fitted
        # road planes provide only roll, pitch and height; annotations resolve
        # planar translation and yaw.
        candidates = _manual_refinement_candidates(paired_clouds, base, ground_constraint)
    else:
        candidates = _initial_candidates(source, target, base)
        reverse_base = np.linalg.inv(base) if base is not None else None
        candidates.extend(np.linalg.inv(item) for item in _initial_candidates(target, source, reverse_base))
    paired_indexes = []
    if paired_clouds:
        source_offset = target_offset = 0
        for pair_source, pair_target in paired_clouds:
            score_source = pair_source if len(pair_source) <= 500 else pair_source[np.linspace(0, len(pair_source) - 1, 500, dtype=np.int64)]
            score_target = pair_target if len(pair_target) <= 500 else pair_target[np.linspace(0, len(pair_target) - 1, 500, dtype=np.int64)]
            paired_indexes.append({
                "source": pair_source,
                "target": pair_target,
                "sourceOffset": source_offset,
                "targetOffset": target_offset,
                "sourceTree": cKDTree(pair_source),
                "targetTree": cKDTree(pair_target),
                "scoreSource": score_source,
                "scoreTarget": score_target,
                "scoreSourceTree": cKDTree(score_source),
                "scoreTargetTree": cKDTree(score_target),
                "targetContour": _angular_contour(pair_target),
                "targetTangentContour": _tangent_contour_reference(pair_target),
            })
            source_offset += len(pair_source)
            target_offset += len(pair_target)

    def candidate_score(matrix: np.ndarray) -> float:
        if not paired_indexes:
            return _coarse_score(coarse_source, coarse_target, coarse_source_tree, coarse_target_tree, matrix)
        if partial_overlap:
            pair_scores = np.asarray([
                _partial_overlap_score(
                    item["scoreSource"], item["scoreTarget"], matrix,
                    item["scoreSourceTree"], item["scoreTargetTree"],
                )
                for item in paired_indexes
            ])
            # Let the majority of independently annotated objects determine
            # the seed without completely ignoring a difficult crop.
            return float(np.median(pair_scores) * .75 + np.mean(pair_scores) * .25)
        # Correspondences from different timestamps must never compete with one
        # another. Score one transform over all synchronized pairs instead.
        return float(np.mean([
            _coarse_score(
                item["scoreSource"], item["scoreTarget"],
                item["scoreSourceTree"], item["scoreTargetTree"], matrix,
            )
            for item in paired_indexes
        ]))

    current = min(candidates, key=candidate_score)
    if ground_constraint:
        current = _constrain_ground_transform(current, *ground_constraint)
    if partial_overlap and paired_indexes:
        for item in paired_indexes:
            initial_contour = _tangent_contour_metrics(
                _transform(item["source"], current), item["targetTangentContour"],
            )
            # A crop whose two views have intrinsically different silhouettes
            # remains useful for its center, but must not dominate a compatible
            # outline. Freeze this uncertainty before optimization so the
            # solver cannot lower its cost by making the shape less compatible.
            item["tangentContourUncertainty"] = float(np.clip(
                .04 + initial_contour["contourShapeError"] * 1.5, .04, .25,
            ))
    if paired_indexes:
        source_covariance = np.concatenate([_covariances(item["source"]) for item in paired_indexes])
        target_covariance = np.concatenate([_covariances(item["target"]) for item in paired_indexes])
    else:
        source_covariance = _covariances(source)
        target_covariance = _covariances(target)
    initial_rmse = None
    converged = False
    correspondence_count = 0
    last_step = float("inf")
    distance_schedule = [maximum_distance, min(maximum_distance, 1.0), min(maximum_distance, .6), min(maximum_distance, .35)]
    for iteration in range(24):
        if paired_indexes:
            forward_distance_parts, backward_distance_parts = [], []
            forward_target_parts, backward_source_parts = [], []
            for item in paired_indexes:
                transformed_pair = _transform(item["source"], current)
                forward_distance, forward_target = item["targetTree"].query(transformed_pair, k=1)
                backward_distance, backward_source = cKDTree(transformed_pair).query(item["target"], k=1)
                forward_distance_parts.append(forward_distance)
                backward_distance_parts.append(backward_distance)
                forward_target_parts.append(forward_target + item["targetOffset"])
                backward_source_parts.append(backward_source + item["sourceOffset"])
            forward_distances = np.concatenate(forward_distance_parts)
            backward_distances = np.concatenate(backward_distance_parts)
            forward_targets = np.concatenate(forward_target_parts)
            backward_sources = np.concatenate(backward_source_parts)
        else:
            transformed = _transform(source, current)
            forward_distances, forward_targets = target_tree.query(transformed, k=1)
            backward_distances, backward_sources = cKDTree(transformed).query(target, k=1)
        stage_distance = distance_schedule[min(len(distance_schedule) - 1, iteration // 6)]
        combined_distances = np.r_[forward_distances, backward_distances]
        adaptive = min(stage_distance, max(.12, float(np.quantile(combined_distances, .72)) * 1.35))
        if not partial_overlap:
            # Whole-frame GICP initialization keeps its original symmetric
            # nearest-neighbor behavior and adaptive distance trimming.
            forward_valid = forward_distances <= adaptive
            backward_valid = backward_distances <= adaptive
            if np.count_nonzero(forward_valid) + np.count_nonzero(backward_valid) < 36:
                raise ValueError("两路雷达重叠点不足，无法建立稳定 GICP 对应")
            source_indices = np.r_[np.flatnonzero(forward_valid), backward_sources[backward_valid]]
            matched_target = np.r_[forward_targets[forward_valid], np.flatnonzero(backward_valid)]
            selected_distances = np.r_[forward_distances[forward_valid], backward_distances[backward_valid]]
        else:
            # Manual refinement is partial-overlap registration. A selected
            # point is usable only when its nearest target maps back to the
            # same local source neighborhood. This rejects object surfaces
            # that one lidar never observed.
            reciprocal_limit = max(.08, min(.25, adaptive * .45))
            candidate_sources, candidate_targets, candidate_distances = [], [], []
            if paired_indexes:
                for item, forward_distance, forward_target, backward_distance, backward_source in zip(
                    paired_indexes,
                    forward_distance_parts,
                    forward_target_parts,
                    backward_distance_parts,
                    backward_source_parts,
                ):
                    local_target = forward_target - item["targetOffset"]
                    local_backward_source = backward_source - item["sourceOffset"]
                    transformed_pair = _transform(item["source"], current)
                    returned_source = local_backward_source[local_target]
                    cycle_distance = np.linalg.norm(
                        transformed_pair - transformed_pair[returned_source], axis=1,
                    )
                    valid = (
                        (forward_distance <= adaptive)
                        & (backward_distance[local_target] <= adaptive)
                        & (cycle_distance <= reciprocal_limit)
                    )
                    local_source = np.flatnonzero(valid)
                    candidate_sources.append(local_source + item["sourceOffset"])
                    candidate_targets.append(local_target[valid] + item["targetOffset"])
                    candidate_distances.append(forward_distance[valid])
            else:
                transformed = _transform(source, current)
                returned_source = backward_sources[forward_targets]
                cycle_distance = np.linalg.norm(transformed - transformed[returned_source], axis=1)
                valid = (
                    (forward_distances <= adaptive)
                    & (backward_distances[forward_targets] <= adaptive)
                    & (cycle_distance <= reciprocal_limit)
                )
                candidate_sources.append(np.flatnonzero(valid))
                candidate_targets.append(forward_targets[valid])
                candidate_distances.append(forward_distances[valid])
            source_candidates = np.concatenate(candidate_sources) if candidate_sources else np.empty(0, dtype=np.int64)
            target_candidates = np.concatenate(candidate_targets) if candidate_targets else np.empty(0, dtype=np.int64)
            distance_candidates = np.concatenate(candidate_distances) if candidate_distances else np.empty(0)
            # One-to-one use prevents a dense scan stripe from pulling many
            # source returns onto one sparse target return.
            chosen = []
            used_sources: set[int] = set()
            used_targets: set[int] = set()
            for candidate_index in np.argsort(distance_candidates):
                source_index = int(source_candidates[candidate_index])
                target_index = int(target_candidates[candidate_index])
                if source_index in used_sources or target_index in used_targets:
                    continue
                used_sources.add(source_index)
                used_targets.add(target_index)
                chosen.append(int(candidate_index))
            if len(chosen) < 36:
                raise ValueError("人工参照物在两路雷达中的共同可见点不足，无法可靠精调")
            chosen_indices = np.asarray(chosen, dtype=np.int64)
            source_indices = source_candidates[chosen_indices]
            matched_target = target_candidates[chosen_indices]
            selected_distances = distance_candidates[chosen_indices]
        correspondence_count = len(source_indices)
        if initial_rmse is None:
            initial_rmse = float(np.sqrt(np.mean(selected_distances ** 2)))
        correspondence_weights = np.ones(correspondence_count, dtype=np.float64)
        if partial_overlap and paired_indexes:
            group_masks = [
                (source_indices >= item["sourceOffset"])
                & (source_indices < item["sourceOffset"] + len(item["source"]))
                for item in paired_indexes
            ]
            active_masks = [mask for mask in group_masks if np.count_nonzero(mask) >= 3]
            if len(active_masks) < min(2, len(paired_indexes)):
                raise ValueError("有效共同可见点集中在单个参照物，无法约束可靠外参")
            for mask in active_masks:
                count = int(np.count_nonzero(mask))
                # Each annotation contributes the same total squared weight,
                # without throwing away dense geometry from better crops.
                correspondence_weights[mask] = math.sqrt(
                    correspondence_count / (len(active_masks) * count),
                )
        rotation = current[:3, :3]
        combined = target_covariance[matched_target] + np.einsum(
            "ij,njk,lk->nil", rotation, source_covariance[source_indices], rotation,
        )
        combined += np.eye(3)[None] * 1e-6
        whitening = np.linalg.cholesky(np.linalg.inv(combined))
        source_fixed = source[source_indices]
        target_fixed = target[matched_target]

        def residual(parameters: np.ndarray) -> np.ndarray:
            matrix = _matrix(parameters)
            difference = _transform(source_fixed, matrix) - target_fixed
            weighted = np.einsum("nji,nj->ni", whitening, difference)
            point_residual = (weighted * correspondence_weights[:, None]).reshape(-1)
            if not partial_overlap or not paired_indexes:
                return point_residual
            # Sliced tangent-plane silhouettes compare distribution quantiles,
            # not point samples. They are stable across different scan-line
            # densities and naturally give far objects yaw leverage in metres.
            contour_residual = np.concatenate([
                _tangent_contour_residual(
                    _transform(item["source"], matrix),
                    item["targetTangentContour"],
                    item["tangentContourUncertainty"],
                )
                for item in paired_indexes
            ])
            return np.r_[point_residual, contour_residual]

        before = _parameters(current)
        optimized = least_squares(residual, before, loss="huber", f_scale=1.0, max_nfev=20)
        current = _matrix(optimized.x)
        if ground_constraint:
            current = _constrain_ground_transform(current, *ground_constraint)
        step = np.linalg.norm(optimized.x - before)
        last_step = float(step)
        if progress:
            constraint = f" · {len(paired_indexes)} 帧对内约束" if paired_indexes else ""
            method = "部分重叠精调" if partial_overlap else "GICP"
            correspondence = "个共同可见对应" if partial_overlap else "对应点"
            progress(15 + int((iteration + 1) / 24 * 75), f"{method}迭代 {iteration + 1} · {correspondence_count} {correspondence}{constraint}", None)
        if iteration >= 5 and step < 5e-4:
            converged = True
            break
    if paired_indexes:
        final_forward_parts, final_backward_parts = [], []
        for item in paired_indexes:
            final_transformed = _transform(item["source"], current)
            final_forward_parts.append(item["targetTree"].query(final_transformed, k=1)[0])
            final_backward_parts.append(cKDTree(final_transformed).query(item["target"], k=1)[0])
        final_forward = np.concatenate(final_forward_parts)
        final_backward = np.concatenate(final_backward_parts)
    else:
        final_transformed = _transform(source, current)
        final_forward, _ = target_tree.query(final_transformed, k=1)
        final_backward, _ = cKDTree(final_transformed).query(target, k=1)
    evaluation_distance = min(maximum_distance, .5)
    final_forward_valid = final_forward <= evaluation_distance
    final_backward_valid = final_backward <= evaluation_distance
    final_inliers = np.r_[final_forward[final_forward_valid], final_backward[final_backward_valid]]
    final_rmse = float(np.sqrt(np.mean(final_inliers ** 2))) if len(final_inliers) else float("inf")
    median_error = float(np.median(final_inliers)) if len(final_inliers) else float("inf")
    p90_error = float(np.quantile(final_inliers, .9)) if len(final_inliers) else float("inf")
    overlap_ratio = float((np.mean(final_forward_valid) + np.mean(final_backward_valid)) * .5)
    pair_metrics = []
    if paired_indexes:
        for index, (item, forward, backward) in enumerate(zip(
            paired_indexes, final_forward_parts, final_backward_parts,
        )):
            forward_valid = forward <= evaluation_distance
            backward_valid = backward <= evaluation_distance
            inliers = np.r_[forward[forward_valid], backward[backward_valid]]
            pair_overlap = float((np.mean(forward_valid) + np.mean(backward_valid)) * .5)
            pair_median = float(np.median(inliers)) if len(inliers) else float("inf")
            contour_metrics = _tangent_contour_metrics(
                _transform(item["source"], current), item["targetTangentContour"],
            )
            pair_metrics.append({
                "index": index + 1,
                "overlapRatio": pair_overlap,
                "medianError": pair_median,
                "p90Error": float(np.quantile(inliers, .9)) if len(inliers) else float("inf"),
                "reliable": False,
                **contour_metrics,
            })
    use_contour_quality = bool(partial_overlap and pair_metrics)
    quality_key = "contourError" if use_contour_quality else "medianError"
    comparable_pair_errors = np.asarray([
        item[quality_key] for item in pair_metrics
        if np.isfinite(item[quality_key])
    ])
    pair_error_spread = 0.0
    if len(comparable_pair_errors):
        pair_error_center = float(np.median(comparable_pair_errors))
        pair_error_spread = float(np.median(np.abs(comparable_pair_errors - pair_error_center)))
        consistency_limit = max(
            .05 if use_contour_quality else .06,
            pair_error_spread * (2.5 if use_contour_quality else 4.5),
        )
        for item in pair_metrics:
            item["reliable"] = bool(
                np.isfinite(item[quality_key])
                and abs(item[quality_key] - pair_error_center) <= consistency_limit
            )
    pair_consistency_ratio = float(np.mean([item["reliable"] for item in pair_metrics])) if pair_metrics else 1.0
    converged = converged or (last_step < .003 and overlap_ratio >= .15 and pair_consistency_ratio >= .5)
    ground_angle_deg, ground_height_error = (
        _ground_diagnostics(current, *ground_constraint) if ground_constraint else (None, None)
    )
    result = {
        "matrix": current,
        "initial_rmse": float(initial_rmse or final_rmse),
        "final_rmse": final_rmse,
        "median_error": median_error,
        "p90_error": p90_error,
        "correspondence_count": int(correspondence_count),
        "converged": converged,
        "iteration_count": iteration + 1,
        "overlap_ratio": overlap_ratio,
        "pair_consistency_ratio": pair_consistency_ratio,
        "pair_error_spread": pair_error_spread,
        "frame_pair_metrics": pair_metrics,
        "ground_angle_deg": ground_angle_deg,
        "ground_height_error": ground_height_error,
    }
    if use_contour_quality:
        contour_errors = np.asarray([item["contourError"] for item in pair_metrics])
        contour_p90_errors = np.asarray([item["contourP90Error"] for item in pair_metrics])
        contour_center_errors = np.asarray([item["contourCenterError"] for item in pair_metrics])
        contour_shape_errors = np.asarray([item["contourShapeError"] for item in pair_metrics])
        # Aggregate annotations equally; point count and scan density must not
        # decide the quality score.
        result.update({
            "contour_error": float(np.median(contour_errors)),
            "contour_p90_error": float(np.median(contour_p90_errors)),
            "contour_center_error": float(np.median(contour_center_errors)),
            "contour_shape_error": float(np.median(contour_shape_errors)),
            "contour_error_spread": pair_error_spread,
        })
    return result


def optimize_lidar_pair(
    request: LidarPairRequest,
    progress: Callable[[int, str, dict | None], None] | None = None,
) -> dict:
    # A lidar pair has one canonical solution. Solving swapped UI selections
    # independently creates two slightly different local minima and an
    # artificial graph-cycle error, so always solve the lexical direction and
    # return the exact inverse when the caller asks for the opposite direction.
    if request.source_lidar_id > request.target_lidar_id:
        base = np.asarray(request.base_matrix, dtype=np.float64).reshape(4, 4) if request.base_matrix else None
        swapped = LidarPairRequest(
            group_key=request.group_key,
            source_lidar_id=request.target_lidar_id,
            target_lidar_id=request.source_lidar_id,
            source_cloud_url=request.target_cloud_url,
            target_cloud_url=request.source_cloud_url,
            base_matrix=np.linalg.inv(base).reshape(-1).tolist() if base is not None else None,
            pairs=[LidarObjectPair(
                annotation_id=pair.annotation_id,
                source_cloud_url=pair.target_cloud_url,
                target_cloud_url=pair.source_cloud_url,
                source_indices=pair.target_indices,
                target_indices=pair.source_indices,
            ) for pair in request.pairs],
            frame_pairs=[LidarPreparedFramePair(
                source_cloud_url=pair.target_cloud_url,
                target_cloud_url=pair.source_cloud_url,
                source_timestamp_ns=pair.target_timestamp_ns,
                target_timestamp_ns=pair.source_timestamp_ns,
                delta_ms=pair.delta_ms,
                origins=["target" if origin == "source" else "source" for origin in pair.origins],
            ) for pair in request.frame_pairs],
            stable_only=request.stable_only,
            max_correspondence_distance=request.max_correspondence_distance,
            source_bag_path=request.source_bag_path,
            source_lidar_topic=request.target_lidar_topic,
            target_lidar_topic=request.source_lidar_topic,
            source_selected_timestamps=request.target_selected_timestamps,
            target_selected_timestamps=request.source_selected_timestamps,
        )
        result = optimize_lidar_pair(swapped, progress)
        matrix = np.linalg.inv(np.asarray(result["matrix"], dtype=np.float64).reshape(4, 4))
        base_for_delta = base if base is not None else np.eye(4)
        result.update({
            "sourceLidarId": request.source_lidar_id,
            "targetLidarId": request.target_lidar_id,
            "matrix": matrix.reshape(-1).tolist(),
            "delta": _parameters(matrix @ np.linalg.inv(base_for_delta)).tolist(),
            "pairDiagnostics": [{
                **item,
                "sourcePointCount": item["targetPointCount"],
                "targetPointCount": item["sourcePointCount"],
                "balancedSourcePointCount": item.get("balancedTargetPointCount"),
                "balancedTargetPointCount": item.get("balancedSourcePointCount"),
            } for item in result.get("pairDiagnostics", [])],
            "framePairDiagnostics": [{
                **item,
                "sourceTimestampNs": item["targetTimestampNs"],
                "targetTimestampNs": item["sourceTimestampNs"],
                "sourceCloudUrl": item["targetCloudUrl"],
                "targetCloudUrl": item["sourceCloudUrl"],
                "origins": ["target" if origin == "source" else "source" for origin in item.get("origins", [])],
            } for item in result.get("framePairDiagnostics", [])],
        })
        return result

    source_all, source_stable = _read_cloud(request.source_cloud_url)
    target_all, target_stable = _read_cloud(request.target_cloud_url)
    source_ground_points = source_all[source_stable] if np.count_nonzero(source_stable) >= 100 else source_all
    target_ground_points = target_all[target_stable] if np.count_nonzero(target_stable) >= 100 else target_all
    source_ground = _fit_ground_plane(source_ground_points)
    target_ground = _fit_ground_plane(target_ground_points)
    ground_constraint = (source_ground, target_ground) if source_ground is not None and target_ground is not None else None
    base = np.asarray(request.base_matrix, dtype=np.float64).reshape(4, 4) if request.base_matrix else None
    pair_diagnostics = []
    frame_pair_diagnostics = []
    registration_pairs: list[tuple[np.ndarray, np.ndarray]] | None = None
    if request.pairs:
        source_groups, target_groups = [], []
        per_group_limit = max(250, 3600 // len(request.pairs))
        for pair in request.pairs:
            source, _ = _read_cloud(pair.source_cloud_url)
            target, _ = _read_cloud(pair.target_cloud_url)
            source_indices = np.asarray(pair.source_indices, dtype=np.int64)
            target_indices = np.asarray(pair.target_indices, dtype=np.int64)
            source_indices = source_indices[(source_indices >= 0) & (source_indices < len(source))]
            target_indices = target_indices[(target_indices >= 0) & (target_indices < len(target))]
            source_group = _voxel_downsample(source[source_indices], .06, per_group_limit)
            target_group = _voxel_downsample(target[target_indices], .06, per_group_limit)
            if len(source_group) < MIN_MANUAL_PAIR_POINTS or len(target_group) < MIN_MANUAL_PAIR_POINTS:
                raise ValueError(
                    f"参照物配对 {pair.annotation_id} 的有效点不足："
                    f"左 {len(source_group)} 点，右 {len(target_group)} 点，至少各需 {MIN_MANUAL_PAIR_POINTS} 点"
                )
            source_groups.append(source_group)
            target_groups.append(target_group)
            pair_diagnostics.append({"annotationId": pair.annotation_id, "sourcePointCount": len(source_group), "targetPointCount": len(target_group)})
        # Preserve every group's useful geometry. Equal observation weight is
        # applied to residuals later; shrinking every group to the smallest
        # annotation would let one tiny crop erase thousands of valid points.
        for diagnostic, source_group, target_group in zip(pair_diagnostics, source_groups, target_groups):
            diagnostic["balancedSourcePointCount"] = len(source_group)
            diagnostic["balancedTargetPointCount"] = len(target_group)
        registration_pairs = list(zip(source_groups, target_groups))
        source_points = np.concatenate(source_groups)
        target_points = np.concatenate(target_groups)
        maximum_distance = min(request.max_correspondence_distance, .8)
    elif request.frame_pairs:
        per_frame_limit = max(700, 9000 // len(request.frame_pairs))
        source_groups, target_groups = [], []
        for index, pair in enumerate(request.frame_pairs):
            source, source_stable = _read_cloud(pair.source_cloud_url)
            target, target_stable = _read_cloud(pair.target_cloud_url)
            if request.stable_only and np.any(source_stable):
                source = source[source_stable]
            if request.stable_only and np.any(target_stable):
                target = target[target_stable]
            source_group = _voxel_downsample(_registration_region(source), .12, per_frame_limit)
            target_group = _voxel_downsample(_registration_region(target), .12, per_frame_limit)
            if len(source_group) < 20 or len(target_group) < 20:
                raise ValueError(f"帧对 {index + 1} 的有效静止点不足")
            source_groups.append(source_group)
            target_groups.append(target_group)
            frame_pair_diagnostics.append({
                "index": index + 1,
                "sourceTimestampNs": str(pair.source_timestamp_ns),
                "targetTimestampNs": str(pair.target_timestamp_ns),
                "deltaMs": pair.delta_ms,
                "origins": pair.origins,
                "sourceCloudUrl": pair.source_cloud_url,
                "targetCloudUrl": pair.target_cloud_url,
            })
        registration_pairs = list(zip(source_groups, target_groups))
        source_points = np.concatenate(source_groups)
        target_points = np.concatenate(target_groups)
        maximum_distance = request.max_correspondence_distance
    elif (request.source_bag_path and request.source_lidar_topic and request.target_lidar_topic and
          request.source_selected_timestamps and request.target_selected_timestamps):
        from .preparation import nearest_lidar_frame_pairs
        frame_pairs = nearest_lidar_frame_pairs(
            request.source_bag_path,
            request.source_lidar_topic,
            request.target_lidar_topic,
            request.source_selected_timestamps,
            request.target_selected_timestamps,
        )
        per_frame_limit = max(700, 9000 // len(frame_pairs))
        source_groups, target_groups = [], []
        for index, pair in enumerate(frame_pairs):
            source, source_stable = _read_cloud_path(pair["sourcePath"])
            target, target_stable = _read_cloud_path(pair["targetPath"])
            if request.stable_only and np.any(source_stable):
                source = source[source_stable]
            if request.stable_only and np.any(target_stable):
                target = target[target_stable]
            source_group = _voxel_downsample(_registration_region(source), .12, per_frame_limit)
            target_group = _voxel_downsample(_registration_region(target), .12, per_frame_limit)
            if len(source_group) < 20 or len(target_group) < 20:
                raise ValueError(f"帧对 {index + 1} 的有效静止点不足")
            source_groups.append(source_group)
            target_groups.append(target_group)
            frame_pair_diagnostics.append({
                "index": index + 1,
                "sourceTimestampNs": str(pair["sourceTimestampNs"]),
                "targetTimestampNs": str(pair["targetTimestampNs"]),
                "deltaMs": pair["deltaMs"],
                "origins": pair["origins"],
                "sourceCloudUrl": _cloud_url(pair["sourcePath"]),
                "targetCloudUrl": _cloud_url(pair["targetPath"]),
            })
        # Preserve pair boundaries. Flattening both sides independently would
        # allow a point from one timestamp to match an unrelated later frame.
        registration_pairs = list(zip(source_groups, target_groups))
        source_points = np.concatenate(source_groups)
        target_points = np.concatenate(target_groups)
        maximum_distance = request.max_correspondence_distance
    else:
        source_points = source_all[source_stable] if request.stable_only and np.any(source_stable) else source_all
        target_points = target_all[target_stable] if request.stable_only and np.any(target_stable) else target_all
        source_points = _voxel_downsample(source_points, .12)
        target_points = _voxel_downsample(target_points, .12)
        maximum_distance = request.max_correspondence_distance
    if progress:
        pairing = f" · {len(frame_pair_diagnostics)} 个去重帧对" if frame_pair_diagnostics else ""
        progress(8, f"已准备 {len(source_points)} × {len(target_points)} 个配准点{pairing}", {
            "framePairCount": len(frame_pair_diagnostics),
        })
    global_diagnostics = {}
    if request.pairs:
        # Manual object refinement retains the partial-overlap correspondence
        # logic. Different lidars need not observe the same object surfaces.
        result = _gicp(
            source_points, target_points, base, maximum_distance, progress,
            paired_clouds=registration_pairs,
            ground_constraint=ground_constraint,
            partial_overlap=True,
        )
    else:
        # ICP/GICP are local optimizers. With no supplied transform, obtain a
        # real global seed from standard FPFH feature matches and RANSAC first.
        # When a transform is supplied, refine that exact transform directly;
        # never mix it with global candidates and accidentally discard it.
        initial = base
        if initial is None:
            try:
                initial, global_diagnostics = _global_fpfh_ransac(source_points, target_points, progress)
            except ValueError as fpfh_error:
                # A single RANSAC hypothesis is brittle for side lidars: their
                # overlap is real, but repetitive road-side structure can give
                # FPFH a high-scoring transform several metres away. Continue
                # with an extrinsic-free search that aligns the two fitted road
                # planes, enumerates yaw/translation hypotheses, and ranks them
                # by symmetric overlap before local refinement.
                if progress:
                    progress(24, f"{fpfh_error}；正在尝试地面约束多候选全局搜索", None)
                fallback = _gicp(
                    source_points,
                    target_points,
                    None,
                    maximum_distance,
                    progress,
                    paired_clouds=registration_pairs,
                    ground_constraint=None,
                )
                initial = fallback["matrix"]
                global_diagnostics = {
                    "globalMethod": "ground-yaw-translation-multiseed",
                    "fpfhFailure": str(fpfh_error),
                    "fallbackOverlapRatio": fallback["overlap_ratio"],
                    "fallbackMedianError": fallback["median_error"],
                }
        result = _small_gicp_registration(
            source_points,
            target_points,
            initial,
            maximum_distance,
            progress,
            paired_clouds=registration_pairs,
        )
    if request.pairs and ground_constraint is not None:
        result["ground_angle_deg"], result["ground_height_error"] = _ground_diagnostics(
            result["matrix"], *ground_constraint,
        )
    matrix = result.pop("matrix")
    translation = float(np.linalg.norm(matrix[:3, 3]))
    if request.pairs:
        failure = None
        if translation > 6.0:
            failure = f"平移 {translation:.2f} m 超出车载雷达合理范围"
        elif result.get("ground_angle_deg") is not None and result["ground_angle_deg"] > 5.0:
            failure = f"地面夹角 {result['ground_angle_deg']:.2f}°，疑似点云翻转"
        elif result.get("ground_height_error") is not None and result["ground_height_error"] > .25:
            failure = f"地面高度误差 {result['ground_height_error']:.2f} m 超出合理范围"
        if failure:
            raise ValueError(f"人工配准结果无效：{failure}；请检查所选参照物或改用其他初值")
    else:
        failure = None
        if translation > 6.0:
            failure = f"平移 {translation:.2f} m 超出车载雷达合理范围"
        elif translation < .15:
            failure = f"平移 {translation:.2f} m 过小，疑似落入重复结构假解"
        elif result["overlap_ratio"] < .08:
            failure = f"有效重叠仅 {result['overlap_ratio'] * 100:.0f}%"
        elif len(frame_pair_diagnostics) >= 3 and result["pair_consistency_ratio"] < .35:
            failure = f"仅 {result['pair_consistency_ratio'] * 100:.0f}% 帧对支持同一外参"
        if failure:
            raise ValueError(f"全局配准初值无效：{failure}；请换一组重叠更明显的静止帧")
    base_for_delta = base if base is not None else np.eye(4)
    delta = _parameters(matrix @ np.linalg.inv(base_for_delta))
    if request.pairs and "contour_error" in result:
        # Manual calibration quality is based on density-independent local
        # outlines. Nearest-point overlap remains available only as a secondary
        # diagnostic for debugging the selected crops.
        quality = (
            "good" if result["contour_error"] <= .12 and result["pair_consistency_ratio"] >= .75
            else "warning" if result["contour_error"] <= .25 and result["pair_consistency_ratio"] >= .5
            else "poor"
        )
    else:
        quality = (
            "good" if result["overlap_ratio"] >= .35 and result["pair_consistency_ratio"] >= .75
            else "warning" if result["overlap_ratio"] >= .12 and result["pair_consistency_ratio"] >= .5
            else "poor"
        )
    return {
        "groupKey": request.group_key,
        "sourceLidarId": request.source_lidar_id,
        "targetLidarId": request.target_lidar_id,
        "transform": "target_from_source",
        "matrix": matrix.reshape(-1).tolist(),
        "delta": delta.tolist(),
        "pairCount": len(request.pairs),
        "pairDiagnostics": pair_diagnostics,
        "framePairCount": len(frame_pair_diagnostics),
        "framePairDiagnostics": frame_pair_diagnostics,
        "algorithmVersion": ALGORITHM_VERSION,
        "qualityStatus": quality,
        **global_diagnostics,
        **result,
    }


def solve_lidar_graph(request: LidarGraphRequest) -> dict:
    lidar_ids = list(dict.fromkeys(request.lidar_ids))
    if request.anchor_lidar_id not in lidar_ids:
        raise ValueError("基准雷达不在节点列表中")
    variable_ids = [item for item in lidar_ids if item != request.anchor_lidar_id]
    variable_index = {item: index for index, item in enumerate(variable_ids)}
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {item: [] for item in lidar_ids}
    for edge in request.edges:
        measurement = np.asarray(edge.matrix, dtype=np.float64).reshape(4, 4)
        adjacency.setdefault(edge.source_lidar_id, []).append((edge.target_lidar_id, measurement))
        adjacency.setdefault(edge.target_lidar_id, []).append((edge.source_lidar_id, np.linalg.inv(measurement)))
    poses = {request.anchor_lidar_id: np.eye(4)}
    queue = [request.anchor_lidar_id]
    while queue:
        source_id = queue.pop(0)
        for target_id, target_from_source in adjacency.get(source_id, []):
            if target_id in poses:
                continue
            poses[target_id] = poses[source_id] @ np.linalg.inv(target_from_source)
            queue.append(target_id)
    disconnected = [item for item in lidar_ids if item not in poses]
    if disconnected:
        raise ValueError("外参图不连通：" + "、".join(disconnected))
    initial = np.concatenate([_parameters(poses[item]) for item in variable_ids]) if variable_ids else np.empty(0)

    def pose(parameters: np.ndarray, lidar_id: str) -> np.ndarray:
        if lidar_id == request.anchor_lidar_id:
            return np.eye(4)
        offset = variable_index[lidar_id] * 6
        return _matrix(parameters[offset:offset + 6])

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = []
        for edge in request.edges:
            measurement = np.asarray(edge.matrix, dtype=np.float64).reshape(4, 4)
            predicted = np.linalg.inv(pose(parameters, edge.target_lidar_id)) @ pose(parameters, edge.source_lidar_id)
            error = np.linalg.inv(measurement) @ predicted
            value = _parameters(error)
            value[:3] *= 2.0
            values.append(value * math.sqrt(edge.weight))
        return np.concatenate(values)

    optimized = least_squares(residual, initial, loss="huber", f_scale=.08, max_nfev=200) if len(initial) else None
    parameters = optimized.x if optimized is not None else initial
    solved = {item: pose(parameters, item).reshape(-1).tolist() for item in lidar_ids}
    edge_diagnostics = []
    for edge in request.edges:
        measurement = np.asarray(edge.matrix, dtype=np.float64).reshape(4, 4)
        predicted = np.linalg.inv(pose(parameters, edge.target_lidar_id)) @ pose(parameters, edge.source_lidar_id)
        value = _parameters(np.linalg.inv(measurement) @ predicted)
        edge_diagnostics.append({
            "edgeId": edge.edge_id,
            "rotationResidualDeg": float(np.linalg.norm(value[:3]) * 180 / math.pi),
            "translationResidualM": float(np.linalg.norm(value[3:])),
            "suspected": bool(np.linalg.norm(value[:3]) > math.radians(1.0) or np.linalg.norm(value[3:]) > .08),
        })
    return {
        "anchorLidarId": request.anchor_lidar_id,
        "poses": solved,
        "edgeDiagnostics": edge_diagnostics,
        "converged": bool(optimized is None or optimized.success),
        "algorithmVersion": "se3-pose-graph-v1",
    }
