from __future__ import annotations

import math
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import distance_transform_edt, minimum_filter
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .nuscenes_fusion import is_virtual_cloud, read_cloud_bytes
from .groups import PREPARED_ROOT
from .waymo_camera_model import prepare_projection as prepare_waymo_projection
from .waymo_camera_model import project as project_waymo_camera


PUBLIC_ROOT = Path(os.environ.get("AUTOCALIB_PUBLIC_ROOT", "/workspace/web/public")).resolve()
MAX_OPTIMIZER_WORKERS = min(
    max(1, os.cpu_count() or 1),
    max(1, int(os.environ.get("AUTOCALIB_OPTIMIZER_CPUS", "12"))),
)
_PAIR_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_OPTIMIZER_WORKERS, thread_name_prefix="calibration-pair")
_PREPARATION_EXECUTOR = ThreadPoolExecutor(
    max_workers=MAX_OPTIMIZER_WORKERS,
    thread_name_prefix="calibration-prepare",
)
_PREPARATION_BATCH_TTL_SECONDS = 300
_PREPARATION_BATCH_LOCK = threading.RLock()
_PREPARATION_BATCHES: dict[str, dict] = {}


def _evaluate_pair_costs(evaluate: Callable, pairs: list[dict]) -> list:
    # Small annotation sets cost less to evaluate than to dispatch thousands
    # of per-pair futures. Retain threaded evaluation for larger workloads.
    if len(pairs) <= 8 and sum(len(pair["points"]) for pair in pairs) <= 8192:
        return list(map(evaluate, pairs))
    return list(_PAIR_EXECUTOR.map(evaluate, pairs))


CALIBRATION_ALGORITHM_VERSION = "truth-audited-adaptive-projected-contour-v33-strict-descent"
SILHOUETTE_METRIC = "visible-point-convex-envelope-iou-v1"
FORWARD = np.asarray([
    [0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1],
], dtype=np.float64)


class CalibrationPair(BaseModel):
    annotation_id: str
    cloud_url: str
    point_indices: list[int]
    mask_width: int = Field(gt=0)
    mask_height: int = Field(gt=0)
    mask_rle: list[int]
    intrinsic: list[float] = Field(min_length=9, max_length=9)
    distortion: list[float]
    distortion_model: str = "rational"
    temporal_projection: dict | None = None


class CalibrationRequest(BaseModel):
    group_key: str
    initialization_mode: str
    base_matrix: list[float] = Field(min_length=16, max_length=16)
    previous_matrix: list[float] | None = None
    pairs: list[CalibrationPair] = Field(min_length=1)
    optimization_batch_id: str | None = Field(default=None, max_length=120)
    optimization_batch_size: int = Field(default=1, ge=1, le=3)


def _resolve_cloud(url: str) -> Path:
    relative = url.split("?", 1)[0].lstrip("/")
    roots_and_paths: list[tuple[Path, str]] = []
    if relative.startswith("data/cache/"):
        roots_and_paths.append((PREPARED_ROOT, relative.removeprefix("data/cache/")))
    roots_and_paths.append((PUBLIC_ROOT, relative))
    for root, candidate in roots_and_paths:
        path = (root / candidate).resolve()
        if root in path.parents and (path.is_file() or is_virtual_cloud(path)):
            return path
    raise ValueError(f"点云文件不可用：{url}")


def _read_cloud(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = read_cloud_bytes(path)
    if raw[:4] != b"ACP1":
        raise ValueError(f"无法识别点云：{path.name}")
    count = struct.unpack_from("<I", raw, 4)[0]
    if len(raw) != 16 + count * 16:
        raise ValueError(f"点云不完整：{path.name}")
    records = np.ndarray((count,), dtype=np.dtype({
        "names": ["x", "y", "z", "stable", "scanline"],
        "formats": ["<f4", "<f4", "<f4", "u1", "u1"],
        "offsets": [0, 4, 8, 12, 15], "itemsize": 16,
    }), buffer=raw, offset=16)
    points = np.column_stack((records["x"], records["y"], records["z"])).astype(np.float64)
    # Preserve absolute point indices, but exclude marked self returns from fits.
    points[records["stable"] == 2] = np.nan
    return points, np.asarray(records["scanline"], dtype=np.uint8)


def _decode_rle(width: int, height: int, runs: list[int]) -> np.ndarray:
    flat = np.zeros(width * height, dtype=np.uint8)
    cursor, value = 0, 0
    for run in runs:
        end = min(flat.size, cursor + max(0, int(run)))
        if value:
            flat[cursor:end] = 1
        cursor, value = end, 1 - value
    return flat.reshape(height, width)


def _scanline_boundary_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Return an ordered, pose-independent boundary from parallel lidar scan lines.

    This geometric fallback is retained until preprocessing supplies real ring
    identifiers. It must not be interpreted as a complete observed silhouette.
    """
    if len(points) < 6:
        return points, 1
    origin = np.median(points, axis=0)
    centered = points - origin
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    vertical_index = int(np.argmax(np.abs(axes[:, 2])))
    remaining = [index for index in range(3) if index != vertical_index]
    horizontal_index = max(remaining, key=lambda index: np.ptp(centered @ axes[index]))
    coordinates = np.column_stack((
        centered @ axes[horizontal_index], centered @ axes[vertical_index],
    ))

    tree = cKDTree(coordinates)
    _, neighbors = tree.query(coordinates, k=2)
    local_vectors = coordinates[neighbors[:, 1]] - coordinates
    lengths = np.linalg.norm(local_vectors, axis=1)
    valid = lengths > 1e-9
    if np.count_nonzero(valid) < 3:
        return points, 1
    angles = np.arctan2(local_vectors[valid, 1], local_vectors[valid, 0])
    direction_angle = .5 * np.arctan2(
        np.mean(np.sin(2 * angles)), np.mean(np.cos(2 * angles)),
    )
    direction = np.asarray([np.cos(direction_angle), np.sin(direction_angle)])
    normal = np.asarray([-direction[1], direction[0]])
    along = coordinates @ direction
    across = coordinates @ normal
    order = np.argsort(across)
    gaps = np.diff(across[order])
    local_spacing = float(np.quantile(lengths[valid], .7))
    split_gap = max(local_spacing * 1.8, float(np.ptp(across)) * .002)
    raw_groups = np.split(order, np.flatnonzero(gaps > split_gap) + 1)
    minimum_line_points = max(3, int(np.ceil(len(points) * .008)))
    groups = [group for group in raw_groups if len(group) >= minimum_line_points]
    if len(groups) < 2:
        line_order = order[np.argsort(along[order])]
        return points[line_order], 1
    groups.sort(key=lambda group: float(np.median(across[group])))

    ordered_indices: list[int] = []
    top = groups[0][np.argsort(along[groups[0]])]
    bottom = groups[-1][np.argsort(along[groups[-1]])][::-1]
    ordered_indices.extend(int(index) for index in top)
    for group in groups[1:-1]:
        line_order = group[np.argsort(along[group])]
        high_position = int(np.ceil((len(line_order) - 1) * .98))
        ordered_indices.append(int(line_order[high_position]))
    ordered_indices.extend(int(index) for index in bottom)
    for group in reversed(groups[1:-1]):
        line_order = group[np.argsort(along[group])]
        low_position = int(np.floor((len(line_order) - 1) * .02))
        ordered_indices.append(int(line_order[low_position]))
    return points[np.asarray(ordered_indices, dtype=np.int64)], len(groups)


def _native_scanline_groups(
    points: np.ndarray,
    scanline_ids: np.ndarray,
) -> list[np.ndarray]:
    """Return native scanlines ordered by elevation and then by azimuth."""
    groups: list[tuple[float, np.ndarray]] = []
    for scanline_id in np.unique(scanline_ids[scanline_ids > 0]):
        indices = np.flatnonzero(scanline_ids == scanline_id)
        if len(indices) < 2:
            continue
        group = points[indices]
        azimuth = np.arctan2(group[:, 1], group[:, 0])
        azimuth_order = np.argsort(azimuth)
        ordered_azimuth = azimuth[azimuth_order]
        cyclic_gaps = np.diff(np.concatenate((ordered_azimuth, ordered_azimuth[:1] + 2 * math.pi)))
        # Start immediately after the largest empty sector. This preserves a
        # short object interval that crosses -pi/pi without relying on input
        # point order or densifying the scanline.
        start = (int(np.argmax(cyclic_gaps)) + 1) % len(azimuth_order)
        ordered_indices = indices[np.roll(azimuth_order, -start)]
        elevation = np.arctan2(
            group[:, 2], np.hypot(group[:, 0], group[:, 1]),
        )
        groups.append((float(np.median(elevation)), ordered_indices))
    groups.sort(key=lambda item: item[0])
    return [indices for _, indices in groups]


def _native_scanline_endpoint_points(
    points: np.ndarray,
    scanline_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return the two observed azimuth endpoints on every native scanline."""
    groups = _native_scanline_groups(points, scanline_ids)
    endpoints = [index for group in groups for index in (int(group[0]), int(group[-1]))]
    if len(endpoints) < 2:
        return np.empty((0, 3), dtype=np.float64), 0
    return points[np.asarray(endpoints, dtype=np.int64)], len(groups)


def _native_scanline_boundary_indices(
    points: np.ndarray,
    scanline_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Extract the actually observed sparse outline using native lidar rings.

    The lowest and highest observed rings constrain the vertical outline, so
    all of their selected samples are retained. Interior rings contribute only
    their two azimuth endpoints. No points are interpolated and the result is
    deliberately not interpreted as a closed camera-visible silhouette.
    """
    groups = _native_scanline_groups(points, scanline_ids)
    if not groups:
        return np.empty(0, dtype=np.int64), 0
    if len(groups) == 1:
        boundary_indices = groups[0]
    else:
        boundary_indices = np.concatenate((
            groups[0],
            *(
                np.asarray([group[0], group[-1]], dtype=np.int64)
                for group in groups[1:-1]
            ),
            groups[-1],
        ))
    return np.asarray(boundary_indices, dtype=np.int64), len(groups)


def _native_scanline_boundary_points(
    points: np.ndarray,
    scanline_ids: np.ndarray,
) -> tuple[np.ndarray, int]:
    boundary_indices, scanline_count = _native_scanline_boundary_indices(
        points, scanline_ids,
    )
    return points[boundary_indices], scanline_count


def _original_mask_geometry(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Keep the annotation unchanged and extract only its one-pixel outer edge."""
    source = mask.astype(np.uint8)
    contours, _ = cv2.findContours(source, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge = np.zeros_like(source)
    cv2.drawContours(edge, contours, -1, 1, 1)
    edge_points = np.column_stack(np.nonzero(edge))[:, ::-1].astype(np.float64)
    return source.copy(), edge.astype(bool), edge_points, 0


def _delta_matrix(delta: np.ndarray) -> np.ndarray:
    rx, ry, rz, tx, ty, tz = delta
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    return np.asarray([
        [cz*cy, cz*sy*sx-sz*cx, cz*sy*cx+sz*sx, tx],
        [sz*cy, sz*sy*sx+cz*cx, sz*sy*cx-cz*sx, ty],
        [-sy, cy*sx, cy*cx, tz], [0, 0, 0, 1],
    ], dtype=np.float64)


def _matrix_delta(matrix: np.ndarray, base: np.ndarray) -> np.ndarray:
    relative = matrix @ np.linalg.inv(base)
    ry = math.asin(float(np.clip(-relative[2, 0], -1, 1)))
    cy = math.cos(ry)
    rx = math.atan2(relative[2, 1], relative[2, 2]) if abs(cy) > 1e-6 else 0.0
    rz = math.atan2(relative[1, 0], relative[0, 0]) if abs(cy) > 1e-6 else math.atan2(-relative[0, 1], relative[1, 1])
    return np.asarray([rx, ry, rz, relative[0, 3], relative[1, 3], relative[2, 3]])


def _aggregate_pair_costs(costs: list[tuple[float, float]], scale: float = 24.0) -> float:
    """Equal-prior, continuously redescending aggregation across annotations."""
    if not costs:
        return scale
    values = np.asarray([max(0.0, cost) for cost, _ in costs], dtype=np.float64)
    weights = np.asarray([max(0.0, weight) for _, weight in costs], dtype=np.float64)
    transformed = scale * np.log1p(values / scale)
    return float(np.sum(transformed * weights) / max(np.sum(weights), 1e-9))


def _consensus_weight(terms: dict) -> float:
    """Continuously reduce only pairs that fail three independent gross-consistency checks."""
    center_excess = max(0.0, (terms["normalized_center_error"] - .45) / .15)
    inside_excess = max(0.0, (.25 - terms["inside_ratio"]) / .15)
    overlap_excess = max(0.0, (.20 - terms["iou"]) / .15)
    severity = center_excess * inside_excess * overlap_excess
    return 1.0 / (1.0 + 4.0 * severity * severity)


def _prepare_pair(pair: CalibrationPair) -> dict:
    cloud, cloud_scanlines = _read_cloud(_resolve_cloud(pair.cloud_url))
    indices = np.asarray(pair.point_indices, dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(cloud))]
    indices = indices[np.isfinite(cloud[indices]).all(axis=1)]
    if len(indices) == 0:
        raise ValueError(f"配对 {pair.annotation_id} 没有有效点")
    points = cloud[indices]
    point_scanlines = cloud_scanlines[indices]
    selected = np.zeros(len(cloud), dtype=bool)
    selected[indices] = True
    background = cloud[(~selected) & np.isfinite(cloud).all(axis=1)]
    mask = _decode_rle(pair.mask_width, pair.mask_height, pair.mask_rle)
    native_boundary_local_indices, native_scanline_count = _native_scanline_boundary_indices(
        points, point_scanlines,
    )
    native_boundary_points = points[native_boundary_local_indices]
    # Preserve only actually observed boundary samples. Sparse lidar does not
    # provide a complete camera-visible silhouette, so these samples must never
    # be densified into a closed contour or reverse-matched to the whole Mask.
    if native_scanline_count >= 2:
        boundary_points = native_boundary_points
        # The v18 solve remains exactly equal-weighted. Projection-derived
        # reliability is estimated only after that solve has produced a good
        # local pose, then frozen for a separate refinement stage.
        boundary_weights = np.ones(len(boundary_points), dtype=np.float64)
        scanline_count = native_scanline_count
        boundary_mode = "native-scanline-boundary"
    else:
        boundary_points, scanline_count = _scanline_boundary_points(points)
        boundary_weights = np.ones(len(boundary_points), dtype=np.float64)
        boundary_mode = "observed-sparse-boundary"
    closed_mask, closed_edge, edge_points, closing_radius = _original_mask_geometry(mask)
    if len(edge_points) < 8:
        raise ValueError(f"配对 {pair.annotation_id} 的图像 Mask 轮廓不足")
    distance = distance_transform_edt(~closed_edge)
    outside_distance = distance_transform_edt(~mask.astype(bool))
    small_width = max(1, math.ceil(pair.mask_width / 4))
    small_height = max(1, math.ceil(pair.mask_height / 4))
    mask_small = cv2.resize(mask, (small_width, small_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    step = max(1, math.ceil(len(points) / 3200))
    background_step = max(1, math.ceil(len(background) / 1800))
    center, center_method, plane_residual = _geometric_center(points)
    pnp_center, pnp_center_method, pnp_center_removed_count = _pnp_object_center(points)
    mask_supports = _directional_supports(edge_points)
    mask_scale = float(np.linalg.norm(np.ptp(edge_points, axis=0)))
    temporal_projection = (
        prepare_waymo_projection(
            pair.temporal_projection,
            np.asarray(pair.intrinsic, dtype=np.float64).reshape(3, 3),
            np.asarray(pair.distortion, dtype=np.float64),
        )
        if pair.temporal_projection else None
    )
    return {
        "id": pair.annotation_id, "points": points[::step], "center": pnp_center.copy(), "mask": mask,
        "boundary_points": boundary_points, "boundary_weights": boundary_weights,
        "scanline_count": scanline_count,
        "boundary_mode": boundary_mode,
        "native_boundary_points": native_boundary_points,
        "native_scanline_count": native_scanline_count,
        "native_boundary_local_indices": native_boundary_local_indices,
        "selected_points_full": points,
        "selected_scanline_ids": point_scanlines,
        "closed_mask": closed_mask,
        "closed_edge_points": edge_points[::max(1, math.ceil(len(edge_points) / 640))],
        "mask_closing_radius": closing_radius,
        "center_method": pnp_center_method, "plane_residual": plane_residual,
        "pnp_center": pnp_center, "pnp_center_method": pnp_center_method,
        "pnp_center_removed_count": pnp_center_removed_count,
        "background": background[::background_step], "outside_distance": outside_distance,
        "mask_small": mask_small, "mask_centroid": np.asarray(_mask_centroid(closed_mask)),
        "mask_bounds": np.quantile(edge_points, [.02, .98], axis=0),
        "mask_supports": mask_supports,
        "mask_scale": max(mask_scale, 1.0),
        "distance": distance, "k": np.asarray(pair.intrinsic).reshape(3, 3),
        "d": np.pad(np.asarray(pair.distortion, dtype=np.float64), (0, max(0, 8-len(pair.distortion))))[:8],
        "distortion_model": pair.distortion_model,
        "distortion_parameter_count": len(pair.distortion),
        "temporal_projection": temporal_projection,
        "range": float(np.median(np.linalg.norm(points, axis=1))),
    }


def _apply_pair_calibration(prepared: dict, pair: CalibrationPair) -> dict:
    """Create a request-private dict while sharing immutable prepared arrays."""
    result = prepared.copy()
    result.update({
        "id": pair.annotation_id,
        "k": np.asarray(pair.intrinsic).reshape(3, 3),
        "d": np.pad(
            np.asarray(pair.distortion, dtype=np.float64),
            (0, max(0, 8 - len(pair.distortion))),
        )[:8],
        "distortion_model": pair.distortion_model,
        "distortion_parameter_count": len(pair.distortion),
        "temporal_projection": (
            prepare_waymo_projection(
                pair.temporal_projection,
                np.asarray(pair.intrinsic, dtype=np.float64).reshape(3, 3),
                np.asarray(pair.distortion, dtype=np.float64),
            )
            if pair.temporal_projection else None
        ),
    })
    return result


def _expire_preparation_batch(batch_id: str) -> None:
    with _PREPARATION_BATCH_LOCK:
        _PREPARATION_BATCHES.pop(batch_id, None)


def _register_preparation_batch(batch_id: str | None, expected_consumers: int) -> None:
    if not batch_id:
        return
    with _PREPARATION_BATCH_LOCK:
        if batch_id in _PREPARATION_BATCHES:
            return
        timer = threading.Timer(
            _PREPARATION_BATCH_TTL_SECONDS,
            _expire_preparation_batch,
            args=(batch_id,),
        )
        timer.daemon = True
        _PREPARATION_BATCHES[batch_id] = {
            "futures": {},
            "remaining": expected_consumers,
            "timer": timer,
        }
        timer.start()


def release_preparation_batch(batch_id: str | None) -> None:
    """Release shared geometry after every model in a frontend batch finishes."""
    if not batch_id:
        return
    with _PREPARATION_BATCH_LOCK:
        state = _PREPARATION_BATCHES.get(batch_id)
        if state is None:
            return
        state["remaining"] -= 1
        if state["remaining"] <= 0:
            _PREPARATION_BATCHES.pop(batch_id, None)
            state["timer"].cancel()


def _prepare_pair_for_batch(pair: CalibrationPair, batch_id: str | None) -> dict:
    if not batch_id:
        return _prepare_pair(pair)
    with _PREPARATION_BATCH_LOCK:
        state = _PREPARATION_BATCHES.get(batch_id)
        if state is None:
            return _prepare_pair(pair)
        future = state["futures"].get(pair.annotation_id)
        if future is None:
            # Geometry producers use a separate pool. Consumers may wait in
            # _PAIR_EXECUTOR without occupying every worker needed to produce
            # the Future they are waiting for.
            future = _PREPARATION_EXECUTOR.submit(_prepare_pair, pair)
            state["futures"][pair.annotation_id] = future
    return _apply_pair_calibration(future.result(), pair)


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_area = float(np.sum(cross) / 2.0)
    if abs(signed_area) < 1e-12:
        return np.mean(points, axis=0)
    return np.asarray([
        np.sum((x + np.roll(x, -1)) * cross),
        np.sum((y + np.roll(y, -1)) * cross),
    ]) / (6.0 * signed_area)


def _geometric_center(points: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Use a planar boundary centroid when observable, otherwise a robust sample center."""
    fallback = np.median(points, axis=0)
    if len(points) < 12:
        return fallback, "median", math.inf
    centered = points - fallback
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    plane_residual = float(np.median(np.abs(centered @ axes[-1])))
    in_plane_scale = float(max(singular_values[-2] / math.sqrt(len(points)), 1e-9))
    planar = (
        singular_values[-1] <= singular_values[-2] * .12
        and plane_residual <= max(.03, in_plane_scale * .04)
    )
    if not planar:
        return fallback, "median", plane_residual
    coordinates = centered @ axes[:2].T
    bounds = np.quantile(coordinates, [.01, .99], axis=0)
    trusted = coordinates[np.all((coordinates >= bounds[0]) & (coordinates <= bounds[1]), axis=1)]
    if len(trusted) < 3:
        return fallback, "median", plane_residual
    try:
        hull = ConvexHull(trusted)
    except QhullError:
        return fallback, "median", plane_residual
    centroid = _polygon_centroid(trusted[hull.vertices])
    return fallback + centroid @ axes[:2], "planar-boundary-centroid", plane_residual


def _remove_remote_micro_components(points: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove only tiny point components that are unmistakably remote.

    A nearest-neighbour singleton test misses two or three mutually adjacent
    bad returns: each bad point makes the others look locally well supported.
    Build a capped k-NN connectivity graph instead, find its dominant object
    component, and discard a small component only when it is also far beyond
    the dominant component's spatial extent.  The capped graph keeps this
    linear in the number of selected points after the k-D tree query.
    """
    count = len(points)
    if count < 8:
        return points, 0
    tree = cKDTree(points)
    distances, neighbors = tree.query(points, k=min(9, count))
    nearest = distances[:, 1]
    positive = nearest[nearest > 1e-9]
    typical_spacing = float(np.median(positive)) if len(positive) else 0.0
    if typical_spacing <= 0.0:
        return points, 0

    connection_radius = typical_spacing * 10.0
    neighbor_count = distances.shape[1] - 1
    sources = np.repeat(np.arange(count, dtype=np.int64), neighbor_count)
    targets = neighbors[:, 1:].reshape(-1).astype(np.int64, copy=False)
    connected = distances[:, 1:].reshape(-1) <= connection_radius
    sources, targets = sources[connected], targets[connected]
    graph = coo_matrix((
        np.ones(len(sources) * 2, dtype=np.uint8),
        (np.concatenate((sources, targets)), np.concatenate((targets, sources))),
    ), shape=(count, count))
    _, labels = connected_components(graph, directed=False)
    unique_labels, sizes = np.unique(labels, return_counts=True)
    dominant_label = int(unique_labels[int(np.argmax(sizes))])
    dominant = labels == dominant_label
    # If the selected surface itself is strongly fragmented, topology alone
    # cannot distinguish a real sparse part from an erroneous component.
    if int(np.count_nonzero(dominant)) < max(8, int(math.ceil(count * .70))):
        return points, 0

    maximum_removal = max(1, int(math.floor(count * .05)))
    dominant_points = points[dominant]
    dominant_extent = float(np.linalg.norm(np.ptp(dominant_points, axis=0)))
    minimum_separation = max(typical_spacing * 25.0, dominant_extent * .50, 1e-6)
    dominant_tree = cKDTree(dominant_points)
    remove = np.zeros(count, dtype=bool)
    for label, size in zip(unique_labels, sizes):
        if int(label) == dominant_label or int(size) > maximum_removal:
            continue
        indices = np.flatnonzero(labels == label)
        separation = float(np.min(dominant_tree.query(points[indices], k=1)[0]))
        if separation > minimum_separation:
            remove[indices] = True
    removed_count = int(np.count_nonzero(remove))
    if removed_count > maximum_removal or count - removed_count < 8:
        return points, 0
    return points[~remove], removed_count


def _pca_object_center(points: np.ndarray) -> tuple[np.ndarray, str, int]:
    """Return a density-independent 3D representative used only by PnP.

    The downstream contour objective intentionally keeps ``_geometric_center``:
    changing both the initialization and refinement correspondence at once
    would conflate two independent algorithm changes.  PCA supplies a
    rotation-equivariant oriented box, while the quantile midpoint is a stable
    fallback for too few or nearly collinear samples.
    """
    quantile_bounds = np.quantile(points, [.10, .90], axis=0)
    fallback = np.mean(quantile_bounds, axis=0)
    if len(points) < 8:
        return fallback, "quantile-envelope-sparse", 0

    # Ordinary lidar boundary samples can be several times farther apart than
    # the median, especially at long range.  Remove only tiny components that
    # are both disconnected from and unmistakably remote from the main object.
    trusted, removed_count = _remove_remote_micro_components(points)

    origin = np.mean(trusted, axis=0)
    centered = trusted - origin
    _, singular_values, axes = np.linalg.svd(centered, full_matrices=False)
    if singular_values[0] <= 1e-9 or singular_values[1] < singular_values[0] * .08:
        return fallback, "quantile-envelope-degenerate", removed_count
    coordinates = centered @ axes.T
    local_center = np.mean(np.asarray([coordinates.min(axis=0), coordinates.max(axis=0)]), axis=0)
    return origin + local_center @ axes, "pca-oriented-box", removed_count


def _xy_boundary_box_center(xy: np.ndarray) -> np.ndarray:
    """Full-enclosure XY box minimizing robust nearest-edge distance.

    A deterministic 0.25-degree grid over rectangle orientations is followed
    by bounded scalar refinement. The 5 cm pseudo-Huber scale is fixed, not
    selected using calibration truth. All supplied points define the extent.
    """
    origin = xy.mean(axis=0)
    q = xy - origin

    def fit(angle: float) -> tuple[float, np.ndarray]:
        cosine, sine = np.cos(angle), np.sin(angle)
        axes = np.asarray([[cosine, sine], [-sine, cosine]])
        local = q @ axes.T
        lower, upper = local.min(axis=0), local.max(axis=0)
        distance = np.minimum(local - lower, upper - local).min(axis=1)
        cost = float(np.mean(.05**2 * (np.sqrt(1 + (distance / .05)**2) - 1)))
        return cost, origin + ((lower + upper) / 2) @ axes

    angles = np.arange(360) * np.pi / 720
    best = angles[np.argmin([fit(angle)[0] for angle in angles])]
    result = minimize_scalar(lambda angle: fit(angle)[0],
                             bounds=(best - np.pi / 720, best + np.pi / 720),
                             method="bounded", options={"xatol": 1e-10})
    return fit(float(result.x))[1]


def _pnp_object_center(points: np.ndarray) -> tuple[np.ndarray, str, int]:
    """XY boundary-fit center and vertical extent midpoint, on one trusted set."""
    trusted, removed = _remove_remote_micro_components(points)
    center = np.empty(3, dtype=np.float64)
    center[:2] = _xy_boundary_box_center(trusted[:, :2])
    center[2] = (trusted[:, 2].min() + trusted[:, 2].max()) / 2
    return center, "xy-boundary-box-z-range-midpoint", removed


def _mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    moments = cv2.moments(mask)
    if moments["m00"] <= 0:
        raise ValueError("图像 Mask 为空")
    return moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]


def _pnp_initialize(pairs: list[dict]) -> tuple[np.ndarray, int, float]:
    if len(pairs) < 4:
        raise ValueError("无初值 PnP 至少需要 4 个完整配对，建议使用 6–10 个以上且覆盖近中远和不同图像区域")
    object_points = np.asarray([
        pair.get("pnp_center", pair["center"]) for pair in pairs
    ], dtype=np.float64)
    image_points = np.asarray([_mask_centroid(pair["mask"]) for pair in pairs], dtype=np.float64)
    centered = object_points - object_points.mean(axis=0)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    if singular_values[1] < max(singular_values[0], 1e-9) * 1e-3:
        raise ValueError("配对的 3D 中心几乎共线，无法可靠估计无初值外参；请增加不同横向位置或距离的物体")
    k, d = pairs[0]["k"], pairs[0]["d"]
    distortion_model = pairs[0]["distortion_model"]
    pnp_image_points = image_points
    pnp_distortion = d
    if distortion_model == "fisheye":
        pnp_image_points = cv2.fisheye.undistortPoints(
            image_points.reshape(-1, 1, 2), k, d[:4].reshape(4, 1), P=k,
        ).reshape(-1, 2)
        pnp_distortion = np.zeros(4, dtype=np.float64)
    # Every annotation is an explicit user correspondence. Use all of them in
    # one solve: no subset sampling, RANSAC, inlier threshold, or silent drop.
    success, rotation_vector, translation = cv2.solvePnP(
        object_points, pnp_image_points, k, pnp_distortion,
        flags=cv2.SOLVEPNP_SQPNP,
    )
    if not success:
        raise ValueError("全量 PnP 初始化失败：无法从全部中心对应得到有效外参")
    success, rotation_vector, translation = cv2.solvePnP(
        object_points, pnp_image_points, k, pnp_distortion,
        rotation_vector, translation, useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise ValueError("全量 PnP 迭代精修失败：请检查中心对应或增加空间分布")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    matrix[:3, 3] = translation.reshape(3)
    if distortion_model == "fisheye":
        projected, _ = cv2.fisheye.projectPoints(
            object_points.reshape(1, -1, 3), rotation_vector, translation,
            k, d[:4].reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            object_points, rotation_vector, translation, k, d,
        )
    reprojection_error = float(np.mean(
        np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    ))
    return matrix, len(pairs), reprojection_error


def _project_with_depth(
    points: np.ndarray,
    matrix: np.ndarray,
    k: np.ndarray,
    d: np.ndarray,
    distortion_model: str = "rational",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    camera = points @ matrix[:3, :3].T + matrix[:3, 3]
    valid = camera[:, 2] > 0.1
    camera = camera[valid]
    if not len(camera):
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, valid
    depth = camera[:, 2]
    x, y = camera[:, 0] / depth, camera[:, 1] / depth
    if distortion_model == "fisheye":
        radius = np.sqrt(x*x + y*y)
        theta = np.arctan(radius)
        theta2 = theta*theta
        k1, k2, k3, k4 = d[:4]
        theta_distorted = theta * (1 + k1*theta2 + k2*theta2**2 + k3*theta2**3 + k4*theta2**4)
        scale = np.divide(theta_distorted, radius, out=np.ones_like(radius), where=radius > 1e-12)
        return k[0, 0]*x*scale + k[0, 2], k[1, 1]*y*scale + k[1, 2], depth, valid
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    denominator = 1 if distortion_model == "radtan" else 1 + k4*r2 + k5*r4 + k6*r6
    radial = (1 + k1*r2 + k2*r4 + k3*r6) / np.where(np.abs(denominator) < 1e-8, np.nan, denominator)
    distorted_x = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    distorted_y = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return (
        k[0, 0]*distorted_x + k[0, 2],
        k[1, 1]*distorted_y + k[1, 2],
        depth,
        valid,
    )


def _project(points: np.ndarray, matrix: np.ndarray, k: np.ndarray, d: np.ndarray, distortion_model: str = "rational") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, _, valid = _project_with_depth(points, matrix, k, d, distortion_model)
    return x, y, valid


def _project_pair_with_depth(
    pair: dict,
    points: np.ndarray,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    temporal_projection = pair.get("temporal_projection")
    if temporal_projection is not None:
        if pair["distortion_model"] not in {"radtan", "rational"}:
            raise ValueError("Waymo CameraModel only supports its native radtan projection")
        return project_waymo_camera(
            points, matrix, pair["k"], pair["d"][:5], temporal_projection,
        )
    return _project_with_depth(
        points, matrix, pair["k"], pair["d"], pair["distortion_model"],
    )


SILHOUETTE_DOWNSAMPLE = 4
PARTIAL_EDGE_FRACTION = 1.0
DEPTH_CELL_SIZE = 12
SUPPORT_DIRECTIONS = np.asarray([
    [1.0, 0.0], [math.sqrt(.5), math.sqrt(.5)],
    [0.0, 1.0], [-math.sqrt(.5), math.sqrt(.5)],
])


def _robust_projection(points: np.ndarray) -> np.ndarray:
    if len(points) < 20:
        return points
    bounds = np.quantile(points, [.01, .99], axis=0)
    keep = (
        (points[:, 0] >= bounds[0, 0]) & (points[:, 0] <= bounds[1, 0])
        & (points[:, 1] >= bounds[0, 1]) & (points[:, 1] <= bounds[1, 1])
    )
    return points[keep]


def _directional_supports(points: np.ndarray) -> np.ndarray:
    points = _robust_projection(points)
    if len(points) < 3:
        return np.full((2, len(SUPPORT_DIRECTIONS)), np.nan)
    projected = points @ SUPPORT_DIRECTIONS.T
    return np.quantile(projected, [.02, .98], axis=0)


def _adaptive_visible_boundary(projected: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize a spacing-aware visible support boundary, preserving concavities and components."""
    projected = _robust_projection(projected)
    if len(projected) < 3:
        return projected
    sample = projected[::max(1, math.ceil(len(projected) / 512))]
    if len(sample) >= 3:
        distances, _ = cKDTree(sample).query(sample, k=2)
        spacing = float(np.quantile(distances[:, 1], .7))
    else:
        spacing = float(SILHOUETTE_DOWNSAMPLE)
    radius = int(np.clip(math.ceil(spacing / SILHOUETTE_DOWNSAMPLE), 1, 5))
    small_height = max(1, math.ceil(shape[0] / SILHOUETTE_DOWNSAMPLE))
    small_width = max(1, math.ceil(shape[1] / SILHOUETTE_DOWNSAMPLE))
    points = np.rint(projected / SILHOUETTE_DOWNSAMPLE).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, small_width - 1)
    points[:, 1] = np.clip(points[:, 1], 0, small_height - 1)
    occupancy = np.zeros((small_height, small_width), dtype=np.uint8)
    occupancy[points[:, 1], points[:, 0]] = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    support = cv2.morphologyEx(cv2.dilate(occupancy, kernel), cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(support, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    trusted = [contour for contour in contours if len(contour) >= 3 and abs(cv2.contourArea(contour)) >= 2.0]
    if not trusted:
        return projected
    return np.concatenate([contour[:, 0, :] for contour in trusted]).astype(np.float64) * SILHOUETTE_DOWNSAMPLE


def _silhouette_metrics(projected: np.ndarray, mask_small: np.ndarray) -> tuple[float, float, float]:
    projected = _robust_projection(projected)
    if len(projected) < 3:
        return 0.0, 0.0, 0.0
    points = np.rint(projected / SILHOUETTE_DOWNSAMPLE).astype(np.int32)
    points[:, 0] = np.clip(points[:, 0], 0, mask_small.shape[1] - 1)
    points[:, 1] = np.clip(points[:, 1], 0, mask_small.shape[0] - 1)
    hull = cv2.convexHull(points.reshape(-1, 1, 2))
    if len(hull) < 3:
        return 0.0, 0.0, 0.0
    silhouette = np.zeros(mask_small.shape, dtype=np.uint8)
    cv2.fillConvexPoly(silhouette, hull, 1)
    intersection = int(np.logical_and(silhouette, mask_small).sum())
    union = int(np.logical_or(silhouette, mask_small).sum())
    envelope_area = int(silhouette.sum())
    mask_area = int(mask_small.sum())
    return (
        intersection / max(union, 1),
        intersection / max(envelope_area, 1),
        intersection / max(mask_area, 1),
    )


def _silhouette_iou(projected: np.ndarray, mask_small: np.ndarray) -> float:
    return _silhouette_metrics(projected, mask_small)[0]


def _sample_distance(distance: np.ndarray, points: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty(0)
    height, width = distance.shape
    x = np.clip(points[:, 0], 0, width - 1.000001)
    y = np.clip(points[:, 1], 0, height - 1.000001)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = x - x0, y - y0
    return (
        distance[y0, x0] * (1-wx) * (1-wy)
        + distance[y0, x1] * wx * (1-wy)
        + distance[y1, x0] * (1-wx) * wy
        + distance[y1, x1] * wx * wy
    )


def _project_all_points(pair: dict, points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project while retaining one output row for every source point."""
    projected = np.full((len(points), 2), np.nan, dtype=np.float64)
    x, y, _, valid = _project_pair_with_depth(pair, points, matrix)
    projected[valid] = np.column_stack((x, y))
    return projected


def _inside_mask(mask: np.ndarray, point: np.ndarray) -> bool:
    if not np.isfinite(point).all():
        return False
    x, y = np.rint(point).astype(np.int64)
    return bool(0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x])


def _directed_mask_edge_gap(
    distance: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    maximum_distance: float,
) -> float | None:
    """Find the first external Mask edge encountered along a projected scanline."""
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length < 1e-6:
        return None
    unit = direction / length
    steps = np.arange(0.0, max(1.0, math.ceil(maximum_distance)) + 1.0)
    samples = origin + steps[:, None] * unit
    in_frame = (
        (samples[:, 0] >= 0) & (samples[:, 0] < distance.shape[1])
        & (samples[:, 1] >= 0) & (samples[:, 1] < distance.shape[0])
    )
    if not np.any(in_frame):
        return None
    sampled_steps = steps[in_frame]
    sampled_distance = _sample_distance(distance, samples[in_frame])
    intersections = np.flatnonzero(sampled_distance <= 1.0)
    return float(sampled_steps[intersections[0]]) if len(intersections) else None


def _projected_gap_weight(gap_pixels: float, spacing_pixels: float) -> tuple[float, float]:
    """Map an edge gap to confidence in units of local lidar spacing."""
    gap_steps = max(0.0, float(gap_pixels)) / max(float(spacing_pixels), 1.0)
    if gap_steps <= 1.0:
        return 1.0, gap_steps
    if gap_steps < 2.0:
        return 2.0 - gap_steps, gap_steps
    return 0.0, gap_steps


def _projected_boundary_reliability(
    pair: dict,
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Estimate observable contour reliability from a fixed, already-good pose.

    Endpoint distance is normalized by local projected lidar spacing. An
    endpoint inside the Mask uses its outward scanline direction to detect a
    prematurely truncated observation. An endpoint outside the Mask remains
    active in the one-sided containment loss, but is not assumed to be an exact
    camera-visible silhouette correspondence. Weights are computed once and
    remain frozen during the following local optimization. No reference
    extrinsic is read here.
    """
    boundary_points = np.asarray(pair.get("boundary_points", ()), dtype=np.float64)
    weights = np.zeros(len(boundary_points), dtype=np.float64)
    local_spacings = np.full(len(boundary_points), np.nan, dtype=np.float64)
    endpoint_diagnostics: list[dict] = []
    if not len(boundary_points):
        return weights, local_spacings, {
            "boundaryPointCount": 0, "reliablePointCount": 0,
            "weakPointCount": 0, "rejectedPointCount": 0,
            "containmentOnlyPointCount": 0, "outsideMaskPointCount": 0,
            "meanWeight": 0.0, "endpoints": endpoint_diagnostics,
        }

    mask = pair["mask"]
    distance = pair["distance"]
    frame_diagonal = float(math.hypot(mask.shape[1], mask.shape[0]))
    source_points = pair.get("selected_points_full")
    source_scanlines = pair.get("selected_scanline_ids")
    boundary_local_indices = pair.get("native_boundary_local_indices")

    if (
        source_points is not None and source_scanlines is not None
        and boundary_local_indices is not None
        and len(boundary_local_indices) == len(boundary_points)
    ):
        source_points = np.asarray(source_points, dtype=np.float64)
        source_scanlines = np.asarray(source_scanlines)
        projected = _project_all_points(pair, source_points, matrix)
        boundary_projected = projected[np.asarray(boundary_local_indices, dtype=np.int64)]
        boundary_position = {
            int(local_index): position
            for position, local_index in enumerate(boundary_local_indices)
        }
        groups = _native_scanline_groups(source_points, source_scanlines)

        def endpoint_confidence(group: np.ndarray, endpoint_at_start: bool) -> None:
            endpoint_offset = 0 if endpoint_at_start else len(group) - 1
            inward_offset = 1 if endpoint_at_start else len(group) - 2
            endpoint_index = int(group[endpoint_offset])
            output_position = boundary_position.get(endpoint_index)
            if output_position is None:
                return
            endpoint = projected[endpoint_index]
            inward = projected[int(group[inward_offset])]
            side = "start" if endpoint_at_start else "end"
            record = {
                "scanlineId": int(source_scanlines[endpoint_index]),
                "side": side,
                "weight": 0.0,
            }
            if not _inside_mask(mask, endpoint):
                record["reason"] = "outside-mask-containment-only"
                endpoint_diagnostics.append(record)
                return
            outward = endpoint - inward
            spacing = float(np.linalg.norm(outward))
            if not np.isfinite(spacing) or spacing < 1.0:
                record["reason"] = "invalid-projected-spacing"
                endpoint_diagnostics.append(record)
                return
            local_spacings[output_position] = spacing
            direction_confidence = .35
            if len(group) >= 3:
                second_inward_offset = 2 if endpoint_at_start else len(group) - 3
                second_inward = projected[int(group[second_inward_offset])]
                previous_direction = inward - second_inward
                previous_length = float(np.linalg.norm(previous_direction))
                if np.isfinite(previous_length) and previous_length >= 1.0:
                    cosine = float(np.dot(outward, previous_direction) / (spacing * previous_length))
                    direction_confidence = float(np.clip((cosine - .25) / .75, 0.0, 1.0))
            maximum_gap = min(
                frame_diagonal,
                max(32.0, spacing * 8.0, float(pair.get("mask_scale", 1.0)) * .75),
            )
            gap = _directed_mask_edge_gap(distance, endpoint, outward, maximum_gap)
            if gap is None:
                record.update({
                    "reason": "no-directed-mask-edge",
                    "spacingPixels": spacing,
                })
                endpoint_diagnostics.append(record)
                return
            gap_weight, gap_steps = _projected_gap_weight(gap, spacing)
            weight = gap_weight * direction_confidence
            weights[output_position] = weight
            record.update({
                "reason": "reliable" if weight >= .5 else "weak-or-missing",
                "weight": float(weight),
                "gapPixels": float(gap),
                "spacingPixels": spacing,
                "gapSteps": float(gap_steps),
                "directionConfidence": float(direction_confidence),
            })
            endpoint_diagnostics.append(record)

        for group in groups:
            endpoint_confidence(group, True)
            endpoint_confidence(group, False)

        # The complete lowest and highest rings are v18's vertical-outline
        # evidence. Non-endpoint ring samples outside the Mask are not known to
        # be camera-visible silhouette points; their one-sided containment term
        # remains active, but they are not promoted to edge correspondences.
        for group in (groups[0], groups[-1]) if groups else ():
            for group_offset, local_index in enumerate(group):
                output_position = boundary_position.get(int(local_index))
                if output_position is None or group_offset in {0, len(group) - 1}:
                    continue
                point = projected[int(local_index)]
                if not _inside_mask(mask, point):
                    continue
                neighbor_distances = []
                for neighbor_offset in (group_offset - 1, group_offset + 1):
                    if 0 <= neighbor_offset < len(group):
                        neighbor = projected[int(group[neighbor_offset])]
                        if np.isfinite(neighbor).all():
                            neighbor_distances.append(float(np.linalg.norm(point - neighbor)))
                if not neighbor_distances:
                    continue
                spacing = max(1.0, float(np.median(neighbor_distances)))
                local_spacings[output_position] = spacing
                gap = float(_sample_distance(distance, point.reshape(1, 2))[0])
                weights[output_position] = _projected_gap_weight(gap, spacing)[0]
    else:
        # Ringless fallback: the inferred 3-D boundary is not guaranteed to be
        # a camera-visible silhouette. Exterior samples remain fully active in
        # the one-sided containment loss, but only interior near-edge samples
        # are promoted to the symmetric contour loss.
        projected = _project_all_points(pair, boundary_points, matrix)
        boundary_projected = projected
        finite = np.isfinite(projected).all(axis=1)
        finite_positions = np.flatnonzero(finite)
        if len(finite_positions) >= 2:
            neighbor_distance = cKDTree(projected[finite]).query(projected[finite], k=2)[0][:, 1]
            finite_projected = projected[finite]
            in_frame = (
                (finite_projected[:, 0] >= 0.0)
                & (finite_projected[:, 0] < mask.shape[1])
                & (finite_projected[:, 1] >= 0.0)
                & (finite_projected[:, 1] < mask.shape[0])
            )
            in_frame_positions = finite_positions[in_frame]
            in_frame_projected = finite_projected[in_frame]
            pixel_x = np.rint(in_frame_projected[:, 0]).astype(np.int64)
            pixel_y = np.rint(in_frame_projected[:, 1]).astype(np.int64)
            rounded_in_frame = (
                (pixel_x >= 0) & (pixel_x < mask.shape[1])
                & (pixel_y >= 0) & (pixel_y < mask.shape[0])
            )
            inside = np.zeros(len(in_frame_projected), dtype=bool)
            inside[rounded_in_frame] = mask[
                pixel_y[rounded_in_frame], pixel_x[rounded_in_frame]
            ].astype(bool)
            contour_positions = in_frame_positions[inside]
            spacings = np.maximum(1.0, neighbor_distance[in_frame][inside])
            gaps = _sample_distance(distance, in_frame_projected[inside])
            gap_steps = np.maximum(0.0, gaps) / spacings
            local_spacings[contour_positions] = spacings
            weights[contour_positions] = np.clip(2.0 - gap_steps, 0.0, 1.0)

    reliable = int(np.count_nonzero(weights >= .5))
    weak = int(np.count_nonzero((weights > 0) & (weights < .5)))
    projected_finite = np.isfinite(boundary_projected).all(axis=1)
    projected_inside = np.zeros(len(boundary_projected), dtype=bool)
    finite_positions = np.flatnonzero(projected_finite)
    finite_pixels = np.rint(boundary_projected[projected_finite]).astype(np.int64)
    finite_in_frame = (
        (finite_pixels[:, 0] >= 0) & (finite_pixels[:, 0] < mask.shape[1])
        & (finite_pixels[:, 1] >= 0) & (finite_pixels[:, 1] < mask.shape[0])
    )
    sampled_positions = finite_positions[finite_in_frame]
    sampled_pixels = finite_pixels[finite_in_frame]
    projected_inside[sampled_positions] = mask[
        sampled_pixels[:, 1], sampled_pixels[:, 0]
    ].astype(bool)
    outside_mask = projected_finite & ~projected_inside
    containment_only = outside_mask & (weights == 0.0)
    return weights, local_spacings, {
        "boundaryPointCount": int(len(weights)),
        "reliablePointCount": reliable,
        "weakPointCount": weak,
        "rejectedPointCount": int(len(weights) - reliable - weak),
        "containmentOnlyPointCount": int(np.count_nonzero(containment_only)),
        "outsideMaskPointCount": int(np.count_nonzero(outside_mask)),
        "meanWeight": float(np.mean(weights)) if len(weights) else 0.0,
        "endpoints": endpoint_diagnostics,
    }


def _projected_step_ratio(
    pairs: list[dict],
    reference_matrix: np.ndarray,
    candidate_matrix: np.ndarray,
) -> tuple[float, float, int]:
    """Measure pose motion in units of each reliable lidar contour spacing."""
    normalized_movements = []
    for pair in pairs:
        weights = np.asarray(pair.get("boundary_weights", ()), dtype=np.float64)
        local_spacings = np.asarray(pair.get("boundary_motion_scales", ()), dtype=np.float64)
        boundary_points = np.asarray(pair.get("boundary_points", ()), dtype=np.float64)
        if not (
            len(boundary_points) == len(weights) == len(local_spacings)
            and len(boundary_points)
        ):
            continue
        reliable = (weights >= .5) & np.isfinite(local_spacings) & (local_spacings >= 1.0)
        if not np.any(reliable):
            continue
        reference = _project_all_points(pair, boundary_points, reference_matrix)
        candidate = _project_all_points(pair, boundary_points, candidate_matrix)
        valid = reliable & np.isfinite(reference).all(axis=1) & np.isfinite(candidate).all(axis=1)
        if not np.any(valid):
            continue
        normalized_movements.append(
            np.linalg.norm(candidate[valid] - reference[valid], axis=1) / local_spacings[valid]
        )
    if not normalized_movements:
        return math.inf, math.inf, 0
    values = np.concatenate(normalized_movements)
    return float(np.median(values)), float(np.quantile(values, .9)), int(len(values))


def _adaptive_refinement_scales(
    pairs: list[dict],
    matrix: np.ndarray,
) -> np.ndarray:
    """Convert a one-spacing image trust region into six pose-axis scales."""
    probes = np.asarray([
        math.radians(.05), math.radians(.05), math.radians(.05),
        .005, .005, .005,
    ], dtype=np.float64)
    # These are only numerical ceilings inherited from the broad forward solve;
    # the actual local scale is determined by projected lidar spacing below.
    ceilings = np.asarray([
        math.radians(20), math.radians(20), math.radians(20), 2.0, 2.0, 2.0,
    ], dtype=np.float64)
    scales = np.empty(6, dtype=np.float64)
    for axis, probe in enumerate(probes):
        delta = np.zeros(6, dtype=np.float64)
        delta[axis] = probe
        ratio, _, count = _projected_step_ratio(
            pairs, matrix, _delta_matrix(delta) @ matrix,
        )
        if count == 0 or not np.isfinite(ratio) or ratio <= 1e-9:
            # No image-space observability means this axis must not be given an
            # arbitrary large search range.
            scales[axis] = 0.0
        else:
            scales[axis] = min(ceilings[axis], probe / ratio)
    return scales


def _dense_closed_polyline(points: np.ndarray, spacing: float = 3.0) -> np.ndarray:
    if len(points) < 2:
        return points
    segments = []
    for index, start in enumerate(points):
        end = points[(index + 1) % len(points)]
        sample_count = int(np.clip(np.ceil(np.linalg.norm(end - start) / spacing), 1, 256))
        segments.append(np.linspace(start, end, sample_count, endpoint=False))
    dense = np.concatenate(segments)
    return dense[::max(1, math.ceil(len(dense) / 1800))]


def _robust_mean(errors: np.ndarray, scale: float) -> float:
    if not len(errors):
        return scale * 8.0
    finite = np.asarray(errors, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return scale * 8.0
    return float(np.mean(scale * (np.sqrt(1 + (finite/scale)**2) - 1)))


def _containment_mean(
    errors: np.ndarray, in_frame: np.ndarray, scale: float, in_frame_scale: float,
) -> float:
    """Soften small mask violations without relaxing invalid/out-of-frame costs.

    The pseudo-Huber family remains quadratic near zero and linear far away.
    A larger in-frame scale reduces the pull from small mask inaccuracies; it
    is not a dead zone or a bound on the allowed projection error.
    """
    errors = np.asarray(errors, dtype=np.float64)
    in_frame = np.asarray(in_frame, dtype=bool)
    if errors.shape != in_frame.shape:
        raise ValueError("Containment errors and in-frame flags must align")
    if scale <= 0 or in_frame_scale <= 0:
        raise ValueError("Containment scales must be positive")
    finite = np.isfinite(errors)
    if not np.any(finite):
        return scale * 8.0
    scales = np.where(in_frame[finite], in_frame_scale, scale)
    d = errors[finite]
    return float(np.mean(scales * (np.sqrt(1.0 + (d / scales) ** 2) - 1.0)))


def _robust_weighted_mean(
    errors: np.ndarray,
    weights: np.ndarray,
    scale: float,
) -> float:
    """Return a confidence-gated robust mean without renormalizing uncertainty."""
    errors = np.asarray(errors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if not len(errors) or len(errors) != len(weights):
        return _robust_mean(errors, scale)
    valid = np.isfinite(errors) & np.isfinite(weights) & (weights >= 0)
    if not np.any(valid):
        return scale * 8.0
    transformed = scale * (np.sqrt(1 + (errors[valid] / scale) ** 2) - 1)
    # Divide by the nominal sample count, not the sum of confidence. If every
    # observed edge is uncertain, the whole edge term should become weaker
    # instead of being normalized back to full strength.
    return float(np.sum(transformed * np.clip(weights[valid], 0.0, 1.0)) / np.count_nonzero(valid))


def _contour_sampling_spacings(pair: dict, matrix: np.ndarray) -> np.ndarray:
    """Pixel sampling intervals, frozen at initialization, without mask/GT use.

    Scan endpoints use their inward neighbor; extreme-ring interiors use the
    next observed ring. Ringless samples use their projected nearest neighbor.
    All selected returns remain unchanged. One pixel is a numerical floor.
    """
    points = np.asarray(pair.get("selected_points_full", pair["points"]))
    boundary = np.asarray(pair.get("boundary_points", np.empty((0, 3))))
    if not len(boundary):
        return np.empty(0, dtype=np.float64)
    if len(points) < 2:
        return np.ones(len(boundary), dtype=np.float64)
    boundary_indices = cKDTree(points).query(boundary)[1]
    x, y, _, valid = _project_pair_with_depth(pair, points, matrix)
    projected = np.full((len(points), 2), np.nan)
    projected[valid] = np.column_stack((x, y))
    finite = np.isfinite(projected).all(axis=1)
    spacing = np.ones(len(points), dtype=np.float64)
    if np.count_nonzero(finite) >= 2:
        spacing[finite] = np.maximum(1.0, cKDTree(projected[finite]).query(
            projected[finite], k=2,
        )[0][:, 1])
    scanlines = pair.get("selected_scanline_ids")
    groups = _native_scanline_groups(points, np.asarray(scanlines)) if scanlines is not None else []
    if len(groups) >= 2:
        for group in groups:
            for endpoint, inward in ((group[0], group[1]), (group[-1], group[-2])):
                if finite[endpoint] and finite[inward]:
                    spacing[endpoint] = max(1.0, float(np.linalg.norm(
                        projected[endpoint] - projected[inward],
                    )))
        for outer, adjacent in ((groups[0], groups[1]), (groups[-1], groups[-2])):
            reference = adjacent[finite[adjacent]]
            interior = outer[1:-1]
            interior = interior[finite[interior]]
            if len(reference) and len(interior):
                spacing[interior] = np.maximum(1.0, cKDTree(projected[reference]).query(
                    projected[interior],
                )[0])
    return spacing[boundary_indices]


def _sampling_log_contour_mean(errors: np.ndarray, weights: np.ndarray,
                               spacings: np.ndarray) -> float:
    """s*log1p(rho_6(d)/s): retain small residuals, attenuate unsupported gaps.

    No hard trimming, no additional fitted coefficient, nominal-count mean.
    The inherited six-pixel robust scale is unchanged; this is not parameter-free.
    """
    errors = np.asarray(errors, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    spacings = np.asarray(spacings, dtype=np.float64)
    if errors.shape != weights.shape or errors.shape != spacings.shape:
        raise ValueError("Contour errors, weights and sampling spacings must align")
    valid = np.isfinite(errors) & np.isfinite(weights) & np.isfinite(spacings) & (weights >= 0) & (spacings > 0)
    if not np.any(valid):
        return 48.0
    cost = 6.0 * (np.sqrt(1.0 + (errors[valid] / 6.0) ** 2) - 1.0)
    scale = np.maximum(1.0, spacings[valid])
    cost = scale * np.log1p(cost / scale)
    return float(np.sum(cost * np.clip(weights[valid], 0.0, 1.0)) / np.count_nonzero(valid))


def _background_visibility(
    pair: dict,
    matrix: np.ndarray,
    selected_points: np.ndarray,
    selected_depths: np.ndarray,
) -> tuple[float, int, int]:
    background = pair.get("background")
    if background is None or not len(background) or not len(selected_points):
        return 0.0, 0, 0
    height, width = pair["mask"].shape
    grid_width = max(1, math.ceil(width / DEPTH_CELL_SIZE))
    grid_height = max(1, math.ceil(height / DEPTH_CELL_SIZE))
    depth_grid = np.full((grid_height, grid_width), np.inf, dtype=np.float64)
    selected_x = np.clip((selected_points[:, 0] / DEPTH_CELL_SIZE).astype(np.int64), 0, grid_width - 1)
    selected_y = np.clip((selected_points[:, 1] / DEPTH_CELL_SIZE).astype(np.int64), 0, grid_height - 1)
    np.minimum.at(depth_grid, (selected_y, selected_x), selected_depths)
    local_surface = minimum_filter(depth_grid, size=3, mode="constant", cval=np.inf)

    x, y, depth, _ = _project_pair_with_depth(pair, background, matrix)
    in_frame = (
        np.isfinite(x) & np.isfinite(y)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    if not np.any(in_frame):
        return 0.0, 0, 0
    x, y, depth = x[in_frame], y[in_frame], depth[in_frame]
    pixel_x = np.clip(np.rint(x).astype(np.int64), 0, width - 1)
    pixel_y = np.clip(np.rint(y).astype(np.int64), 0, height - 1)
    inside = pair["mask"][pixel_y, pixel_x].astype(bool)
    if not np.any(inside):
        return 0.0, 0, 0
    x, y, depth = x[inside], y[inside], depth[inside]
    cell_x = np.clip((x / DEPTH_CELL_SIZE).astype(np.int64), 0, grid_width - 1)
    cell_y = np.clip((y / DEPTH_CELL_SIZE).astype(np.int64), 0, grid_height - 1)
    surface_depth = local_surface[cell_y, cell_x]
    comparable = np.isfinite(surface_depth)
    if not np.any(comparable):
        return 0.0, 0, 0
    depth = depth[comparable]
    surface_depth = surface_depth[comparable]
    tolerance = np.maximum(.35, surface_depth * .015)
    intruding = depth < surface_depth - tolerance
    occluded = depth > surface_depth + tolerance
    return float(np.mean(intruding)), int(np.count_nonzero(intruding)), int(np.count_nonzero(occluded))


def _pair_terms(
    pair: dict, matrix: np.ndarray, include_diagnostics: bool = True,
    *, objective_only: bool = False,
) -> dict:
    points = pair["points"]
    height, width = pair["mask"].shape
    x, y, depth, _ = _project_pair_with_depth(pair, points, matrix)
    finite = np.isfinite(x) & np.isfinite(y)
    in_frame = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)

    frame_diagonal = float(math.hypot(width, height))
    valid_errors = np.full(len(x), frame_diagonal, dtype=np.float64)
    if np.any(in_frame):
        visible = np.column_stack((x[in_frame], y[in_frame]))
        outside_distance = pair.get("outside_distance")
        if outside_distance is None:
            outside_distance = distance_transform_edt(~pair["mask"].astype(bool))
        valid_errors[in_frame] = _sample_distance(outside_distance, visible)
        pixel_x = np.clip(np.rint(x[in_frame]).astype(np.int64), 0, width - 1)
        pixel_y = np.clip(np.rint(y[in_frame]).astype(np.int64), 0, height - 1)
        inside_count = int(np.count_nonzero(pair["mask"][pixel_y, pixel_x]))
        visible_depths = depth[in_frame]
    else:
        visible = np.empty((0, 2), dtype=np.float64)
        visible_depths = np.empty(0, dtype=np.float64)
        inside_count = 0

    out_of_frame = finite & ~in_frame
    if np.any(out_of_frame):
        clipped_x = np.clip(x[out_of_frame], 0, width - 1)
        clipped_y = np.clip(y[out_of_frame], 0, height - 1)
        valid_errors[out_of_frame] = 24.0 + np.hypot(
            x[out_of_frame] - clipped_x, y[out_of_frame] - clipped_y,
        )
    missing = max(0, len(points) - len(x))
    containment_errors = np.concatenate((valid_errors, np.full(missing, frame_diagonal)))
    inside_ratio = inside_count / max(len(points), 1)

    boundary_points = pair.get("boundary_points")
    edge_sampling_spacings = None
    observed_sparse_boundary = pair.get("boundary_mode") in {
        "observed-sparse-boundary", "native-scanline-endpoints",
        "native-scanline-boundary", "native-scanline-confidence-boundary",
    }
    minimum_boundary_points = 1 if observed_sparse_boundary else 3
    if boundary_points is not None and len(boundary_points) >= minimum_boundary_points:
        boundary_x, boundary_y, _, boundary_valid = _project_pair_with_depth(
            pair, boundary_points, matrix,
        )
        source_boundary_weights = np.asarray(
            pair.get("boundary_weights", np.ones(len(boundary_points))),
            dtype=np.float64,
        )
        if len(source_boundary_weights) != len(boundary_points):
            source_boundary_weights = np.ones(len(boundary_points), dtype=np.float64)
        projected_boundary_weights = source_boundary_weights[boundary_valid]
        projected_boundary = np.column_stack((boundary_x, boundary_y))
        finite_boundary = np.isfinite(projected_boundary).all(axis=1)
        projected_boundary = projected_boundary[finite_boundary]
        projected_boundary_weights = projected_boundary_weights[finite_boundary]
        sampling_spacings = pair.get("contour_sampling_spacings")
        if observed_sparse_boundary and sampling_spacings is not None:
            sampling_spacings = np.asarray(sampling_spacings, dtype=np.float64)
            if len(sampling_spacings) != len(boundary_points):
                raise ValueError("Contour sampling spacings do not match boundary points")
            edge_sampling_spacings = sampling_spacings[boundary_valid][finite_boundary]
        contour = (
            projected_boundary
            if observed_sparse_boundary
            else _dense_closed_polyline(projected_boundary)
        )
        edge_weights = (
            projected_boundary_weights
            if observed_sparse_boundary
            else np.ones(len(contour), dtype=np.float64)
        )
    else:
        projected_boundary = _adaptive_visible_boundary(visible, (height, width))
        contour = projected_boundary
        edge_weights = np.ones(len(contour), dtype=np.float64)
    contour_in_frame = (
        np.isfinite(contour).all(axis=1)
        & (contour[:, 0] >= 0) & (contour[:, 0] < width)
        & (contour[:, 1] >= 0) & (contour[:, 1] < height)
    ) if len(contour) else np.zeros(0, dtype=bool)
    edge_errors = np.full(len(contour), frame_diagonal, dtype=np.float64)
    if np.any(contour_in_frame):
        edge_errors[contour_in_frame] = _sample_distance(pair["distance"], contour[contour_in_frame])
    if np.any(~contour_in_frame):
        outside = contour[~contour_in_frame]
        edge_errors[~contour_in_frame] = 24.0 + np.hypot(
            outside[:, 0] - np.clip(outside[:, 0], 0, width - 1),
            outside[:, 1] - np.clip(outside[:, 1], 0, height - 1),
        )
    if not observed_sparse_boundary and len(edge_errors) < 6:
        edge_errors = np.concatenate((edge_errors, np.full(6 - len(edge_errors), frame_diagonal)))
        edge_weights = np.concatenate((edge_weights, np.ones(6 - len(edge_weights))))
    if objective_only and pair.get("objective_spacing") is not None and edge_sampling_spacings is not None:
        # The current joint objective consumes only these quantities. Keep
        # legacy fallback and full diagnostics on the original path below.
        return {
            "objective_spacing": pair["objective_spacing"],
            "containment_errors": containment_errors,
            "edge_errors": edge_errors,
            "edge_weights": edge_weights,
            "edge_sampling_spacings": edge_sampling_spacings,
        }
    mask_edge_points = pair.get("closed_edge_points")
    if (
        not observed_sparse_boundary
        and mask_edge_points is not None and len(contour) >= 2
    ):
        reverse_edge_errors = cKDTree(contour).query(mask_edge_points)[0]
    else:
        reverse_edge_errors = np.empty(0, dtype=np.float64)

    support_source = projected_boundary if len(projected_boundary) >= 3 else (
        np.column_stack((x[finite], y[finite])) if np.any(finite) else np.empty((0, 2))
    )
    projected_supports = _directional_supports(support_source)
    mask_supports = pair.get("mask_supports")
    if mask_supports is None:
        mask_y, mask_x = np.nonzero(pair["mask"])
        mask_supports = _directional_supports(np.column_stack((mask_x, mask_y)))
    support_errors = np.abs(projected_supports - mask_supports).reshape(-1)
    center_x, center_y, _, _ = _project_pair_with_depth(
        pair, pair["center"].reshape(1, 3), matrix,
    )
    # dict.get evaluates its default eagerly. Avoid rescanning the full mask
    # on every optimizer trial when preparation already cached its centroid.
    mask_centroid = pair.get("mask_centroid")
    if mask_centroid is None:
        mask_centroid = np.asarray(_mask_centroid(pair["mask"]))
    center_error = (
        float(np.linalg.norm(
            np.asarray([center_x[0], center_y[0]])
            - np.asarray(mask_centroid)
        ))
        if len(center_x) and np.isfinite(center_x[0]) and np.isfinite(center_y[0]) else frame_diagonal
    )
    mask_scale = float(pair.get("mask_scale", frame_diagonal))
    normalized_center_error = center_error / max(mask_scale, 1.0)
    normalized_support_errors = support_errors / max(mask_scale, 1.0)
    if observed_sparse_boundary:
        # Observed sparse samples provide no evidence that the lidar covers the
        # complete top, bottom, left, or right extent visible to the camera.
        normalized_support_errors = np.zeros_like(normalized_support_errors)

    if include_diagnostics:
        background_ratio, background_count, occluded_count = _background_visibility(
            pair, matrix, visible, visible_depths,
        )
        mask_small = pair.get("mask_small")
        iou, envelope_precision, mask_coverage = (
            _silhouette_metrics(visible, mask_small) if mask_small is not None else (0.0, 0.0, 0.0)
        )
    else:
        # These values are reported only after optimization and do not
        # participate in _pair_cost. Skipping them during Powell's thousands
        # of trial evaluations preserves the exact objective value.
        background_ratio = 0.0
        background_count = occluded_count = 0
        iou = envelope_precision = mask_coverage = 0.0
    return {
        "valid": len(visible) > 0,
        "boundary_mode": pair.get("boundary_mode", "geometric-envelope-fallback"),
        "containment_errors": containment_errors,
        "objective_spacing": pair.get("objective_spacing"),
        "containment_in_frame": np.concatenate((in_frame, np.zeros(missing, dtype=bool))),
        "inside_ratio": float(inside_ratio),
        "edge_errors": edge_errors,
        "edge_weights": edge_weights,
        "edge_sampling_spacings": edge_sampling_spacings,
        "reverse_edge_errors": reverse_edge_errors,
        "support_errors": support_errors,
        "center_error": center_error,
        "normalized_center_error": normalized_center_error,
        "normalized_support_errors": normalized_support_errors,
        "mask_scale": mask_scale,
        "background_intrusion_ratio": background_ratio,
        "background_intrusion_count": background_count,
        "occluded_background_count": occluded_count,
        "iou": iou,
        "envelope_precision": envelope_precision,
        "mask_coverage": mask_coverage,
    }


def _trusted_edge_samples(
    errors: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(errors):
        return np.full(1, 768.0), np.ones(1, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if weights is None or len(weights) != len(errors):
        weights = np.ones(len(errors), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
    keep = max(1, min(len(errors), math.ceil(len(errors) * PARTIAL_EDGE_FRACTION)))
    selected = np.argpartition(errors, keep - 1)[:keep]
    return errors[selected], weights[selected]


def _trusted_edge_errors(errors: np.ndarray) -> np.ndarray:
    return _trusted_edge_samples(errors)[0]


def _confidence_weighted_mean(errors: np.ndarray, weights: np.ndarray) -> float | None:
    """No trusted support is missing evidence, not zero reprojection error."""
    if not np.any(np.asarray(weights) > 0):
        return None
    return float(np.average(errors, weights=weights))


def _joint_pair_cost(terms: dict) -> float:
    """One dimensionless objective in both stages; reliability is frozen per round.

    All containment returns count, including invalid/out-of-frame penalties.
    Contours average over positive-weight samples, not rejected candidates.
    Individual confidence weights still attenuate force (no weight-sum
    normalization). Equal-weight first-stage costs remain unchanged.
    Centers initialize PnP and remain diagnostic only in geometric alignment.
    """
    scale = terms.get("objective_spacing")
    spacings = terms.get("edge_sampling_spacings")
    if scale is None or spacings is None:
        # Legacy envelope geometry has no observed-sample spacing model.
        return _pair_cost({**terms, "normalized_center_error": 0.0}, edge_weight=1.0, inside_target=.90,
                          containment_scale=8.0, sampling_contour=True,
                          in_frame_containment_scale=16.0)
    scale = max(1.0, float(scale))
    errors = np.asarray(terms["containment_errors"], dtype=np.float64)
    containment = float(np.mean(np.log1p((errors / scale) ** 2) / 2)) if len(errors) else 8.0
    edge_errors = np.asarray(terms["edge_errors"], dtype=np.float64)
    edge_scale = np.maximum(1.0, np.asarray(spacings, dtype=np.float64))
    weights = np.asarray(terms["edge_weights"], dtype=np.float64)
    active_count = int(np.count_nonzero(weights > 0))
    edge = (float(np.sum((np.sqrt(1 + (edge_errors / edge_scale) ** 2) - 1)
                        * weights) / active_count) if active_count else 0.0)
    if not len(edge_errors):
        edge = 8.0  # Preserve the empty/invalid-projection sentinel.
    return containment + edge


def _mean_pair_costs(costs: list[tuple[float, float]]) -> float:
    """Arithmetic annotation mean, without a second pixel-scale robustifier."""
    if not costs or sum(weight for _, weight in costs) <= 0:
        return 24.0
    return float(np.average([cost for cost, _ in costs],
                            weights=[weight for _, weight in costs]))


def _pair_cost(
    terms: dict,
    *,
    edge_weight: float,
    inside_target: float,
    containment_scale: float,
    sampling_contour: bool = False,
    in_frame_containment_scale: float | None = None,
) -> float:
    containment = _robust_mean(terms["containment_errors"], containment_scale)
    if in_frame_containment_scale is not None:
        containment = _containment_mean(
            terms["containment_errors"], terms["containment_in_frame"],
            containment_scale, in_frame_containment_scale,
        )
    trusted_edges, trusted_weights = _trusted_edge_samples(
        terms["edge_errors"], terms.get("edge_weights"),
    )
    edge = _robust_weighted_mean(trusted_edges, trusted_weights, 6.0)
    if sampling_contour and terms.get("edge_sampling_spacings") is not None:
        edge = _sampling_log_contour_mean(
            terms["edge_errors"], terms["edge_weights"], terms["edge_sampling_spacings"],
        )
    reverse_errors = terms.get("reverse_edge_errors", np.empty(0))
    reverse_edge = _robust_mean(reverse_errors, 6.0) if len(reverse_errors) else edge
    bidirectional_edge = edge * .75 + reverse_edge * .25
    center_constraint = (max(0.0, terms["normalized_center_error"] - .20) / .10) ** 2
    extent_constraint = float(np.mean(
        (np.maximum(0.0, terms["normalized_support_errors"] - .15) / .10) ** 2
    ))
    # Sparse lidar samples do not cover the complete camera-visible object.
    # Containment already penalizes actual outside distance; a fixed inside
    # ratio target biases a correct projection inward. Background visibility is
    # retained as a diagnostic until it has enough support counts to be robust.
    return (
        containment
        + bidirectional_edge * edge_weight
        + center_constraint
        + extent_constraint
    )


def _diagnostics(pairs: list[dict], matrix: np.ndarray) -> dict:
    weighted_edge = weighted_iou = weighted_precision = weighted_coverage = 0.0
    weighted_inside = weighted_outside = weighted_background = weight_sum = 0.0
    confidence_weighted_edge = boundary_confidence = 0.0
    confidence_pair_weight = 0.0
    all_edges = []
    intrusion_count = occluded_count = 0
    for pair in pairs:
        terms = _pair_terms(pair, matrix)
        edge_errors, edge_weights = _trusted_edge_samples(
            terms["edge_errors"], terms.get("edge_weights"),
        )
        containment_errors = terms["containment_errors"]
        weight = pair["weight"]
        weighted_edge += float(np.mean(edge_errors)) * weight
        confidence_mean = _confidence_weighted_mean(edge_errors, edge_weights)
        if confidence_mean is not None:
            confidence_weighted_edge += confidence_mean * weight
            confidence_pair_weight += weight
        boundary_confidence += float(np.mean(edge_weights)) * weight
        weighted_iou += terms["iou"] * weight
        weighted_precision += terms["envelope_precision"] * weight
        weighted_coverage += terms["mask_coverage"] * weight
        weighted_inside += terms["inside_ratio"] * weight
        weighted_outside += float(np.mean(containment_errors)) * weight
        weighted_background += terms["background_intrusion_ratio"] * weight
        weight_sum += weight
        all_edges.append(edge_errors)
        intrusion_count += terms["background_intrusion_count"]
        occluded_count += terms["occluded_background_count"]
    joined = np.concatenate(all_edges) if all_edges else np.full(1, 96.0)
    divisor = max(weight_sum, 1e-9)
    return {
        "mean": weighted_edge / divisor,
        "confidence_weighted_mean": (confidence_weighted_edge / confidence_pair_weight
                                     if confidence_pair_weight > 0 else None),
        "mean_boundary_confidence": boundary_confidence / divisor,
        "median": float(np.median(joined)),
        "p90": float(np.quantile(joined, .9)),
        "iou": weighted_iou / divisor,
        "envelope_precision": weighted_precision / divisor,
        "mask_coverage": weighted_coverage / divisor,
        "matched": float(np.mean(joined <= 12.0)),
        "inside_ratio": weighted_inside / divisor,
        "outside_error": weighted_outside / divisor,
        "background_intrusion_ratio": weighted_background / divisor,
        "background_intrusion_count": intrusion_count,
        "occluded_background_count": occluded_count,
    }


def _pair_diagnostics(pairs: list[dict], matrix: np.ndarray) -> list[dict]:
    summaries = []
    for pair in pairs:
        terms = _pair_terms(pair, matrix)
        edges, edge_weights = _trusted_edge_samples(
            terms["edge_errors"], terms.get("edge_weights"),
        )
        containment = terms["containment_errors"]
        summaries.append({
            "annotationId": pair["id"],
            "insideRatio": float(terms["inside_ratio"]),
            "containmentError": float(np.mean(containment)),
            "medianEdgeError": float(np.median(edges)),
            "p90EdgeError": float(np.quantile(edges, .9)),
            "confidenceWeightedMeanEdgeError": _confidence_weighted_mean(edges, edge_weights),
            "meanBoundaryConfidence": float(np.mean(edge_weights)),
            "reverseEdgeError": float(np.median(terms["reverse_edge_errors"])) if len(terms["reverse_edge_errors"]) else None,
            "scanlineCount": int(pair.get("scanline_count", 0)),
            "nativeScanlineCount": int(pair.get("native_scanline_count", 0)),
            "boundaryMode": pair.get("boundary_mode", "geometric-envelope-fallback"),
            "boundaryPointCount": int(len(pair.get("boundary_points", ()))),
            "projectedContourReliability": pair.get("projected_contour_reliability"),
            "pnpCenterMethod": pair.get("pnp_center_method", "legacy-geometric-center"),
            "pnpCenterRemovedPointCount": int(pair.get("pnp_center_removed_count", 0)),
            "maskClosingRadius": int(pair.get("mask_closing_radius", 0)),
            "centerError": float(terms["center_error"]),
            "directionalExtentError": float(np.mean(terms["support_errors"])),
            "iou": float(terms["iou"]),
            "envelopePrecision": float(terms["envelope_precision"]),
            "maskCoverage": float(terms["mask_coverage"]),
            "suspected": False,
        })
    if not summaries:
        return summaries
    inside = np.asarray([item["insideRatio"] for item in summaries])
    outside = np.asarray([item["containmentError"] for item in summaries])
    edge = np.asarray([item["medianEdgeError"] for item in summaries])

    def upper_limit(values: np.ndarray, absolute: float) -> float:
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        return max(absolute, median + max(3.0 * 1.4826 * mad, 3.0))

    inside_limit = min(.55, float(np.median(inside)) - .20)
    outside_limit = upper_limit(outside, 18.0)
    edge_limit = upper_limit(edge, 45.0)
    for item in summaries:
        item["suspected"] = bool(
            item["insideRatio"] < inside_limit
            or item["containmentError"] > outside_limit
            or item["medianEdgeError"] > edge_limit
        )
    return summaries


def optimize(request: CalibrationRequest, progress: Callable[[int, str, dict | None], None] | None = None) -> dict:
    def report(percent: int, message: str, details: dict | None = None) -> None:
        if progress:
            progress(percent, message, details)

    _register_preparation_batch(request.optimization_batch_id, request.optimization_batch_size)
    report(2, f"正在读取并准备 {len(request.pairs)} 个配对")
    prepared_count = 0
    prepared_lock = threading.Lock()

    def prepare(pair: CalibrationPair) -> dict:
        nonlocal prepared_count
        result = _prepare_pair_for_batch(pair, request.optimization_batch_id)
        with prepared_lock:
            prepared_count += 1
            report(5 + round(15 * prepared_count / len(request.pairs)), f"已准备 {prepared_count}/{len(request.pairs)} 个配对")
        return result

    pairs = list(_PAIR_EXECUTOR.map(prepare, request.pairs))
    for pair in pairs:
        # A saved human annotation is an equally trusted observation. Point
        # count and object range must not silently reduce a pair's influence.
        pair["weight"] = 1.0

    if request.initialization_mode == "forward":
        report(22, f"正在使用全部 {len(pairs)} 组中心对应执行单次 PnP（无 RANSAC）")
        base, pnp_used_count, pnp_error = _pnp_initialize(pairs)
        report(30, f"全量 PnP 初始化完成：使用 {pnp_used_count}/{len(pairs)} 组，平均中心重投影误差 {pnp_error:.2f} px", {
            "pnpUsedPointCount": pnp_used_count, "pnpReprojectionError": pnp_error,
        })
    else:
        base = np.asarray(request.base_matrix, dtype=np.float64).reshape(4, 4)
        pnp_used_count = None
        pnp_error = None
        report(26, "已载入现有外参初值")

    for pair in pairs:
        pair["contour_sampling_spacings"] = _contour_sampling_spacings(pair, base)
        spacings = pair["contour_sampling_spacings"]
        valid_spacings = spacings[np.isfinite(spacings) & (spacings > 0)]
        # Freeze from ALL original boundary samples, not the subset visible at
        # an optimizer trial pose. This remains defined behind the camera.
        pair["objective_spacing"] = max(1.0, float(np.median(valid_spacings))) if len(valid_spacings) else 1.0

    def full_objective(delta: np.ndarray) -> float:
        key = np.asarray(delta, dtype=np.float64).tobytes()
        if key in full_objective_cache:
            return full_objective_cache[key]
        matrix = _delta_matrix(np.asarray(delta)) @ base
        def evaluate(pair: dict) -> tuple[float, float]:
            terms = _pair_terms(pair, matrix, include_diagnostics=False, objective_only=True)
            cost = _joint_pair_cost(terms)
            return cost, pair["weight"]
        pair_costs = _evaluate_pair_costs(evaluate, pairs)
        # The initializer defines only the search origin, not a prior or bound.
        value = _mean_pair_costs(pair_costs)
        full_objective_cache[key] = value
        return value

    full_objective_cache: dict[bytes, float] = {}
    full_seeds = [np.zeros(6)]
    if request.initialization_mode != "forward" and request.previous_matrix:
        previous = _matrix_delta(np.asarray(request.previous_matrix).reshape(4, 4), base)
        full_seeds.append(previous)

    report(34, "第一阶段：约束稀疏点云进入 Mask，并过滤被目标遮挡的后方点")
    full_iterations = 0

    def full_progress(_: np.ndarray) -> None:
        nonlocal full_iterations
        full_iterations += 1
        report(min(68, 34 + full_iterations * 2), f"稀疏点云包含匹配第 {full_iterations} 轮")

    full_candidates = [minimize(
        full_objective, seed, method="Powell", callback=full_progress,
        options={"xtol": 1e-5, "ftol": 1e-5, "maxiter": 260},
    ) for seed in full_seeds]
    full_best = min(full_candidates, key=lambda result: result.fun)
    full_matrix = _delta_matrix(full_best.x) @ base
    full_diagnostics = _diagnostics(pairs, full_matrix)
    report(70, (
        f"稀疏包含匹配完成：Mask 内点 {full_diagnostics['inside_ratio']*100:.1f}%"
        f"，平均越界 {full_diagnostics['outside_error']:.2f} px"
        f"，可见背景误入 {full_diagnostics['background_intrusion_ratio']*100:.1f}%"
    ))

    def prepare_reliability(reference_matrix: np.ndarray) -> list[dict]:
        def estimate(pair: dict) -> dict:
            weights, local_spacings, diagnostics = _projected_boundary_reliability(
                pair, reference_matrix,
            )
            refined_pair = pair.copy()
            refined_pair["boundary_weights"] = weights
            refined_pair["boundary_motion_scales"] = local_spacings
            refined_pair["projected_contour_reliability"] = diagnostics
            return refined_pair
        return list(_PAIR_EXECUTOR.map(estimate, pairs))

    def summarize_reliability(refined_pairs: list[dict]) -> tuple[int, int, int, int, int, float]:
        diagnostics = [
            pair["projected_contour_reliability"] for pair in refined_pairs
        ]
        reliable = sum(item["reliablePointCount"] for item in diagnostics)
        weak = sum(item["weakPointCount"] for item in diagnostics)
        rejected = sum(item["rejectedPointCount"] for item in diagnostics)
        containment_only = sum(item["containmentOnlyPointCount"] for item in diagnostics)
        total = reliable + weak + rejected
        mean_weight = (
            float(np.mean([
                weight
                for pair in refined_pairs
                for weight in np.asarray(pair["boundary_weights"], dtype=np.float64)
            ]))
            if total else 0.0
        )
        return reliable, weak, rejected, containment_only, total, mean_weight

    original_diagnostics = _diagnostics(pairs, base)
    matrix = full_matrix
    partial_matrix = full_matrix
    partial_diagnostics = full_diagnostics
    partial_refinement_accepted = False
    projected_refinement_rounds: list[dict] = []
    partial_iterations = 0
    partial_optimizer_success = True
    last_candidate_objective_improved = False
    # Computational safeguard; convergence is decided by the existing
    # projected-step and geometric acceptance checks, not a best-round search.
    maximum_refinement_rounds = 20

    for refinement_round in range(maximum_refinement_rounds):
        refinement_pairs = prepare_reliability(matrix)
        reliable, weak, rejected, containment_only, _, _ = summarize_reliability(refinement_pairs)
        if reliable == 0:
            projected_refinement_rounds.append({
                "round": refinement_round + 1,
                "accepted": False,
                "reason": "no-reliable-projected-contour",
            })
            break
        axis_scales = _adaptive_refinement_scales(refinement_pairs, matrix)
        if not np.any(axis_scales > 0):
            projected_refinement_rounds.append({
                "round": refinement_round + 1,
                "accepted": False,
                "reason": "no-observable-pose-axis",
            })
            break

        report(min(95, 72 + refinement_round * 6), (
            f"第二阶段第 {refinement_round + 1} 轮：按投影点间距冻结轮廓可靠性"
            f"（强 {reliable}，弱 {weak}，仅包含约束 {containment_only}，"
            f"其余无轮廓约束 {max(0, rejected - containment_only)}）"
        ))
        reference_matrix = matrix
        objective_cache: dict[bytes, float] = {}

        def partial_objective(normalized_delta: np.ndarray) -> float:
            normalized_delta = np.asarray(normalized_delta, dtype=np.float64)
            key = normalized_delta.tobytes()
            if key in objective_cache:
                return objective_cache[key]
            delta = normalized_delta * axis_scales
            candidate_matrix = _delta_matrix(delta) @ reference_matrix

            def evaluate(pair: dict) -> tuple[float, float]:
                terms = _pair_terms(pair, candidate_matrix, include_diagnostics=False, objective_only=True)
                cost = _joint_pair_cost(terms)
                return cost, pair["weight"]

            pair_costs = _evaluate_pair_costs(evaluate, refinement_pairs)
            value = _mean_pair_costs(pair_costs)
            objective_cache[key] = value
            return value

        def partial_progress(_: np.ndarray) -> None:
            nonlocal partial_iterations
            partial_iterations += 1
            report(
                min(95, 72 + refinement_round * 6 + min(5, partial_iterations)),
                f"投影位移自适应精调：第 {refinement_round + 1} 轮",
            )

        partial_baseline = partial_objective(np.zeros(6))
        partial_best = minimize(
            partial_objective,
            np.zeros(6),
            method="Powell",
            bounds=[(-1.0, 1.0)] * 6,
            callback=partial_progress,
            options={"xtol": 1e-5, "ftol": 1e-5, "maxiter": 120},
        )
        partial_optimizer_success = partial_optimizer_success and bool(partial_best.success)
        last_candidate_objective_improved = bool(
            partial_best.fun < partial_baseline
        )
        proposed_delta = np.asarray(partial_best.x, dtype=np.float64) * axis_scales
        proposed_matrix = _delta_matrix(proposed_delta) @ reference_matrix
        proposed_median_step, proposed_p90_step, movement_count = _projected_step_ratio(
            refinement_pairs, reference_matrix, proposed_matrix,
        )
        partial_matrix = proposed_matrix
        partial_diagnostics = _diagnostics(refinement_pairs, proposed_matrix)

        step_scale = 1.0
        if np.isfinite(proposed_median_step) and proposed_median_step > 1.0:
            step_scale = 1.0 / proposed_median_step
        accepted = False
        accepted_objective = partial_baseline
        accepted_median_step = 0.0
        accepted_p90_step = 0.0
        accepted_diagnostics = None
        accepted_matrix = reference_matrix
        accepted_step_scale = 0.0

        # The reliability classification remains frozen throughout this inner
        # solve and line search. A step that crosses more than one local lidar
        # spacing is shortened along the same direction instead of rejected.
        for _ in range(9):
            trial_delta = proposed_delta * step_scale
            trial_matrix = _delta_matrix(trial_delta) @ reference_matrix
            median_step, p90_step, trusted_count = _projected_step_ratio(
                refinement_pairs, reference_matrix, trial_matrix,
            )
            if trusted_count == 0 or not np.isfinite(median_step):
                break
            if median_step > 1.0001:
                step_scale *= min(.8, 1.0 / median_step)
                continue
            normalized_trial = np.divide(
                trial_delta,
                axis_scales,
                out=np.zeros(6, dtype=np.float64),
                where=axis_scales > 0,
            )
            trial_objective = partial_objective(normalized_trial)
            trial_diagnostics = _diagnostics(refinement_pairs, trial_matrix)
            safe = bool(
                trial_objective < partial_baseline
                and trial_diagnostics["inside_ratio"] >= full_diagnostics["inside_ratio"] - .005
                and trial_diagnostics["outside_error"] <= full_diagnostics["outside_error"] + .35
                and trial_diagnostics["background_intrusion_ratio"] <= full_diagnostics["background_intrusion_ratio"] + .01
            )
            if safe:
                accepted = True
                accepted_objective = trial_objective
                accepted_median_step = median_step
                accepted_p90_step = p90_step
                accepted_diagnostics = trial_diagnostics
                accepted_matrix = trial_matrix
                accepted_step_scale = step_scale
                break
            step_scale *= .5

        projected_refinement_rounds.append({
            "round": refinement_round + 1,
            "reliablePointCount": int(reliable),
            "weakPointCount": int(weak),
            "rejectedPointCount": int(rejected),
            "containmentOnlyPointCount": int(containment_only),
            "rotationAxisScaleDegrees": np.degrees(axis_scales[:3]).tolist(),
            "translationAxisScaleMeters": axis_scales[3:].tolist(),
            "proposedMedianStepRatio": float(proposed_median_step),
            "proposedP90StepRatio": float(proposed_p90_step),
            "trustedMovementPointCount": int(movement_count),
            "acceptedStepScale": float(accepted_step_scale),
            "acceptedMedianStepRatio": float(accepted_median_step),
            "acceptedP90StepRatio": float(accepted_p90_step),
            "objectiveBefore": float(partial_baseline),
            "objectiveAfter": float(accepted_objective),
            "optimizerIterations": int(getattr(partial_best, "nit", 0)),
            "accepted": accepted,
        })
        if not accepted:
            break
        matrix = accepted_matrix
        partial_matrix = accepted_matrix
        partial_diagnostics = accepted_diagnostics
        partial_refinement_accepted = True
        if accepted_median_step < .02:
            break

    # Recompute reliability once at the selected pose for final diagnostics.
    # It is not followed by another pose update, so reporting cannot influence
    # the accepted result.
    refinement_pairs = prepare_reliability(matrix)
    reliable_boundary_count, weak_boundary_count, rejected_boundary_count, containment_only_boundary_count, total_boundary_count, mean_projected_reliability = summarize_reliability(
        refinement_pairs,
    )
    final_diagnostics = _diagnostics(refinement_pairs, matrix)
    pair_diagnostics = _pair_diagnostics(refinement_pairs, matrix)
    suggested_review_ids = [item["annotationId"] for item in pair_diagnostics if item["suspected"]]
    if final_diagnostics["inside_ratio"] >= .82 and final_diagnostics["outside_error"] <= 5.0 and final_diagnostics["p90"] <= 40.0:
        quality_status = "good"
    elif final_diagnostics["inside_ratio"] >= .65 and final_diagnostics["outside_error"] <= 15.0:
        quality_status = "warning"
    else:
        quality_status = "poor"
    quality_warnings: list[str] = []
    centers = np.asarray([pair["center"] for pair in pairs], dtype=np.float64)
    singular_values = np.linalg.svd(centers - centers.mean(axis=0), compute_uv=False)
    if len(pairs) < 6:
        quality_warnings.append("配对数量较少，建议增加到至少 6 对并覆盖近、中、远距离")
    if singular_values[-1] < max(singular_values[0], 1e-9) * .02:
        quality_warnings.append("物体中心接近共面，建议增加不同高度或纵深位置的物体")
    if pnp_error is not None and pnp_error > 20.0:
        quality_warnings.append(
            f"全部中心对应的平均 PnP 重投影误差为 {pnp_error:.1f}px；"
            "当前未使用 RANSAC，所有配对均参与求解"
        )
    if quality_status == "poor":
        quality_warnings.append("整体投影质量较差，请先查看建议复核的配对，再考虑增加标注")
    elif quality_status == "warning":
        quality_warnings.append("结果可用但仍有较大残差，建议在不同位置增加配对后重新计算")
    if suggested_review_ids:
        quality_warnings.append(f"有 {len(suggested_review_ids)} 个配对与整体结果偏差较大，建议人工复核")
    report(98, (
        f"自适应投影轮廓精调完成：Mask 内点 {final_diagnostics['inside_ratio']*100:.1f}%"
        f"，平均越界 {final_diagnostics['outside_error']:.2f} px"
        f"，已忽略后方遮挡点 {final_diagnostics['occluded_background_count']}"
        if partial_refinement_accepted else
        "自适应投影轮廓精调未通过包含或可见性约束，已自动保留 v18 结果"
    ))

    return {
        "matrix": matrix.reshape(-1).tolist(),
        "fullMatchMatrix": full_matrix.reshape(-1).tolist(),
        "delta": _matrix_delta(matrix, base).tolist(),
        "originalError": float(original_diagnostics["mean"]),
        "fullMatchError": float(full_diagnostics["mean"]),
        "optimizedError": float(final_diagnostics["mean"]),
        "confidenceWeightedEdgeError": final_diagnostics["confidence_weighted_mean"],
        "meanBoundaryConfidence": float(final_diagnostics["mean_boundary_confidence"]),
        "originalIoU": float(original_diagnostics["iou"]),
        "fullMatchIoU": float(full_diagnostics["iou"]),
        "optimizedIoU": float(final_diagnostics["iou"]),
        "silhouetteMetric": SILHOUETTE_METRIC,
        "envelopePrecision": float(final_diagnostics["envelope_precision"]),
        "maskCoverage": float(final_diagnostics["mask_coverage"]),
        "originalInsideRatio": float(original_diagnostics["inside_ratio"]),
        "fullMatchInsideRatio": float(full_diagnostics["inside_ratio"]),
        "optimizedInsideRatio": float(final_diagnostics["inside_ratio"]),
        "originalContainmentError": float(original_diagnostics["outside_error"]),
        "fullMatchContainmentError": float(full_diagnostics["outside_error"]),
        "optimizedContainmentError": float(final_diagnostics["outside_error"]),
        "backgroundIntrusionRatio": float(final_diagnostics["background_intrusion_ratio"]),
        "backgroundIntrusionCount": int(final_diagnostics["background_intrusion_count"]),
        "occludedBackgroundCount": int(final_diagnostics["occluded_background_count"]),
        "medianEdgeError": float(final_diagnostics["median"]),
        "p90EdgeError": float(final_diagnostics["p90"]),
        "partialMatchRatio": float(final_diagnostics["matched"]),
        "partialRefinementAccepted": partial_refinement_accepted,
        "projectedRefinementCandidateMatrix": partial_matrix.reshape(-1).tolist(),
        "projectedRefinementCandidate": {
            "objectiveImproved": last_candidate_objective_improved,
            "insideRatio": float(partial_diagnostics["inside_ratio"]),
            "containmentError": float(partial_diagnostics["outside_error"]),
            "iou": float(partial_diagnostics["iou"]),
            "medianEdgeError": float(partial_diagnostics["median"]),
            "p90EdgeError": float(partial_diagnostics["p90"]),
        },
        "projectedContourReliability": {
            "boundaryPointCount": int(total_boundary_count),
            "reliablePointCount": int(reliable_boundary_count),
            "weakPointCount": int(weak_boundary_count),
            "rejectedPointCount": int(rejected_boundary_count),
            "containmentOnlyPointCount": int(containment_only_boundary_count),
            "meanWeight": mean_projected_reliability,
        },
        "projectedRefinementRounds": projected_refinement_rounds,
        "pairCount": len(pairs),
        "iterationCount": int(getattr(full_best, "nit", 0) + partial_iterations),
        "fullMatchIterationCount": int(getattr(full_best, "nit", 0)),
        "converged": bool(full_best.success and partial_optimizer_success),
        "groupKey": request.group_key,
        # Legacy field retained for stored-result/UI compatibility. With the
        # no-RANSAC solver it means the number of correspondences actually used.
        "pnpInlierCount": pnp_used_count,
        "pnpUsedPointCount": pnp_used_count,
        "pnpReprojectionError": pnp_error,
        "algorithmVersion": CALIBRATION_ALGORITHM_VERSION,
        "projectionModel": (
            "waymo-camera-model-v1"
            if any(pair.get("temporal_projection") is not None for pair in pairs)
            else "opencv-static-v1"
        ),
        "distortionModel": pairs[0]["distortion_model"],
        "distortionParameterCount": int(pairs[0].get("distortion_parameter_count", len(pairs[0]["d"]))),
        "qualityStatus": quality_status,
        "qualityWarnings": quality_warnings,
        "pairDiagnostics": pair_diagnostics,
        "suggestedReviewAnnotationIds": suggested_review_ids,
    }
