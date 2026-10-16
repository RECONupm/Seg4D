"""Numerical point-cloud features and label postprocessing used by SEG4D.

The functions in this module deliberately have no CloudCompare or Tkinter
dependencies, which keeps the scientific calculations testable in isolation.
"""

from collections import OrderedDict
from itertools import combinations

import numpy as np

try:
    from scipy.spatial import cKDTree as _SciPyKDTree
except ImportError:  # Lightweight fallback for development/test environments.
    _SciPyKDTree = None


CUSTOM_FEATURE_NAMES = (
    "Point Feature Histogram",
    "Direction",
    "Grid Relative Elevation",
    "Relative Elevation",
    "Elevation Variance",
    "Elevation Difference",
    "Normalized Elevation",
)

_SCALAR_FEATURE_KEYS = {
    "Direction": "direction",
    "Grid Relative Elevation": "grid_relative_elevation",
    "Relative Elevation": "relative_elevation",
    "Elevation Variance": "elevation_variance",
    "Elevation Difference": "elevation_difference",
    "Normalized Elevation": "normalized_elevation",
}


class _NumpyKDTree:
    """Small compatibility fallback; production CloudCompare ships SciPy."""

    def __init__(self, points):
        self.points = np.asarray(points, dtype=np.float64)

    def query_ball_point(self, query_points, radius):
        query_points = np.atleast_2d(query_points)
        radius_squared = float(radius) ** 2
        return [
            np.flatnonzero(
                np.sum((self.points - point) ** 2, axis=1) <= radius_squared
            ).tolist()
            for point in query_points
        ]

    def query(self, query_points, k):
        query_points = np.atleast_2d(query_points)
        squared_distances = np.sum(
            (query_points[:, None, :] - self.points[None, :, :]) ** 2,
            axis=2,
        )
        indices = np.argsort(squared_distances, axis=1, kind="stable")[:, :k]
        distances = np.take_along_axis(squared_distances, indices, axis=1) ** 0.5
        return distances, indices


def _neighbor_tree(points):
    return _SciPyKDTree(points) if _SciPyKDTree is not None else _NumpyKDTree(points)


def _validated_points(points):
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must be an N x 3 array")
    if not np.all(np.isfinite(values)):
        raise ValueError("point coordinates must be finite")
    return values


def _normal_from_neighborhood(points, indices):
    if len(indices) < 3:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    neighborhood = points[np.asarray(indices, dtype=np.intp)]
    centered = neighborhood - neighborhood.mean(axis=0)
    covariance = centered.T @ centered / len(neighborhood)
    _, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    length = np.linalg.norm(normal)
    if length <= np.finfo(np.float64).eps:
        return np.array((0.0, 0.0, 1.0), dtype=np.float64)
    normal = normal / length
    # PCA normals have an arbitrary sign. Orient them consistently upward so
    # Direction remains comparable between runs and point-cloud orderings.
    if normal[2] < 0:
        normal = -normal
    return normal


def _compute_normals(points, neighborhoods):
    return np.vstack(
        [_normal_from_neighborhood(points, indices) for indices in neighborhoods]
    )


def _limited_neighborhood(center_index, indices, points, maximum):
    indices = np.asarray(indices, dtype=np.intp)
    if maximum is None or len(indices) <= maximum:
        return indices
    squared_distances = np.sum(
        (points[indices] - points[center_index]) ** 2, axis=1
    )
    order = np.argsort(squared_distances, kind="stable")[:maximum]
    return indices[order]


def _pfh_histogram(points, normals, indices, bins):
    """Return the classical PFH joint histogram (bins ** 3 values)."""
    alpha_values = []
    phi_values = []
    theta_values = []
    epsilon = np.finfo(np.float64).eps

    for first, second in combinations(indices.tolist(), 2):
        difference = points[second] - points[first]
        distance = np.linalg.norm(difference)
        if distance <= epsilon:
            continue
        direction = difference / distance
        first_normal = normals[first]
        second_normal = normals[second]

        first_angle = np.arccos(np.clip(np.dot(first_normal, direction), -1, 1))
        second_angle = np.arccos(np.clip(np.dot(second_normal, direction), -1, 1))
        if first_angle > second_angle:
            first_normal, second_normal = second_normal, first_normal
            direction = -direction

        u_axis = first_normal
        v_axis = np.cross(direction, u_axis)
        v_length = np.linalg.norm(v_axis)
        if v_length <= epsilon:
            continue
        v_axis /= v_length
        w_axis = np.cross(u_axis, v_axis)

        alpha_values.append(np.clip(np.dot(v_axis, second_normal), -1, 1))
        phi_values.append(np.clip(np.dot(u_axis, direction), -1, 1))
        theta_values.append(
            np.arctan2(
                np.dot(w_axis, second_normal),
                np.dot(u_axis, second_normal),
            )
        )

    if not alpha_values:
        return np.zeros(bins ** 3, dtype=np.float64)
    histogram, _ = np.histogramdd(
        np.column_stack((alpha_values, phi_values, theta_values)),
        bins=(bins, bins, bins),
        range=((-1, 1), (-1, 1), (-np.pi, np.pi)),
    )
    histogram = histogram.ravel()
    return histogram / histogram.sum()


def _grid_relative_elevation(points, cell_size):
    """Estimate height above local ground with the paper's 5 x 5 grid idea."""
    xy = points[:, :2]
    origin = xy.min(axis=0)
    cells = np.floor((xy - origin) / cell_size).astype(np.int64)

    ground_by_cell = {}
    for cell, elevation in zip(map(tuple, cells), points[:, 2]):
        current = ground_by_cell.get(cell)
        if current is None or elevation < current:
            ground_by_cell[cell] = float(elevation)

    corrected_ground = {}
    for cell, ground in ground_by_cell.items():
        nearby = [
            ground_by_cell[(cell[0] + dx, cell[1] + dy)]
            for dx in range(-2, 3)
            for dy in range(-2, 3)
            if (cell[0] + dx, cell[1] + dy) in ground_by_cell
            and (dx != 0 or dy != 0)
        ]
        if len(nearby) < 3:
            corrected_ground[cell] = ground
            continue

        nearby = np.asarray(nearby, dtype=np.float64)
        median = np.median(nearby)
        mad = np.median(np.abs(nearby - median))
        if mad > 0:
            nearby = nearby[np.abs(nearby - median) <= 3.0 * 1.4826 * mad]
        local_mean = float(nearby.mean()) if len(nearby) else float(median)
        local_std = float(nearby.std()) if len(nearby) else 0.0
        tolerance = max(2.0 * local_std, np.finfo(np.float64).eps)
        corrected_ground[cell] = (
            local_mean if abs(ground - local_mean) > tolerance else ground
        )

    ground = np.fromiter(
        (corrected_ground[tuple(cell)] for cell in cells),
        dtype=np.float64,
        count=len(points),
    )
    return np.maximum(points[:, 2] - ground, 0.0)


def compute_custom_features(
    points,
    radii,
    selected_features,
    *,
    pfh_bins=5,
    pfh_max_neighbors=32,
    progress_callback=None,
):
    """Compute selected paper-derived features for every point and radius.

    Elevation features from the segment-based article are adapted to each
    point's spherical neighborhood. PFH uses the standard 5 x 5 x 5 joint
    histogram; its neighborhood is capped to make runtime and memory bounded.
    """
    points = _validated_points(points)
    selected = tuple(selected_features)
    unknown = sorted(set(selected) - set(CUSTOM_FEATURE_NAMES))
    if unknown:
        raise ValueError(f"Unsupported custom features: {unknown}")
    radii = [float(radius) for radius in radii]
    if not radii or any(radius <= 0 for radius in radii):
        raise ValueError("Every search radius must be greater than zero")
    if int(pfh_bins) < 2:
        raise ValueError("pfh_bins must be at least 2")
    if pfh_max_neighbors is not None and int(pfh_max_neighbors) < 3:
        raise ValueError("pfh_max_neighbors must be at least 3")

    result = OrderedDict()
    if len(points) == 0:
        return result
    tree = _neighbor_tree(points)
    total_steps = len(radii) * max(1, len(selected))
    completed_steps = 0

    for radius in radii:
        neighborhoods = tree.query_ball_point(points, radius)
        needs_normals = bool(
            {"Direction", "Point Feature Histogram"}.intersection(selected)
        )
        normals = _compute_normals(points, neighborhoods) if needs_normals else None

        if "Direction" in selected:
            result[f"direction_({radius})"] = np.abs(normals[:, 2])
            completed_steps += 1
            if progress_callback:
                progress_callback(completed_steps / total_steps)

        if "Point Feature Histogram" in selected:
            bins = int(pfh_bins)
            descriptors = np.zeros((len(points), bins ** 3), dtype=np.float64)
            for index, indices in enumerate(neighborhoods):
                limited = _limited_neighborhood(
                    index, indices, points, pfh_max_neighbors
                )
                descriptors[index] = _pfh_histogram(
                    points, normals, limited, bins
                )
            for bin_index in range(bins ** 3):
                result[f"pfh_{bin_index:03d}_({radius})"] = descriptors[:, bin_index]
            completed_steps += 1
            if progress_callback:
                progress_callback(completed_steps / total_steps)

        if "Grid Relative Elevation" in selected:
            result[f"grid_relative_elevation_({radius})"] = (
                _grid_relative_elevation(points, radius)
            )
            completed_steps += 1
            if progress_callback:
                progress_callback(completed_steps / total_steps)

        local_requested = [
            name
            for name in (
                "Relative Elevation",
                "Elevation Variance",
                "Elevation Difference",
                "Normalized Elevation",
            )
            if name in selected
        ]
        if local_requested:
            arrays = {
                name: np.zeros(len(points), dtype=np.float64)
                for name in local_requested
            }
            for index, indices in enumerate(neighborhoods):
                elevations = points[np.asarray(indices, dtype=np.intp), 2]
                lowest = float(elevations.min())
                highest = float(elevations.max())
                average = float(elevations.mean())
                if "Relative Elevation" in arrays:
                    arrays["Relative Elevation"][index] = points[index, 2] - lowest
                if "Elevation Variance" in arrays:
                    arrays["Elevation Variance"][index] = elevations.var()
                if "Elevation Difference" in arrays:
                    arrays["Elevation Difference"][index] = highest - lowest
                if "Normalized Elevation" in arrays:
                    arrays["Normalized Elevation"][index] = average - lowest
            for name in local_requested:
                result[f"{_SCALAR_FEATURE_KEYS[name]}_({radius})"] = arrays[name]
                completed_steps += 1
                if progress_callback:
                    progress_callback(completed_steps / total_steps)

    return result


def spatial_majority_filter(
    points,
    labels,
    *,
    neighbors=16,
    minimum_agreement=0.70,
    iterations=1,
):
    """Suppress isolated class labels using a conservative 3-D k-NN vote."""
    points = _validated_points(points)
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) != len(points):
        raise ValueError("labels must contain one value per point")
    neighbors = int(neighbors)
    iterations = int(iterations)
    minimum_agreement = float(minimum_agreement)
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    if iterations < 1:
        raise ValueError("iterations must be at least 1")
    if not 0.5 <= minimum_agreement <= 1.0:
        raise ValueError("minimum_agreement must be between 0.5 and 1.0")
    if len(points) < 2:
        return labels.copy()

    count = min(neighbors + 1, len(points))
    tree = _neighbor_tree(points)
    _, indices = tree.query(points, k=count)
    if indices.ndim == 1:
        indices = indices[:, None]
    neighbor_indices = indices[:, 1:]
    filtered = labels.copy()

    for _ in range(iterations):
        updated = filtered.copy()
        for index, local_indices in enumerate(neighbor_indices):
            local_labels = filtered[local_indices]
            values, counts = np.unique(local_labels, return_counts=True)
            winner_position = int(np.argmax(counts))
            winner = values[winner_position]
            agreement = counts[winner_position] / len(local_labels)
            current_support = np.count_nonzero(local_labels == filtered[index])
            if (
                winner != filtered[index]
                and agreement >= minimum_agreement
                and current_support < counts[winner_position]
            ):
                updated[index] = winner
        filtered = updated
    return filtered
