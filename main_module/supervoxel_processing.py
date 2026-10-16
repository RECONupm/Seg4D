"""VCCS-inspired supervoxels, LCCP neighborhoods and hybrid descriptors.

This is a NumPy implementation of the two-level graphical model described by
Liao et al. (Remote Sensing 2022, 14, 1516).  It intentionally exposes the
approximation choices needed outside PCL and keeps all calculations independent
from CloudCompare's UI.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from point_cloud_processing import _neighbor_tree, _validated_points


@dataclass(frozen=True)
class SupervoxelResult:
    point_labels: np.ndarray
    patch_labels: np.ndarray
    centroids: np.ndarray
    normals: np.ndarray
    adjacency: tuple
    convex_adjacency: tuple
    point_features: dict
    supervoxel_features: dict


class _DisjointSet:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value):
        value = int(value)
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

    def union(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def _normal_and_eigenvalues(points):
    if len(points) < 3:
        return np.array((0.0, 0.0, 1.0)), np.zeros(3)
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    normal = eigenvectors[:, order[-1]]
    normal /= max(np.linalg.norm(normal), np.finfo(float).eps)
    if normal[2] < 0:
        normal = -normal
    return normal, eigenvalues


def _voxelize(points, colors, resolution):
    origin = points.min(axis=0)
    keys = np.floor((points - origin) / resolution).astype(np.int64)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse, minlength=len(unique_keys)).astype(np.int64)
    centroids = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis], minlength=len(unique_keys))
            / counts
            for axis in range(3)
        ]
    )
    voxel_colors = None
    if colors is not None:
        voxel_colors = np.column_stack(
            [
                np.bincount(
                    inverse,
                    weights=colors[:, axis],
                    minlength=len(unique_keys),
                )
                / counts
                for axis in range(3)
            ]
        )
    return origin, unique_keys, inverse, counts, centroids, voxel_colors


def _voxel_normals(points, voxel_centroids, radius):
    tree = _neighbor_tree(points)
    neighborhoods = tree.query_ball_point(voxel_centroids, radius)
    normals = np.zeros_like(voxel_centroids)
    for index, neighbors in enumerate(neighborhoods):
        normal, _ = _normal_and_eigenvalues(points[np.asarray(neighbors, dtype=int)])
        normals[index] = normal
    return normals


def _initial_seed_indices(voxel_centroids, origin, seed_resolution):
    seed_cells = np.floor((voxel_centroids - origin) / seed_resolution).astype(int)
    _, groups = np.unique(seed_cells, axis=0, return_inverse=True)
    seed_indices = []
    for group in range(groups.max() + 1):
        members = np.flatnonzero(groups == group)
        center = voxel_centroids[members].mean(axis=0)
        closest = np.argmin(np.sum((voxel_centroids[members] - center) ** 2, axis=1))
        seed_indices.append(members[closest])
    return np.asarray(seed_indices, dtype=int)


def _assign_voxels(
    voxel_centroids,
    voxel_normals,
    voxel_colors,
    seed_centroids,
    seed_normals,
    seed_colors,
    seed_resolution,
    spatial_weight,
    normal_weight,
    color_weight,
):
    candidate_count = min(12, len(seed_centroids))
    _, candidates = _neighbor_tree(seed_centroids).query(
        voxel_centroids, k=candidate_count
    )
    if candidates.ndim == 1:
        candidates = candidates[:, None]
    spatial = np.linalg.norm(
        voxel_centroids[:, None, :] - seed_centroids[candidates], axis=2
    ) / seed_resolution
    normal_similarity = np.abs(
        np.einsum("ij,ikj->ik", voxel_normals, seed_normals[candidates])
    )
    distance = spatial_weight * spatial + normal_weight * (1.0 - normal_similarity)
    if color_weight > 0 and voxel_colors is not None and seed_colors is not None:
        color_scale = max(float(np.ptp(voxel_colors, axis=0).max()), 1.0)
        color_distance = np.linalg.norm(
            voxel_colors[:, None, :] - seed_colors[candidates], axis=2
        ) / (np.sqrt(3.0) * color_scale)
        distance += color_weight * color_distance
    return candidates[np.arange(len(voxel_centroids)), np.argmin(distance, axis=1)]


def _update_seeds(
    labels,
    voxel_centroids,
    voxel_normals,
    voxel_colors,
    counts,
):
    occupied = np.unique(labels)
    remap = np.full(labels.max() + 1, -1, dtype=int)
    remap[occupied] = np.arange(len(occupied))
    labels = remap[labels]
    seed_centroids = np.zeros((len(occupied), 3), dtype=float)
    seed_normals = np.zeros((len(occupied), 3), dtype=float)
    seed_colors = (
        np.zeros((len(occupied), 3), dtype=float)
        if voxel_colors is not None
        else None
    )
    for label in range(len(occupied)):
        members = np.flatnonzero(labels == label)
        weights = counts[members].astype(float)
        seed_centroids[label] = np.average(
            voxel_centroids[members], axis=0, weights=weights
        )
        reference = voxel_normals[members[0]]
        member_normals = voxel_normals[members].copy()
        member_normals[np.dot(member_normals, reference) < 0] *= -1
        normal = np.average(member_normals, axis=0, weights=weights)
        seed_normals[label] = normal / max(
            np.linalg.norm(normal), np.finfo(float).eps
        )
        if seed_colors is not None:
            seed_colors[label] = np.average(
                voxel_colors[members], axis=0, weights=weights
            )
    return labels, seed_centroids, seed_normals, seed_colors


_FORWARD_NEIGHBORS = tuple(
    offset
    for offset in product((-1, 0, 1), repeat=3)
    if offset != (0, 0, 0) and offset > (0, 0, 0)
)


def _voxel_edges(keys):
    lookup = {tuple(key): index for index, key in enumerate(keys)}
    edges = []
    for index, key in enumerate(keys):
        key = tuple(key)
        for offset in _FORWARD_NEIGHBORS:
            neighbor = (
                key[0] + offset[0],
                key[1] + offset[1],
                key[2] + offset[2],
            )
            neighbor_index = lookup.get(neighbor)
            if neighbor_index is not None:
                edges.append((index, neighbor_index))
    return edges


def _enforce_connectivity(labels, edges, counts, minimum_points):
    components = _DisjointSet(len(labels))
    for first, second in edges:
        if labels[first] == labels[second]:
            components.union(first, second)
    roots = np.fromiter(
        (components.find(index) for index in range(len(labels))),
        dtype=int,
        count=len(labels),
    )
    _, component_labels = np.unique(roots, return_inverse=True)
    component_counts = np.bincount(component_labels, weights=counts)
    component_neighbors = [set() for _ in range(len(component_counts))]
    for first, second in edges:
        first_component = component_labels[first]
        second_component = component_labels[second]
        if first_component != second_component:
            component_neighbors[first_component].add(second_component)
            component_neighbors[second_component].add(first_component)

    merged = np.arange(len(component_counts), dtype=int)
    for component in np.argsort(component_counts):
        if component_counts[component] >= minimum_points:
            continue
        candidates = [
            candidate
            for candidate in component_neighbors[component]
            if component_counts[candidate] > component_counts[component]
        ]
        if candidates:
            target = max(candidates, key=lambda value: component_counts[value])
            merged[component] = target
    for component in range(len(merged)):
        while merged[component] != merged[merged[component]]:
            merged[component] = merged[merged[component]]
    _, connected_labels = np.unique(merged[component_labels], return_inverse=True)
    return connected_labels


def _supervoxel_edges(voxel_labels, voxel_edges):
    return tuple(
        sorted(
            {
                tuple(sorted((int(voxel_labels[first]), int(voxel_labels[second]))))
                for first, second in voxel_edges
                if voxel_labels[first] != voxel_labels[second]
            }
        )
    )


def _shape_features(eigenvalues):
    first, second, third = eigenvalues
    epsilon = np.finfo(float).eps
    denominator = max(first, epsilon)
    total = max(first + second + third, epsilon)
    return {
        "SV Curvature": third / total,
        "SV Linearity": (first - second) / denominator,
        "SV Planarity": (second - third) / denominator,
        "SV Scattering": third / denominator,
        "SV Anisotropy": (first - third) / denominator,
    }


def compute_supervoxels(
    points,
    *,
    colors=None,
    voxel_resolution=0.10,
    seed_resolution=0.50,
    normal_radius=0.30,
    spatial_weight=0.40,
    normal_weight=0.60,
    color_weight=0.0,
    iterations=5,
    minimum_points=10,
    convexity_threshold=0.0,
    density_neighbors=16,
    progress_callback=None,
):
    """Compute VCCS-like connected supervoxels and LCCP feature neighborhoods."""
    points = _validated_points(points)
    if len(points) == 0:
        raise ValueError("The point cloud is empty")
    colors = None if colors is None else np.asarray(colors, dtype=float)
    if colors is not None and (colors.ndim != 2 or colors.shape != (len(points), 3)):
        raise ValueError("colors must be an N x 3 array")
    voxel_resolution = float(voxel_resolution)
    seed_resolution = float(seed_resolution)
    normal_radius = float(normal_radius)
    iterations = int(iterations)
    minimum_points = int(minimum_points)
    density_neighbors = int(density_neighbors)
    weights = np.asarray((spatial_weight, normal_weight, color_weight), dtype=float)
    if voxel_resolution <= 0 or seed_resolution <= 0 or normal_radius <= 0:
        raise ValueError("All resolutions and radii must be greater than zero")
    if seed_resolution < voxel_resolution:
        raise ValueError("Seed resolution must not be smaller than voxel resolution")
    if iterations < 1 or minimum_points < 1 or density_neighbors < 1:
        raise ValueError("Iterations, minimum points and density neighbors must be positive")
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Feature weights must be non-negative and not all zero")
    weights /= weights.sum()
    spatial_weight, normal_weight, color_weight = weights
    if color_weight > 0 and colors is None:
        raise ValueError("A positive color weight requires RGB data")

    (
        origin,
        voxel_keys,
        point_to_voxel,
        voxel_counts,
        voxel_centroids,
        voxel_colors,
    ) = _voxelize(points, colors, voxel_resolution)
    if progress_callback:
        progress_callback(0.10, "Estimating voxel normals...")
    voxel_normals = _voxel_normals(points, voxel_centroids, normal_radius)
    seed_indices = _initial_seed_indices(voxel_centroids, origin, seed_resolution)
    seed_centroids = voxel_centroids[seed_indices].copy()
    seed_normals = voxel_normals[seed_indices].copy()
    seed_colors = voxel_colors[seed_indices].copy() if voxel_colors is not None else None

    labels = np.zeros(len(voxel_centroids), dtype=int)
    for iteration in range(iterations):
        labels = _assign_voxels(
            voxel_centroids,
            voxel_normals,
            voxel_colors,
            seed_centroids,
            seed_normals,
            seed_colors,
            seed_resolution,
            spatial_weight,
            normal_weight,
            color_weight,
        )
        labels, seed_centroids, seed_normals, seed_colors = _update_seeds(
            labels,
            voxel_centroids,
            voxel_normals,
            voxel_colors,
            voxel_counts,
        )
        if progress_callback:
            progress_callback(
                0.15 + 0.45 * (iteration + 1) / iterations,
                f"Refining VCCS clusters ({iteration + 1}/{iterations})...",
            )

    voxel_edges = _voxel_edges(voxel_keys)
    labels = _enforce_connectivity(labels, voxel_edges, voxel_counts, minimum_points)
    point_labels = labels[point_to_voxel]
    supervoxel_count = int(point_labels.max()) + 1
    indices_by_label = [
        np.flatnonzero(point_labels == label) for label in range(supervoxel_count)
    ]
    centroids = np.vstack(
        [points[indices].mean(axis=0) for indices in indices_by_label]
    )
    normals = np.zeros_like(centroids)
    own_eigenvalues = np.zeros_like(centroids)
    for label, indices in enumerate(indices_by_label):
        normals[label], own_eigenvalues[label] = _normal_and_eigenvalues(points[indices])

    adjacency = _supervoxel_edges(labels, voxel_edges)
    convex_edges = []
    convex_neighbors = [set() for _ in range(supervoxel_count)]
    for first, second in adjacency:
        direction = centroids[first] - centroids[second]
        length = np.linalg.norm(direction)
        if length <= np.finfo(float).eps:
            continue
        direction /= length
        delta_alpha = np.dot(normals[first], direction) - np.dot(
            normals[second], direction
        )
        if delta_alpha > float(convexity_threshold):
            convex_edges.append((first, second))
            convex_neighbors[first].add(second)
            convex_neighbors[second].add(first)

    patches = _DisjointSet(supervoxel_count)
    for first, second in convex_edges:
        patches.union(first, second)
    patch_roots = np.fromiter(
        (patches.find(label) for label in range(supervoxel_count)),
        dtype=int,
        count=supervoxel_count,
    )
    _, patch_by_supervoxel = np.unique(patch_roots, return_inverse=True)
    point_patch_labels = patch_by_supervoxel[point_labels]

    if progress_callback:
        progress_callback(0.72, "Computing LCCP-constrained eigen-features...")
    feature_names = (
        "SV Curvature",
        "SV Linearity",
        "SV Planarity",
        "SV Scattering",
        "SV Anisotropy",
    )
    supervoxel_features = {
        name: np.zeros(supervoxel_count, dtype=float) for name in feature_names
    }
    for label in range(supervoxel_count):
        neighborhood_labels = [label, *sorted(convex_neighbors[label])]
        neighborhood_indices = np.concatenate(
            [indices_by_label[value] for value in neighborhood_labels]
        )
        _, eigenvalues = _normal_and_eigenvalues(points[neighborhood_indices])
        shape = _shape_features(eigenvalues)
        for name in feature_names:
            supervoxel_features[name][label] = shape[name]

    supervoxel_features["SV Direction"] = np.abs(normals[:, 2])
    supervoxel_features["SV Point Count"] = np.asarray(
        [len(indices) for indices in indices_by_label], dtype=float
    )
    supervoxel_features["SV Adjacent Count"] = np.asarray(
        [
            sum(label in edge for edge in adjacency)
            for label in range(supervoxel_count)
        ],
        dtype=float,
    )
    supervoxel_features["SV Convex Neighbor Count"] = np.asarray(
        [len(values) for values in convex_neighbors], dtype=float
    )
    query_count = min(density_neighbors, len(points))
    distances, _ = _neighbor_tree(points).query(centroids, k=query_count)
    if distances.ndim == 1:
        distances = distances[:, None]
    supervoxel_features["SV Local Density"] = distances.mean(axis=1)

    if colors is not None:
        for channel, name in enumerate(("SV Mean R", "SV Mean G", "SV Mean B")):
            supervoxel_features[name] = np.asarray(
                [colors[indices, channel].mean() for indices in indices_by_label]
            )

    point_features = {
        "Supervoxel ID": point_labels.astype(float),
        "LCCP Patch ID": point_patch_labels.astype(float),
    }
    for name, values in supervoxel_features.items():
        point_features[name] = values[point_labels]
    if progress_callback:
        progress_callback(1.0, f"Computed {supervoxel_count} supervoxels.")

    return SupervoxelResult(
        point_labels=point_labels,
        patch_labels=point_patch_labels,
        centroids=centroids,
        normals=normals,
        adjacency=adjacency,
        convex_adjacency=tuple(convex_edges),
        point_features=point_features,
        supervoxel_features=supervoxel_features,
    )


def aggregate_features_by_supervoxel(feature_values, labels):
    """Average point descriptors inside each supervoxel and broadcast them."""
    original_ndim = np.asarray(feature_values).ndim
    values = np.asarray(feature_values, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if values.ndim == 1:
        values = values[:, None]
    if len(values) != len(labels):
        raise ValueError("feature_values and labels must have the same length")
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    aggregated = np.column_stack(
        [
            np.bincount(inverse, weights=values[:, column]) / counts
            for column in range(values.shape[1])
        ]
    )
    broadcast = aggregated[inverse]
    return broadcast[:, 0] if original_ndim == 1 else broadcast


def reduce_features_by_supervoxel(feature_values, labels):
    """Return sorted supervoxel IDs and one mean feature vector per ID."""
    values = np.asarray(feature_values, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(values) != len(labels):
        raise ValueError("feature_values and labels must have the same length")
    unique_labels, inverse = np.unique(labels, return_inverse=True)
    counts = np.bincount(inverse).astype(float)
    reduced = np.column_stack(
        [
            np.bincount(inverse, weights=values[:, column]) / counts
            for column in range(values.shape[1])
        ]
    )
    return unique_labels, reduced


def majority_labels_by_supervoxel(classifications, labels):
    """Return a deterministic majority class for every sorted supervoxel ID."""
    classifications = np.asarray(classifications).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(classifications) != len(labels):
        raise ValueError("classifications and labels must have the same length")
    unique_labels = np.unique(labels)
    majority = []
    for label in unique_labels:
        values, counts = np.unique(classifications[labels == label], return_counts=True)
        majority.append(values[np.argmax(counts)])
    return unique_labels, np.asarray(majority)


def broadcast_supervoxel_predictions(supervoxel_ids, predictions, point_labels):
    """Project one prediction per supervoxel back to the original points."""
    supervoxel_ids = np.asarray(supervoxel_ids, dtype=int).reshape(-1)
    predictions = np.asarray(predictions).reshape(-1)
    point_labels = np.asarray(point_labels, dtype=int).reshape(-1)
    if len(supervoxel_ids) != len(predictions):
        raise ValueError("supervoxel_ids and predictions must have the same length")
    order = np.argsort(supervoxel_ids)
    sorted_ids = supervoxel_ids[order]
    locations = np.searchsorted(sorted_ids, point_labels)
    valid = locations < len(sorted_ids)
    if np.any(valid):
        valid &= sorted_ids[np.minimum(locations, len(sorted_ids) - 1)] == point_labels
    if not np.all(valid):
        missing = np.unique(point_labels[~valid]).tolist()
        raise ValueError(f"Missing predictions for supervoxels: {missing}")
    return predictions[order][locations]
