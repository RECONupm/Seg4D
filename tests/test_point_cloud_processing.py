import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "main_module"))

from point_cloud_processing import (  # noqa: E402
    compute_custom_features,
    spatial_majority_filter,
)
from supervoxel_processing import (  # noqa: E402
    aggregate_features_by_supervoxel,
    broadcast_supervoxel_predictions,
    compute_supervoxels,
    majority_labels_by_supervoxel,
    reduce_features_by_supervoxel,
)


class PointCloudFeatureTests(unittest.TestCase):
    def test_horizontal_plane_has_vertical_normals(self):
        points = np.array(
            [[x, y, 4.0] for x in range(3) for y in range(3)], dtype=float
        )
        features = compute_custom_features(points, [2.0], ["Direction"])
        np.testing.assert_allclose(features["direction_(2.0)"], 1.0)

    def test_local_elevation_formulas(self):
        points = np.array(
            [[0, 0, 0], [0, 0, 1], [0, 0, 2]], dtype=float
        )
        features = compute_custom_features(
            points,
            [3.0],
            [
                "Relative Elevation",
                "Elevation Variance",
                "Elevation Difference",
                "Normalized Elevation",
            ],
        )
        np.testing.assert_allclose(
            features["relative_elevation_(3.0)"], [0, 1, 2]
        )
        np.testing.assert_allclose(features["elevation_variance_(3.0)"], 2 / 3)
        np.testing.assert_allclose(features["elevation_difference_(3.0)"], 2)
        np.testing.assert_allclose(features["normalized_elevation_(3.0)"], 1)

    def test_pfh_has_125_normalized_bins(self):
        points = np.array(
            [[x, y, 0.0] for x in range(3) for y in range(3)], dtype=float
        )
        features = compute_custom_features(
            points, [2.0], ["Point Feature Histogram"]
        )
        self.assertEqual(len(features), 125)
        descriptor = np.column_stack(tuple(features.values()))
        np.testing.assert_allclose(descriptor.sum(axis=1), 1.0)


class SpatialPostprocessingTests(unittest.TestCase):
    def test_isolated_label_is_replaced(self):
        points = np.column_stack((np.arange(9), np.zeros(9), np.zeros(9)))
        labels = np.ones(9, dtype=int)
        labels[4] = 7
        filtered = spatial_majority_filter(
            points, labels, neighbors=4, minimum_agreement=0.75
        )
        np.testing.assert_array_equal(filtered, np.ones(9, dtype=int))

    def test_stable_regions_are_unchanged(self):
        points = np.column_stack((np.arange(8), np.zeros(8), np.zeros(8)))
        labels = np.array([1, 1, 1, 1, 2, 2, 2, 2])
        filtered = spatial_majority_filter(
            points, labels, neighbors=2, minimum_agreement=1.0
        )
        np.testing.assert_array_equal(filtered, labels)


class SupervoxelTests(unittest.TestCase):
    def test_disconnected_surfaces_get_connected_supervoxels(self):
        first = np.array(
            [[x, y, 0.0] for x in (0.0, 0.1, 0.2) for y in (0.0, 0.1, 0.2)]
        )
        second = first + np.array((3.0, 0.0, 1.0))
        points = np.vstack((first, second))
        result = compute_supervoxels(
            points,
            voxel_resolution=0.09,
            seed_resolution=0.5,
            normal_radius=0.25,
            minimum_points=2,
            iterations=2,
        )
        self.assertGreaterEqual(len(np.unique(result.point_labels)), 2)
        self.assertTrue(
            set(
                (
                    "SV Curvature",
                    "SV Linearity",
                    "SV Planarity",
                    "SV Scattering",
                    "SV Anisotropy",
                )
            ).issubset(result.point_features)
        )
        for values in result.point_features.values():
            self.assertTrue(np.all(np.isfinite(values)))

    def test_feature_aggregation_is_broadcast_by_supervoxel(self):
        values = np.array([1.0, 3.0, 10.0, 14.0])
        labels = np.array([2, 2, 7, 7])
        aggregated = aggregate_features_by_supervoxel(values, labels)
        np.testing.assert_allclose(aggregated, [2.0, 2.0, 12.0, 12.0])

    def test_supervoxel_reduction_and_prediction_broadcast(self):
        values = np.array([[1.0, 2.0], [3.0, 4.0], [8.0, 10.0]])
        labels = np.array([5, 5, 9])
        ids, reduced = reduce_features_by_supervoxel(values, labels)
        np.testing.assert_array_equal(ids, [5, 9])
        np.testing.assert_allclose(reduced, [[2.0, 3.0], [8.0, 10.0]])
        _, classes = majority_labels_by_supervoxel([2, 2, 7], labels)
        np.testing.assert_array_equal(classes, [2, 7])
        broadcast = broadcast_supervoxel_predictions(ids, [4, 8], labels)
        np.testing.assert_array_equal(broadcast, [4, 4, 8])


if __name__ == "__main__":
    unittest.main()
