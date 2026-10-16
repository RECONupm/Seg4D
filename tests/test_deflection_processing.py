"""Analytic and coordinate-invariance regressions for beam deflection."""

import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main_module"))
from deflection_processing import (
    analyze_beam, chord_deflection, extract_beam_profile,
    polynomial_chord_deflection,
)


class ChordDeflectionTests(unittest.TestCase):
    def test_straight_inclined_beams_have_no_false_deflection(self):
        x = np.linspace(0, 10, 1001) ** 1.1
        for slope in (-2, -0.5, 0, 0.5, 2):
            for fraction in (0, 0.05, 0.5):
                with self.subTest(slope=slope, fraction=fraction):
                    result = chord_deflection(x, slope * x + 7, fraction)
                    self.assertEqual(result["maximum"], 0)
                    self.assertEqual(result["relative"], np.inf)
                    np.testing.assert_allclose(result["z_ends"], slope * np.array([x.min(), x.max()]) + 7)

    def test_known_sag_and_tilt(self):
        x = np.linspace(0, 10, 1001)
        result = chord_deflection(x, 0.5 * x - 0.08 * x * (10 - x) / 25, 0)
        self.assertAlmostEqual(result["maximum"], 0.08)
        self.assertAlmostEqual(result["x_maximum"], 5)
        self.assertAlmostEqual(result["length"], np.hypot(10, 5))

    def test_perpendicular_measure_is_invariant_to_in_plane_rotation(self):
        s = np.linspace(0, 10, 1001)
        sag = -0.08 * s * (10 - s) / 25
        for angle in (0, 20, 45, 70):
            radians = np.radians(angle)
            x = s * np.cos(radians) - sag * np.sin(radians)
            z = s * np.sin(radians) + sag * np.cos(radians)
            result = chord_deflection(x, z, 0, "Perpendicular")
            self.assertAlmostEqual(result["maximum"], 0.08)
            self.assertAlmostEqual(result["length"], 10)
            self.assertAlmostEqual(result["maximum_station"], 5)

    def test_support_regions_extrapolate_to_actual_ends(self):
        x = np.linspace(0, 10, 1001)
        z = 0.5 * x - 0.08 * x * (10 - x) / 25
        result = chord_deflection(x, z)
        # The finite support window approximates a smooth curved end locally;
        # its bias must remain small, rather than the old 125 mm tilt error.
        # A local linear support estimate retains an O(window**2) curvature
        # bias (about 0.13 mm here), not an exact polynomial endpoint.
        self.assertLess(abs(result["maximum"] - 0.08), 0.0002)

    def test_endpoint_outlier_does_not_move_reference_chord(self):
        x = np.linspace(0, 10, 1001)
        z = 0.5 * x
        z[25] += 1
        result = chord_deflection(x, z)
        np.testing.assert_allclose(result["z_ends"], [0, 5], atol=1e-7)

    def test_unordered_points_and_duplicate_endpoints(self):
        x = np.array([10, 0, 5, 0, 10, 7], dtype=float)
        z = 0.5 * x
        z[2] -= 0.1
        result = chord_deflection(x, z, 0)
        self.assertAlmostEqual(result["maximum"], 0.1)

    def test_exact_polynomial_extremum_between_sample_locations(self):
        # p(x)=0.5*x - 0.01*x*(10-x)*(1+0.1*x), on [0,10].
        curve = np.polynomial.Polynomial([0, 0.4, 0, 0.001])
        stationary = np.sqrt(0.1 / 0.003)
        result = polynomial_chord_deflection(curve, 0, 10)
        self.assertAlmostEqual(result["x_maximum"], stationary)
        self.assertAlmostEqual(result["maximum"], 0.1 * stationary - 0.001 * stationary ** 3)

    def test_polynomial_perpendicular_distance(self):
        curve = np.polynomial.Polynomial([0, 0.5 - 0.032, 0.0032])
        result = polynomial_chord_deflection(curve, 0, 10, "Perpendicular")
        self.assertAlmostEqual(result["maximum"], 0.08 / np.hypot(1, 0.5))

    def test_invalid_profiles_and_support_width(self):
        for x, z in (([0], [0]), ([1, 1], [0, 1]), ([0, 1], [0]),
                     ([[0, 1]], [[0, 1]]), ([np.nan, np.inf], [0, 1])):
            with self.subTest(x=x), self.assertRaises(ValueError):
                chord_deflection(x, z)
        for fraction in (-0.1, 0.51, np.nan, np.inf):
            with self.assertRaises(ValueError):
                chord_deflection([0, 1], [0, 1], fraction)
        with self.assertRaises(ValueError):
            chord_deflection([0, 1], [0, 1], direction="unknown")


class BeamProfileTests(unittest.TestCase):
    @staticmethod
    def beam(angle=0, offset=(0, 0, 0), slope=0.5):
        station = np.linspace(0, 10, 201)
        across = np.array([-0.04, 0, 0.04])
        s, y = np.meshgrid(station, across)
        radians = np.radians(angle)
        x = s * np.cos(radians) - y * np.sin(radians)
        y = s * np.sin(radians) + y * np.cos(radians)
        z = slope * s - 0.08 * s * (10 - s) / 25
        return np.column_stack((x.ravel(), y.ravel(), z.ravel())) + offset

    def test_shuffled_cloud_azimuth_and_utm_origin(self):
        base = analyze_beam(self.beam(), degree=2)
        rotated = self.beam(67, (500000, 4500000, 1000))
        np.random.default_rng(42).shuffle(rotated)
        result = analyze_beam(rotated, degree=2)
        self.assertAlmostEqual(result["data"]["maximum"], base["data"]["maximum"], places=8)
        self.assertAlmostEqual(result["fit"]["maximum"], 0.08, places=8)
        np.testing.assert_allclose(result["skeleton"], rotated[result["indices"]])

    def test_steep_beam_with_wide_xy_footprint(self):
        # The horizontal span is shorter than the width; 3D direction must
        # still identify the actual beam axis, unlike the longest XY side.
        z = np.linspace(0, 10, 201)
        x, y = np.meshgrid(z * 0.01, [-0.2, 0, 0.2])
        points = np.column_stack((x.ravel(), y.ravel(), np.tile(z, 3)))
        result = analyze_beam(points, degree=1)
        self.assertEqual(result["data"]["maximum"], 0)
        self.assertAlmostEqual(result["data"]["length"], np.hypot(10, 0.1))

    def test_vertical_cloud_is_rejected_explicitly(self):
        points = np.column_stack((np.zeros(20), np.zeros(20), np.arange(20)))
        with self.assertRaisesRegex(ValueError, "vertical"):
            extract_beam_profile(points, 0.02)

    def test_data_can_succeed_when_optional_fit_has_insufficient_points(self):
        points = np.array([[0, 0, 0], [1, 0, 0.4], [2, 0, 1]])
        result = analyze_beam(points, degree=4, require_fit=False)
        self.assertIsNone(result["fit"])
        self.assertTrue(result["fit_error"])
        self.assertAlmostEqual(result["data"]["maximum"], 0.1)
        with self.assertRaisesRegex(ValueError, "Polynomial fit unavailable"):
            analyze_beam(points, degree=4)

    def test_finite_coordinate_filter_preserves_original_indices(self):
        points = self.beam()
        points = np.insert(points, 10, [np.nan, 0, 0], axis=0)
        result = analyze_beam(points, degree=2)
        self.assertNotIn(10, result["indices"])
        np.testing.assert_allclose(result["skeleton"], points[result["indices"]])

    def test_invalid_inputs(self):
        for tolerance in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                analyze_beam(self.beam(), tolerance=tolerance)
        with self.assertRaises(ValueError):
            analyze_beam(np.zeros((20, 4)))
        with self.assertRaises(ValueError):
            analyze_beam(self.beam(), degree=0)


if __name__ == "__main__":
    unittest.main()
