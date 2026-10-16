"""Exercise the CloudCompare adapter using a small in-memory host double."""

import copy
import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
pycc_stub = types.ModuleType("pycc")
spec = importlib.util.spec_from_file_location(
    "deflection_interface_test_target", ROOT / "geometric-based_methods/analysis_of_deflections.py")
gui = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"pycc": pycc_stub}):
    spec.loader.exec_module(gui)


class Scalar:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def asArray(self):
        return self.values

    def computeMinAndMax(self):
        pass


class Cloud:
    def __init__(self, x, y, z, name="Beam", scale=1.0):
        self.coordinates = np.column_stack((x, y, z))
        self.name, self.scale = name, scale
        self.shift = (-500000, -4500000, 0)
        self.fields = {"Unused NaN feature": Scalar(np.full(len(x), np.nan))}

    def points(self):
        return self.coordinates

    def getName(self):
        return self.name

    def setName(self, value):
        self.name = value

    def size(self):
        return len(self.coordinates)

    def clone(self):
        return copy.deepcopy(self)

    def getGlobalScale(self):
        return self.scale

    def setGlobalScale(self, value):
        self.scale = value

    def getGlobalShift(self):
        return self.shift

    def setGlobalShift(self, value):
        self.shift = value

    def getScalarFieldIndexByName(self, name):
        return list(self.fields).index(name) if name in self.fields else -1

    def deleteScalarField(self, index):
        del self.fields[list(self.fields)[index]]

    def addScalarField(self, name):
        self.fields[name] = Scalar(np.zeros(self.size()))
        return list(self.fields).index(name)

    def getScalarField(self, index):
        return list(self.fields.values())[index]


class Folder:
    def __init__(self, children):
        self.children = children

    def getChildrenNumber(self):
        return len(self.children)

    def getChild(self, index):
        return self.children[index]


class Host:
    def __init__(self, selected):
        self.selected, self.added = selected, []

    def getSelectedEntities(self):
        return self.selected

    def addToDB(self, cloud):
        self.added.append(cloud)


class Progress:
    def setMethodTitle(self, title):
        pass

    def setInfo(self, message):
        pass

    def update(self, percent):
        pass


class DeflectionInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.app = object.__new__(gui.GUI_deflection)
        (ROOT / ".build").mkdir(exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / ".build")
        self.addCleanup(self.folder.cleanup)

    def run_analysis(self, host, mode="Data", direction="Vertical", degree=2, load_axis=False):
        with patch.object(gui.pycc, "GetInstance", return_value=host, create=True), \
             patch.object(gui.pycc, "ccPointCloud", Cloud, create=True):
            return self.app._analyse_selection(0.02, degree, 300, 5, mode, direction,
                                               load_axis, self.folder.name, Progress())

    @staticmethod
    def beam(scale=1):
        x = np.linspace(0, 10, 201)
        return Cloud(x * scale, np.zeros(len(x)),
                     (0.5 * x - 0.08 * x * (10 - x) / 25) * scale, scale=scale)

    def test_data_and_fit_reports_match_selected_scalar_and_own_length(self):
        for mode in ("Data", "Fit"):
            with self.subTest(mode=mode):
                original = self.beam()
                original.fields["Maximum deflection"] = Scalar(np.full(original.size(), 999))
                host = Host([original])
                count, errors, directory = self.run_analysis(host, mode)
                self.assertEqual((count, errors), (1, []))
                result = host.added[0]
                np.testing.assert_array_equal(result.points(), original.points())
                self.assertTrue((original.fields["Maximum deflection"].values == 999).all())
                self.assertTrue(np.isnan(result.fields["Unused NaN feature"].values).all())
                with (directory / "deflection_analysis.txt").open(encoding="utf-8") as stream:
                    reader = csv.DictReader(stream, delimiter="\t")
                    row = next(reader)
                    self.assertEqual(reader.fieldnames[:12], list(gui.LEGACY_REPORT_COLUMNS))
                name = "point" if mode == "Data" else "polynomial"
                self.assertAlmostEqual(float(row["Length"]), float(row[f"Length from {name} chord"]))
                self.assertAlmostEqual(float(row["Selected maximum allowed deflection"]), float(row["Length"]) / 300)
                self.assertAlmostEqual(float(row["Selected deflection"]),
                                       float(result.fields["Maximum deflection"].values[0]), places=7)
                self.assertTrue((directory / "Beam_0.png").is_file())
        self.assertEqual(plt.get_fignums(), [])

    def test_global_scale_skeleton_metadata_and_perpendicular_measurement(self):
        original = self.beam(scale=2)
        host = Host([original])
        count, errors, _ = self.run_analysis(host, mode="Fit", direction="Perpendicular", load_axis=True)
        self.assertEqual((count, errors), (1, []))
        result, skeleton = host.added
        np.testing.assert_array_equal(skeleton.points(), original.points())
        self.assertEqual(skeleton.getGlobalShift(), original.getGlobalShift())
        self.assertEqual(skeleton.getGlobalScale(), 2)
        self.assertAlmostEqual(float(result.fields["Maximum deflection"].values[0]),
                               0.08 / np.hypot(1, 0.5), places=7)

    def test_bad_folder_child_is_reported_and_other_beam_still_processed(self):
        host = Host([Folder([object(), self.beam()])])
        count, errors, directory = self.run_analysis(host)
        self.assertEqual(count, 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("not a point cloud", errors[0][1])
        self.assertTrue((directory / "deflection_analysis_errors.txt").is_file())

    def test_data_survives_fit_unavailability(self):
        host = Host([Cloud([0, 1, 2], [0, 0, 0], [0, 0.4, 1])])
        count, errors, _ = self.run_analysis(host, degree=4)
        self.assertEqual(count, 1)
        self.assertIn("Fit unavailable", errors[0][1])
        self.assertAlmostEqual(float(host.added[0].fields["Maximum deflection"].values[0]), 0.1, places=7)

    def test_straight_beam_reports_zero_and_undefined_scalar_ratio(self):
        x = np.linspace(0, 10, 201)
        host = Host([Cloud(x, np.zeros(len(x)), 0.5 * x)])
        count, errors, directory = self.run_analysis(host)
        self.assertEqual((count, errors), (1, []))
        self.assertEqual(float(host.added[0].fields["Maximum deflection"].values[0]), 0)
        self.assertEqual(float(host.added[0].fields["Is deflected"].values[0]), 0)
        self.assertTrue(np.isnan(host.added[0].fields["Relative_deflection"].values).all())
        with (directory / "deflection_analysis.txt").open(encoding="utf-8") as stream:
            row = next(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(row["Relative deflection from point data"], "inf")

    def test_parameter_validation_and_empty_selection(self):
        for parameters in ((np.nan, 2, 300, 5), (0.02, 2, np.inf, 5),
                           (0.02, 0, 300, 5), (0.02, 2, 300, 0)):
            with self.assertRaises(ValueError):
                self.app._validate_parameters(*parameters)
        with self.assertRaisesRegex(ValueError, "Select a beam"):
            self.run_analysis(Host([]))


if __name__ == "__main__":
    unittest.main()
