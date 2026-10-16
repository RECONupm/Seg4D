"""CloudCompare interface for beam chord-deflection analysis."""

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pycc
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

MODULE_DIRECTORY = Path(__file__).resolve().parents[1] / "main_module"
if str(MODULE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(MODULE_DIRECTORY))
from deflection_processing import (
    DEFLECTION_DIRECTIONS, analyze_beam, chord_deflection,
    polynomial_chord_deflection,
)

LEGACY_REPORT_COLUMNS = (
    "Identifier", "Length", "Deflection from point data relative to chord",
    "Deflection from polynomial data relative to chord",
    "Distance to maximum deflection point from point data",
    "Distance to maximum deflection point from polynomial data",
    "Inflection points (vertical coordinates)", "Inflection points (horizontal coordinates)",
    "Relative deflection from point data", "Relative deflection from polynomial data",
    "Is within the relative deflection tolerance according to the data from the points?",
    "Is within the relative deflection tolerance according to the curve fitting?",
)


class GUI_deflection(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.parameters = {"tolerance": 0.02, "degree": 4,
                           "relative_deflection": 300, "endpoint_percentage": 5.0}
        self.output_directory = os.getcwd()
        self._running = False

    def main_frame(self, window):
        window.title("Analysis of deflections")
        window.resizable(False, False)
        panel = ttk.Frame(window, padding=10)
        panel.pack(fill="both", expand=True)
        labels = ["Thickness threshold (coordinate units):", "Polynomial degree:",
                  "Relative deflection limit (L/300: enter 300):",
                  "End region used to estimate supports (%):"]
        entries = []
        for row, (label, key) in enumerate(zip(labels, self.parameters)):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=3)
            entry = ttk.Entry(panel, width=12)
            entry.insert(0, str(self.parameters[key]))
            entry.grid(row=row, column=1, sticky="ew", pady=3)
            entries.append(entry)
        ttk.Label(panel, text="Deflection direction:").grid(row=4, column=0, sticky="w")
        direction = ttk.Combobox(panel, values=DEFLECTION_DIRECTIONS, state="readonly")
        direction.set("Vertical")
        direction.grid(row=4, column=1, sticky="ew", pady=3)
        ttk.Label(panel, text="Input used for scalar fields:").grid(row=5, column=0, sticky="w")
        source = ttk.Combobox(panel, values=("Data", "Fit"), state="readonly")
        source.set("Data")
        source.grid(row=5, column=1, sticky="ew", pady=3)
        load_axis = tk.BooleanVar(master=window, value=False)
        ttk.Checkbutton(panel, text="Load the points of the main axis", variable=load_axis).grid(
            row=6, column=0, columnspan=2, sticky="w", pady=3)
        ttk.Label(panel, text="Path for saving results:").grid(row=7, column=0, sticky="w")
        output = ttk.Entry(panel, width=36)
        output.insert(0, self.output_directory)
        output.grid(row=7, column=1, sticky="ew")

        def browse():
            directory = filedialog.askdirectory(parent=window, initialdir=self.output_directory)
            if directory:
                output.delete(0, tk.END)
                output.insert(0, directory)
                self.output_directory = directory

        ttk.Button(panel, text="...", command=browse).grid(row=7, column=2)
        ttk.Label(panel, text="Select a beam cloud or a folder of beams. Use one longitudinal\n"
                  "surface/profile per beam, with Z vertical. Perpendicular is measured\n"
                  "in the vertical longitudinal plane. The result is geometry\n"
                  "relative to the end chord, not change since an earlier survey.").grid(
                      row=8, column=0, columnspan=3, sticky="w", pady=8)

        def close():
            if not self._running:
                window.destroy()

        def run():
            if self._running:
                return
            progress = None
            try:
                tolerance = float(entries[0].get())
                degree = int(entries[1].get())
                denominator = float(entries[2].get())
                percentage = float(entries[3].get())
                self._validate_parameters(tolerance, degree, denominator, percentage)
                directory = Path(output.get().strip())
                if not output.get().strip() or not directory.is_dir():
                    raise ValueError("Select an existing output directory")
                self.output_directory = str(directory.resolve())
                self.parameters.update(tolerance=tolerance, degree=degree,
                                       relative_deflection=denominator, endpoint_percentage=percentage)
                self._running = True
                run_button.configure(state="disabled")
                cancel_button.configure(state="disabled")
                progress = pycc.ccProgressDialog()
                progress.start()
                count, errors, result_directory = self._analyse_selection(
                    tolerance, degree, denominator, percentage, source.get(),
                    direction.get(), load_axis.get(), self.output_directory, progress)
                messagebox.showinfo("Deflection analysis", f"Analysed beams: {count}\n"
                                    f"Warnings/errors: {len(errors)}\nResults: {result_directory}", parent=window)
            except Exception as error:
                traceback.print_exc()
                messagebox.showerror("Deflection analysis", str(error), parent=window)
            finally:
                if progress is not None:
                    progress.stop()
                self._running = False
                run_button.configure(state="normal")
                cancel_button.configure(state="normal")

        run_button = ttk.Button(panel, text="Run", command=run)
        run_button.grid(row=9, column=0, pady=5)
        cancel_button = ttk.Button(panel, text="Cancel", command=close)
        cancel_button.grid(row=9, column=1, pady=5)
        window.protocol("WM_DELETE_WINDOW", close)

    @staticmethod
    def _validate_parameters(tolerance, degree, denominator, percentage):
        if not np.isfinite([tolerance, denominator, percentage]).all():
            raise ValueError("All numeric parameters must be finite")
        if tolerance <= 0 or degree < 1 or denominator <= 0 or not 0 < percentage <= 50:
            raise ValueError("Thickness and relative denominator must be positive; degree >= 1; support region in (0, 50]%")

    @staticmethod
    def _set_scalar_field(cloud, name, value):
        # L/d is infinite for a straight profile. Preserve that in the text
        # report, using CloudCompare's undefined value (NaN) in the scalar field.
        if name == "Relative_deflection" and not np.isfinite(value):
            value = np.nan
        elif not np.isfinite(value) or abs(value) > np.finfo(np.float32).max:
            raise ValueError(f"The result for '{name}' cannot be represented as a scalar field")
        existing = cloud.getScalarFieldIndexByName(name)
        if existing >= 0:
            cloud.deleteScalarField(existing)
        index = cloud.addScalarField(name)
        if index < 0:
            raise RuntimeError(f"Could not create scalar field '{name}'")
        scalar = cloud.getScalarField(index)
        scalar.asArray()[:] = value
        scalar.computeMinAndMax()

    @staticmethod
    def _plot(result, identifier, path):
        x, z = result["points"][:, 0], result["points"][:, 2]
        order = np.argsort(x)
        plotted = order[np.linspace(0, len(order) - 1, min(len(order), 10000), dtype=int)]
        figure, (axes, deflections) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
        try:
            axes.scatter(x[plotted], z[plotted], s=3, label="Data points")
            for name, values in (("Data", result["data"]), ("Fit", result["fit"])):
                if values is None:
                    continue
                axes.plot(values["x_ends"], values["z_ends"], linestyle="--", label=f"Reference chord ({name})")
                axes.scatter(values["x_maximum"], values["z_maximum"], label=f"Maximum ({name})")
                residual = ((values["z_maximum"] - values["z_ends"][0])
                            - values["slope"] * (values["x_maximum"] - values["x_ends"][0]))
                if values["direction"] == "Perpendicular":
                    residual *= values["span"] / values["length"]
                deflections.scatter(values["x_maximum"], residual, label=f"Maximum ({name})")
            deflections.scatter(x[plotted], result["data"]["signed_deflection"][plotted],
                                s=3, label="Data deflection")
            if result["curve"] is not None:
                samples = np.linspace(x.min(), x.max(), 1000)
                heights = result["curve"](samples)
                axes.plot(samples, heights, color="red", label="Polynomial curve")
                fitted = result["fit"]
                residuals = ((heights - fitted["z_ends"][0])
                             - fitted["slope"] * (samples - fitted["x_ends"][0]))
                if fitted["direction"] == "Perpendicular":
                    residuals *= fitted["span"] / fitted["length"]
                deflections.plot(samples, residuals, color="red", label="Fit deflection")
                roots = result["curve"].deriv(2).roots()
                roots = roots[np.isclose(roots.imag, 0, atol=1e-10, rtol=0)].real
                roots = roots[(roots > x.min()) & (roots < x.max())]
                if len(roots):
                    axes.scatter(roots, result["curve"](roots), marker="x", label="Inflection points")
            axes.set(ylabel="Elevation relative to local origin",
                     title=f"{identifier}: {result['data']['direction']} deflection")
            axes.legend()
            axes.grid(True)
            deflections.axhline(0, color="black", linestyle="--", linewidth=0.8)
            deflections.set(xlabel="Horizontal longitudinal coordinate relative to local origin",
                            ylabel=f"Signed {result['data']['direction'].lower()} deflection")
            deflections.grid(True)
            deflections.legend()
            figure.tight_layout()
            figure.savefig(path)
        finally:
            plt.close(figure)

    @staticmethod
    def _report_row(identifier, source_name, result, mode, denominator, percentage, degree, scale):
        observed, fitted = result["data"], result["fit"]
        selected = observed if mode == "Data" else fitted
        roots = []
        if result["curve"] is not None:
            candidates = result["curve"].deriv(2).roots()
            roots = candidates[np.isclose(candidates.imag, 0, atol=1e-10, rtol=0)].real
            roots = roots[(roots > observed["x_ends"][0]) & (roots < observed["x_ends"][1])].tolist()
        row = {"Identifier": identifier, "Source cloud": source_name,
               "Direction": observed["direction"], "Selected input": mode,
               "Length": selected["length"], "Horizontal span": observed["span"],
               "Selected deflection": selected["maximum"], "Relative limit denominator": denominator,
               "Selected maximum allowed deflection": selected["length"] / denominator,
               "Inflection points (vertical coordinates)": json.dumps(
                   [] if not roots else result["curve"](roots).tolist()),
               "Inflection points (horizontal coordinates)": json.dumps(roots),
               "Local origin": json.dumps(result["origin"].tolist()),
               "Horizontal longitudinal axis": json.dumps(result["horizontal_axis"].tolist()),
               "CloudCompare global scale": scale, "Support region percentage": percentage,
               "Polynomial degree": degree, "Fit error": result["fit_error"]}
        for name, values in (("point", observed), ("polynomial", fitted)):
            row[f"Length from {name} chord"] = "" if values is None else values["length"]
            row[f"Deflection from {name} data relative to chord"] = "" if values is None else values["maximum"]
            row[f"Distance to maximum deflection point from {name} data"] = "" if values is None else values["maximum_station"]
            row[f"Relative deflection from {name} data"] = "" if values is None else values["relative"]
            condition_column = LEGACY_REPORT_COLUMNS[10 if name == "point" else 11]
            row[condition_column] = (
                "" if values is None else values["maximum"] <= values["length"] / denominator)
        ordered = {name: row[name] for name in LEGACY_REPORT_COLUMNS}
        ordered.update(row)
        return ordered

    def _analyse_selection(self, tolerance, degree, denominator, percentage, mode,
                           direction, load_axis, output_directory, progress):
        self._validate_parameters(tolerance, degree, denominator, percentage)
        if mode not in ("Data", "Fit") or direction not in DEFLECTION_DIRECTIONS:
            raise ValueError("Select a valid input and deflection direction")
        cc = pycc.GetInstance()
        selected = list(cc.getSelectedEntities())
        if not selected:
            raise ValueError("Select a beam cloud or a folder containing beam clouds")
        clouds = []
        for entity in selected:
            if callable(getattr(entity, "points", None)):
                clouds.append(entity)
            else:
                clouds.extend(entity.getChild(i) for i in range(entity.getChildrenNumber()))
        # Selecting both a folder and its child must not analyse the same cloud twice.
        clouds = list({cloud.getUniqueID() if callable(getattr(cloud, "getUniqueID", None))
                       else id(cloud): cloud for cloud in clouds}.values())
        if not clouds:
            raise ValueError("The selected folder contains no entities")
        directory = Path(output_directory) / ("deflection_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        directory.mkdir()
        rows, errors = [], []
        progress.setMethodTitle("Analysis of beam deflections")
        progress.update(0)
        for index, cloud in enumerate(clouds):
            identifier = f"Beam_{index}"
            progress.setInfo(f"{identifier}: calculating profile and deflection...")
            try:
                if not callable(getattr(cloud, "points", None)):
                    raise ValueError("The selected entity is not a point cloud")
                coordinates = np.asarray(cloud.points(), dtype=np.float64)
                scale = float(cloud.getGlobalScale()) if hasattr(cloud, "getGlobalScale") else 1.0
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("The point cloud has an invalid global scale")
                result = analyze_beam(coordinates / scale, tolerance, degree,
                                      percentage / 100, direction, require_fit=(mode == "Fit"))
                values = result["data"] if mode == "Data" else result["fit"]
                source_name = cloud.getName()
                clone = cloud.clone()
                clone.setName(f"{source_name} [deflection: {mode}, {direction}]")
                self._set_scalar_field(clone, "Is deflected", float(values["maximum"] > values["length"] / denominator))
                self._set_scalar_field(clone, "Maximum deflection", values["maximum"])
                self._set_scalar_field(clone, "Relative_deflection", values["relative"])
                skeleton = None
                if load_axis:
                    points = coordinates[result["indices"]]
                    skeleton = pycc.ccPointCloud(points[:, 0], points[:, 1], points[:, 2])
                    skeleton.setName(f"Skeleton_of_{source_name}")
                    for property_name in ("GlobalShift", "GlobalScale"):
                        if hasattr(cloud, "get" + property_name) and hasattr(skeleton, "set" + property_name):
                            getattr(skeleton, "set" + property_name)(getattr(cloud, "get" + property_name)())
                row = self._report_row(identifier, source_name, result, mode, denominator, percentage, degree, scale)
                cc.addToDB(clone)
                if skeleton is not None:
                    cc.addToDB(skeleton)
                rows.append(row)
                if result["fit_error"]:
                    errors.append((identifier, "Data was analysed; Fit unavailable: " + result["fit_error"]))
                try:
                    self._plot(result, identifier, directory / f"{identifier}.png")
                except Exception as error:
                    errors.append((identifier, "Plot unavailable: " + str(error)))
            except Exception as error:
                errors.append((identifier, str(error)))
            progress.update(100 * (index + 1) / len(clouds))
        with (directory / "deflection_analysis.txt").open("w", encoding="utf-8", newline="") as stream:
            if rows:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            else:
                stream.write("Identifier\n")
        with (directory / "deflection_analysis_errors.txt").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(("Identifier", "Error"))
            writer.writerows(errors)
        progress.setInfo(f"Analysed {len(rows)} beam(s). Results: {directory}")
        return len(rows), errors, directory


if __name__ == "__main__":
    window = tk.Tk()
    app = GUI_deflection(window)
    app.main_frame(window)
    window.mainloop()
