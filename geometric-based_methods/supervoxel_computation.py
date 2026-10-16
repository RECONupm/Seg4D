# -*- coding: utf-8 -*-
"""CloudCompare GUI for VCCS/LCCP supervoxel computation."""

import os
import sys
import traceback

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

import pycc


script_directory = os.path.abspath(__file__)
additional_modules_directory = os.path.join(
    os.path.dirname(os.path.dirname(script_directory)), "main_module"
)
sys.path.insert(0, additional_modules_directory)

from main import P2p_getdata, check_input, get_point_clouds_name
from supervoxel_processing import compute_supervoxels


name_list = get_point_clouds_name()


class GUI_supervoxels(tk.Frame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.parameters = {
            "voxel_resolution": 0.10,
            "seed_resolution": 0.50,
            "normal_radius": 0.30,
            "spatial_weight": 0.40,
            "normal_weight": 0.60,
            "color_weight": 0.00,
            "iterations": 5,
            "minimum_points": 10,
            "convexity_threshold": 0.00,
            "density_neighbors": 16,
        }

    @staticmethod
    def _rgb_from_cloud(cloud):
        colors = cloud.colors()
        if colors is None:
            return None
        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[0] != cloud.size() or colors.shape[1] < 3:
            raise ValueError("The native RGB table has an invalid shape")
        return colors[:, :3].copy()

    @staticmethod
    def _add_or_replace_scalar_field(cloud, name, values):
        values = np.asarray(values, dtype=np.float32).reshape(-1)
        if values.size != cloud.size():
            raise ValueError(f"Scalar field '{name}' does not match the cloud size")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Scalar field '{name}' contains non-finite values")
        existing = cloud.getScalarFieldIndexByName(name)
        if existing >= 0:
            cloud.deleteScalarField(existing)
        index = cloud.addScalarField(name)
        if index < 0:
            raise RuntimeError(f"CloudCompare could not create '{name}'")
        scalar = cloud.getScalarField(index)
        scalar.asArray()[:] = values
        scalar.computeMinAndMax()
        return index

    def main_frame(self, window):
        window.title("Supervoxel computation (VCCS + LCCP)")
        window.resizable(False, False)
        window.attributes('-toolwindow', -1)

        panel = ttk.Frame(window, padding=10)
        panel.pack(fill="both", expand=True)

        ttk.Label(panel, text="Point cloud:").grid(
            row=0, column=0, sticky="w", padx=4, pady=3
        )
        cloud_combo = ttk.Combobox(panel, values=name_list, width=34)
        cloud_combo.grid(row=0, column=1, columnspan=2, sticky="ew", padx=4, pady=3)
        cloud_combo.set("Select the point cloud:")

        fields = (
            ("Voxel resolution (m):", "voxel_resolution"),
            ("Seed / supervoxel resolution (m):", "seed_resolution"),
            ("Normal estimation radius (m):", "normal_radius"),
            ("Spatial weight:", "spatial_weight"),
            ("Normal weight:", "normal_weight"),
            ("RGB weight:", "color_weight"),
            ("VCCS refinement iterations:", "iterations"),
            ("Minimum points per supervoxel:", "minimum_points"),
            ("LCCP convexity threshold:", "convexity_threshold"),
            ("Neighbors for local density:", "density_neighbors"),
        )
        entries = {}
        for row, (label, key) in enumerate(fields, start=1):
            ttk.Label(panel, text=label).grid(
                row=row, column=0, sticky="w", padx=4, pady=3
            )
            entry = ttk.Entry(panel, width=16)
            entry.insert(0, self.parameters[key])
            entry.grid(row=row, column=1, sticky="e", padx=4, pady=3)
            entries[key] = entry

        use_rgb = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            panel,
            text="Use native RGB in VCCS (when available)",
            variable=use_rgb,
        ).grid(row=len(fields) + 1, column=0, columnspan=3, sticky="w", padx=4, pady=5)

        explanation = (
            "Outputs: Supervoxel ID, LCCP Patch ID, curvature, linearity, "
            "planarity, scattering, anisotropy, direction, density and "
            "adjacency fields. Values are shared by all points in a supervoxel."
        )
        ttk.Label(
            panel,
            text=explanation,
            wraplength=520,
            justify="left",
        ).grid(
            row=len(fields) + 2,
            column=0,
            columnspan=3,
            sticky="w",
            padx=4,
            pady=(4, 8),
        )

        def run():
            try:
                parameters = {
                    "voxel_resolution": float(entries["voxel_resolution"].get()),
                    "seed_resolution": float(entries["seed_resolution"].get()),
                    "normal_radius": float(entries["normal_radius"].get()),
                    "spatial_weight": float(entries["spatial_weight"].get()),
                    "normal_weight": float(entries["normal_weight"].get()),
                    "color_weight": float(entries["color_weight"].get()),
                    "iterations": int(entries["iterations"].get()),
                    "minimum_points": int(entries["minimum_points"].get()),
                    "convexity_threshold": float(
                        entries["convexity_threshold"].get()
                    ),
                    "density_neighbors": int(entries["density_neighbors"].get()),
                }
            except ValueError:
                messagebox.showerror(
                    "Invalid parameters",
                    "Check that resolutions and weights are numeric, and that "
                    "iterations, point count and neighbors are integers.",
                    parent=window,
                )
                return

            progress = pycc.ccProgressDialog()
            progress.start()
            progress.setMethodTitle("Preparing supervoxel computation...")
            try:
                cloud = check_input(name_list, cloud_combo.get())
                dataframe = P2p_getdata(cloud, False, True, False)
                points = np.asarray(
                    dataframe[["X", "Y", "Z"]].values, dtype=np.float64
                )
                colors = self._rgb_from_cloud(cloud) if use_rgb.get() else None
                if use_rgb.get() and colors is None:
                    raise ValueError(
                        "RGB use was enabled, but the selected cloud has no native colors."
                    )
                if not use_rgb.get():
                    parameters["color_weight"] = 0.0

                def update(fraction, title):
                    progress.setMethodTitle(title)
                    progress.update(float(fraction) * 100.0)

                result = compute_supervoxels(
                    points,
                    colors=colors,
                    progress_callback=update,
                    **parameters,
                )
                last_index = -1
                for field_name, values in result.point_features.items():
                    last_index = self._add_or_replace_scalar_field(
                        cloud, field_name, values
                    )
                if last_index >= 0:
                    cloud.setCurrentDisplayedScalarField(last_index)
                pycc.GetInstance().updateUI()
                self.parameters.update(parameters)
                print(
                    f"Computed {len(result.centroids)} supervoxels, "
                    f"{len(result.adjacency)} adjacency edges and "
                    f"{len(result.convex_adjacency)} convex LCCP edges."
                )
                progress.stop()
                window.destroy()
            except Exception as error:
                progress.stop()
                messagebox.showerror(
                    "Supervoxel computation error", str(error), parent=window
                )

        button_row = len(fields) + 3
        ttk.Button(panel, text="Run", command=run).grid(
            row=button_row, column=1, sticky="e", padx=4, pady=4
        )
        ttk.Button(panel, text="Cancel", command=window.destroy).grid(
            row=button_row, column=2, sticky="w", padx=4, pady=4
        )


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GUI_supervoxels(root)
        app.main_frame(root)
        root.mainloop()
    except Exception as error:
        print("An error occurred during supervoxel computation:", error)
        traceback.print_exc()
