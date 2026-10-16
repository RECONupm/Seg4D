"""Seg4D window for the replaceable Point Transformer V3 console runtime."""

import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd
import pycc


CURRENT_DIRECTORY = Path(__file__).resolve().parent
MAIN_MODULE = CURRENT_DIRECTORY.parent / "main_module"
sys.path.insert(0, str(MAIN_MODULE))

from main import P2p_getdata, check_input, get_point_clouds_name
from main_gui import show_features_window
from execution_logging import (
    append_execution_exception,
    append_execution_result,
    execution_failure_message,
    start_execution_log,
)


RUNTIME_EXECUTABLE = (
    CURRENT_DIRECTORY
    / "point_transformer_v3_runtime"
    / "point_transformer_v3.exe"
)
PROGRESS_FILENAME = "progress.json"
EXECUTION_LOG_FILENAME = "deep_learning_execution.log"


class GUI_dl_v3(tk.Frame):
    """Training and prediction UI that communicates exclusively with the V3 EXE."""

    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)
        self.features2include = []
        self.output_directory = os.getcwd()
        self.process = None
        self.log_file = None
        self.execution_log_path = None
        self.progress_window = None
        self.progress_value = None
        self.progress_message = None
        self.progress_button = None
        self.loss_canvas = None
        self.accuracy_canvas = None
        self.metric_summary = None
        self.metric_history = []
        self.training_epoch_count = 0
        self.metrics_log_offset = 0
        self.metrics_log_remainder = ""
        self.completion_callback = None
        self.temporary_files = []

    @staticmethod
    def _point_cloud_names():
        names = get_point_clouds_name()
        return [names] if isinstance(names, str) else list(names)

    @staticmethod
    def _normalized_cloud(cloud, require_classification):
        frame = P2p_getdata(cloud, False, True, True).copy()
        frame.rename(
            columns={column: column.replace(" ", "_") for column in frame.columns},
            inplace=True,
        )
        if "Classification" not in frame.columns:
            if require_classification:
                raise ValueError(
                    "The training and evaluation clouds need a Classification "
                    "scalar field."
                )
            frame["Classification"] = 0
        frame["Dummy"] = 0
        return frame

    @staticmethod
    def _write_point_table(frame, path):
        # An empty field is ambiguous in a space-delimited file because
        # numpy.loadtxt collapses consecutive whitespace. Keep every column
        # aligned by writing missing values as an explicit numeric NaN token.
        frame.to_csv(
            path,
            sep=" ",
            header=True,
            index=False,
            na_rep="nan",
        )

    def _select_features(self, cloud_name):
        show_features_window(
            self,
            self._point_cloud_names(),
            cloud_name,
            excluded_feature={
                "Classification",
                "Dummy",
                "X",
                "Y",
                "Z",
            },
        )

    @staticmethod
    def _choose_directory(entry):
        directory = filedialog.askdirectory()
        if directory:
            entry.delete(0, tk.END)
            entry.insert(0, directory)

    @staticmethod
    def _choose_file(entry, filetypes):
        filename = filedialog.askopenfilename(filetypes=filetypes)
        if filename:
            entry.delete(0, tk.END)
            entry.insert(0, filename)

    def _runtime_error(self):
        if RUNTIME_EXECUTABLE.is_file():
            return None
        return (
            "Point Transformer V3 is not installed.\n\n"
            f"Expected executable:\n{RUNTIME_EXECUTABLE}\n\n"
            "Build it with F5 → Build Point Transformer V3 executable, or copy "
            "a newer point_transformer_v3_runtime folder into segmentation_methods."
        )

    def _start_process(self, command, output_directory, completion_callback):
        runtime_error = self._runtime_error()
        if runtime_error:
            raise FileNotFoundError(runtime_error)
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        progress_path = output / PROGRESS_FILENAME
        if progress_path.exists():
            progress_path.unlink()
        process_command = [str(RUNTIME_EXECUTABLE), *command]
        self.execution_log_path = Path(
            start_execution_log(
                output,
                EXECUTION_LOG_FILENAME,
                "Point Transformer V3 console runtime",
                command=process_command,
                working_directory=RUNTIME_EXECUTABLE.parent,
                details={"Runtime command": command[0] if command else "unknown"},
            )
        )
        self.log_file = self.execution_log_path.open(
            "a", encoding="utf-8", errors="replace"
        )
        startup_info = subprocess.STARTUPINFO()
        startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        try:
            self.process = subprocess.Popen(
                process_command,
                cwd=str(RUNTIME_EXECUTABLE.parent),
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                startupinfo=startup_info,
            )
        except Exception:
            self.log_file.close()
            self.log_file = None
            raise
        self.completion_callback = completion_callback
        is_training = bool(command and command[0] == "train")
        self.metric_history = []
        self.training_epoch_count = 0
        self.metrics_log_offset = 0
        self.metrics_log_remainder = ""
        if is_training and "--epoch" in command:
            epoch_index = command.index("--epoch") + 1
            self.training_epoch_count = int(command[epoch_index])
        self.progress_window = tk.Toplevel(self.winfo_toplevel())
        self.progress_window.title("Point Transformer V3")
        self.progress_window.transient(self.winfo_toplevel())
        self.progress_window.resizable(False, False)
        self.progress_window.protocol("WM_DELETE_WINDOW", self._cancel_process)
        panel = ttk.Frame(self.progress_window, padding=12)
        panel.pack(fill="both", expand=True)
        self.progress_message = tk.StringVar(value="Starting console runtime...")
        ttk.Label(
            panel,
            textvariable=self.progress_message,
            width=88 if is_training else 58,
        ).pack(
            anchor="w"
        )
        self.progress_value = tk.DoubleVar(value=0)
        ttk.Progressbar(
            panel,
            variable=self.progress_value,
            maximum=100,
            length=700 if is_training else 430,
        ).pack(fill="x", pady=10)

        if is_training:
            chart_panel = ttk.Frame(panel)
            chart_panel.pack(fill="both", expand=True, pady=(0, 8))
            self.loss_canvas = tk.Canvas(
                chart_panel,
                width=340,
                height=220,
                background="white",
                highlightthickness=1,
                highlightbackground="#b8b8b8",
            )
            self.loss_canvas.grid(row=0, column=0, padx=(0, 5))
            self.accuracy_canvas = tk.Canvas(
                chart_panel,
                width=340,
                height=220,
                background="white",
                highlightthickness=1,
                highlightbackground="#b8b8b8",
            )
            self.accuracy_canvas.grid(row=0, column=1, padx=(5, 0))
            self.metric_summary = tk.StringVar(
                value="Metrics will appear after the first epoch."
            )
            ttk.Label(panel, textvariable=self.metric_summary).pack(anchor="w")
            self._redraw_metric_charts()

        self.progress_button = ttk.Button(
            panel, text="Cancel", command=self._cancel_process
        )
        self.progress_button.pack(
            side="right"
        )
        self.progress_window.update_idletasks()
        self.progress_window.grab_set()
        self.progress_window.lift()
        self.progress_window.focus_force()
        self.after(250, lambda: self._poll_process(progress_path))

    @staticmethod
    def _chart_series(history, metric):
        epochs = []
        training = []
        evaluation = []
        for record in history:
            try:
                epoch = int(record["epoch"])
                training_value = float(record["training"][metric])
                evaluation_value = float(record["evaluation"][metric])
            except (KeyError, TypeError, ValueError):
                continue
            epochs.append(epoch)
            training.append(training_value)
            evaluation.append(evaluation_value)
        return epochs, training, evaluation

    def _draw_metric_chart(self, canvas, title, metric, fixed_maximum=None):
        if canvas is None:
            return
        canvas.delete("all")
        width = int(canvas.cget("width"))
        height = int(canvas.cget("height"))
        left, top, right, bottom = 46, 32, 12, 30
        plot_width = width - left - right
        plot_height = height - top - bottom
        canvas.create_text(
            left,
            12,
            text=title,
            anchor="w",
            font=("TkDefaultFont", 10, "bold"),
        )

        epochs, training, evaluation = self._chart_series(
            self.metric_history, metric
        )
        observed = training + evaluation
        maximum = fixed_maximum or max(1.0, max(observed, default=1.0) * 1.1)
        for tick in range(5):
            value = maximum * tick / 4
            y = top + plot_height * (1 - tick / 4)
            canvas.create_line(left, y, width - right, y, fill="#e6e6e6")
            label = f"{value * 100:.0f}%" if fixed_maximum else f"{value:.2f}"
            canvas.create_text(left - 6, y, text=label, anchor="e")
        canvas.create_line(left, top, left, height - bottom, fill="#666666")
        canvas.create_line(
            left, height - bottom, width - right, height - bottom, fill="#666666"
        )

        total_epochs = max(
            self.training_epoch_count,
            max(epochs, default=1),
            1,
        )
        canvas.create_text(left, height - 10, text="1", anchor="center")
        canvas.create_text(
            width - right,
            height - 10,
            text=str(total_epochs),
            anchor="center",
        )
        canvas.create_text(
            left + plot_width / 2,
            height - 10,
            text="Epoch",
            anchor="center",
        )

        colors = (("#2878b5", training), ("#e67e22", evaluation))
        for color, values in colors:
            points = []
            for epoch, value in zip(epochs, values):
                x_ratio = (
                    0
                    if total_epochs == 1
                    else (epoch - 1) / (total_epochs - 1)
                )
                x = left + plot_width * x_ratio
                y = top + plot_height * (
                    1 - min(max(value, 0.0), maximum) / maximum
                )
                points.append((x, y))
            if len(points) > 1:
                flattened = [coordinate for point in points for coordinate in point]
                canvas.create_line(*flattened, fill=color, width=2)
            for x, y in points:
                canvas.create_oval(
                    x - 3,
                    y - 3,
                    x + 3,
                    y + 3,
                    fill=color,
                    outline="",
                )

        legend_x = width - right - 150
        canvas.create_line(legend_x, 13, legend_x + 18, 13, fill="#2878b5", width=2)
        canvas.create_text(legend_x + 23, 13, text="Train", anchor="w")
        canvas.create_line(
            legend_x + 68,
            13,
            legend_x + 86,
            13,
            fill="#e67e22",
            width=2,
        )
        canvas.create_text(legend_x + 91, 13, text="Evaluation", anchor="w")

    def _redraw_metric_charts(self):
        self._draw_metric_chart(self.loss_canvas, "Loss", "loss")
        self._draw_metric_chart(
            self.accuracy_canvas,
            "Accuracy",
            "accuracy",
            fixed_maximum=1.0,
        )
        if self.metric_summary is not None and self.metric_history:
            latest = self.metric_history[-1]
            try:
                self.metric_summary.set(
                    f"Epoch {latest['epoch']}/{self.training_epoch_count}  |  "
                    f"Loss: train {latest['training']['loss']:.4f}, "
                    f"evaluation {latest['evaluation']['loss']:.4f}  |  "
                    f"Accuracy: train {latest['training']['accuracy']:.1%}, "
                    f"evaluation {latest['evaluation']['accuracy']:.1%}"
                )
            except (KeyError, TypeError, ValueError):
                pass

    def _read_metrics_from_execution_log(self):
        if self.execution_log_path is None or self.loss_canvas is None:
            return
        try:
            with self.execution_log_path.open(
                "r", encoding="utf-8", errors="replace"
            ) as log_file:
                log_file.seek(self.metrics_log_offset)
                chunk = log_file.read()
                self.metrics_log_offset = log_file.tell()
        except OSError:
            return
        if not chunk:
            return

        chunk = self.metrics_log_remainder + chunk
        lines = chunk.splitlines(keepends=True)
        self.metrics_log_remainder = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self.metrics_log_remainder = lines.pop()

        records_by_epoch = {
            int(record["epoch"]): record
            for record in self.metric_history
            if isinstance(record, dict) and "epoch" in record
        }
        changed = False
        for line in lines:
            candidate = line.strip()
            if not candidate.startswith("{"):
                continue
            try:
                record = json.loads(candidate)
                epoch = int(record["epoch"])
                float(record["training"]["loss"])
                float(record["training"]["accuracy"])
                float(record["evaluation"]["loss"])
                float(record["evaluation"]["accuracy"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            if records_by_epoch.get(epoch) != record:
                records_by_epoch[epoch] = record
                changed = True
        if changed:
            self.metric_history = [
                records_by_epoch[epoch] for epoch in sorted(records_by_epoch)
            ]
            self._redraw_metric_charts()

    def _close_progress_window(self):
        if self.progress_window and self.progress_window.winfo_exists():
            try:
                self.progress_window.grab_release()
            except tk.TclError:
                pass
            self.progress_window.destroy()
        self.progress_window = None
        self.progress_button = None
        self.loss_canvas = None
        self.accuracy_canvas = None
        self.metric_summary = None

    def _poll_process(self, progress_path):
        if self.process is None:
            return
        self._read_metrics_from_execution_log()
        if progress_path.is_file():
            try:
                with progress_path.open(encoding="utf-8") as progress_file:
                    status = json.load(progress_file)
                self.progress_value.set(float(status.get("percent", 0)))
                self.progress_message.set(status.get("message", "Processing..."))
            except (OSError, ValueError, TypeError):
                pass
        return_code = self.process.poll()
        if return_code is None:
            self.after(250, lambda: self._poll_process(progress_path))
            return

        self.log_file.close()
        self.log_file = None
        log_path = self.execution_log_path
        append_execution_result(log_path, return_code)
        callback = self.completion_callback
        self.completion_callback = None
        self.process = None
        keep_training_charts = self.loss_canvas is not None and return_code == 0
        if not keep_training_charts:
            self._close_progress_window()
        if return_code != 0:
            messagebox.showerror(
                "Point Transformer V3",
                execution_failure_message(
                    "The Point Transformer V3 console runtime failed.",
                    log_path,
                    return_code,
                ),
                parent=self.winfo_toplevel(),
            )
            return
        try:
            callback()
            if keep_training_charts and self.progress_window:
                self.progress_value.set(100)
                self.progress_message.set(
                    "Training completed. Review the final metrics and close this window."
                )
                self.progress_button.configure(
                    text="Close", command=self._close_progress_window
                )
                self.progress_window.protocol(
                    "WM_DELETE_WINDOW", self._close_progress_window
                )
        except Exception as error:
            self._close_progress_window()
            traceback.print_exc()
            append_execution_exception(
                log_path, "postprocessing the Point Transformer V3 result", error
            )
            messagebox.showerror(
                "Point Transformer V3",
                execution_failure_message(str(error), log_path),
                parent=self.winfo_toplevel(),
            )

    def _cancel_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            if self.progress_message:
                self.progress_message.set("Cancelling...")
            return
        self._close_progress_window()

    def _training_files(
        self, training_name, evaluation_name, output_directory
    ):
        names = self._point_cloud_names()
        training = self._normalized_cloud(
            check_input(names, training_name), True
        )
        evaluation = self._normalized_cloud(
            check_input(names, evaluation_name), True
        )
        if list(training.columns) != list(evaluation.columns):
            raise ValueError(
                "Training and evaluation clouds need identical scalar fields."
            )
        labels = training["Classification"].to_numpy(dtype=float)
        evaluation_labels = evaluation["Classification"].to_numpy(dtype=float)
        if (
            not np.isfinite(labels).all()
            or not np.isfinite(evaluation_labels).all()
            or not (labels == labels.round()).all()
            or not (evaluation_labels == evaluation_labels.round()).all()
        ):
            raise ValueError(
                "Training and evaluation Classification values must be finite "
                "integers. The integer class codes do not need to begin at 0 "
                "or be consecutive."
            )
        classes = sorted(set(labels.astype(int).tolist()))
        if len(classes) < 2:
            raise ValueError(
                "The training cloud must contain at least two classes."
            )
        evaluation_classes = set(evaluation_labels.astype(int).tolist())
        if not evaluation_classes.issubset(classes):
            raise ValueError(
                "The evaluation cloud contains classes absent from training."
            )
        candidates = [
            column for column in training.columns if column != "Classification"
        ]
        coordinates = ["X", "Y", "Z"]
        missing_coordinates = [
            coordinate for coordinate in coordinates if coordinate not in candidates
        ]
        if missing_coordinates:
            raise ValueError(
                "The point clouds need X, Y and Z coordinates."
            )
        optional_features = [
            feature.replace(" ", "_")
            for feature in self.features2include
            if feature not in {"Classification", "Dummy", "X", "Y", "Z"}
        ]
        missing = [
            feature for feature in optional_features if feature not in candidates
        ]
        if missing:
            raise ValueError(f"Selected features are missing: {missing}")
        selected_features = coordinates + list(dict.fromkeys(optional_features))
        indices = [str(candidates.index(feature)) for feature in selected_features]
        output = Path(output_directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        training_path = output / "input_point_cloud_training.txt"
        evaluation_path = output / "input_point_cloud_testing.txt"
        features_path = output / "features.txt"
        self._write_point_table(training, training_path)
        self._write_point_table(evaluation, evaluation_path)
        features_path.write_text(",".join(indices), encoding="utf-8")
        self.temporary_files = [training_path, evaluation_path]
        return (
            training_path,
            evaluation_path,
            features_path,
            list(training.columns).index("Classification"),
        )

    def _run_training(
        self,
        training_name,
        evaluation_name,
        training_system,
        batch_size,
        epochs,
        output_directory,
    ):
        self.execution_log_path = None
        try:
            training_system = training_system.strip().lower()
            batch_size = int(batch_size)
            epochs = int(epochs)
            if training_system not in {"base", "extreme", "custom"}:
                raise ValueError("Select Base, Extreme or Custom.")
            if batch_size < 1:
                raise ValueError("The batch size must be at least 1.")
            if epochs < 1:
                raise ValueError("The number of epochs must be at least 1.")
            train_path, test_path, features_path, label_index = self._training_files(
                training_name, evaluation_name, output_directory
            )
            features = features_path.read_text(encoding="utf-8").strip()
            command = [
                "train",
                "--train", str(train_path),
                "--test", str(test_path),
                "--features", features,
                "--labels", str(label_index),
                "--training-system", training_system,
                "--batch-size", str(batch_size),
                "--epoch", str(epochs),
                "--output", str(Path(output_directory).resolve()),
            ]

            def completed():
                for path in self.temporary_files:
                    path.unlink(missing_ok=True)

            self._start_process(command, output_directory, completed)
        except Exception as error:
            if self.execution_log_path is None:
                self.execution_log_path = Path(
                    start_execution_log(
                        output_directory,
                        EXECUTION_LOG_FILENAME,
                        "Point Transformer V3 training preparation",
                    )
                )
            append_execution_exception(
                self.execution_log_path,
                "preparing or starting Point Transformer V3 training",
                error,
            )
            messagebox.showerror(
                "Point Transformer V3",
                execution_failure_message(str(error), self.execution_log_path),
                parent=self.winfo_toplevel(),
            )

    def _run_prediction(
        self,
        model_path,
        feature_path,
        cloud_name,
        maximum_points,
        output_directory,
    ):
        self.execution_log_path = None
        try:
            maximum_points = int(maximum_points)
            model = Path(model_path).expanduser().resolve()
            features_file = Path(feature_path).expanduser().resolve()
            if not model.is_file() or not features_file.is_file():
                raise FileNotFoundError("Select a valid model and feature file.")
            names = self._point_cloud_names()
            frame = self._normalized_cloud(check_input(names, cloud_name), False)
            output = Path(output_directory).expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            input_path = output / "input_point_cloud_prediction.txt"
            result_path = output / "predictions.txt"
            self._write_point_table(frame, input_path)
            self.temporary_files = [input_path]
            features = features_file.read_text(encoding="utf-8").strip()
            command = [
                "predict",
                "--model", str(model),
                "--features", features,
                "--input", str(input_path),
                "--size", str(maximum_points),
                "--output", str(result_path),
            ]

            def completed():
                result = pd.read_csv(result_path, sep=r"\s+")
                cloud = pycc.ccPointCloud(
                    result["X"].to_numpy(),
                    result["Y"].to_numpy(),
                    result["Z"].to_numpy(),
                )
                cloud.setName("Results_from_segmentation")
                prediction_column = (
                    "Predictions"
                    if "Predictions" in result.columns
                    else "prediction"
                )
                scalar_index = cloud.addScalarField(
                    "Classification", result[prediction_column].to_numpy()
                )
                cloud.setCurrentDisplayedScalarField(scalar_index)
                cloud.getScalarField(scalar_index).computeMinAndMax()
                instance = pycc.GetInstance()
                instance.addToDB(cloud)
                instance.updateUI()
                for path in self.temporary_files:
                    path.unlink(missing_ok=True)
                messagebox.showinfo(
                    "Point Transformer V3",
                    "Prediction completed and added to CloudCompare.",
                    parent=self.winfo_toplevel(),
                )

            self._start_process(command, output_directory, completed)
        except Exception as error:
            if self.execution_log_path is None:
                self.execution_log_path = Path(
                    start_execution_log(
                        output_directory,
                        EXECUTION_LOG_FILENAME,
                        "Point Transformer V3 prediction preparation",
                    )
                )
            append_execution_exception(
                self.execution_log_path,
                "preparing or starting Point Transformer V3 prediction",
                error,
            )
            messagebox.showerror(
                "Point Transformer V3",
                execution_failure_message(str(error), self.execution_log_path),
                parent=self.winfo_toplevel(),
            )

    @staticmethod
    def _entry(parent, row, default, width=32):
        entry = ttk.Entry(parent, width=width)
        entry.grid(row=row, column=1, sticky="ew", padx=5, pady=4)
        entry.insert(0, default)
        return entry

    def main_frame(self, root):
        root.title("Point Transformer V3")
        root.resizable(False, False)
        root.attributes("-toolwindow", -1)
        names = self._point_cloud_names()
        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True, padx=8, pady=8)
        training_tab = ttk.Frame(notebook, padding=10)
        prediction_tab = ttk.Frame(notebook, padding=10)
        notebook.add(training_tab, text="Training")
        notebook.add(prediction_tab, text="Prediction")

        training_labels = (
            "Choose point cloud for training:",
            "Choose point cloud for testing:",
            "Select the features to include:",
            "Training system:",
            "Batch size:",
            "Number of epochs:",
            "Choose output directory:",
        )
        for row, text in enumerate(training_labels):
            ttk.Label(training_tab, text=text).grid(
                row=row, column=0, sticky="w", padx=5, pady=4
            )
        training_cloud = ttk.Combobox(
            training_tab, values=names, state="readonly", width=29
        )
        training_cloud.grid(row=0, column=1, sticky="ew", padx=5, pady=4)
        evaluation_cloud = ttk.Combobox(
            training_tab, values=names, state="readonly", width=29
        )
        evaluation_cloud.grid(row=1, column=1, sticky="ew", padx=5, pady=4)
        if names:
            training_cloud.current(0)
            evaluation_cloud.current(min(1, len(names) - 1))
        ttk.Button(
            training_tab,
            text="...",
            command=lambda: self._select_features(training_cloud.get()),
        ).grid(row=2, column=1, sticky="e", padx=5, pady=4)
        training_system = ttk.Combobox(
            training_tab,
            values=("Base", "Extreme", "Custom"),
            state="readonly",
            width=29,
        )
        training_system.grid(row=3, column=1, sticky="ew", padx=5, pady=4)
        training_system.set("Custom")
        training_batch = self._entry(training_tab, 4, "12", 12)
        training_epochs = self._entry(training_tab, 5, "10", 12)
        training_output = self._entry(training_tab, 6, self.output_directory)

        custom_values = {"batch_size": "12", "epochs": "10"}
        preset_values = {
            "Base": {"batch_size": "12", "epochs": "800"},
            "Extreme": {"batch_size": "24", "epochs": "100"},
        }

        def update_training_system(event=None):
            selected = training_system.get()
            if selected == "Custom":
                values = custom_values
                state = "normal"
            else:
                if str(training_batch.cget("state")) == "normal":
                    custom_values["batch_size"] = training_batch.get()
                    custom_values["epochs"] = training_epochs.get()
                values = preset_values[selected]
                state = "disabled"
            for entry, key in (
                (training_batch, "batch_size"),
                (training_epochs, "epochs"),
            ):
                entry.configure(state="normal")
                entry.delete(0, tk.END)
                entry.insert(0, values[key])
                entry.configure(state=state)

        training_system.bind("<<ComboboxSelected>>", update_training_system)
        preset_description = ttk.Label(
            training_tab,
            text=(
                "Custom: lightweight Windows model; epochs and batch size "
                "are editable."
            ),
            foreground="#555555",
        )
        preset_description.grid(
            row=7, column=0, columnspan=3, sticky="w", padx=5, pady=(4, 0)
        )

        def update_preset_description(event=None):
            descriptions = {
                "Base": (
                    "Base: official PTv3 capacity; 800 epochs, global batch 12."
                ),
                "Extreme": (
                    "Extreme adapted: official capacity; 100 epochs, global "
                    "batch 24, single dataset (without PPT)."
                ),
                "Custom": (
                    "Custom: lightweight Windows model; epochs and batch size "
                    "are editable."
                ),
            }
            preset_description.configure(text=descriptions[training_system.get()])

        training_system.bind(
            "<<ComboboxSelected>>", update_preset_description, add="+"
        )
        ttk.Button(
            training_tab,
            text="...",
            command=lambda: self._choose_directory(training_output),
        ).grid(row=6, column=2, padx=5)
        ttk.Button(
            training_tab,
            text="Run",
            command=lambda: self._run_training(
                training_cloud.get(),
                evaluation_cloud.get(),
                training_system.get(),
                training_batch.get(),
                training_epochs.get(),
                training_output.get(),
            ),
        ).grid(row=8, column=1, sticky="e", padx=5, pady=8)
        ttk.Button(training_tab, text="Cancel", command=root.destroy).grid(
            row=8, column=2, padx=5, pady=8
        )

        prediction_labels = (
            "Choose the trained model (.pt model):",
            "Load the feature file:",
            "Choose point cloud for classifying:",
            "Maximum points per GPU batch:",
            "Choose output directory:",
        )
        for row, text in enumerate(prediction_labels):
            ttk.Label(prediction_tab, text=text).grid(
                row=row, column=0, sticky="w", padx=5, pady=4
            )
        model_entry = self._entry(prediction_tab, 0, "")
        feature_entry = self._entry(prediction_tab, 1, "")
        prediction_cloud = ttk.Combobox(
            prediction_tab, values=names, state="readonly", width=29
        )
        prediction_cloud.grid(row=2, column=1, sticky="ew", padx=5, pady=4)
        if names:
            prediction_cloud.current(0)
        prediction_points = self._entry(prediction_tab, 3, "8000", 12)
        prediction_output = self._entry(
            prediction_tab, 4, self.output_directory
        )
        ttk.Button(
            prediction_tab,
            text="...",
            command=lambda: self._choose_file(model_entry, [("Model", "*.pt")]),
        ).grid(row=0, column=2, padx=5)
        ttk.Button(
            prediction_tab,
            text="...",
            command=lambda: self._choose_file(
                feature_entry, [("Feature file", "*.txt")]
            ),
        ).grid(row=1, column=2, padx=5)
        ttk.Button(
            prediction_tab,
            text="...",
            command=lambda: self._choose_directory(prediction_output),
        ).grid(row=4, column=2, padx=5)
        ttk.Button(
            prediction_tab,
            text="Run",
            command=lambda: self._run_prediction(
                model_entry.get(),
                feature_entry.get(),
                prediction_cloud.get(),
                prediction_points.get(),
                prediction_output.get(),
            ),
        ).grid(row=5, column=1, sticky="e", padx=5, pady=8)
        ttk.Button(prediction_tab, text="Cancel", command=root.destroy).grid(
            row=5, column=2, padx=5, pady=8
        )

    def show_frame(self, root):
        self.main_frame(root)
        self.grid(row=1, column=0, pady=10)

    def hide_frame(self):
        self.grid_forget()


if __name__ == "__main__":
    root = tk.Tk()
    GUI_dl_v3(root).main_frame(root)
    root.mainloop()
