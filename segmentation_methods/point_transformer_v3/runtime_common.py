"""Shared file, batching and progress helpers for the PTv3 command-line tools."""

import json
import os

import numpy as np
import torch


def read_point_table(path):
    with open(path, "r", encoding="utf-8-sig") as input_file:
        header = input_file.readline().strip().split()
    if not header:
        raise ValueError(f"The point-cloud file has no header: {path}")
    values = np.loadtxt(path, skiprows=1, ndmin=2)
    if values.shape[1] != len(header):
        raise ValueError(
            f"The header has {len(header)} fields but the data has "
            f"{values.shape[1]} columns: {path}"
        )
    return header, values.astype(np.float32, copy=False)


def encode_class_labels(training_values, evaluation_values):
    """Encode arbitrary integer class codes for cross-entropy training."""
    training_values = np.asarray(training_values)
    evaluation_values = np.asarray(evaluation_values)
    if (
        not np.isfinite(training_values).all()
        or not np.isfinite(evaluation_values).all()
        or not np.equal(training_values, np.rint(training_values)).all()
        or not np.equal(evaluation_values, np.rint(evaluation_values)).all()
    ):
        raise ValueError(
            "Training and evaluation Classification values must be finite integers."
        )
    original_training = training_values.astype(np.int64)
    original_evaluation = evaluation_values.astype(np.int64)
    class_values = np.unique(original_training)
    if class_values.size < 2:
        raise ValueError("The training cloud must contain at least two classes.")
    if not np.isin(original_evaluation, class_values).all():
        raise ValueError(
            "The evaluation cloud contains classes absent from training."
        )
    encoded_training = np.searchsorted(class_values, original_training).astype(
        np.int64
    )
    encoded_evaluation = np.searchsorted(
        class_values, original_evaluation
    ).astype(np.int64)
    return encoded_training, encoded_evaluation, class_values


def parse_feature_indices(value):
    if value is None:
        return []
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("Feature indices cannot contain duplicates.")
    return indices


def resolve_feature_columns(column_count, label_column, feature_indices):
    candidate_columns = [
        index for index in range(column_count) if index != label_column
    ]
    if not feature_indices:
        raise ValueError("At least one feature must be selected.")
    invalid = [
        index
        for index in feature_indices
        if index < 0 or index >= len(candidate_columns)
    ]
    if invalid:
        raise ValueError(f"Feature indices outside the valid range: {invalid}")
    return [candidate_columns[index] for index in feature_indices]


def coordinate_columns(header):
    normalized = {name.lower(): index for index, name in enumerate(header)}
    try:
        return [normalized["x"], normalized["y"], normalized["z"]]
    except KeyError as error:
        raise ValueError("The input must contain X, Y and Z columns.") from error


def require_finite_fields(values, field_names, dataset_name):
    """Reject missing or infinite values only in fields used by the model."""
    values = np.asarray(values)
    invalid = ~np.isfinite(values)
    if not invalid.any():
        return
    row_index, column_index = np.argwhere(invalid)[0]
    invalid_count = int(invalid.sum())
    field_name = field_names[column_index]
    raise ValueError(
        f"{dataset_name} contains {invalid_count} missing or infinite value(s) "
        f"in the selected fields. The first is at data row {row_index + 1}, "
        f"field '{field_name}'. Recompute that field or deselect it."
    )


def compute_feature_statistics(values):
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation[standard_deviation < 1e-8] = 1.0
    return mean.astype(np.float32), standard_deviation.astype(np.float32)


def compute_grid_size(coordinates):
    extent = np.ptp(coordinates, axis=0)
    return max(float(extent.max()) / 2048.0, 1e-4)


def spatial_patches(coordinates, maximum_points, shuffle=False):
    maximum_points = int(maximum_points)
    if maximum_points < 32:
        raise ValueError("The maximum points per batch must be at least 32.")
    order = np.lexsort(
        (coordinates[:, 2], coordinates[:, 1], coordinates[:, 0])
    )
    patches = [
        order[start : start + maximum_points]
        for start in range(0, len(order), maximum_points)
    ]
    if len(patches) > 1 and len(patches[-1]) < 32:
        patches[-2] = np.concatenate((patches[-2], patches[-1]))
        patches.pop()
    if shuffle:
        np.random.shuffle(patches)
    return patches


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Point Transformer V3 for Windows requires an NVIDIA GPU and a "
            "CUDA-enabled PyTorch runtime."
        )
    return torch.device("cuda")


def write_progress(output_directory, percent, message):
    os.makedirs(output_directory, exist_ok=True)
    final_path = os.path.join(output_directory, "progress.json")
    temporary_path = final_path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as output_file:
        json.dump({"percent": float(percent), "message": message}, output_file)
    os.replace(temporary_path, final_path)


def tensor(values, device, dtype=torch.float32):
    return torch.as_tensor(values, dtype=dtype, device=device)


def cuda_out_of_memory_message(error, maximum_points):
    return RuntimeError(
        "CUDA ran out of memory. Reduce 'GPU consumption / maximum points per "
        f"batch' below {maximum_points}. Original error: {error}"
    )
