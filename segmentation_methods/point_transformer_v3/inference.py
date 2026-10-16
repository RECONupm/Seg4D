"""Run a trained Seg4D Point Transformer V3 model on a point cloud."""

import argparse
import os
import sys

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from point_transformer_v3.network import build_model_from_checkpoint
    from point_transformer_v3.runtime_common import (
        coordinate_columns,
        cuda_out_of_memory_message,
        parse_feature_indices,
        read_point_table,
        require_finite_fields,
        require_cuda,
        resolve_feature_columns,
        spatial_patches,
        tensor,
        write_progress,
    )
else:
    from .network import build_model_from_checkpoint
    from .runtime_common import (
        coordinate_columns,
        cuda_out_of_memory_message,
        parse_feature_indices,
        read_point_table,
        require_finite_fields,
        require_cuda,
        resolve_feature_columns,
        spatial_patches,
        tensor,
        write_progress,
    )


def parse_arguments(arguments=None):
    parser = argparse.ArgumentParser(description="Run Seg4D Point Transformer V3.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--size", type=int, default=20000)
    return parser.parse_args(arguments)


def main(command_line_arguments=None):
    arguments = parse_arguments(command_line_arguments)
    output_directory = os.path.dirname(os.path.abspath(arguments.output))
    write_progress(output_directory, 3, "Loading PTv3 model and point cloud...")
    device = require_cuda()
    checkpoint = torch.load(arguments.model, map_location="cpu")
    if checkpoint.get("format") != "seg4d-point-transformer-v3":
        raise ValueError("The selected model is not a Seg4D Point Transformer V3 model.")

    header, values = read_point_table(arguments.input)
    label_column = next(
        (
            index
            for index, name in enumerate(header)
            if name.lower() == "classification"
        ),
        None,
    )
    feature_indices = parse_feature_indices(arguments.features)
    feature_columns = resolve_feature_columns(
        len(header), label_column, feature_indices
    )
    if feature_indices != list(checkpoint["feature_indices"]):
        raise ValueError(
            "The selected feature file does not match the features used to train "
            "this Point Transformer V3 model."
        )
    if len(feature_columns) != checkpoint["input_channels"]:
        raise ValueError("The input feature count does not match the trained model.")

    coordinate_indices = coordinate_columns(header)
    coordinates = values[:, coordinate_indices]
    features = values[:, feature_columns]
    require_finite_fields(
        coordinates,
        [header[index] for index in coordinate_indices],
        "The point cloud coordinates",
    )
    require_finite_fields(
        features,
        [header[index] for index in feature_columns],
        "The selected prediction features",
    )
    features = (
        features - np.asarray(checkpoint["feature_mean"])
    ) / np.asarray(checkpoint["feature_standard_deviation"])
    model = build_model_from_checkpoint(checkpoint, device)
    model.eval()
    predictions = np.empty(len(values), dtype=np.int64)
    patches = spatial_patches(coordinates, arguments.size)

    try:
        with torch.no_grad():
            for patch_number, indices in enumerate(patches, start=1):
                write_progress(
                    output_directory,
                    10 + 80 * (patch_number - 1) / len(patches),
                    f"Predicting patch {patch_number}/{len(patches)}...",
                )
                logits = model(
                    tensor(coordinates[indices], device),
                    tensor(features[indices], device),
                )
                class_indices = logits.argmax(dim=1).cpu().numpy()
                predictions[indices] = np.asarray(
                    checkpoint["class_values"], dtype=np.int64
                )[class_indices]
    except torch.cuda.OutOfMemoryError as error:
        raise cuda_out_of_memory_message(error, arguments.size) from error

    output_values = np.column_stack((values, predictions))
    formats = ["%.6f"] * values.shape[1] + ["%d"]
    np.savetxt(
        arguments.output,
        output_values,
        fmt=formats,
        delimiter=" ",
        comments="",
        header=" ".join(header + ["Predictions"]),
    )
    write_progress(output_directory, 100, "Point Transformer V3 prediction completed.")


if __name__ == "__main__":
    main()
