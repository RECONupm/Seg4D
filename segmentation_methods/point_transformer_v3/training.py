"""Train Point Transformer V3 using Seg4D's space-delimited point-cloud files."""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as functional

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from point_transformer_v3.network import PointTransformerV3Segmenter
    from point_transformer_v3.training_presets import resolve_training_preset
    from point_transformer_v3.runtime_common import (
        compute_feature_statistics,
        compute_grid_size,
        coordinate_columns,
        cuda_out_of_memory_message,
        encode_class_labels,
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
    from .network import PointTransformerV3Segmenter
    from .training_presets import resolve_training_preset
    from .runtime_common import (
        compute_feature_statistics,
        compute_grid_size,
        coordinate_columns,
        cuda_out_of_memory_message,
        encode_class_labels,
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
    parser = argparse.ArgumentParser(description="Train Seg4D Point Transformer V3.")
    parser.add_argument("--train", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--labels", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--training-system",
        choices=("base", "extreme", "custom"),
        default="custom",
    )
    parser.add_argument("--size", type=int)
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args(arguments)


def run_epoch(
    model,
    coordinates,
    features,
    labels,
    maximum_points,
    batch_size,
    device,
    optimizer=None,
    scheduler=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    point_count = 0
    patches = spatial_patches(coordinates, maximum_points, shuffle=training)

    for batch_start in range(0, len(patches), batch_size):
        patch_batch = patches[batch_start : batch_start + batch_size]
        batch_point_count = sum(len(indices) for indices in patch_batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
        for indices in patch_batch:
            coordinates_tensor = tensor(coordinates[indices], device)
            features_tensor = tensor(features[indices], device)
            labels_tensor = tensor(labels[indices], device, dtype=torch.long)
            logits = model(coordinates_tensor, features_tensor)
            loss_sum = functional.cross_entropy(
                logits, labels_tensor, reduction="sum"
            )
            if training:
                (loss_sum / batch_point_count).backward()
            total_loss += float(loss_sum.detach())
            correct += int((logits.argmax(dim=1) == labels_tensor).sum())
            point_count += len(indices)
        if training:
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

    return {
        "loss": total_loss / max(point_count, 1),
        "accuracy": correct / max(point_count, 1),
    }


def self_test():
    device = require_cuda()
    model = PointTransformerV3Segmenter(3, 2, 0.05).to(device)
    model.eval()
    coordinates = torch.rand((128, 3), device=device)
    with torch.no_grad():
        output = model(coordinates, coordinates)
    if output.shape != (128, 2):
        raise RuntimeError(f"Unexpected PTv3 output shape: {output.shape}")
    print("Point Transformer V3 Windows self-test passed.")


def main(command_line_arguments=None):
    arguments = parse_arguments(command_line_arguments)
    preset = resolve_training_preset(
        arguments.training_system,
        epochs=arguments.epoch,
        batch_size=arguments.batch_size,
        maximum_points=arguments.size,
    )
    epoch_count = preset["epochs"]
    batch_size = preset["batch_size"]
    maximum_points = preset["maximum_points"]

    os.makedirs(arguments.output, exist_ok=True)
    write_progress(arguments.output, 2, "Loading point clouds...")
    train_header, train_values = read_point_table(arguments.train)
    test_header, test_values = read_point_table(arguments.test)
    if train_header != test_header:
        raise ValueError("Training and evaluation point clouds need identical fields.")
    if arguments.labels < 0 or arguments.labels >= len(train_header):
        raise ValueError("The Classification column index is invalid.")

    feature_indices = parse_feature_indices(arguments.features)
    feature_columns = resolve_feature_columns(
        len(train_header), arguments.labels, feature_indices
    )
    coordinate_indices = coordinate_columns(train_header)
    train_coordinates = train_values[:, coordinate_indices]
    test_coordinates = test_values[:, coordinate_indices]
    train_features = train_values[:, feature_columns]
    test_features = test_values[:, feature_columns]
    coordinate_names = [train_header[index] for index in coordinate_indices]
    feature_names = [train_header[index] for index in feature_columns]
    require_finite_fields(
        train_coordinates, coordinate_names, "The training point cloud"
    )
    require_finite_fields(
        test_coordinates, coordinate_names, "The evaluation point cloud"
    )
    require_finite_fields(
        train_features, feature_names, "The selected training features"
    )
    require_finite_fields(
        test_features, feature_names, "The selected evaluation features"
    )
    train_labels, test_labels, class_values = encode_class_labels(
        train_values[:, arguments.labels],
        test_values[:, arguments.labels],
    )

    feature_mean, feature_standard_deviation = compute_feature_statistics(
        train_features
    )
    train_features = (
        train_features - feature_mean
    ) / feature_standard_deviation
    test_features = (
        test_features - feature_mean
    ) / feature_standard_deviation
    grid_size = (
        compute_grid_size(train_coordinates)
        if preset["grid_size"] is None
        else preset["grid_size"]
    )

    device = require_cuda()
    model = PointTransformerV3Segmenter(
        input_channels=len(feature_columns),
        number_of_classes=len(class_values),
        grid_size=grid_size,
        model_config=preset["model_config"],
    ).to(device)
    block_learning_rate = preset["block_learning_rate"]
    if block_learning_rate is None:
        parameter_groups = model.parameters()
        maximum_learning_rates = preset["learning_rate"]
    else:
        block_parameters = []
        other_parameters = []
        for parameter_name, parameter in model.named_parameters():
            target = (
                block_parameters if "block" in parameter_name else other_parameters
            )
            target.append(parameter)
        parameter_groups = [
            {"params": other_parameters, "lr": preset["learning_rate"]},
            {"params": block_parameters, "lr": block_learning_rate},
        ]
        maximum_learning_rates = [
            preset["learning_rate"],
            block_learning_rate,
        ]
    optimizer = torch.optim.AdamW(
        parameter_groups,
        lr=preset["learning_rate"],
        weight_decay=preset["weight_decay"],
    )
    training_patch_count = len(
        spatial_patches(train_coordinates, maximum_points, shuffle=False)
    )
    steps_per_epoch = max(math.ceil(training_patch_count / batch_size), 1)
    if preset["scheduler"] == "one-cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=maximum_learning_rates,
            epochs=epoch_count,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.05,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=1000.0,
        )
        scheduler_per_batch = scheduler
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epoch_count
        )
        scheduler_per_batch = None

    history = []
    best_accuracy = -1.0
    best_state = None
    started_at = time.monotonic()
    try:
        for epoch_number in range(1, epoch_count + 1):
            write_progress(
                arguments.output,
                5 + 85 * (epoch_number - 1) / epoch_count,
                f"Training epoch {epoch_number}/{epoch_count}...",
            )
            training_metrics = run_epoch(
                model,
                train_coordinates,
                train_features,
                train_labels,
                maximum_points,
                batch_size,
                device,
                optimizer,
                scheduler_per_batch,
            )
            with torch.no_grad():
                evaluation_metrics = run_epoch(
                    model,
                    test_coordinates,
                    test_features,
                    test_labels,
                    maximum_points,
                    batch_size,
                    device,
                )
            if scheduler_per_batch is None:
                scheduler.step()
            record = {
                "epoch": epoch_number,
                "training": training_metrics,
                "evaluation": evaluation_metrics,
            }
            history.append(record)
            print(json.dumps(record), flush=True)
            if evaluation_metrics["accuracy"] > best_accuracy:
                best_accuracy = evaluation_metrics["accuracy"]
                best_state = {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                }
    except torch.cuda.OutOfMemoryError as error:
        raise cuda_out_of_memory_message(error, maximum_points) from error

    write_progress(arguments.output, 93, "Saving the best PTv3 model...")
    checkpoint = {
        "format": "seg4d-point-transformer-v3",
        "format_version": 1,
        "input_channels": len(feature_columns),
        "number_of_classes": len(class_values),
        "grid_size": grid_size,
        "training_system": preset["name"],
        "training_configuration": {
            key: value
            for key, value in preset.items()
            if key != "model_config"
        },
        "model_config": model.checkpoint_configuration(),
        "model_state": best_state,
        "feature_indices": feature_indices,
        "feature_columns": feature_columns,
        "feature_names": [train_header[index] for index in feature_columns],
        "feature_mean": feature_mean,
        "feature_standard_deviation": feature_standard_deviation,
        "class_values": class_values,
        "history": history,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    torch.save(checkpoint, os.path.join(arguments.output, "model.pt"))
    with open(
        os.path.join(arguments.output, "training_history.json"),
        "w",
        encoding="utf-8",
    ) as history_file:
        json.dump(history, history_file, indent=2)
    write_progress(arguments.output, 100, "Point Transformer V3 training completed.")


if __name__ == "__main__":
    main()
