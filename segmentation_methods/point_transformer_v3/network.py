"""Semantic-segmentation head around the official detached PTv3 backbone."""

import torch
from torch import nn

from .model import PointTransformerV3
from .training_presets import CUSTOM_MODEL_CONFIG


class PointTransformerV3Segmenter(nn.Module):
    """Point Transformer V3 backbone followed by a per-point classification head."""

    def __init__(self, input_channels, number_of_classes, grid_size, model_config=None):
        super().__init__()
        configuration = dict(CUSTOM_MODEL_CONFIG)
        if model_config:
            configuration.update(model_config)
        self.input_channels = int(input_channels)
        self.number_of_classes = int(number_of_classes)
        self.grid_size = float(grid_size)
        self.model_config = configuration
        self.backbone = PointTransformerV3(
            in_channels=self.input_channels,
            **configuration,
        )
        self.classifier = nn.Linear(
            configuration["dec_channels"][0], self.number_of_classes
        )

    def forward(self, coordinates, features):
        batch = torch.zeros(
            coordinates.shape[0], dtype=torch.long, device=coordinates.device
        )
        point = self.backbone(
            {
                "coord": coordinates,
                "feat": features,
                "batch": batch,
                "grid_size": self.grid_size,
            }
        )
        return self.classifier(point.feat)

    def checkpoint_configuration(self):
        configuration = {}
        for key, value in self.model_config.items():
            configuration[key] = list(value) if isinstance(value, tuple) else value
        return configuration


def build_model_from_checkpoint(checkpoint, device):
    model = PointTransformerV3Segmenter(
        input_channels=checkpoint["input_channels"],
        number_of_classes=checkpoint["number_of_classes"],
        grid_size=checkpoint["grid_size"],
        model_config=checkpoint.get("model_config"),
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)
