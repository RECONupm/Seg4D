"""Small compatibility helpers used by the detached Pointcept PTv3 model."""

import torch
from torch import nn


class Dict(dict):
    """Dictionary with attribute access, matching the subset used from addict."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError as error:
            raise AttributeError(name) from error


class DropPath(nn.Module):
    """Per-sample stochastic depth compatible with timm.models.layers.DropPath."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, values):
        if self.drop_prob == 0.0 or not self.training:
            return values
        keep_probability = 1.0 - self.drop_prob
        shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        random_tensor = keep_probability + torch.rand(
            shape, dtype=values.dtype, device=values.device
        )
        random_tensor.floor_()
        return values.div(keep_probability) * random_tensor
