"""Training presets exposed by the Seg4D Point Transformer V3 interface."""

from copy import deepcopy


CUSTOM_MODEL_CONFIG = {
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (1, 1, 2, 2, 1),
    "enc_channels": (16, 32, 64, 128, 256),
    "enc_num_head": (1, 2, 4, 8, 16),
    "enc_patch_size": (128, 128, 128, 128, 128),
    "dec_depths": (1, 1, 1, 1),
    "dec_channels": (32, 32, 64, 128),
    "dec_num_head": (2, 2, 4, 8),
    "dec_patch_size": (128, 128, 128, 128),
    "drop_path": 0.1,
    "enable_flash": False,
    "enable_rpe": False,
    "upcast_attention": True,
    "upcast_softmax": True,
    "cls_mode": False,
}


BASE_MODEL_CONFIG = {
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (2, 2, 2, 6, 2),
    "enc_channels": (32, 64, 128, 256, 512),
    "enc_num_head": (2, 4, 8, 16, 32),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "dec_depths": (2, 2, 2, 2),
    "dec_channels": (64, 64, 128, 256),
    "dec_num_head": (4, 4, 8, 16),
    "dec_patch_size": (1024, 1024, 1024, 1024),
    "drop_path": 0.3,
    # The paper uses FlashAttention. The detached Windows runtime uses the
    # equivalent standard attention path because FlashAttention is unavailable.
    "enable_flash": False,
    "enable_rpe": False,
    "upcast_attention": True,
    "upcast_softmax": True,
    "cls_mode": False,
}


EXTREME_MODEL_CONFIG = {
    "order": ("z", "z-trans", "hilbert", "hilbert-trans"),
    "stride": (2, 2, 2, 2),
    "enc_depths": (3, 3, 3, 6, 3),
    "enc_channels": (48, 96, 192, 384, 512),
    "enc_num_head": (3, 6, 12, 24, 32),
    "enc_patch_size": (1024, 1024, 1024, 1024, 1024),
    "dec_depths": (3, 3, 3, 3),
    "dec_channels": (64, 96, 192, 384),
    "dec_num_head": (4, 6, 12, 24),
    "dec_patch_size": (1024, 1024, 1024, 1024),
    "drop_path": 0.3,
    # Extreme adapted keeps the official PTv3 backbone capacity but omits PPT,
    # PDNorm and FlashAttention so it can train one custom dataset on Windows.
    "enable_flash": False,
    "enable_rpe": False,
    "upcast_attention": True,
    "upcast_softmax": True,
    "cls_mode": False,
}


TRAINING_PRESETS = {
    "custom": {
        "display_name": "Custom",
        "epochs": 10,
        "batch_size": 12,
        "maximum_points": 8000,
        "grid_size": None,
        "learning_rate": 1e-3,
        "block_learning_rate": None,
        "weight_decay": 0.01,
        "scheduler": "cosine",
        "model_config": CUSTOM_MODEL_CONFIG,
    },
    "base": {
        "display_name": "Base",
        "epochs": 800,
        "batch_size": 12,
        "maximum_points": 102400,
        "grid_size": 0.02,
        "learning_rate": 0.006,
        "block_learning_rate": 0.0006,
        "weight_decay": 0.05,
        "scheduler": "one-cycle",
        "model_config": BASE_MODEL_CONFIG,
    },
    "extreme": {
        "display_name": "Extreme",
        "epochs": 100,
        "batch_size": 24,
        "maximum_points": 204800,
        "grid_size": 0.02,
        "learning_rate": 0.005,
        "block_learning_rate": 0.0005,
        "weight_decay": 0.05,
        "scheduler": "one-cycle",
        "model_config": EXTREME_MODEL_CONFIG,
    },
}


def resolve_training_preset(name, epochs=None, batch_size=None, maximum_points=None):
    """Return a validated, independent configuration for a training run."""
    normalized_name = str(name).strip().lower()
    if normalized_name not in TRAINING_PRESETS:
        choices = ", ".join(TRAINING_PRESETS)
        raise ValueError(
            f"Unknown training system '{name}'. Expected one of: {choices}."
        )

    preset = deepcopy(TRAINING_PRESETS[normalized_name])
    preset["name"] = normalized_name
    if normalized_name == "custom":
        if epochs is not None:
            preset["epochs"] = int(epochs)
        if batch_size is not None:
            preset["batch_size"] = int(batch_size)
        if maximum_points is not None:
            preset["maximum_points"] = int(maximum_points)

    if preset["epochs"] < 1:
        raise ValueError("The number of epochs must be at least 1.")
    if preset["batch_size"] < 1:
        raise ValueError("The batch size must be at least 1.")
    if preset["maximum_points"] < 32:
        raise ValueError("The maximum points per spatial patch must be at least 32.")
    return preset
