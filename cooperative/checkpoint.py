"""Checkpoint loading with an explicit original-pretrained bootstrap policy."""

from collections.abc import Mapping
from os import PathLike
import torch


def load_checkpoint(model, checkpoint, allow_pretrained=False):
    """Load all existing parameters strictly; optionally allow missing TARC.

    ``allow_pretrained`` is only for beginning temporal training from an
    original Coopernaut checkpoint. Inference should leave it False. A partial
    temporal checkpoint and any unexpected/base-model keys are always errors.
    Returns PyTorch's incompatible-key record (empty, except an allowed full
    TARC initialization when bootstrapping).
    """
    if not isinstance(allow_pretrained, bool):
        raise ValueError("allow_pretrained must be a boolean")
    if isinstance(checkpoint, (str, bytes, PathLike)):
        checkpoint = torch.load(checkpoint, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a state mapping or checkpoint path")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint.state_dict must be a mapping")
    expected = model.state_dict()
    missing = set(expected) - set(state)
    unexpected = set(state) - set(expected)
    tarc_keys = {key for key in expected if key.startswith("tarc.")}
    allowed_missing = tarc_keys if allow_pretrained and tarc_keys and not (tarc_keys & set(state)) else set()
    if unexpected or missing - allowed_missing:
        details = []
        if missing - allowed_missing:
            details.append("missing keys: " + ", ".join(sorted(missing - allowed_missing)))
        if unexpected:
            details.append("unexpected keys: " + ", ".join(sorted(unexpected)))
        raise RuntimeError("Checkpoint is incompatible (" + "; ".join(details) + ")")
    # Check shapes before mutation. load_state_dict(strict=False) does reject
    # mismatches, but may already have copied earlier matching parameters.
    for key, value in state.items():
        if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape:
            raise RuntimeError("Checkpoint tensor shape/type mismatch for %s" % key)
    return model.load_state_dict(state, strict=not bool(missing))
