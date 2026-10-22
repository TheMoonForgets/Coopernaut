"""Training-only derivative correction for native max-pooling's zero floor."""
import torch


def zero_floor_backward(pooled_features):
    """Keep native forward bits; stop gradients where its output is zero.

    Open3D initializes MAX accumulators at zero, but its backward may select a
    negative input as a maximum. For max(0, candidates), the negative region has
    derivative zero. At exactly zero this selects the valid zero subgradient.
    Call only in the new training path, after the unmodified native operator.
    """
    if not isinstance(pooled_features, torch.Tensor) or not pooled_features.is_floating_point():
        raise ValueError('Native pooled features must be floating point tensors')
    if not bool(torch.isfinite(pooled_features).all()) or bool((pooled_features < 0).any()):
        raise ValueError('Expected finite nonnegative native MAX output with a zero floor')
    return torch.where(pooled_features > 0, pooled_features, pooled_features.detach())
