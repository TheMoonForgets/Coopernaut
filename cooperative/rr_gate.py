"""Fixed reliability/risk gate applied before cross-vehicle voxel pooling."""

import torch


def _tensor(value, like=None):
    dtype = like.dtype if like is not None and like.is_floating_point() else torch.get_default_dtype()
    device = None if like is None else like.device
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _finite(name, value):
    if not bool(torch.isfinite(value).all()):
        raise ValueError("%s must contain finite values" % name)


def risk_from_ttc(ttc_s, ego_speed_mps):
    """Return risk and valid-TTC mask; missing/invalid TTC uses m/s proxy."""
    like = ttc_s if isinstance(ttc_s, torch.Tensor) else ego_speed_mps if isinstance(ego_speed_mps, torch.Tensor) else None
    speed = _tensor(ego_speed_mps, like=like)
    _finite("ego_speed_mps", speed)
    if bool((speed < 0).any()):
        raise ValueError("ego_speed_mps must be nonnegative")
    proxy = ((speed - 3.0) / 7.0).clamp(0.0, 1.0)
    if ttc_s is None:
        return proxy, torch.zeros_like(proxy, dtype=torch.bool)
    ttc = _tensor(ttc_s, like=speed)
    ttc, proxy = torch.broadcast_tensors(ttc, proxy)
    valid = torch.isfinite(ttc) & (ttc >= 0)
    # Invalid TTC is intentionally supported. Replace it before arithmetic so
    # NaN/inf cannot contaminate a backward graph through the unused branch.
    safe_ttc = torch.where(valid, ttc, torch.full_like(ttc, 4.0))
    risk = ((4.0 - safe_ttc) / (4.0 - 1.5)).clamp(0.0, 1.0)
    return torch.where(valid, risk, proxy), valid


def compute_gate(quality, age_s, risk):
    like = next((item for item in (quality, age_s, risk) if isinstance(item, torch.Tensor)), None)
    quality, age, risk = torch.broadcast_tensors(_tensor(quality, like), _tensor(age_s, like), _tensor(risk, like))
    for name, value in (("quality", quality), ("age_s", age), ("risk", risk)):
        _finite(name, value)
    if bool(((quality < 0) | (quality > 1)).any()):
        raise ValueError("quality must be in [0, 1]")
    if bool((age < 0).any()):
        raise ValueError("age_s must be nonnegative")
    if bool(((risk < 0) | (risk > 1)).any()):
        raise ValueError("risk must be in [0, 1]")
    reliability = quality * torch.exp(-age / 0.5)
    return 0.2 + 0.8 * reliability * (1.0 - 0.5 * risk * (1.0 - reliability))


def apply_log_gate(features, gate):
    """Apply one gate per message, broadcasting over points and channels."""
    if not isinstance(features, torch.Tensor) or not features.is_floating_point() or features.ndim < 2:
        raise ValueError("features must be a floating-point tensor [..., N, C]")
    _finite("features", features)
    gate = _tensor(gate, features)
    _finite("gate", gate)
    if bool(((gate < 0.2) | (gate > 1)).any()):
        raise ValueError("gate must be in [0.2, 1]")
    # A message-level gate may use the prefix shape [...], or singleton axes.
    prefix = features.shape[:-2]
    if gate.ndim <= len(prefix):
        try:
            # broadcast_tensors is available in the repository's original
            # PyTorch 1.7 environment; broadcast_to was introduced later.
            gate = torch.broadcast_tensors(gate, features.new_empty(prefix))[0]
        except RuntimeError as exc:
            raise ValueError("gate must broadcast to the message dimensions") from exc
        gate = gate[..., None, None]
    else:
        if gate.shape[-2:] != (1, 1) or gate.shape[:-2] != prefix:
            raise ValueError("gate must provide one scalar per message, not per point/channel")
    return features + 0.1 * torch.log(gate)
