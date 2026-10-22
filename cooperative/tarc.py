"""Temporal sparse-feature compensation; inputs are already in current ego axes.

This module neither transforms nor extrapolates geometry. History belongs to one
sender, is sorted oldest to newest, and contains only messages already delivered.
Experiment choices live in TemporalConfig; there are no fallback hyperparameters.
"""

import math

import torch
from torch import nn


def _soft_match(query_xyz, source_xyz, source_features, radius, weighting,
                temperature=None):
    """Radius-limited distance weighting, independent of sparse point indices."""
    count = query_xyz.shape[0]
    if source_xyz.shape[0] == 0:
        return (source_features.new_zeros((count, source_features.shape[-1])),
                torch.zeros(count, dtype=torch.bool, device=query_xyz.device))
    # Squared distance avoids cdist's device/version-dependent compute modes.
    squared = (query_xyz[:, None, :] - source_xyz[None, :, :]).square().sum(-1)
    candidates = squared <= radius * radius
    supported = candidates.any(-1)
    epsilon = torch.finfo(source_features.dtype).eps
    if weighting == "inverse_distance":
        # Exact coincidences share weight equally, without sqrt(0) gradients.
        coincident = candidates & (squared == 0)
        has_coincident = coincident.any(-1, keepdim=True)
        inverse = squared.clamp_min(epsilon * epsilon).rsqrt()
        weights = torch.where(candidates, inverse, torch.zeros_like(inverse))
        weights = torch.where(has_coincident, coincident.to(weights.dtype), weights)
    elif weighting == "softmax_squared_distance":
        logits = (-squared / temperature).masked_fill(~candidates, float("-inf"))
        maximum = logits.max(-1, keepdim=True)[0]
        maximum = torch.where(supported[:, None], maximum, torch.zeros_like(maximum))
        weights = torch.exp(logits - maximum)
    else:
        raise ValueError("Unknown temporal match_weighting: {}".format(weighting))
    weights = weights / weights.sum(-1, keepdim=True).clamp_min(epsilon)
    return weights.matmul(source_features), supported


def _activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError("temporal.activation must be relu or gelu")


class TARC(nn.Module):
    """Four-frame, 128-to-64, four-head temporal residual compensation.

    Quality uses 66 values: mean context (64), normalized historical/anchor
    feature disagreement (1), and older-frame local-match fraction (1). The
    latter two omit the anchor self-match and invalid frames. With no historical
    support both are zero. Quality does not use a current reference or absolute
    anchor age; RR-Gate accounts for the anchor's actual age separately.
    """

    feature_dim = 128
    context_dim = 64
    max_history = 4

    def __init__(self, cfg):
        super().__init__()
        self.match_config = {
            "radius": cfg.match_radius_m,
            "weighting": cfg.match_weighting,
            "temperature": getattr(cfg, "match_temperature", None),
        }
        self.time_encoding = cfg.time_encoding
        self.feature_projection = nn.Linear(self.feature_dim, self.context_dim)
        if self.time_encoding == "linear":
            self.time_projection = nn.Linear(1, self.context_dim)
        elif self.time_encoding == "sinusoidal":
            # Dimension/frequency formula is the ordinary 64-D sinusoidal code.
            frequency = torch.exp(torch.arange(0, self.context_dim, 2).float()
                                  * (-math.log(10000.0) / self.context_dim))
            self.register_buffer("time_frequency", frequency)
        else:
            raise ValueError("temporal.time_encoding must be linear or sinusoidal")
        self.attention = nn.MultiheadAttention(self.context_dim, 4, dropout=0.0)
        self.residual = nn.Sequential(
            nn.Linear(self.feature_dim + self.context_dim + 1,
                      cfg.residual_hidden_dim),
            _activation(cfg.activation),
            nn.Linear(cfg.residual_hidden_dim, self.feature_dim),
        )
        self.quality_head = nn.Sequential(
            nn.Linear(self.context_dim + 2, cfg.quality_hidden_dim),
            _activation(cfg.activation),
            nn.Linear(cfg.quality_hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _encode_time(self, relative_age):
        if self.time_encoding == "linear":
            return self.time_projection(relative_age[:, None])
        angles = relative_age[:, None] * self.time_frequency[None, :]
        return torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    @staticmethod
    def reliability_loss(output, reference_features, reference_xyz, sigma):
        """Same supervision entry point on the module and at package level."""
        return reliability_loss(output, reference_features, reference_xyz, sigma)

    def _validate(self, features, xyz, ages_s, valid):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [H,N,128]")
        history, count, _ = features.shape
        if history > self.max_history:
            raise ValueError("TARC accepts at most four history frames")
        if xyz.shape != (history, count, 3):
            raise ValueError("xyz must have shape [H,N,3] matching features")
        if ages_s.shape != (history,) or valid.shape != (history,):
            raise ValueError("ages_s and valid must have shape [H]")
        if valid.dtype != torch.bool:
            raise ValueError("valid must be a boolean tensor")
        if not features.is_floating_point() or not xyz.is_floating_point():
            raise ValueError("features and xyz must be floating point tensors")
        if not ages_s.is_floating_point():
            raise ValueError("ages_s must be floating point seconds")
        if not (features.device == xyz.device == ages_s.device == valid.device):
            raise ValueError("all TARC input tensors must use one device")
        if not (features.dtype == xyz.dtype == ages_s.dtype):
            raise ValueError("features, xyz and ages_s must use one floating dtype")
        if bool(valid.any()):
            if count == 0:
                raise ValueError("a valid frame must contain sparse points")
            selected_age = ages_s[valid]
            if not bool(torch.isfinite(selected_age).all()) or bool((selected_age < 0).any()):
                raise ValueError("valid message ages must be finite and nonnegative")
            if bool((selected_age[1:] > selected_age[:-1]).any()):
                raise ValueError("history must be sorted oldest to newest")
            if not bool(torch.isfinite(features[valid]).all()):
                raise ValueError("valid frame features must be finite")
            if not bool(torch.isfinite(xyz[valid]).all()):
                raise ValueError("valid frame coordinates must be finite")

    def forward(self, features, xyz, ages_s, valid):
        """Return compensated features anchored to the newest valid coordinates.

        ``match_support`` is boolean [H,N], preserving input frame order;
        ``context`` is [N,64], ``quality`` and ``valid`` are scalar tensors.
        Invalid padded frames are never read, even if their payload contains NaN.
        A one-frame history returns the exact anchor features, without residual.
        """
        self._validate(features, xyz, ages_s, valid)
        indices = torch.nonzero(valid, as_tuple=False).flatten()
        if indices.numel() == 0:
            # Retain a zero-valued parameter graph for empty-batch quality loss.
            zero = self.quality_head[2].weight.sum().to(features) * 0.0
            return {
                "features": features.new_empty((0, self.feature_dim)),
                "xyz": xyz.new_empty((0, 3)), "quality": zero,
                "valid": valid.new_tensor(False),
                "context": features.new_empty((0, self.context_dim)),
                "match_support": valid.new_zeros((features.shape[0], 0)),
                "match_config": dict(self.match_config),
            }

        anchor_index = int(indices[-1].item())
        anchor = features[anchor_index]
        anchor_xyz = xyz[anchor_index]
        matched, masks = [], []
        for index in range(features.shape[0]):
            if not bool(valid[index]):
                matched.append(torch.zeros_like(anchor))
                masks.append(valid.new_zeros(anchor.shape[0]))
            elif index == anchor_index:
                matched.append(anchor)
                masks.append(valid.new_ones(anchor.shape[0]))
            else:
                feature, mask = _soft_match(anchor_xyz, xyz[index], features[index],
                                           **self.match_config)
                matched.append(feature)
                masks.append(mask)
        matched = torch.stack(matched, dim=0)  # [H,N,128]
        support = torch.stack(masks, dim=0)     # [H,N]
        # Replace invalid ages before encoding; NaN padding must not reach autograd.
        relative_age = torch.where(valid, ages_s - ages_s[anchor_index],
                                   torch.zeros_like(ages_s))
        encoded = self.feature_projection(matched)
        encoded = encoded + self._encode_time(relative_age)[:, None, :]
        query = encoded[anchor_index:anchor_index + 1]
        context, _ = self.attention(query, encoded, encoded,
                                    key_padding_mask=~support.transpose(0, 1),
                                    need_weights=False)
        context = context[0]
        if indices.numel() == 1:
            compensated = anchor
        else:
            age = ages_s[anchor_index].expand(anchor.shape[0], 1)
            compensated = anchor + self.residual(torch.cat((anchor, context, age), -1))

        historical_support = support.clone()
        historical_support[anchor_index] = False
        epsilon = torch.finfo(features.dtype).eps
        disagreement = ((matched - anchor[None]).abs().sum(-1)
                        / (anchor.abs().sum(-1) + epsilon)[None])
        disagreement = disagreement.masked_fill(~historical_support, 0.0)
        supported_count = historical_support.sum().to(features.dtype)
        consistency = disagreement.sum() / supported_count.clamp_min(1.0)
        opportunities = max(int(indices.numel()) - 1, 0) * anchor.shape[0]
        fraction = supported_count / max(opportunities, 1)
        summary = torch.cat((context.mean(0), consistency.reshape(1), fraction.reshape(1)))
        quality = self.quality_head(summary).reshape(())
        return {
            "features": compensated, "xyz": anchor_xyz, "quality": quality,
            "valid": valid.new_tensor(True), "context": context,
            "match_support": support, "match_config": dict(self.match_config),
        }


def reliability_loss(output, reference_features, reference_xyz, sigma):
    """Message quality MSE against a detached normalized feature-error target.

    Current reference data is supervision only. Neither reference tensors nor
    the error-derived target receives a gradient. No correspondence yields a
    graph-connected zero and matched count zero, not a fabricated quality label.
    """
    if isinstance(sigma, bool) or not math.isfinite(float(sigma)) or sigma <= 0:
        raise ValueError("sigma must be explicitly supplied, finite and positive")
    quality = output["quality"]
    zero = quality.sum() * 0.0
    if not bool(output["valid"]):
        return zero, 0
    if reference_features.ndim != 2 or reference_features.shape[-1] != TARC.feature_dim:
        raise ValueError("reference_features must have shape [N,128]")
    if reference_xyz.shape != (reference_features.shape[0], 3):
        raise ValueError("reference_xyz must have shape [N,3]")
    if reference_features.device != output["features"].device or reference_xyz.device != quality.device:
        raise ValueError("reference and output tensors must use one device")
    if reference_features.dtype != output["features"].dtype or reference_xyz.dtype != output["xyz"].dtype:
        raise ValueError("reference and output tensors must use one dtype")
    if not bool(torch.isfinite(reference_features).all()) or not bool(torch.isfinite(reference_xyz).all()):
        raise ValueError("reference features and coordinates must be finite")
    with torch.no_grad():
        reference, matched = _soft_match(output["xyz"].detach(), reference_xyz.detach(),
                                         reference_features.detach(), **output["match_config"])
        count = int(matched.sum().item())
        if count == 0:
            return zero, 0
        epsilon = torch.finfo(reference.dtype).eps
        error = ((output["features"].detach()[matched] - reference[matched]).abs().sum(-1)
                 / (reference[matched].abs().sum(-1) + epsilon)).mean()
        target = torch.exp(-error / sigma)
    return (quality - target).square(), count
