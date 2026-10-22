"""Explicit experiment configuration for the optional temporal path."""

from dataclasses import asdict, dataclass
from collections.abc import Mapping
import math


def _get(config, key, default=None):
    value = config.get(key, default) if isinstance(config, Mapping) else getattr(config, key, default)
    if isinstance(value, Mapping) and 'value' in value and set(value).issubset({'value', 'desc'}):
        return value["value"]
    return value


def _number(name, value, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("temporal.%s must be a finite number" % name)
    if positive and value <= 0:
        raise ValueError("temporal.%s must be positive" % name)
    if nonnegative and value < 0:
        raise ValueError("temporal.%s must be nonnegative" % name)
    return value


def _integer(name, value, minimum=1):
    _number(name, value)
    if not isinstance(value, int) or value < minimum:
        raise ValueError("temporal.%s must be an integer >= %d" % (name, minimum))
    return value


@dataclass(frozen=True)
class TemporalConfig:
    delay_ms: int
    match_radius_m: float
    match_weighting: str
    residual_hidden_dim: int
    quality_hidden_dim: int
    activation: str
    time_encoding: str
    rr_gate: bool = False
    match_temperature: float = None
    beta: float = None
    sigma: float = None
    weight_decay: float = None
    epochs: int = None
    batch_size: int = None

    # Architecture and RR constants are fixed by the agreed method.
    feature_dim = 128
    hidden_dim = 64
    num_heads = 4
    history_length = 4

    def __post_init__(self):
        _integer("delay_ms", self.delay_ms, minimum=0)
        if self.delay_ms % 50:
            raise ValueError("temporal.delay_ms must be a multiple of the simulator's 50 ms tick")
        _number("match_radius_m", self.match_radius_m, positive=True)
        if self.match_weighting not in ("inverse_distance", "softmax_squared_distance"):
            raise ValueError("temporal.match_weighting must be inverse_distance or softmax_squared_distance")
        if self.match_weighting == "softmax_squared_distance" and self.match_temperature is None:
            raise ValueError("temporal.match_temperature is required for softmax_squared_distance")
        if self.match_temperature is not None:
            _number("match_temperature", self.match_temperature, positive=True)
        _integer("residual_hidden_dim", self.residual_hidden_dim)
        _integer("quality_hidden_dim", self.quality_hidden_dim)
        if self.activation not in ("relu", "gelu"):
            raise ValueError("temporal.activation must be relu or gelu")
        if self.time_encoding not in ("linear", "sinusoidal"):
            raise ValueError("temporal.time_encoding must be linear or sinusoidal")
        if not isinstance(self.rr_gate, bool):
            raise ValueError("temporal.rr_gate must be a boolean")
        for name in ("beta", "weight_decay"):
            if getattr(self, name) is not None:
                _number(name, getattr(self, name), nonnegative=True)
        if self.sigma is not None:
            _number("sigma", self.sigma, positive=True)
        for name in ("epochs", "batch_size"):
            if getattr(self, name) is not None:
                _integer(name, getattr(self, name))

    @classmethod
    def from_config(cls, config):
        """Return None for old configs; validate all settings when enabled.

        Supports Namespace, plain mappings, and original W&B ``value`` fields.
        Disabled configs are intentionally not inspected beyond ``enabled``.
        """
        temporal = _get(config, "temporal")
        if temporal is None:
            return None
        if not isinstance(temporal, Mapping):
            raise ValueError("config.temporal must be a mapping")
        enabled = _get(temporal, "enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("temporal.enabled must be a boolean")
        if not enabled:
            return None
        required = ("delay_ms", "match_radius_m", "match_weighting", "residual_hidden_dim",
                    "quality_hidden_dim", "activation", "time_encoding")
        missing = [name for name in required if _get(temporal, name) is None]
        if missing:
            raise ValueError("Enabled temporal configuration requires: " + ", ".join(missing))
        allowed = set(cls.__dataclass_fields__) | {"enabled"}
        unknown = set(temporal) - allowed
        if unknown:
            raise ValueError("Unknown or fixed temporal settings: " + ", ".join(sorted(unknown)))
        for key, expected in (("npoints", 2048), ("nblocks", 2), ("input_dim", 3)):
            actual = _get(config, key)
            if isinstance(actual, bool) or actual != expected:
                raise ValueError("Enabled temporal path requires config.%s=%s" % (key, expected))
        if _get(config, "cpt") is not True:
            raise ValueError("Enabled temporal path requires config.cpt=True")
        for name in ("shared", "earlyfusion"):
            if _get(config, name, False):
                raise ValueError("Enabled temporal path requires config.%s=False to prevent fused-current-input leakage" % name)
        values = {key: _get(temporal, key) for key in temporal if key != "enabled"}
        return cls(**values)

    def as_dict(self):
        return dict(enabled=True, **{key: value for key, value in asdict(self).items() if value is not None})

    def require_training(self):
        """Require explicit training hyperparameters without supplying defaults."""
        names = ("beta", "sigma", "weight_decay", "epochs", "batch_size")
        missing = [name for name in names if getattr(self, name) is None]
        if missing:
            raise ValueError("Temporal training requires explicit: " + ", ".join(missing))
        return {name: getattr(self, name) for name in names}
