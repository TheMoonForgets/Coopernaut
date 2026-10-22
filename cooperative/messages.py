"""Causal inference-only sparse-message history (50 ms simulation ticks)."""

from dataclasses import dataclass
import heapq
import itertools
import torch


@dataclass(frozen=True)
class SparseMessage:
    sender_id: object
    capture_tick: int
    arrival_tick: int
    sensor_frame: int
    features: torch.Tensor
    xyz: torch.Tensor
    sender_world: torch.Tensor


def _tick(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer simulation frame" % name)
    return value


def _snapshot(name, value):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError("%s must be a floating-point tensor" % name)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("%s must contain finite values" % name)
    return value.detach().clone()


class CausalMessageBuffer:
    """Snapshots messages; exposes only delivered history for each sender.

    This intentionally detaches payloads and must never be used by training.
    Cached XYZ stays in the sender's capture coordinate system. The caller
    aligns a delivered snapshot using its stored capture pose, exactly once.
    """

    tick_ms = 50
    history_length = 4

    def __init__(self, delay_ms):
        if isinstance(delay_ms, bool) or not isinstance(delay_ms, int) or delay_ms < 0 or delay_ms % self.tick_ms:
            raise ValueError("delay_ms must be a nonnegative integer multiple of 50 ms")
        self.delay_ms = delay_ms
        self.delay_ticks = delay_ms // self.tick_ms
        self.reset()

    def reset(self):
        self._pending = []
        self._history = {}
        self._seen_frames = set()
        self._seen_ticks = set()
        self._sequence = itertools.count()
        self._now_tick = None

    def advance(self, now_tick):
        now_tick = _tick("now_tick", now_tick)
        if self._now_tick is not None and now_tick < self._now_tick:
            raise ValueError("Simulation time moved backwards; reset the message buffer for a new episode")
        self._now_tick = now_tick
        while self._pending and self._pending[0][0] <= now_tick:
            _, _, message = heapq.heappop(self._pending)
            history = self._history.setdefault(message.sender_id, [])
            history.append(message)
            history.sort(key=lambda item: item.capture_tick)
            del history[:-self.history_length]

    def publish(self, sender_id, capture_tick, features, xyz, sender_world, sensor_frame=None):
        """Publish a capture once; return False for an already seen capture.

        Both sensor identity and capture tick are checked, so a repeated packet
        cannot become another historical frame by changing a publish ID.
        """
        if sender_id is None:
            raise ValueError("sender_id must identify a real vehicle")
        try:
            hash(sender_id)
        except TypeError as exc:
            raise ValueError("sender_id must be hashable") from exc
        capture_tick = _tick("capture_tick", capture_tick)
        sensor_frame = capture_tick if sensor_frame is None else _tick("sensor_frame", sensor_frame)
        if self._now_tick is not None and capture_tick > self._now_tick:
            raise ValueError("Cannot publish a capture from the future")
        identity, tick_identity = (sender_id, sensor_frame), (sender_id, capture_tick)
        if identity in self._seen_frames or tick_identity in self._seen_ticks:
            return False
        if not isinstance(features, torch.Tensor) or features.ndim != 2 or features.shape[-1] != 128:
            raise ValueError("features must have shape [N, 128]")
        if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape != (features.shape[0], 3):
            raise ValueError("xyz must have shape [N, 3] matching features")
        if not isinstance(sender_world, torch.Tensor) or sender_world.shape != (4, 4):
            raise ValueError("sender_world must have shape [4, 4]")
        message = SparseMessage(sender_id, capture_tick, capture_tick + self.delay_ticks, sensor_frame,
                                _snapshot("features", features), _snapshot("xyz", xyz),
                                _snapshot("sender_world", sender_world))
        self._seen_frames.add(identity)
        self._seen_ticks.add(tick_identity)
        heapq.heappush(self._pending, (message.arrival_tick, next(self._sequence), message))
        if self._now_tick is not None:
            self.advance(self._now_tick)
        return True

    def consume_history(self, sender_id):
        """Return the latest four delivered captures, oldest first.

        Returns a new list; consumers must treat message tensors as immutable.
        Age in seconds is ``(now_tick - message.capture_tick) * 0.05``.
        """
        if self._now_tick is None:
            return []
        history = list(self._history.get(sender_id, ()))
        if any(not (item.capture_tick <= item.arrival_tick <= self._now_tick) for item in history):
            raise RuntimeError("Internal message causality violation")
        return history
