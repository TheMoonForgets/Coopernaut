"""Explicit per-decision publication traces for reproducible temporal datasets.

Only a successful inference-buffer publication belongs in ``published``. Empty
publication lists still record the actual neighbor slots for that decision.
No capture, neighbor selection, arrival, or log directory is inferred here.
"""
import json
from pathlib import Path
import uuid


def _frame(name, value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("%s must be a nonnegative integer frame" % name)
    return value


def _sender(value, optional=False):
    if optional and value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("sender_id must be a nonnegative integer or nonempty string")
    if isinstance(value, int) and value < 0 or isinstance(value, str) and not value:
        raise ValueError("sender_id must be a nonnegative integer or nonempty string")
    return value


def baseline_publication_record(context, seen):
    """Record the original CPT's observed encoder inputs after its forward.

    The caller supplies a persistent ``seen={}`` and calls this only after a
    successful original forward. A missing historical capture pose cannot
    become a trace publication. Deduplication follows the inference buffer's
    sender/capture and sender/sensor identities. Episode changes reset the
    caller-owned identities. No encoder, tensor operation, or RNG is invoked.
    """
    if not isinstance(seen, dict):
        raise ValueError("baseline publication seen state must be a dictionary")
    episode = context["episode_id"]
    if not isinstance(episode, str) or not episode:
        raise ValueError("episode_id must be a nonempty string")
    now = _frame("decision_tick", context["now_tick"])
    sources = context["slots"]
    if not isinstance(sources, list):
        raise ValueError("context slots must be a list")
    same_episode = seen.get("episode_id") == episode
    captures = set(seen.get("capture_ticks", ())) if same_episode else set()
    frames = set(seen.get("sensor_frames", ())) if same_episode else set()
    slots, published = [], []
    for source in sources:
        sender = _sender(source["sender_id"], optional=True)
        if sender is not None and sender in slots:
            raise ValueError("slots must not repeat a sender_id")
        slots.append(sender)
        if sender is None:
            continue
        capture = _frame("capture_tick", source["capture_tick"])
        sensor_frame = _frame("sensor_frame", source["sensor_frame"])
        if capture != sensor_frame:
            raise ValueError("Schema 1 requires capture_tick == sensor_frame (absolute world frame IDs)")
        if capture > now or sensor_frame > now:
            raise ValueError("Baseline source capture/sensor frame must not be in the future")
        if source.get("sender_world") is None:
            continue
        capture_identity, frame_identity = (sender, capture), (sender, sensor_frame)
        if capture_identity in captures or frame_identity in frames:
            continue
        captures.add(capture_identity)
        frames.add(frame_identity)
        published.append(dict(sender_id=sender, capture_tick=capture,
                              sensor_frame=sensor_frame, publish_tick=now))
    seen.clear()
    seen.update(episode_id=episode, capture_ticks=captures, sensor_frames=frames)
    return dict(schema_version=1, episode_id=episode, decision_tick=now,
                slots=slots, published=published)


class PublicationRecorder:
    """Write one new JSONL trace, flushing the header and every decision.

    ``log_dir`` must be explicitly chosen by the caller. A UUID4 filename is
    generated without consuming Python or Torch RNG, then opened exclusively;
    an existing trace can never be overwritten. ``path`` identifies the file
    for the explicit episode-to-trace training manifest.
    """

    schema_version = 1
    tick_ms = 50

    def __init__(self, log_dir, episode_id, delay_ms):
        if log_dir is None or not str(log_dir).strip():
            raise ValueError("publication log_dir must be explicitly provided")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("episode_id must be a nonempty string")
        _frame("delay_ms", delay_ms)
        if delay_ms % self.tick_ms:
            raise ValueError("delay_ms must be a multiple of 50 ms")
        self.log_dir = Path(log_dir)
        self.episode_id = episode_id
        self.delay_ms = delay_ms
        self._last_tick = None
        self._seen_capture_ticks = set()
        self._seen_sensor_frames = set()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / ("publications-%s.jsonl" % uuid.uuid4().hex)
        self._stream = self.path.open("x", encoding="utf-8", newline="\n")
        try:
            self._append(dict(event="header", schema_version=self.schema_version,
                              episode_id=episode_id, tick_ms=self.tick_ms,
                              delay_ms=delay_ms))
        except BaseException:
            self._stream.close()
            raise

    @property
    def closed(self):
        return self._stream.closed

    def _append(self, row):
        self._stream.write(json.dumps(row, allow_nan=False, sort_keys=True,
                                      separators=(",", ":")) + "\n")
        self._stream.flush()

    def write(self, record):
        """Validate and append a decision; invalid rows leave state unchanged."""
        if self.closed:
            raise ValueError("Publication recorder is closed")
        fields = {"schema_version", "episode_id", "decision_tick", "slots", "published"}
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("Publication record must contain exactly: " + ", ".join(sorted(fields)))
        if type(record["schema_version"]) is not int or record["schema_version"] != self.schema_version:
            raise ValueError("Unsupported publication schema_version")
        if record["episode_id"] != self.episode_id:
            raise ValueError("Publication episode_id does not match the recorder episode")
        now = _frame("decision_tick", record["decision_tick"])
        if self._last_tick is not None and now <= self._last_tick:
            raise ValueError("decision_tick must increase strictly; duplicate decisions are forbidden")
        slots = record["slots"]
        if not isinstance(slots, list):
            raise ValueError("slots must be a list in the actual neighbor order")
        for sender in slots:
            _sender(sender, optional=True)
        real_slots = [sender for sender in slots if sender is not None]
        if len(real_slots) != len(set(real_slots)):
            raise ValueError("slots must not repeat a sender_id")
        if not isinstance(record["published"], list):
            raise ValueError("published must be a list, including when empty")
        captures, frames, published = set(), set(), []
        message_fields = {"sender_id", "capture_tick", "sensor_frame", "publish_tick"}
        for message in record["published"]:
            if not isinstance(message, dict) or set(message) != message_fields:
                raise ValueError("Each published message requires sender_id/capture_tick/sensor_frame/publish_tick")
            sender = _sender(message["sender_id"])
            if sender not in real_slots:
                raise ValueError("Published sender_id must be present in the decision slots")
            capture = _frame("capture_tick", message["capture_tick"])
            sensor_frame = _frame("sensor_frame", message["sensor_frame"])
            publish = _frame("publish_tick", message["publish_tick"])
            if capture != sensor_frame:
                raise ValueError("Schema 1 requires capture_tick == sensor_frame (absolute world frame IDs)")
            if publish != now or capture > publish or sensor_frame > publish:
                raise ValueError("Publication must satisfy capture/sensor_frame <= publish_tick = decision_tick")
            capture_identity, frame_identity = (sender, capture), (sender, sensor_frame)
            if (capture_identity in captures or capture_identity in self._seen_capture_ticks or
                    frame_identity in frames or frame_identity in self._seen_sensor_frames):
                raise ValueError("Duplicate capture/sensor frame is not a new successful publication")
            captures.add(capture_identity)
            frames.add(frame_identity)
            published.append(dict(sender_id=sender, capture_tick=capture,
                                  sensor_frame=sensor_frame, publish_tick=publish))
        normalized = dict(schema_version=self.schema_version, episode_id=self.episode_id,
                          decision_tick=now, slots=list(slots), published=published)
        self._append(normalized)
        self._last_tick = now
        self._seen_capture_ticks.update(captures)
        self._seen_sensor_frames.update(frames)

    def close(self):
        self._stream.close()

    def __enter__(self):
        if self.closed:
            raise ValueError("Publication recorder is closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

