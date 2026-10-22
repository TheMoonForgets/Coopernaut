"""Causal, sender-keyed four-message samples for TARC fine-tuning.

The existing loader is deliberately not edited.  With temporal mode disabled,
this wrapper delegates both samples and collation to it without changing the
legacy twelve-item tuple.  Native simulator dependencies are imported lazily.
"""

import json
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cooperative.config import TemporalConfig


TICK_SECONDS = 0.05
HISTORY_LENGTH = 4


def _integer_tick(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("{} must be a non-negative integer".format(name))
    return value


def _sender_id(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("sender_id must be a non-negative integer or decimal string")
    if isinstance(value, str) and not value.isdecimal():
        raise ValueError("sender_id must be a non-negative integer or decimal string")
    if int(value) < 0:
        raise ValueError("sender_id must be non-negative")
    return str(int(value))


class PublicationTrace:
    """Index actual online publication decisions; never infer archive traffic."""

    def __init__(self, path):
        self.path = Path(path)
        self.decisions = {}
        self.publications = defaultdict(list)
        seen = {}
        seen_captures = set()
        episode_id = None
        previous_tick = None
        try:
            handle = self.path.open(encoding="utf-8")
        except OSError as exc:
            raise ValueError("Cannot read publication trace {}: {}".format(self.path, exc)) from exc
        with handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("publication row must be an object")
                    if row.get("event") == "header":
                        if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                            raise ValueError("header schema_version must be 1")
                        if type(row.get("tick_ms")) is not int or row["tick_ms"] != 50:
                            raise ValueError("header tick_ms must be 50")
                        identity = row.get("episode_id")
                        if not isinstance(identity, str) or not identity:
                            raise ValueError("header episode_id must be a non-empty string")
                        if episode_id is not None and identity != episode_id:
                            raise ValueError("header episode_id does not match trace episode")
                        episode_id = identity
                        continue
                    if type(row.get("schema_version")) is not int or row["schema_version"] != 1:
                        raise ValueError("schema_version must be 1")
                    identity = row["episode_id"]
                    if not isinstance(identity, str) or not identity:
                        raise ValueError("episode_id must be a non-empty string")
                    if episode_id is None:
                        episode_id = identity
                    if identity != episode_id:
                        raise ValueError("one trace cannot combine multiple episode IDs")
                    tick = _integer_tick(row["decision_tick"], "decision_tick")
                    if previous_tick is not None and tick <= previous_tick:
                        raise ValueError("decision_tick must increase strictly")
                    previous_tick = tick
                    if tick in self.decisions:
                        raise ValueError("duplicate decision_tick {}".format(tick))
                    if not isinstance(row["slots"], list) or not isinstance(row["published"], list):
                        raise ValueError("slots and published must be lists")
                    slots = [None if item is None else _sender_id(item) for item in row["slots"]]
                    present = [item for item in slots if item is not None]
                    if len(present) != len(set(present)):
                        raise ValueError("duplicate sender in slots")
                    self.decisions[tick] = slots
                    for record in row["published"]:
                        sender = _sender_id(record["sender_id"])
                        capture = _integer_tick(record["capture_tick"], "capture_tick")
                        sensor = _integer_tick(record["sensor_frame"], "sensor_frame")
                        publish = _integer_tick(record["publish_tick"], "publish_tick")
                        if publish != tick or capture > publish or sensor > publish:
                            raise ValueError("publication must satisfy capture/sensor_frame <= publish_tick = decision_tick")
                        if capture != sensor:
                            raise ValueError("schema_version 1 requires capture_tick == sensor_frame in absolute world ticks")
                        if sender not in present:
                            raise ValueError("published sender must occur in that decision's slots")
                        key = sender, sensor
                        canonical = {"sender_id": sender, "capture_tick": capture,
                                     "sensor_frame": sensor, "publish_tick": publish}
                        if key in seen or (sender, capture) in seen_captures:
                            raise ValueError("duplicate successful publication capture/sensor identity")
                        seen[key] = canonical
                        seen_captures.add((sender, capture))
                        self.publications[sender].append(canonical)
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError("Invalid publication trace {} line {}: {}".format(
                        self.path, line_number, exc)) from exc
        for sender, records in self.publications.items():
            captures = [record["capture_tick"] for record in records]
            if len(captures) != len(set(captures)):
                raise ValueError("Sender {} declares multiple true frames for one capture in {}".format(
                    sender, self.path))
            records.sort(key=lambda record: record["capture_tick"])
        self.capture_ticks = {sender: [record["capture_tick"] for record in records]
                              for sender, records in self.publications.items()}

    def slots(self, target, count):
        if target not in self.decisions:
            raise ValueError("Missing publication decision for target frame {} in {}".format(target, self.path))
        slots = self.decisions[target]
        if len(slots) != count:
            raise ValueError("Logged slot count {} does not match max_num_neighbors {} at frame {}".format(
                len(slots), count, target))
        return slots

    def history(self, sender, target, delay_ms):
        """Select the four newest eligible records before touching raw data."""
        sender = str(sender)
        records = self.publications.get(sender, [])
        cutoff = target - delay_ms / 50
        position = bisect_right(self.capture_ticks.get(sender, []), cutoff)
        selected = []
        for index in range(position - 1, -1, -1):
            record = records[index]
            if record["publish_tick"] <= target:
                selected.append(record)
                if len(selected) == HISTORY_LENGTH:
                    break
        return list(reversed(selected))


class PublicationManifest:
    def __init__(self, path):
        if path is None:
            raise ValueError("Enabled offline TARC requires config.publication_manifest; archived frames cannot infer publications")
        self.path = Path(path).resolve()
        try:
            with self.path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
        except (OSError, ValueError) as exc:
            raise ValueError("Cannot read publication manifest {}: {}".format(self.path, exc)) from exc
        if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), dict):
            raise ValueError("Publication manifest requires an episodes mapping")
        self.paths, self.traces = {}, {}
        for episode, trace in manifest["episodes"].items():
            if not isinstance(episode, str) or not isinstance(trace, str):
                raise ValueError("Publication manifest episode/trace paths must be strings")
            root = self.path.parent
            episode_path = (root / episode).resolve()
            if episode_path in self.paths:
                raise ValueError("Duplicate resolved episode path in publication manifest")
            self.paths[episode_path] = (root / trace).resolve()

    def for_measurement(self, measurement_path):
        parent = Path(measurement_path).resolve().parent
        episode = parent.parent if parent.name == "measurements" else parent
        if episode not in self.paths:
            raise ValueError("Missing publication manifest mapping for resolved episode {}".format(episode))
        if episode not in self.traces:
            self.traces[episode] = PublicationTrace(self.paths[episode])
        return self.traces[episode]


def causal_history_frames(frames, target_frame, delay_ms, limit=HISTORY_LENGTH):
    """Return distinct delivered capture frames, in oldest-to-newest order.

    The filenames contain CARLA world frame indices, not contiguous dataset
    indices or milliseconds.  Comparing in milliseconds also handles a delay
    that is not a whole simulator tick without rounding it down.
    """
    if delay_ms < 0:
        raise ValueError("delay_ms must be non-negative")
    if limit <= 0:
        return []
    return sorted({int(frame) for frame in frames
                   if int(frame) * 50 + delay_ms <= int(target_frame) * 50})[-limit:]


class TemporalLiDARDataset(Dataset):
    def __init__(self, config, legacy_dataset=None, runtime=None, include_reference=True):
        if not isinstance(include_reference, bool):
            raise ValueError("include_reference must be boolean")
        self.include_reference = include_reference
        self.temporal_config = TemporalConfig.from_config(config)
        self.publications = (PublicationManifest(getattr(config, "publication_manifest", None))
                             if self.temporal_config is not None else None)
        if legacy_dataset is None:
            from utils.point_transformer_loader import LiDARDataset
            legacy_dataset = LiDARDataset(config)
        self.legacy = legacy_dataset
        self._capture_metadata = {}
        self._runtime = runtime
        if self.temporal_config is not None:
            if self.legacy.earlyfusion or self.legacy.shared:
                raise ValueError("TARC requires sparse CPT inputs: shared and earlyfusion must be false")

    def _get_runtime(self):
        if self._runtime is not None:
            return self._runtime
        # Do not retain module objects on the dataset: spawn-based workers must
        # be able to pickle it. Python itself caches these imports per worker.
        from AutoCastSim.AVR import Utils
        from AutoCastSim.AVR.PCProcess import LidarPreprocessor
        return Utils, LidarPreprocessor

    def __len__(self):
        return len(self.legacy)

    def _measurement(self, episode, frame):
        with open(self.legacy.measurements[episode][frame], encoding="utf-8") as handle:
            return json.load(handle)

    def _world_matrix(self, actor, inverse=False):
        utils, _ = self._get_runtime()
        transform = utils.convert_json_to_transform(actor["transform"])
        # Preserve the original loader's ground-plane pose convention.
        transform.location.z = 0
        matrix = utils.TransformMatrix_WorldCoords(transform)
        return np.asarray(matrix.inversematrix() if inverse else matrix.matrix,
                          dtype=np.float32)

    def _cloud(self, episode, sender, frame, actor):
        utils, preprocessor = self._get_runtime()
        paths = self.legacy._lidars[episode].get(str(sender), {})
        path = paths.get(frame)
        if path is None:
            return None
        points = np.load(path).copy()
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            return None
        compensation = 2 * abs(actor["bounding_box"]["extent_z"]) + utils.LidarRoofTopDistance
        points[:, 2] = -points[:, 2] + abs(compensation)
        points = utils.pc_to_car_alignment(points)
        points = preprocessor.Sparse_Quantize(points)
        points = np.unique(points, axis=0)
        if len(points) == 0:
            return None
        return np.asarray(self.legacy._pad_or_truncate(points, self.legacy.npoints),
                          dtype=np.float32)

    def _load_rgb(self, episode, vehicle, frame):
        from PIL import Image
        with Image.open(self.legacy._bev_rgbs[episode][vehicle][frame]) as image:
            return self.legacy.transform(image.copy())

    def _history(self, episode, sender, now, ego_inverse, trace):
        delivered = trace.history(sender, now, self.temporal_config.delay_ms)
        # A declared publication must have its own raw payload and capture pose.
        # Never replace a broken record with another archived sensor frame.
        checked = []
        for record in delivered:
            frame = record["capture_tick"]
            identity = episode, sender, frame
            if identity not in self._capture_metadata:
                if frame not in self.legacy.measurements[episode]:
                    raise ValueError("Published capture {} for sender {} lacks a measurement/capture pose".format(frame, sender))
                actor = self._measurement(episode, frame).get("other_actors", {}).get(str(sender))
                if actor is None or not isinstance(actor.get("transform"), dict) or not actor["transform"]:
                    raise ValueError("Published capture {} for sender {} lacks a capture pose".format(frame, sender))
                path = self.legacy._lidars[episode].get(str(sender), {}).get(frame)
                if path is None or not Path(path).is_file():
                    raise ValueError("Published capture {} for sender {} lacks raw LiDAR".format(frame, sender))
                self._capture_metadata[identity] = actor
            actor = self._capture_metadata[identity]
            checked.append((record, actor))
        result = []
        for record, actor in checked:
            frame = record["capture_tick"]
            cloud = self._cloud(episode, sender, frame, actor)
            if cloud is None:
                raise ValueError("Published capture {} for sender {} has empty/invalid raw LiDAR".format(frame, sender))
            result.append((frame, cloud, ego_inverse @ self._world_matrix(actor), actor))
        return result

    def __getitem__(self, index):
        if self.temporal_config is None:
            return self.legacy[index]
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.legacy.indexer, index, "right") - 1)
        frame = list(self.legacy.measurements[episode])[index - self.legacy.indexer[episode]]
        trace = self.publications.for_measurement(self.legacy.measurements[episode][frame])
        selected = trace.slots(frame, self.legacy.max_num_neighbors)
        measurement = self._measurement(episode, frame)
        vehicle = str(measurement["ego_vehicle_id"])
        parsed = self.legacy.get_measurement(measurement, vehicle)
        if not parsed:
            raise ValueError("Invalid ego measurement at episode {} frame {}".format(episode, frame))
        speed, brake, has_plan, command, control, actors = parsed
        ego_actor = actors[vehicle]
        ego_lidar = self._cloud(episode, vehicle, frame, ego_actor)
        if ego_lidar is None:
            raise ValueError("Missing/empty ego LiDAR at episode {} frame {}".format(episode, frame))
        ego_inverse = self._world_matrix(ego_actor, inverse=True)
        neighbors, points = self.legacy.max_num_neighbors, self.legacy.npoints
        meta = {
            "history_lidar": np.zeros((neighbors, 4, points, 3), np.float32),
            "history_transform": np.tile(np.eye(4, dtype=np.float32), (neighbors, 4, 1, 1)),
            "history_age_s": np.zeros((neighbors, 4), np.float32),
            "history_valid": np.zeros((neighbors, 4), np.bool_),
            "reference_lidar": np.zeros((neighbors, points, 3), np.float32),
            "reference_transform": np.tile(np.eye(4, dtype=np.float32), (neighbors, 1, 1)),
            "reference_valid": np.zeros(neighbors, np.bool_),
            "ego_speed_mps": np.asarray(ego_actor["velocity"], np.float32),
            "history_capture_tick": np.full((neighbors, 4), -1, np.int64),
            "sender_id": np.full(neighbors, -1, np.int64),
        }
        others = np.zeros((neighbors, points, 3), np.float32)
        other_speed = np.zeros(neighbors, np.float32)
        other_transform = np.tile(np.eye(4, dtype=np.float32), (neighbors, 1, 1))
        valid_count = 0
        for slot, sender in enumerate(selected):
            if sender is None:
                continue
            if sender == vehicle:
                raise ValueError("Publication trace cannot select the ego vehicle as a remote sender")
            meta["sender_id"][slot] = int(sender)
            history = self._history(episode, sender, frame, ego_inverse, trace)
            start = HISTORY_LENGTH - len(history)
            for position, (capture, cloud, transform, actor) in enumerate(history, start):
                meta["history_lidar"][slot, position] = cloud
                meta["history_transform"][slot, position] = transform
                meta["history_age_s"][slot, position] = (frame - capture) * TICK_SECONDS
                meta["history_valid"][slot, position] = True
                meta["history_capture_tick"][slot, position] = capture
            if history:
                _, others[slot], other_transform[slot], anchor_actor = history[-1]
                other_speed[slot] = anchor_actor["velocity"] / 30.0
                valid_count += 1
            if self.include_reference and sender in actors:
                reference = self._cloud(episode, sender, frame, actors[sender])
                if reference is not None:
                    meta["reference_lidar"][slot] = reference
                    meta["reference_transform"][slot] = ego_inverse @ self._world_matrix(actors[sender])
                    meta["reference_valid"][slot] = True
        sample = (self._load_rgb(episode, vehicle, frame), ego_lidar, speed, brake,
                  has_plan, command, control, others, other_speed, other_transform,
                  ego_inverse, valid_count)
        return sample + (meta,)

    def collate_fn(self, samples):
        if self.temporal_config is None:
            return self.legacy.collate_fn(samples)
        legacy = self.legacy.collate_fn([sample[:12] for sample in samples])
        metadata = {key: torch.from_numpy(np.stack([sample[12][key] for sample in samples]))
                    for key in samples[0][12]}
        return legacy + (metadata,)


def get_data_loader(config, shuffle=True, drop_last=True, include_reference=True):
    dataset = TemporalLiDARDataset(config, include_reference=include_reference)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      drop_last=drop_last, num_workers=config.num_dataloader_workers,
                      collate_fn=dataset.collate_fn)
