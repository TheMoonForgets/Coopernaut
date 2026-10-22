"""Synthetic-only loader tests; no simulator or dataset download is needed."""

from collections import defaultdict
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from utils.temporal_point_transformer_loader import TemporalLiDARDataset, causal_history_frames


def config(enabled=True, delay_ms=100):
    # These values are tiny synthetic fixture choices, not experiment defaults.
    return SimpleNamespace(npoints=2048, nblocks=2, input_dim=3, cpt=True,
                           temporal={"enabled": enabled, "delay_ms": delay_ms,
                                     "match_radius_m": 1.0, "match_weighting": "inverse_distance",
                                     "residual_hidden_dim": 8, "quality_hidden_dim": 8,
                                     "activation": "relu", "time_encoding": "linear"})


class FakeUtils:
    LidarRoofTopDistance = 0.0

    @staticmethod
    def pc_to_car_alignment(points):
        return points

    @staticmethod
    def convert_json_to_transform(values):
        return SimpleNamespace(location=SimpleNamespace(**values))

    @staticmethod
    def TransformMatrix_WorldCoords(transform):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = [transform.location.x, transform.location.y, transform.location.z]
        return SimpleNamespace(matrix=matrix, inversematrix=lambda: np.linalg.inv(matrix))


class FakePreprocessor:
    Sparse_Quantize = staticmethod(lambda points: points)


class FakeLegacy:
    earlyfusion = False
    shared = False
    max_num_neighbors = 1
    npoints = 2

    def __init__(self, root, episodes=1):
        self.measurements, self._lidars = [], []
        self.lengths = []
        self.sentinel = tuple(object() for _ in range(12))
        for episode in range(episodes):
            measurement_paths, clouds = {}, defaultdict(dict)
            folder = Path(root) / str(episode)
            folder.mkdir()
            for frame in range(10, 16):
                actors = {
                    "1": self.actor(frame, 12.0),
                    "2": self.actor(2 * frame, 7.5),
                }
                measurement = {"ego_vehicle_id": "1", "other_actors": actors,
                               "control": [0.2, 0.0, -0.1]}
                path = folder / "{}.json".format(frame)
                path.write_text(json.dumps(measurement), encoding="utf-8")
                measurement_paths[frame] = str(path)
                for sender in actors:
                    cloud = np.array([[frame + episode * 100, 1, -1],
                                      [frame + episode * 100, 2, -2]], np.float32)
                    cloud_path = folder / "{}_{}.npy".format(sender, frame)
                    np.save(cloud_path, cloud)
                    clouds[sender][frame] = str(cloud_path)
            self.measurements.append(measurement_paths)
            self._lidars.append(clouds)
            self.lengths.append(6)
        self.indexer = np.cumsum([0] + self.lengths)

    @staticmethod
    def actor(x, speed):
        return {"transform": {"x": x, "y": 0, "z": 9},
                "bounding_box": {"extent_z": 0}, "velocity": speed}

    def __len__(self):
        return sum(self.lengths)

    def __getitem__(self, index):
        return self.sentinel

    @staticmethod
    def get_measurement(measurement, vehicle):
        return (measurement["other_actors"][vehicle]["velocity"] / 30.0,
                0.0, True, 3, measurement["control"], measurement["other_actors"])

    @staticmethod
    def _cal_dist(first, second):
        return np.linalg.norm([first[axis] - second[axis] for axis in ("x", "y", "z")])

    @staticmethod
    def _pad_or_truncate(points, count):
        return points[:count]

    @staticmethod
    def collate_fn(samples):
        return tuple(torch.as_tensor(np.stack(values)) for values in zip(*samples))


class TestDataset(TemporalLiDARDataset):
    def _load_rgb(self, episode, vehicle, frame):
        return np.zeros((3, 2, 2), np.float32)


class TemporalLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.legacy = FakeLegacy(self.temporary.name, episodes=2)
        self.manifest = Path(self.temporary.name) / "publications.json"
        self.manifest.write_text(json.dumps({"episodes": {
            str(episode): "trace-{}.jsonl".format(episode) for episode in range(2)}}), encoding="utf-8")
        for episode in range(2):
            self.write_trace(episode)

    def trace_path(self, episode):
        return Path(self.temporary.name) / "trace-{}.jsonl".format(episode)

    def write_trace(self, episode, published_frames=None, slots=None, omit=()):
        if published_frames is None:
            published_frames = range(10, 16)
        if slots is None:
            slots = [2]
        rows = [{"event": "header", "schema_version": 1,
                 "tick_ms": 50, "episode_id": str(episode)}]
        for frame in range(10, 16):
            if frame in omit:
                continue
            rows.append({"schema_version": 1, "episode_id": str(episode),
                         "decision_tick": frame, "slots": slots,
                         "published": [{"sender_id": 2, "capture_tick": frame,
                                        "sensor_frame": frame, "publish_tick": frame}]
                         if frame in published_frames else []})
        self.trace_path(episode).write_text("\n".join(json.dumps(row) for row in rows) + "\n",
                                          encoding="utf-8")

    def edit_trace(self, episode, edit):
        path = self.trace_path(episode)
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        edit(rows)
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    def dataset(self, include_reference=True, **kwargs):
        settings = config(**kwargs)
        settings.publication_manifest = str(self.manifest)
        return TestDataset(settings, legacy_dataset=self.legacy,
                           runtime=(FakeUtils, FakePreprocessor),
                           include_reference=include_reference)

    def test_disabled_delegates_exact_legacy_sample_and_collation(self):
        dataset = self.dataset(enabled=False)
        self.assertIs(dataset[0], self.legacy.sentinel)
        marker = object()
        self.legacy.collate_fn = lambda samples: marker
        self.assertIs(dataset.collate_fn([dataset[0]]), marker)

    def test_causality_and_true_frame_age(self):
        sample = self.dataset()[5]  # target world frame 15
        self.assertEqual(len(sample), 13)
        metadata = sample[12]
        np.testing.assert_array_equal(metadata["history_capture_tick"], [[10, 11, 12, 13]])
        np.testing.assert_allclose(metadata["history_age_s"], [[0.25, 0.2, 0.15, 0.1]])
        np.testing.assert_array_equal(metadata["history_valid"], [[True] * 4])
        np.testing.assert_allclose(metadata["history_transform"][0, :, 0, 3], [5, 7, 9, 11])
        np.testing.assert_allclose(metadata["history_transform"][0, :, 2, 3], 0)
        self.assertEqual(float(metadata["ego_speed_mps"]), 12.0)
        self.assertEqual(metadata["sender_id"].tolist(), [2])
        self.assertEqual(sample[11], 1)
        # The original offline flip-Z convention remains in use.
        np.testing.assert_array_equal(metadata["history_lidar"][0, 0, :, 2], [1, 2])
        self.assertEqual(float(metadata["reference_lidar"][0, 0, 0]), 15.0)

    def test_absent_frames_are_not_duplicated(self):
        self.write_trace(0, published_frames=[10, 13, 14, 15])
        missing = Path(self.legacy.measurements[0][12])
        measurement = json.loads(missing.read_text(encoding="utf-8"))
        del measurement["other_actors"]["2"]
        missing.write_text(json.dumps(measurement), encoding="utf-8")
        del self.legacy._lidars[0]["2"][11]
        metadata = self.dataset()[5][12]
        np.testing.assert_array_equal(metadata["history_capture_tick"], [[-1, -1, 10, 13]])
        np.testing.assert_array_equal(metadata["history_valid"], [[False, False, True, True]])
        np.testing.assert_array_equal(metadata["history_lidar"][0, :2], 0)

    def test_sparse_actual_publications_do_not_use_continuous_archive(self):
        self.write_trace(0, published_frames=[10, 12])
        metadata = self.dataset()[5][12]
        np.testing.assert_array_equal(metadata["history_capture_tick"], [[-1, -1, 10, 12]])
        np.testing.assert_array_equal(metadata["history_valid"], [[False, False, True, True]])
        np.testing.assert_array_equal(metadata["history_lidar"][0, 2:, 0, 0], [10, 12])
        self.assertTrue(Path(self.legacy._lidars[0]["2"][13]).is_file())

    def test_logged_slot_order_and_nulls_replace_fresh_selection(self):
        self.legacy.max_num_neighbors = 2
        self.write_trace(0, slots=[None, 2])
        metadata = self.dataset()[5][12]
        self.assertEqual(metadata["sender_id"].tolist(), [-1, 2])
        self.assertFalse(metadata["history_valid"][0].any())
        self.assertTrue(metadata["history_valid"][1].all())

    def test_delayed_publication_cannot_expose_already_old_archived_frame(self):
        self.write_trace(0, published_frames=[])

        def edit(rows):
            rows[5]["published"] = [{"sender_id": 2, "capture_tick": 10,
                                      "sensor_frame": 10, "publish_tick": 14}]

        self.edit_trace(0, edit)
        dataset = self.dataset()
        self.assertFalse(dataset[3][12]["history_valid"].any())  # frame 13
        np.testing.assert_array_equal(dataset[5][12]["history_capture_tick"], [[-1, -1, -1, 10]])

    def test_repeated_successful_publication_is_rejected(self):
        self.write_trace(0, published_frames=[10])

        def edit(rows):
            rows[2]["published"] = [{"sender_id": 2, "capture_tick": 10,
                                      "sensor_frame": 10, "publish_tick": 11}]

        self.edit_trace(0, edit)
        with self.assertRaisesRegex(ValueError, "duplicate successful publication"):
            self.dataset()[5]

    def test_future_decision_cannot_backdate_a_publication(self):
        self.write_trace(0, published_frames=[])

        def edit(rows):
            rows[-1]["published"] = [{"sender_id": 2, "capture_tick": 10,
                                       "sensor_frame": 10, "publish_tick": 10}]

        self.edit_trace(0, edit)
        with self.assertRaisesRegex(ValueError, "publish_tick = decision_tick"):
            self.dataset()[3]

    def test_future_sensor_frame_is_rejected(self):
        def edit(rows):
            rows[1]["published"][0]["sensor_frame"] = 99

        self.edit_trace(0, edit)
        with self.assertRaisesRegex(ValueError, "sensor_frame"):
            self.dataset()[5]

    def test_schema_one_capture_and_sensor_world_ticks_must_match(self):
        def edit(rows):
            rows[-1]["published"][0]["capture_tick"] = 14

        self.edit_trace(0, edit)
        with self.assertRaisesRegex(ValueError, "capture_tick == sensor_frame"):
            self.dataset()[5]

    def test_header_tick_schema_and_episode_must_match(self):
        for key, value, expected in (("tick_ms", 100, "tick_ms"),
                                     ("schema_version", True, "schema_version"),
                                     ("episode_id", "wrong-episode", "multiple episode")):
            self.write_trace(0)
            self.edit_trace(0, lambda rows: rows[0].update({key: value}))
            with self.assertRaisesRegex(ValueError, expected):
                self.dataset()[5]

    def test_decision_ticks_must_increase(self):
        def edit(rows):
            rows[2], rows[3] = rows[3], rows[2]

        self.edit_trace(0, edit)
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            self.dataset()[5]

    def test_manifest_is_required_only_for_enabled_loader(self):
        with self.assertRaisesRegex(ValueError, "requires config.publication_manifest"):
            TestDataset(config(), legacy_dataset=self.legacy, runtime=(FakeUtils, FakePreprocessor))
        self.manifest.unlink()
        self.assertIs(self.dataset(enabled=False)[0], self.legacy.sentinel)
        with self.assertRaisesRegex(ValueError, "Cannot read publication manifest"):
            self.dataset()

    def test_missing_manifest_episode_mapping_is_an_error(self):
        self.manifest.write_text(json.dumps({"episodes": {}}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Missing publication manifest mapping"):
            self.dataset()[5]

    def test_missing_target_decision_or_trace_is_an_error(self):
        self.write_trace(0, omit=[15])
        with self.assertRaisesRegex(ValueError, "Missing publication decision.*15"):
            self.dataset()[5]
        self.trace_path(0).unlink()
        with self.assertRaisesRegex(ValueError, "Cannot read publication trace"):
            self.dataset()[5]

    def test_declared_missing_raw_payload_is_an_error(self):
        del self.legacy._lidars[0]["2"][12]
        with self.assertRaisesRegex(ValueError, "Published capture 12.*lacks raw LiDAR"):
            self.dataset()[5]

    def test_irrelevant_old_publication_payload_is_not_read(self):
        del self.legacy._lidars[0]["2"][10]
        del self.legacy._lidars[0]["2"][11]
        metadata = self.dataset(delay_ms=0)[5][12]
        np.testing.assert_array_equal(metadata["history_capture_tick"], [[12, 13, 14, 15]])

    def test_declared_missing_capture_pose_is_an_error(self):
        path = Path(self.legacy.measurements[0][12])
        measurement = json.loads(path.read_text(encoding="utf-8"))
        del measurement["other_actors"]["2"]
        path.write_text(json.dumps(measurement), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Published capture 12.*lacks a capture pose"):
            self.dataset()[5]

    def test_measurements_subdirectory_uses_exact_episode_mapping(self):
        folder = Path(self.temporary.name) / "0" / "measurements"
        folder.mkdir()
        for frame, old in list(self.legacy.measurements[0].items()):
            new = folder / Path(old).name
            new.write_bytes(Path(old).read_bytes())
            self.legacy.measurements[0][frame] = str(new)
        self.assertTrue(self.dataset()[5][12]["history_valid"].all())

    def test_histories_do_not_cross_episode(self):
        metadata = self.dataset()[11][12]
        self.assertEqual(metadata["history_lidar"][0, 0, 0, 0], 110)
        np.testing.assert_array_equal(metadata["history_capture_tick"], [[10, 11, 12, 13]])

    def test_undelivered_history_is_all_invalid(self):
        sample = self.dataset(delay_ms=500)[0]
        self.assertFalse(sample[12]["history_valid"].any())
        self.assertTrue(sample[12]["reference_valid"][0])
        self.assertEqual(sample[11], 0)
        np.testing.assert_array_equal(sample[7], 0)

    def test_enabled_collates_thirteenth_tensor_dictionary(self):
        dataset = self.dataset()
        batch = dataset.collate_fn([dataset[4], dataset[5]])
        self.assertEqual(len(batch), 13)
        self.assertEqual(batch[12]["history_lidar"].shape, (2, 1, 4, 2, 3))
        self.assertEqual(batch[12]["history_transform"].shape, (2, 1, 4, 4, 4))
        self.assertEqual(batch[12]["history_valid"].dtype, torch.bool)
        self.assertEqual(batch[12]["ego_speed_mps"].shape, (2,))

    def test_validation_does_not_read_or_sample_current_remote_reference(self):
        dataset = self.dataset(include_reference=False)
        read = []
        original = dataset._cloud

        def traced(episode, sender, frame, actor):
            read.append((sender, frame))
            return original(episode, sender, frame, actor)

        dataset._cloud = traced
        metadata = dataset[5][12]
        self.assertFalse(metadata["reference_valid"].any())
        self.assertNotIn(("2", 15), read)
        self.assertTrue(all(frame <= 13 for sender, frame in read if sender == "2"))

    def test_shared_or_earlyfusion_rejected_only_when_enabled(self):
        for flag in ("shared", "earlyfusion"):
            setattr(self.legacy, flag, True)
            with self.assertRaisesRegex(ValueError, "sparse CPT"):
                self.dataset()
            self.assertIs(self.dataset(enabled=False)[0], self.legacy.sentinel)
            setattr(self.legacy, flag, False)

    def test_world_frames_not_dataset_indices_or_duplicates(self):
        self.assertEqual(causal_history_frames([100, 100, 104, 107, 110], 110, 100),
                         [100, 104, 107])
        self.assertEqual(causal_history_frames([109], 110, 51), [])


if __name__ == "__main__":
    unittest.main()
