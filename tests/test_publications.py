"""Synthetic publication trace schema, lifecycle and causality checks."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import torch
from cooperative.publications import PublicationRecorder, baseline_publication_record


def row(tick=5, sender=7, capture=3, episode="episode"):
    return dict(schema_version=1, episode_id=episode, decision_tick=tick,
                slots=[sender, None], published=[dict(sender_id=sender,
                capture_tick=capture, sensor_frame=capture, publish_tick=tick)])


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


class PublicationTests(unittest.TestCase):
    def test_header_and_every_decision_are_flushed_and_context_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 100) as recorder:
                self.assertFalse(recorder.closed)
                self.assertEqual(read_rows(recorder.path), [dict(event="header", schema_version=1,
                    episode_id="episode", tick_ms=50, delay_ms=100)])
                recorder.write(row())
                empty = dict(schema_version=1, episode_id="episode", decision_tick=6,
                             slots=[None, 8], published=[])
                recorder.write(empty)
                self.assertEqual(read_rows(recorder.path)[1:], [row(), empty])
            self.assertTrue(recorder.closed)
            recorder.close()  # Idempotent cleanup.
            with self.assertRaisesRegex(ValueError, "closed"):
                recorder.write(row(7))

    def test_explicit_directory_unique_files_and_no_overwrite_on_collision(self):
        with self.assertRaisesRegex(ValueError, "explicit"):
            PublicationRecorder(None, "episode", 100)
        with self.assertRaisesRegex(ValueError, "explicit"):
            PublicationRecorder("", "episode", 100)
        with tempfile.TemporaryDirectory() as directory:
            nested = Path(directory) / "explicit" / "traces"
            first = PublicationRecorder(nested, "episode", 100)
            second = PublicationRecorder(nested, "episode", 100)
            self.assertNotEqual(first.path, second.path)
            first.close()
            second.close()
            fixed = UUID("00000000-0000-0000-0000-000000000001")
            with patch("cooperative.publications.uuid.uuid4", return_value=fixed):
                with PublicationRecorder(nested, "episode", 100) as recorder:
                    recorder.write(row())
                saved = recorder.path.read_bytes()
                with self.assertRaises(FileExistsError):
                    PublicationRecorder(nested, "episode", 100)
                self.assertEqual(recorder.path.read_bytes(), saved)

    def test_recording_does_not_consume_python_torch_or_cuda_rng(self):
        python_state = random.getstate()
        torch_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 100) as recorder:
                recorder.write(row())
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        if cuda_state is not None:
            self.assertTrue(all(torch.equal(left, right) for left, right in
                                zip(cuda_state, torch.cuda.get_rng_state_all())))

    def test_schema_missing_episode_and_sender_mismatch_are_rejected(self):
        invalid = []
        missing = row()
        del missing["published"]
        invalid.append(missing)
        invalid.extend([dict(row(), extra="unknown"), dict(row(), schema_version=True),
                        dict(row(), schema_version=2), row(episode="other"),
                        dict(row(), slots=[8]), dict(row(), slots=[7, 7]),
                        dict(row(), slots=[None]), dict(row(), slots="7"),
                        dict(row(), slots=[False]), dict(row(), published=None)])
        bad_message = row()
        bad_message["published"][0]["sender_id"] = None
        invalid.append(bad_message)
        bad_message = row()
        del bad_message["published"][0]["sensor_frame"]
        invalid.append(bad_message)
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 100) as recorder:
                for value in invalid:
                    with self.subTest(value=value):
                        with self.assertRaises(ValueError):
                            recorder.write(value)
                        self.assertEqual(len(read_rows(recorder.path)), 1)
                # Invalid decisions do not change the last written tick.
                recorder.write(row(tick=1, capture=0))

    def test_future_fractional_negative_and_inconsistent_frames_fail(self):
        invalid = []
        for tick in (-1, 2.5, True):
            invalid.append(row(tick=tick))
        for name, value in (("capture_tick", 6), ("sensor_frame", 6),
                            ("sensor_frame", 2),
                            ("publish_tick", 4), ("capture_tick", 1.5),
                            ("sensor_frame", -1), ("publish_tick", True)):
            message = row()
            message["published"][0][name] = value
            invalid.append(message)
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 100) as recorder:
                for value in invalid:
                    with self.subTest(value=value):
                        with self.assertRaises(ValueError):
                            recorder.write(value)
                self.assertEqual(len(read_rows(recorder.path)), 1)

    def test_decisions_strictly_increase_and_successful_capture_never_repeated(self):
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 100) as recorder:
                recorder.write(row())
                for duplicate in (row(tick=5, capture=4), row(tick=4, capture=4),
                                  row(tick=6, capture=3)):
                    with self.assertRaisesRegex(ValueError, "duplicate|Duplicate|increase"):
                        recorder.write(duplicate)
                duplicate = row(tick=6, capture=4)
                duplicate["published"].append(dict(duplicate["published"][0]))
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    recorder.write(duplicate)
                recorder.write(row(tick=6, capture=4))
                self.assertEqual([item["decision_tick"] for item in read_rows(recorder.path)[1:]], [5, 6])

    def test_different_senders_may_publish_same_frame_and_empty_slots_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 0) as recorder:
                value = row(tick=0, capture=0)
                value["slots"] = [7, "vehicle-eight"]
                value["published"].append(dict(sender_id="vehicle-eight", capture_tick=0,
                                                sensor_frame=0, publish_tick=0))
                recorder.write(value)
                recorder.write(dict(schema_version=1, episode_id="episode", decision_tick=1,
                                    slots=[], published=[]))
                self.assertEqual(read_rows(recorder.path)[1], value)

    def test_constructor_validates_episode_and_tick_delay(self):
        with tempfile.TemporaryDirectory() as directory:
            for episode, delay in (("", 100), (None, 100), ("episode", 75),
                                   ("episode", -1), ("episode", True)):
                with self.subTest(episode=episode, delay=delay):
                    with self.assertRaises(ValueError):
                        PublicationRecorder(directory, episode, delay)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_baseline_helper_records_real_captures_and_keeps_slot_order_on_dedup(self):
        seen = {}
        sources = [dict(sender_id=7, capture_tick=3, sensor_frame=3, sender_world=object()),
                   dict(sender_id=None),
                   dict(sender_id=8, capture_tick=4, sensor_frame=4, sender_world=None)]
        context = dict(episode_id="episode", now_tick=5, slots=sources)
        first = baseline_publication_record(context, seen)
        self.assertEqual(first["slots"], [7, None, 8])
        self.assertEqual(first["published"], row()["published"])
        context["now_tick"] = 6
        context["slots"] = [sources[2], sources[0], sources[1]]
        second = baseline_publication_record(context, seen)
        self.assertEqual(second["slots"], [8, 7, None])
        self.assertEqual(second["published"], [])
        # The previously unobservable source becomes a true publication only
        # once its capture-compatible pose is available.
        sources[2]["sender_world"] = object()
        context["now_tick"] = 7
        third = baseline_publication_record(context, seen)
        self.assertEqual([item["sender_id"] for item in third["published"]], [8])
        with tempfile.TemporaryDirectory() as directory:
            with PublicationRecorder(directory, "episode", 0) as recorder:
                for record in (first, second, third):
                    recorder.write(record)
                self.assertEqual(len(read_rows(recorder.path)), 4)

    def test_baseline_helper_episode_reset_and_rng_are_independent_of_model(self):
        seen = {}
        context = dict(episode_id="episode", now_tick=0, slots=[
            dict(sender_id=7, capture_tick=0, sensor_frame=0, sender_world=object())])
        python_state, torch_state = random.getstate(), torch.get_rng_state()
        self.assertEqual(len(baseline_publication_record(context, seen)["published"]), 1)
        self.assertEqual(baseline_publication_record(context, seen)["published"], [])
        context["episode_id"] = "next"
        self.assertEqual(len(baseline_publication_record(context, seen)["published"]), 1)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_baseline_helper_invalid_frame_does_not_mutate_seen(self):
        seen = {}
        context = dict(episode_id="episode", now_tick=0, slots=[
            dict(sender_id=7, capture_tick=1, sensor_frame=1, sender_world=object())])
        with self.assertRaisesRegex(ValueError, "future"):
            baseline_publication_record(context, seen)
        self.assertEqual(seen, {})
        context["now_tick"] = 2
        context["slots"][0]["sensor_frame"] = 0
        with self.assertRaisesRegex(ValueError, "capture_tick == sensor_frame"):
            baseline_publication_record(context, seen)
        self.assertEqual(seen, {})


if __name__ == "__main__":
    unittest.main()
