"""Capture-pose bookkeeping tests; no CARLA or simulator mocks are used."""
import unittest
import torch
from cooperative.online import OnlineContextBuilder


def pose(x):
    value = torch.eye(4)
    value[0, 3] = x
    return value


class OnlineContextTests(unittest.TestCase):
    def setUp(self):
        self.builder = OnlineContextBuilder()

    def build(self, tick, poses, slots, episode="episode", ttc=None):
        return self.builder.build(episode, tick, torch.eye(4), poses, slots,
                                  ego_speed_mps=6.5, ttc_s=ttc)

    def test_capture_pose_is_actual_historical_snapshot_not_current_pose(self):
        observed = pose(10)
        first = self.build(100, {7: observed}, [{"sender_id": 7, "sensor_frame": 99}])
        self.assertIsNone(first["slots"][0]["sender_world"])
        self.assertFalse(first["slots"][0]["capture_pose_available"])
        observed[0, 3] = 999
        second = self.build(101, {7: pose(20)}, [{"sender_id": 7, "sensor_frame": 100}])
        source = second["slots"][0]
        self.assertEqual(source["capture_tick"], 100)
        self.assertEqual(source["sensor_frame"], 100)
        self.assertTrue(source["capture_pose_available"])
        self.assertEqual(source["sender_world"][0, 3].item(), 10)
        self.assertEqual(second["now_tick"], 101)
        self.assertEqual(second["ego_speed_mps"], 6.5)

    def test_slot_permutation_uses_sender_identity_and_keeps_ttc_order(self):
        self.build(10, {1: pose(1), 2: pose(2)}, [])
        slots = [{"sender_id": 2, "sensor_frame": 10}, {"sender_id": None},
                 {"sender_id": 1, "sensor_frame": 10}]
        result = self.build(11, {1: pose(11), 2: pose(12)}, slots,
                            ttc=[1.5, float("nan"), 4.0])
        self.assertEqual([item["sender_id"] for item in result["slots"]], [2, None, 1])
        self.assertEqual(result["slots"][0]["sender_world"][0, 3].item(), 2)
        self.assertEqual(result["slots"][2]["sender_world"][0, 3].item(), 1)
        self.assertEqual(result["ttc_s"][0], 1.5)

    def test_unobserved_sender_or_frame_never_invents_capture_pose(self):
        self.build(10, {1: pose(1)}, [])
        result = self.build(12, {1: pose(12), 2: pose(22)}, [
            {"sender_id": 2, "sensor_frame": 10},
            {"sender_id": 1, "sensor_frame": 11}])
        self.assertTrue(all(item["sender_world"] is None for item in result["slots"]))

    def test_future_fractional_and_negative_frames_fail(self):
        for frame in (11, 9.5, -1):
            with self.subTest(frame=frame):
                with self.assertRaisesRegex(ValueError, "sensor frame"):
                    self.build(10, {1: pose(1)}, [{"sender_id": 1, "sensor_frame": frame}])
        with self.assertRaisesRegex(ValueError, "snapshot frame"):
            self.build(1.5, {}, [])

    def test_episode_change_and_explicit_reset_drop_old_pose_history(self):
        self.build(10, {1: pose(1)}, [])
        with self.assertRaisesRegex(ValueError, "backwards"):
            self.build(1, {}, [])
        result = self.build(12, {1: pose(2)}, [{"sender_id": 1, "sensor_frame": 10}], episode="next")
        self.assertIsNone(result["slots"][0]["sender_world"])
        self.builder.reset()
        result = self.build(1, {}, [{"sender_id": 1, "sensor_frame": 0}])
        self.assertIsNone(result["slots"][0]["sender_world"])

    def test_invalid_pose_speed_and_ttc_shape_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "pose"):
            self.build(1, {1: torch.zeros(3, 3)}, [])
        with self.assertRaisesRegex(ValueError, "pose"):
            self.build(1, {1: torch.full((4, 4), float("nan"))}, [])
        for speed in (-1., float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "speed"):
                self.builder.build("episode", 1, torch.eye(4), {}, [], speed)
        with self.assertRaisesRegex(ValueError, "TTC"):
            self.build(2, {}, [{"sender_id": None}], ttc=[])


if __name__ == "__main__":
    unittest.main()
