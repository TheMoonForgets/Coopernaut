import unittest
import torch
from cooperative.messages import CausalMessageBuffer


class MessagesTests(unittest.TestCase):
    def setUp(self):
        self.features = torch.ones(2, 128, requires_grad=True)
        self.xyz = torch.zeros(2, 3)
        self.pose = torch.eye(4)

    def publish(self, buffer, sender, tick, sensor_frame=None):
        return buffer.publish(sender, tick, self.features, self.xyz, self.pose, sensor_frame)

    def test_no_pending_or_future_exposure_and_actual_age(self):
        buffer = CausalMessageBuffer(100)
        buffer.advance(10)
        self.publish(buffer, 1, 10)
        self.assertEqual(buffer.consume_history(1), [])
        buffer.advance(11)
        self.assertEqual(buffer.consume_history(1), [])
        buffer.advance(15)
        message, = buffer.consume_history(1)
        self.assertEqual(message.arrival_tick, 12)
        self.assertAlmostEqual((15 - message.capture_tick) * 0.05, 0.25)
        with self.assertRaisesRegex(ValueError, "future"):
            self.publish(buffer, 1, 16)

    def test_zero_delay_arrives_immediately(self):
        buffer = CausalMessageBuffer(0)
        buffer.advance(10)
        self.publish(buffer, 1, 10)
        self.assertEqual(len(buffer.consume_history(1)), 1)

    def test_snapshot_detached_and_capture_xyz_never_mutated(self):
        buffer = CausalMessageBuffer(0)
        buffer.advance(10)
        self.publish(buffer, 1, 10)
        with torch.no_grad():
            self.features.add_(8)
            self.xyz.add_(3)
            self.pose.add_(2)
        message, = buffer.consume_history(1)
        self.assertFalse(message.features.requires_grad)
        self.assertTrue(torch.equal(message.features, torch.ones(2, 128)))
        self.assertTrue(torch.equal(message.xyz, torch.zeros(2, 3)))
        self.assertTrue(torch.equal(message.sender_world, torch.eye(4)))

    def test_deduplication_uses_sender_and_true_frame(self):
        buffer = CausalMessageBuffer(50)
        buffer.advance(10)
        self.assertTrue(self.publish(buffer, 8, 8))
        self.assertFalse(self.publish(buffer, 8, 8))
        self.assertFalse(self.publish(buffer, 8, 9, sensor_frame=8))
        self.assertFalse(self.publish(buffer, 8, 8, sensor_frame=9))
        self.assertTrue(self.publish(buffer, 9, 8))
        self.assertEqual(len(buffer.consume_history(8)), 1)
        self.assertEqual(len(buffer.consume_history(9)), 1)
        # A slot swap changes neither vehicle's history.
        for sender in (9, 8):
            self.assertEqual(buffer.consume_history(sender)[0].sender_id, sender)

    def test_latest_four_ordered_captures_and_old_out_of_order_packet(self):
        buffer = CausalMessageBuffer(100)
        buffer.advance(20)
        for tick in (12, 11, 14, 13, 15, 10):
            self.publish(buffer, "vehicle", tick)
        self.assertEqual([m.capture_tick for m in buffer.consume_history("vehicle")], [12, 13, 14, 15])

    def test_simultaneous_arrivals_do_not_compare_payload_tensors(self):
        buffer = CausalMessageBuffer(100)
        self.publish(buffer, 1, 0)
        self.publish(buffer, 2, 0)
        buffer.advance(2)
        self.assertEqual(len(buffer.consume_history(1)), 1)
        self.assertEqual(len(buffer.consume_history(2)), 1)

    def test_reset_clears_pending_delivered_and_dedup_identity(self):
        buffer = CausalMessageBuffer(50)
        buffer.advance(5)
        self.publish(buffer, 1, 3)
        self.publish(buffer, 1, 5)
        with self.assertRaisesRegex(ValueError, "reset"):
            buffer.advance(0)
        buffer.reset()
        buffer.advance(5)
        self.assertEqual(buffer.consume_history(1), [])
        self.assertTrue(self.publish(buffer, 1, 3))
        self.assertEqual([m.capture_tick for m in buffer.consume_history(1)], [3])

    def test_invalid_payload_and_unrepresentable_delay_rejected(self):
        with self.assertRaises(ValueError):
            CausalMessageBuffer(75)
        buffer = CausalMessageBuffer(0)
        with self.assertRaisesRegex(ValueError, "features"):
            buffer.publish(1, 0, torch.zeros(2, 127), self.xyz, self.pose)
        self.features = torch.full((2, 128), float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.publish(buffer, 1, 0)


if __name__ == "__main__":
    unittest.main()
