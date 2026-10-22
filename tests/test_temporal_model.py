"""Real temporal model flow with labelled lightweight dependency stubs.

The backbone/aggregation and voxel operator are stubs to make these tests
independent of Open3D and CARLA. TARC, message delivery, gate, coordinate
transforms, ROI/sampling, grouping, control head and training reference handling
are production code. These tests make no claim about native voxel gradients,
original encoder accuracy, a trained checkpoint, or simulator trajectories.
"""
import importlib.util
import random
import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from cooperative.checkpoint import load_checkpoint
from cooperative.rr_gate import compute_gate
from cooperative.tarc import reliability_loss


ROOT = Path(__file__).resolve().parents[1]


class StubBackbone(nn.Module):
    """Small differentiable encoder, not a substitute for original FPS/KNN."""
    def __init__(self, config):
        super().__init__()
        self.projection = nn.Linear(3, 128)

    def forward(self, lidar):
        xyz = lidar[:, :4, :3]
        features = torch.relu(self.projection(xyz)) + 1.0
        return features, xyz, []


class StatefulStubBackbone(StubBackbone):
    """BN/dropout expose reference-encoding state and RNG contamination."""
    def __init__(self, config):
        super().__init__(config)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(.3)
        self.aux_dropout = nn.Dropout(.2)

    def forward(self, lidar):
        features, xyz, _ = super().forward(lidar)
        shape = features.shape
        features = self.bn(features.reshape(-1, 128)).reshape(shape)
        features = self.aux_dropout(self.dropout(features))
        return features + 5.0, xyz, []


class StubTransitionDown(nn.Module):
    """Mix all points so both ego and remote contributions have gradients."""
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.projection = nn.Linear(channels[0] - 3, channels[-1])
        self.input_counts = []

    def forward(self, xyz, features):
        self.input_counts.append(features.shape[1])
        mixed = self.projection(features.mean(1, keepdim=True))
        return xyz, mixed.expand(-1, xyz.shape[1], -1)


class StubTransformer(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, xyz, features):
        return features, None


class StubVoxelPooling(nn.Module):
    """Identity grouping plus the native zero floor; no voxel merging claimed."""
    def __init__(self, **kwargs):
        super().__init__()
        self.calls = []

    def forward(self, xyz, features, voxel_size):
        self.calls.append((xyz.detach().clone(), features.detach().clone(), voxel_size))
        return Namespace(pooled_positions=xyz, pooled_features=features.clamp_min(0))


def _load_model_module():
    package = types.ModuleType("_temporal_model_test")
    package.__path__ = []
    pointnet = types.ModuleType("_temporal_model_test.pointnet_util")
    pointnet.PointNetFeaturePropagation = nn.Identity
    pointnet.PointNetSetAbstraction = nn.Identity
    transformer = types.ModuleType("_temporal_model_test.transformer_block")
    transformer.TransformerBlock = StubTransformer
    open3d = types.ModuleType("open3d")
    open3d.__path__ = []
    ml = types.ModuleType("open3d.ml")
    ml.__path__ = []
    ml_torch = types.ModuleType("open3d.ml.torch")
    ml_torch.layers = Namespace(VoxelPooling=StubVoxelPooling)
    open3d.ml, ml.torch = ml, ml_torch
    modules = {"_temporal_model_test": package,
               "_temporal_model_test.pointnet_util": pointnet,
               "_temporal_model_test.transformer_block": transformer,
               "open3d": open3d, "open3d.ml": ml, "open3d.ml.torch": ml_torch}
    spec = importlib.util.spec_from_file_location("_temporal_model_test.point_transformer",
                                                ROOT / "models" / "point_transformer.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    module.Backbone = StubBackbone
    module.TransitionDown = StubTransitionDown
    module.TransformerBlock = StubTransformer
    return module


MODEL = _load_model_module()
spec = importlib.util.spec_from_file_location("_original_cooperative_fixture",
                                            ROOT / "tests" / "fixtures" / "original_cooperative_forward.py")
ORIGINAL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ORIGINAL)
ORIGINAL.Backbone = StubBackbone
ORIGINAL.TransitionDown = StubTransitionDown
ORIGINAL.TransformerBlock = StubTransformer
ORIGINAL.ml3d = Namespace(layers=Namespace(VoxelPooling=StubVoxelPooling))


def config(enabled=True, delay_ms=100, rr_gate=False):
    temporal = dict(enabled=enabled, delay_ms=delay_ms, match_radius_m=2.0,
                    match_weighting="inverse_distance", residual_hidden_dim=16,
                    quality_hidden_dim=8, activation="gelu", time_encoding="linear",
                    rr_gate=rr_gate)
    return Namespace(npoints=2048, nblocks=2, nneighbor=2, num_output=3,
                     input_dim=3, transformer_dim=8, cpt=True, temporal=temporal)


def lidar(batch=1, neighbors=None, device="cpu"):
    points = torch.zeros(batch, 2048, 3, device=device)
    points[:, :4, 0] = torch.tensor([-1.5, 0., 1.5, 3.], device=device)
    points[:, :, 2] = -1.
    if neighbors is not None:
        points = points[:, None].expand(-1, neighbors, -1, -1).clone()
    return points


def batch_data(batch=1, neighbors=1, delivered=2, delay_ms=100):
    history = lidar(batch, neighbors)[:, :, None].expand(-1, -1, 4, -1, -1).clone()
    valid = torch.zeros(batch, neighbors, 4, dtype=torch.bool)
    if delivered:
        valid[:, :, -delivered:] = True
    ages = (torch.tensor([.15, .10, .05, 0.]) + delay_ms / 1000.)[None, None].expand(batch, neighbors, -1).clone()
    return dict(history_lidar=history,
                history_transform=torch.eye(4).expand(batch, neighbors, 4, 4, 4).clone(),
                history_age_s=ages, history_valid=valid,
                reference_lidar=lidar(batch, neighbors),
                reference_transform=torch.eye(4).expand(batch, neighbors, 4, 4).clone(),
                reference_valid=torch.ones(batch, neighbors, dtype=torch.bool),
                ego_speed_mps=torch.full((batch,), 6.5))


def context(now, slots, episode="episode"):
    return dict(episode_id=episode, now_tick=now, slots=slots,
                ego_inverse=torch.eye(4), ego_speed_mps=6.5)


def source(sender, frame, x=0., available=True):
    matrix = torch.eye(4)
    matrix[0, 3] = x
    return dict(sender_id=sender, capture_tick=frame, sensor_frame=frame,
                sender_world=matrix if available else None)


class TemporalModelTests(unittest.TestCase):
    def setUp(self):
        self.python_rng = random.getstate()
        self.torch_rng = torch.get_rng_state()
        self.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        random.seed(17)
        torch.manual_seed(17)

    def tearDown(self):
        random.setstate(self.python_rng)
        torch.set_rng_state(self.torch_rng)
        if self.cuda_rng is not None:
            torch.cuda.set_rng_state_all(self.cuda_rng)

    def assertControlsClose(self, first, second, exact=False):
        for left, right in zip(first, second):
            if exact:
                self.assertTrue(torch.equal(left, right))
            else:
                self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))

    def test_disabled_initialization_keys_values_and_rng_equal_frozen_original(self):
        initial = torch.get_rng_state()
        original = ORIGINAL.CooperativePointTransformer(config(False))
        after_original = torch.get_rng_state()
        torch.set_rng_state(initial)
        before_python = random.getstate()
        current = MODEL.CooperativePointTransformer(config(False))
        self.assertTrue(torch.equal(after_original, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())
        self.assertEqual(set(original.state_dict()), set(current.state_dict()))
        self.assertFalse(any(key.startswith("tarc.") for key in current.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, current.state_dict()[key]), key)

    def test_disabled_routes_all_original_arguments_without_consuming_rng(self):
        model = MODEL.CooperativePointTransformer(config(False))
        inputs = (object(), object(), object(), object())
        marker = object()
        before = torch.get_rng_state()
        before_python = random.getstate()
        with patch.object(model, "_forward_legacy", return_value=marker) as call:
            self.assertIs(model(*inputs, temporal_batch={}), marker)
            call.assert_called_once_with(*inputs)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(before_python, random.getstate())

    @unittest.skipUnless(torch.cuda.is_available(), "Original forward hard-codes CUDA; native architecture is stubbed")
    def test_disabled_forward_output_and_rng_equal_frozen_original_on_cuda(self):
        original = ORIGINAL.CooperativePointTransformer(config(False)).cuda().eval()
        current = MODEL.CooperativePointTransformer(config(False)).cuda().eval()
        load_checkpoint(current, original.state_dict())
        args = (lidar(device="cuda"), torch.tensor([.2], device="cuda"),
                lidar(neighbors=2, device="cuda"), torch.eye(4, device="cuda").expand(1, 2, 4, 4))
        before = random.getstate()
        original_result = original(*args)
        after = random.getstate()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all()
        random.setstate(before)
        current_result = current(*args)
        self.assertControlsClose(original_result, current_result, exact=True)
        self.assertEqual(after, random.getstate())
        self.assertTrue(torch.equal(cpu_rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())))

    @unittest.skipUnless(torch.cuda.is_available(), "Original forward hard-codes CUDA; native architecture is stubbed")
    def test_one_frame_gate_off_matches_baseline_forward_and_rng_on_cuda(self):
        baseline = MODEL.CooperativePointTransformer(config(False)).cuda().eval()
        temporal = MODEL.CooperativePointTransformer(config(True, delay_ms=0)).cuda().eval()
        load_checkpoint(temporal, baseline.state_dict(), allow_pretrained=True)
        args = (lidar(device="cuda"), torch.tensor([.2], device="cuda"),
                lidar(neighbors=1, device="cuda"), torch.eye(4, device="cuda").expand(1, 1, 4, 4))
        before_python = random.getstate()
        before_torch = torch.get_rng_state()
        before_cuda = torch.cuda.get_rng_state_all()
        expected = baseline(*args)
        after_python = random.getstate()
        after_torch = torch.get_rng_state()
        after_cuda = torch.cuda.get_rng_state_all()
        random.setstate(before_python)
        torch.set_rng_state(before_torch)
        torch.cuda.set_rng_state_all(before_cuda)
        actual = temporal(*args, temporal_context=context(10, [source(7, 10)]))
        self.assertControlsClose(expected, actual, exact=True)
        self.assertEqual(after_python, random.getstate())
        self.assertTrue(torch.equal(after_torch, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(after_cuda, torch.cuda.get_rng_state_all())))

    def test_online_history_follows_sender_across_slot_swap_and_uses_actual_age(self):
        model = MODEL.CooperativePointTransformer(config()).eval()
        ego, speed, others = lidar(), torch.tensor([.2]), lidar(neighbors=2)
        identity = torch.eye(4).expand(1, 2, 4, 4)
        model(ego, speed, others, identity,
              temporal_context=context(10, [source(1, 10, x=1), source(2, 10, x=2)]))
        self.assertTrue(all(not row["valid"] for row in model.temporal_trace))
        self.assertEqual(model.temporal_publications['slots'], [1, 2])
        self.assertEqual([entry['capture_tick'] for entry in model.temporal_publications['published']], [10, 10])
        # No newly captured pose is available, but old delivered snapshots remain valid.
        model(ego, speed, others.flip(1), identity,
              temporal_context=context(15, [source(2, 15, available=False), source(1, 15, available=False)]))
        self.assertEqual([row["sender_id"] for row in model.temporal_trace], [2, 1])
        self.assertEqual(model.temporal_publications['published'], [])
        for row, offset in zip(model.temporal_trace, (2., 1.)):
            self.assertEqual(row["capture_ticks"], [10])
            self.assertAlmostEqual(row["age_s"].item(), .25)
            self.assertTrue(torch.equal(row["prefusion_xyz"][:, 0], others[0, 0, :4, 0] + offset))
        # Repeated capture does not provide another history frame.
        model(ego, speed, others, identity,
              temporal_context=context(16, [source(1, 10, x=99), source(2, 10, x=99)]))
        self.assertTrue(all(row["capture_ticks"] == [10] for row in model.temporal_trace))
        self.assertEqual(model.temporal_publications['published'], [])
        model(ego, speed, others, identity,
              temporal_context=context(1, [source(1, 1), source(2, 1)], episode="next"))
        self.assertTrue(all(not row["valid"] for row in model.temporal_trace))

    def test_gate_penalty_is_applied_before_pooling_using_physical_speed(self):
        model = MODEL.CooperativePointTransformer(config(rr_gate=True, delay_ms=0)).eval()
        model(lidar(), torch.tensor([6.5 / 30.]), lidar(neighbors=1), torch.eye(4)[None, None],
              temporal_context=context(4, [source(7, 4)]))
        row, = model.temporal_trace
        self.assertAlmostEqual(row["gate"]["risk"].item(), .5)
        expected_gate = compute_gate(row["quality"], row["age_s"], .5)
        self.assertTrue(torch.equal(row["gate"]["gate"], expected_gate))
        expected = row["compensated_features"] + .1 * torch.log(expected_gate)
        self.assertTrue(torch.equal(row["prefusion_features"], expected))
        pooled_xyz, pooled_features, size = model.voxel_pooling.calls[-1]
        self.assertTrue(torch.equal(pooled_features, expected))
        self.assertTrue(torch.equal(pooled_xyz, row["prefusion_xyz"]))
        self.assertEqual(size, .5)

    def test_no_delivered_batch_and_all_remote_roi_empty_use_same_ego_head(self):
        model = MODEL.CooperativePointTransformer(config()).eval()
        ego, speed = lidar(), torch.tensor([.2])
        empty = batch_data(delivered=0)
        missing_result = model(ego, speed, None, None, temporal_batch=empty)
        self.assertEqual(model.aggr_transition_down.input_counts[-1], 4)
        self.assertEqual(model.voxel_pooling.calls, [])
        far = batch_data(delivered=1)
        far["history_transform"][:, :, :, 0, 3] = 1000
        far_result = model(ego, speed, None, None, temporal_batch=far)
        self.assertEqual(model.aggr_transition_down.input_counts[-1], 4)
        self.assertControlsClose(missing_result, far_result, exact=True)
        self.assertTrue(all(torch.isfinite(control).all() for control in far_result))

    def test_mixed_batch_empty_and_nonempty_preserves_sample_order(self):
        model = MODEL.CooperativePointTransformer(config()).eval()
        batch = batch_data(batch=2, delivered=1)
        batch["history_valid"][0] = False
        ego, speed = lidar(2), torch.tensor([.1, .3])
        before = random.getstate()
        together = model(ego, speed, None, None, temporal_batch=batch)
        random.setstate(before)
        separately = []
        for i in range(2):
            single = {key: value[i:i+1] for key, value in batch.items()}
            separately.append(model(ego[i:i+1], speed[i:i+1], None, None, temporal_batch=single))
        for channel in range(3):
            self.assertTrue(torch.allclose(together[channel], torch.cat([item[channel] for item in separately]), atol=1e-6))

    def test_eval_does_not_read_current_supervision_and_rejects_undelivered_history(self):
        model = MODEL.CooperativePointTransformer(config()).eval()
        batch = batch_data(delivered=2)
        batch["reference_lidar"].fill_(float("nan"))
        batch["reference_transform"].fill_(float("nan"))
        before = random.getstate()
        actual = model(lidar(), torch.tensor([.2]), None, None, temporal_batch=batch)
        random.setstate(before)
        for name in ("reference_lidar", "reference_transform", "reference_valid"):
            del batch[name]
        expected = model(lidar(), torch.tensor([.2]), None, None, temporal_batch=batch)
        self.assertControlsClose(actual, expected, exact=True)
        self.assertEqual(model.temporal_aux, [])
        batch["history_age_s"][batch["history_valid"]] = .05
        with self.assertRaisesRegex(ValueError, "not yet delivered"):
            model(lidar(), torch.tensor([.2]), None, None, temporal_batch=batch)

    def test_training_control_and_quality_gradients_without_reference_leakage(self):
        model = MODEL.CooperativePointTransformer(config(rr_gate=True)).train()
        batch = batch_data(delivered=2)
        batch["history_lidar"].requires_grad_()
        batch["reference_lidar"].requires_grad_()
        predictions = model(lidar(), torch.tensor([.2]), None, None, temporal_batch=batch)
        record, = model.temporal_aux
        quality_loss, matches = reliability_loss(record["output"], record["reference_features"],
                                                record["reference_xyz"], sigma=.5)
        self.assertEqual(matches, 4)
        loss = sum(channel.abs().mean() for channel in predictions) + .2 * quality_loss
        loss.backward()
        for module in (model.backbone, model.backbone_other, model.tarc.residual, model.tarc.quality_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(grad).all() for grad in gradients))
            self.assertGreater(sum(grad.abs().sum().item() for grad in gradients), 0)
        self.assertIsNotNone(batch["history_lidar"].grad)
        self.assertIsNone(batch["reference_lidar"].grad)
        self.assertFalse(record["reference_features"].requires_grad)
        self.assertFalse(record["reference_xyz"].requires_grad)

    def test_training_reference_pass_preserves_history_encoder_bn_rng_and_mixed_modes(self):
        original = MODEL.Backbone
        try:
            MODEL.Backbone = StatefulStubBackbone
            with_reference = MODEL.CooperativePointTransformer(config()).train()
            without_reference = MODEL.CooperativePointTransformer(config()).train()
        finally:
            MODEL.Backbone = original
        load_checkpoint(without_reference, with_reference.state_dict())
        for model in (with_reference, without_reference):
            model.backbone_other.aux_dropout.eval()
        batch = batch_data(delivered=2)
        batch["reference_lidar"] = batch["reference_lidar"] + 30
        no_reference = dict(batch, reference_valid=torch.zeros(1, 1, dtype=torch.bool))
        initial_torch, initial_python = torch.get_rng_state(), random.getstate()
        expected = without_reference(lidar(), torch.tensor([.2]), None, None, temporal_batch=no_reference)
        expected_torch, expected_python = torch.get_rng_state(), random.getstate()
        torch.set_rng_state(initial_torch)
        random.setstate(initial_python)
        actual = with_reference(lidar(), torch.tensor([.2]), None, None, temporal_batch=batch)
        self.assertControlsClose(actual, expected, exact=True)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch))
        self.assertEqual(random.getstate(), expected_python)
        self.assertTrue(torch.equal(with_reference.backbone_other.bn.running_mean,
                                    without_reference.backbone_other.bn.running_mean))
        self.assertTrue(torch.equal(with_reference.backbone_other.bn.running_var,
                                    without_reference.backbone_other.bn.running_var))
        self.assertEqual(with_reference.backbone_other.bn.num_batches_tracked,
                         without_reference.backbone_other.bn.num_batches_tracked)
        self.assertTrue(with_reference.backbone_other.training)
        self.assertTrue(with_reference.backbone_other.dropout.training)
        self.assertTrue(with_reference.backbone_other.bn.training)
        self.assertFalse(with_reference.backbone_other.aux_dropout.training)

    def test_temporal_cache_rejected_during_training_and_context_is_required(self):
        model = MODEL.CooperativePointTransformer(config()).train()
        args = (lidar(), torch.tensor([.2]), lidar(neighbors=1), torch.eye(4)[None, None])
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(*args)
        with self.assertRaisesRegex(ValueError, "inference-only"):
            model(*args, temporal_context=context(1, [source(7, 1)]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            model(*args, temporal_context={}, temporal_batch={})


if __name__ == "__main__":
    unittest.main()
