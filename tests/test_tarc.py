"""Synthetic-only tests: these constants are fixtures, not experiment defaults."""

import unittest
from types import SimpleNamespace

import torch

from cooperative.tarc import TARC, _soft_match, reliability_loss


def config(**changes):
    settings = dict(match_radius_m=0.4, match_weighting="inverse_distance",
                    match_temperature=None, residual_hidden_dim=32,
                    quality_hidden_dim=16, activation="relu", time_encoding="linear")
    settings.update(changes)
    return SimpleNamespace(**settings)


def history(count=4, points=3):
    # Distinct aligned points prevent cross-point averaging in these fixtures.
    xyz = torch.arange(points).float()[:, None].expand(-1, 3).clone() * 2.0
    xyz = xyz.unsqueeze(0).repeat(count, 1, 1)
    features = torch.arange(count * points * 128).float().reshape(count, points, 128) / 1000.0
    ages = torch.arange(count - 1, -1, -1).float() * 0.05
    return features, xyz, ages, torch.ones(count, dtype=torch.bool)


class SoftMatchingTests(unittest.TestCase):
    def test_exact_aligned_coordinates_ignore_point_indices(self):
        features, xyz, _, _ = history(1, 3)
        permutation = torch.tensor([2, 0, 1])
        matched, support = _soft_match(xyz[0], xyz[0, permutation],
                                       features[0, permutation], 0.4, "inverse_distance")
        self.assertTrue(support.all())
        self.assertTrue(torch.equal(matched, features[0]))

    def test_inverse_distance_interpolation_and_empty_radius(self):
        query = torch.tensor([[1., 0., 0.], [100., 0., 0.]])
        source = torch.tensor([[0., 0., 0.], [3., 0., 0.]])
        features = torch.tensor([[3.], [9.]])
        matched, support = _soft_match(query, source, features, 3., "inverse_distance")
        self.assertTrue(torch.equal(support, torch.tensor([True, False])))
        self.assertTrue(torch.allclose(matched, torch.tensor([[5.], [0.]])))

    def test_softmax_squared_and_all_unmatched_have_finite_gradients(self):
        query = torch.tensor([[0., 0., 0.], [100., 0., 0.]], requires_grad=True)
        source = torch.tensor([[0., 0., 0.], [1., 0., 0.]], requires_grad=True)
        features = torch.tensor([[2.], [10.]], requires_grad=True)
        matched, support = _soft_match(query, source, features, 2.,
                                       "softmax_squared_distance", temperature=1.)
        expected = (2. + 10. * torch.exp(torch.tensor(-1.))) / (1. + torch.exp(torch.tensor(-1.)))
        self.assertTrue(torch.allclose(matched[0, 0], expected))
        self.assertEqual(matched[1, 0].item(), 0.)
        self.assertFalse(support[1])
        matched.sum().backward()
        for tensor in (query, source, features):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_exact_coincident_inverse_match_has_finite_coordinate_gradient(self):
        query = torch.zeros(1, 3, requires_grad=True)
        source = torch.zeros(2, 3, requires_grad=True)
        features = torch.tensor([[2.], [6.]], requires_grad=True)
        matched, _ = _soft_match(query, source, features, 1., "inverse_distance")
        self.assertEqual(matched.item(), 4.)
        matched.sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertTrue(torch.isfinite(source.grad).all())


class TARCTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TARC(config()).eval()

    def test_latest_valid_anchor_not_last_padded_slot(self):
        features, xyz, ages, valid = history(4)
        valid[-1] = False
        features[-1] = float("nan")
        xyz[-1] = float("nan")
        ages[-1] = float("nan")
        output = self.model(features, xyz, ages, valid)
        self.assertTrue(torch.equal(output["xyz"], xyz[2]))
        self.assertTrue(torch.isfinite(output["features"]).all())
        self.assertEqual(output["context"].shape, (3, 64))
        self.assertEqual(output["match_support"].shape, (4, 3))
        self.assertFalse(output["match_support"][-1].any())
        self.assertEqual(output["quality"].shape, torch.Size([]))

    def test_one_frame_is_exact_passthrough_even_with_large_residual_bias(self):
        with torch.no_grad():
            self.model.residual[-1].bias.fill_(100.)
        features, xyz, ages, valid = history(4)
        valid[:] = False
        valid[1] = True
        output = self.model(features, xyz, ages, valid)
        self.assertTrue(torch.equal(output["features"], features[1]))
        self.assertTrue(torch.equal(output["xyz"], xyz[1]))
        self.assertTrue(torch.isfinite(output["quality"]))

    def test_missing_frames_do_not_act_as_duplicate_observations(self):
        features, xyz, ages, valid = history(4)
        valid[1:3] = False
        padded = self.model(features, xyz, ages, valid)
        compact = self.model(features[valid], xyz[valid], ages[valid], valid[valid])
        for name in ("features", "context", "quality"):
            self.assertTrue(torch.allclose(padded[name], compact[name], atol=1e-6), name)

    def test_independent_history_and_anchor_point_permutations(self):
        args = list(history(4, 3))
        original = self.model(*args)
        permutation = torch.tensor([2, 0, 1])
        args[0][0] = args[0][0, permutation]
        args[1][0] = args[1][0, permutation]
        # A distinct permutation for the output anchor must permute the outputs.
        args[0][-1] = args[0][-1, permutation]
        args[1][-1] = args[1][-1, permutation]
        changed = self.model(*args)
        self.assertTrue(torch.allclose(changed["features"], original["features"][permutation], atol=1e-6))
        self.assertTrue(torch.allclose(changed["context"], original["context"][permutation], atol=1e-6))
        self.assertTrue(torch.allclose(changed["quality"], original["quality"], atol=1e-6))

    def test_unmatched_historical_frames_are_masked_per_anchor(self):
        features, xyz, ages, valid = history(2)
        xyz[0, 1:] += 100.
        output = self.model(features, xyz, ages, valid)
        self.assertTrue(torch.equal(output["match_support"],
                                    torch.tensor([[True, False, False], [True, True, True]])))
        changed = features.clone()
        changed[0, 1:] += 1000.
        second = self.model(changed, xyz, ages, valid)
        self.assertTrue(torch.allclose(output["features"], second["features"]))
        self.assertTrue(torch.allclose(output["quality"], second["quality"]))

    def test_quality_input_is_documented_66_value_summary(self):
        features, xyz, ages, valid = history(2)
        features[0] = 2.
        features[1] = 1.
        xyz[0, 1:] += 100.
        captured = []
        hook = self.model.quality_head[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach()))
        try:
            output = self.model(features, xyz, ages, valid)
        finally:
            hook.remove()
        summary = captured[0]
        self.assertEqual(summary.shape, (66,))
        self.assertTrue(torch.allclose(summary[:64], output["context"].mean(0)))
        self.assertAlmostEqual(summary[64].item(), 1., places=6)
        self.assertAlmostEqual(summary[65].item(), 1. / 3., places=6)

    def test_relative_time_context_unchanged_by_common_absolute_age_shift(self):
        features, xyz, ages, valid = history(3)
        first = self.model(features, xyz, ages, valid)
        second = self.model(features, xyz, ages + .5, valid)
        self.assertTrue(torch.allclose(first["context"], second["context"], atol=1e-6))
        self.assertTrue(torch.allclose(first["quality"], second["quality"], atol=1e-6))

    def test_common_rigid_alignment_preserves_feature_matching(self):
        features, xyz, ages, valid = history(3)
        first = self.model(features, xyz, ages, valid)
        rotation = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        aligned = xyz.matmul(rotation.t()) + torch.tensor([20., -10., 3.])
        second = self.model(features, aligned, ages, valid)
        self.assertTrue(torch.allclose(first["features"], second["features"]))
        self.assertTrue(torch.allclose(first["quality"], second["quality"]))
        self.assertTrue(torch.equal(second["xyz"], aligned[-1]))

    def test_empty_history_is_safe_and_zero_loss_has_graph(self):
        features, xyz, ages, valid = history(4)
        valid[:] = False
        features[:] = float("nan")
        xyz[:] = float("nan")
        ages[:] = float("nan")
        output = self.model(features, xyz, ages, valid)
        self.assertFalse(output["valid"])
        self.assertEqual(output["features"].shape, (0, 128))
        self.assertEqual(output["xyz"].shape, (0, 3))
        self.assertEqual(output["quality"].item(), 0.)
        loss, count = reliability_loss(output, torch.empty(0, 128), torch.empty(0, 3), sigma=.5)
        self.assertEqual(count, 0)
        self.assertEqual(loss.item(), 0.)
        loss.backward()

    def test_zero_frames_and_double_dtype(self):
        model = TARC(config(time_encoding="sinusoidal")).double()
        output = model(torch.empty(0, 3, 128, dtype=torch.double),
                       torch.empty(0, 3, 3, dtype=torch.double),
                       torch.empty(0, dtype=torch.double), torch.empty(0, dtype=torch.bool))
        self.assertEqual(output["quality"].dtype, torch.double)
        self.assertEqual(output["features"].shape, (0, 128))
        args = history(3)
        args = tuple(a.double() if a.is_floating_point() else a for a in args)
        self.assertTrue(torch.isfinite(model(*args)["features"]).all())

    def test_zero_residual_is_identity(self):
        with torch.no_grad():
            self.model.residual[-1].weight.zero_()
            self.model.residual[-1].bias.zero_()
        args = history()
        output = self.model(*args)
        self.assertTrue(torch.equal(output["features"], args[0][-1]))

    def test_time_validation(self):
        for bad in (torch.tensor([0., .1]), torch.tensor([.1, -.1]),
                    torch.tensor([float("nan"), 0.])):
            args = list(history(2))
            args[2] = bad
            with self.assertRaises(ValueError):
                self.model(*args)

    def test_control_and_quality_backward_and_detached_reference(self):
        features, xyz, ages, valid = history(4)
        features.requires_grad_()
        reference = (features[-1].detach() + .1).requires_grad_()
        reference_xyz = xyz[-1].detach().clone().requires_grad_()
        output = self.model(features, xyz, ages, valid)
        quality_loss, count = reliability_loss(output, reference, reference_xyz, sigma=.5)
        self.assertEqual(count, 3)
        loss = output["features"].square().mean() + quality_loss
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        for module in (self.model.residual, self.model.attention, self.model.quality_head):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0.)
        self.assertIsNone(reference.grad)
        self.assertIsNone(reference_xyz.grad)

    def test_quality_target_detached_from_compensated_feature_branch(self):
        output = self.model(*history(2))
        output["features"].retain_grad()
        loss, _ = reliability_loss(output, output["features"].detach() + 1.,
                                   output["xyz"].detach(), sigma=.5)
        loss.backward()
        # q uses context; only the target would otherwise reach the residual.
        self.assertIsNone(output["features"].grad)
        self.assertIsNone(self.model.residual[-1].weight.grad)
        self.assertIsNotNone(self.model.quality_head[2].weight.grad)

    def test_no_reference_matches_skips_supervision(self):
        output = self.model(*history(2))
        loss, count = reliability_loss(output, torch.ones(3, 128),
                                       output["xyz"] + 1000., sigma=.5)
        self.assertEqual(count, 0)
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.equal(self.model.quality_head[2].weight.grad,
                                    torch.zeros_like(self.model.quality_head[2].weight)))

    def test_quality_target_normalized_error_formula(self):
        output = self.model(*history(1))
        reference = torch.ones_like(output["features"])
        output["features"] = reference * 2.
        output["quality"] = torch.tensor(.2, requires_grad=True)
        loss, count = reliability_loss(output, reference, output["xyz"], sigma=.5)
        self.assertEqual(count, 3)
        expected = (.2 - torch.exp(torch.tensor(-2.))) ** 2
        self.assertTrue(torch.allclose(loss, expected))
        for sigma in (0., -1., float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                reliability_loss(output, reference, output["xyz"], sigma=sigma)


if __name__ == "__main__":
    unittest.main()
