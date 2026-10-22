import unittest
import torch
from cooperative.rr_gate import risk_from_ttc, compute_gate, apply_log_gate


class GateTests(unittest.TestCase):
    def test_ttc_boundaries_and_speed_proxy_for_invalid_ttc(self):
        ttc = torch.tensor([0.0, 1.5, 2.75, 4.0, float("nan"), float("inf"), -1.0])
        risk, valid = risk_from_ttc(ttc, 6.5)
        self.assertTrue(torch.equal(valid, torch.tensor([True, True, True, True, False, False, False])))
        self.assertTrue(torch.allclose(risk, torch.tensor([1., 1., .5, 0., .5, .5, .5])))
        risk, valid = risk_from_ttc(None, torch.tensor([0., 3., 6.5, 10., 15.]))
        self.assertTrue(torch.equal(risk, torch.tensor([0., 0., .5, 1., 1.])))
        self.assertFalse(valid.any())

    def test_broadcasting_and_exact_gate_formula(self):
        q = torch.tensor([0., .5, 1.], dtype=torch.float64)
        age = torch.tensor([0., .25, .5], dtype=torch.float64)
        risk = torch.tensor([[0.], [1.]], dtype=torch.float64)
        gate = compute_gate(q, age, risk)
        reliability = q * torch.exp(-age / .5)
        expected = .2 + .8 * reliability * (1 - .5 * risk * (1-reliability))
        self.assertEqual(gate.shape, (2, 3))
        self.assertEqual(gate.dtype, torch.float64)
        self.assertTrue(torch.equal(gate, expected))
        self.assertEqual(compute_gate(1., 0., 1.).item(), 1.)

    def test_higher_age_and_risk_never_increase_gate(self):
        ages = torch.linspace(0, 2, 21)
        low = compute_gate(.7, ages, 0.)
        high = compute_gate(.7, ages, 1.)
        self.assertTrue(torch.all(low[1:] <= low[:-1]))
        self.assertTrue(torch.all(high <= low))
        self.assertTrue(torch.all((high >= .2) & (high <= 1.)))

    def test_additive_log_penalty_with_negative_features_and_neutral_gate(self):
        features = torch.tensor([[-2., 0., 2.], [1., -1., 3.]])
        self.assertTrue(torch.equal(apply_log_gate(features, 1.), features))
        expected = features + .1 * torch.log(torch.tensor(.5))
        self.assertTrue(torch.equal(apply_log_gate(features, .5), expected))
        batch = features.expand(2, -1, -1)
        gated = apply_log_gate(batch, torch.tensor([1., .5]))
        self.assertTrue(torch.equal(gated[0], features))
        self.assertTrue(torch.equal(gated[1], expected))
        with self.assertRaisesRegex(ValueError, "per message"):
            apply_log_gate(features, torch.ones(2, 3))

    def test_invalid_quality_age_and_speed_are_errors(self):
        for q, age, risk in ((float("nan"), 0., 0.), (-.1, 0., 0.), (1.1, 0., 0.),
                             (.5, -1., 0.), (.5, float("inf"), 0.), (.5, 0., 2.)):
            with self.subTest(q=q, age=age, risk=risk):
                with self.assertRaises(ValueError):
                    compute_gate(q, age, risk)
        for speed in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError):
                risk_from_ttc(2., speed)

    def test_gate_retains_differentiability(self):
        quality = torch.tensor(.7, requires_grad=True)
        features = torch.ones(2, 128, requires_grad=True)
        apply_log_gate(features, compute_gate(quality, .2, .5)).sum().backward()
        self.assertTrue(torch.isfinite(quality.grad))
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(quality.grad.item(), 0)


if __name__ == "__main__":
    unittest.main()
