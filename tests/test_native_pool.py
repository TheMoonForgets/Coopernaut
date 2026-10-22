"""Optional real Open3D CPU-op tests, separate from model-flow test doubles."""
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from cooperative.tarc import TARC
from cooperative.pooling import zero_floor_backward


class NativePoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import open3d
            built = str(open3d._build_config.get('Pytorch_VERSION',
                        open3d._build_config.get('PYTORCH_VERSION', '')))
            if built and built.split('+')[0] != torch.__version__.split('+')[0]:
                raise unittest.SkipTest('Open3D was built for Torch %s, runtime is %s' % (built, torch.__version__))
            library = Path(open3d.__file__).parent / 'cpu' / 'open3d_torch_ops.so'
            if not library.is_file():
                raise unittest.SkipTest('Open3D native CPU operator is not installed')
            torch.ops.load_library(str(library))
        except unittest.SkipTest:
            raise
        except (ImportError, OSError, RuntimeError, AttributeError) as exc:
            raise unittest.SkipTest('Compatible Open3D native operator unavailable: %s' % exc)

    @staticmethod
    def pool(xyz, features):
        return torch.ops.open3d.voxel_pooling(xyz, features, 0.5, 'average', 'max', False)

    def test_real_forward_zero_floor_and_positive_argmax_gradient(self):
        xyz = torch.tensor([[0.1, 0.1, -0.5], [0.2, 0.1, -0.5]])
        features = torch.tensor([[-2.0, 1.0], [-1.0, 3.0]], requires_grad=True)
        positions, pooled = self.pool(xyz, features)
        self.assertTrue(torch.equal(pooled, torch.tensor([[0.0, 3.0]])))
        self.assertTrue(torch.allclose(positions, xyz.mean(0, keepdim=True)))
        # Only the positive unique maximum is used for this backward assertion.
        pooled[:, 1].sum().backward()
        self.assertTrue(torch.equal(features.grad[:, 1], torch.tensor([0.0, 1.0])))

    def test_real_native_pool_connects_control_proxy_to_tarc(self):
        cfg = SimpleNamespace(match_radius_m=1.0, match_weighting='inverse_distance',
                              match_temperature=None, time_encoding='linear',
                              activation='relu', residual_hidden_dim=16, quality_hidden_dim=8)
        torch.manual_seed(12)  # Synthetic test fixture, never an experiment default.
        tarc = TARC(cfg)
        features = (torch.randn(2, 2, 128) + 20).requires_grad_()
        xyz = torch.tensor([[[0.1, 0.1, -0.5], [0.2, 0.1, -0.5]]] * 2)
        result = tarc(features, xyz, torch.tensor([0.1, 0.0]), torch.tensor([True, True]))
        _, pooled = self.pool(result['xyz'], result['features'])
        self.assertTrue(pooled.requires_grad)
        zero_floor_backward(pooled).sum().backward()
        self.assertIsNotNone(tarc.residual[-1].weight.grad)
        self.assertTrue(torch.isfinite(tarc.residual[-1].weight.grad).all())
        self.assertGreater(float(tarc.residual[-1].weight.grad.abs().sum()), 0)
        self.assertGreater(float(features.grad.abs().sum()), 0)

    def test_zero_floor_correction_matches_real_native_finite_difference(self):
        xyz = torch.tensor([[0.1, 0.1, -0.5], [0.2, 0.1, -0.5]])
        features = torch.tensor([[-2.0, 1.0], [-1.0, 3.0]], requires_grad=True)
        _, native = self.pool(xyz, features)
        corrected = zero_floor_backward(native)
        self.assertTrue(torch.equal(native, corrected))
        corrected.sum().backward()
        numeric = torch.zeros_like(features)
        epsilon = 0.001
        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                high, low = features.detach().clone(), features.detach().clone()
                high[i, j] += epsilon
                low[i, j] -= epsilon
                numeric[i, j] = (self.pool(xyz, high)[1].sum() -
                                 self.pool(xyz, low)[1].sum()) / (2 * epsilon)
        self.assertTrue(torch.allclose(features.grad, numeric, atol=0.0002, rtol=0))


if __name__ == '__main__':
    unittest.main()
