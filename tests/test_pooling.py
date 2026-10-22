"""Zero-floor gradient rule independent of any installed native operator."""
import unittest
import torch
from cooperative.pooling import zero_floor_backward


class IncorrectNativeMax(torch.autograd.Function):
    """A test double reproducing the observed native negative-region gradient."""
    @staticmethod
    def forward(ctx, features):
        maximum, indices = features.max(dim=0)
        ctx.save_for_backward(indices)
        ctx.shape = features.shape
        return maximum.clamp_min(0)

    @staticmethod
    def backward(ctx, gradient):
        indices, = ctx.saved_tensors
        result = gradient.new_zeros(ctx.shape)
        result.scatter_(0, indices.unsqueeze(0), gradient.unsqueeze(0))
        return result


class PoolingGradientTests(unittest.TestCase):
    def test_forward_bits_and_correct_negative_region_derivative(self):
        features = torch.tensor([[-2., 1., 0.], [-1., 3., -1.]], requires_grad=True)
        native = IncorrectNativeMax.apply(features)
        corrected = zero_floor_backward(native)
        self.assertTrue(torch.equal(native, corrected))
        corrected.sum().backward()
        self.assertTrue(torch.equal(features.grad, torch.tensor([[0., 0., 0.], [0., 1., 0.]])))

    def test_signed_zero_forward_is_preserved(self):
        pooled = torch.tensor([-0., 0., 2.], requires_grad=True)
        corrected = zero_floor_backward(pooled)
        self.assertTrue(torch.equal(torch.signbit(pooled), torch.signbit(corrected)))
        corrected.sum().backward()
        self.assertTrue(torch.equal(pooled.grad, torch.tensor([0., 0., 1.])))

    def test_non_native_or_invalid_output_fails(self):
        for invalid in (torch.tensor([-1.]), torch.tensor([float('nan')]), torch.tensor([float('inf')])):
            with self.assertRaises(ValueError):
                zero_floor_backward(invalid)


if __name__ == '__main__':
    unittest.main()
