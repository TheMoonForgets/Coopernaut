import os
import tempfile
import unittest
import torch
from torch import nn
from cooperative.checkpoint import load_checkpoint


class ToyModel(nn.Module):
    def __init__(self, temporal=True):
        super().__init__()
        self.backbone = nn.Linear(2, 2)
        if temporal:
            self.tarc = nn.Linear(2, 2)


class CheckpointTests(unittest.TestCase):
    def test_legacy_exact_loading_and_nested_path(self):
        source, target = ToyModel(False), ToyModel(False)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.pt")
            torch.save({"state_dict": source.state_dict()}, path)
            result = load_checkpoint(target, path)
        self.assertFalse(result.missing_keys)
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(target.state_dict()[key], value))

    def test_bootstrap_only_explicit_and_only_missing_full_tarc(self):
        base, temporal = ToyModel(False), ToyModel(True)
        with self.assertRaisesRegex(RuntimeError, "tarc"):
            load_checkpoint(temporal, base.state_dict())
        tarc_before = temporal.tarc.weight.detach().clone()
        result = load_checkpoint(temporal, base.state_dict(), allow_pretrained=True)
        self.assertEqual(set(result.missing_keys), {"tarc.weight", "tarc.bias"})
        self.assertTrue(torch.equal(tarc_before, temporal.tarc.weight))
        self.assertTrue(torch.equal(base.backbone.weight, temporal.backbone.weight))

    def test_existing_missing_unexpected_and_partial_tarc_fail(self):
        model = ToyModel()
        invalid = dict(model.state_dict())
        del invalid["backbone.bias"]
        with self.assertRaisesRegex(RuntimeError, "backbone.bias"):
            load_checkpoint(model, invalid, allow_pretrained=True)
        invalid = dict(model.state_dict(), unexpected=torch.tensor(1.))
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            load_checkpoint(model, invalid, allow_pretrained=True)
        invalid = dict(model.state_dict())
        del invalid["tarc.bias"]
        with self.assertRaisesRegex(RuntimeError, "tarc.bias"):
            load_checkpoint(model, invalid, allow_pretrained=True)

    def test_shape_error_does_not_partially_mutate_model(self):
        model = ToyModel()
        before = {key: value.detach().clone() for key, value in model.state_dict().items()}
        invalid = {key: torch.ones_like(value) * 99 for key, value in before.items()}
        invalid["tarc.bias"] = torch.ones(3)
        with self.assertRaisesRegex(RuntimeError, "shape"):
            load_checkpoint(model, invalid)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))


if __name__ == "__main__":
    unittest.main()
