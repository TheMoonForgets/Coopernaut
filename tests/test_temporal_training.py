"""Real TARC gradients with a tiny synthetic encoder; no Coopernaut training."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch import nn

from cooperative.config import TemporalConfig
from cooperative.tarc import TARC
from training.train_tarc import (TrainingSettings, argument_parser, bind_training_config, build_optimizer,
                                 evaluate, forward_loss, load_config, save_runtime_config, seed_everything)


def synthetic_config():
    # Synthetic-only settings; no experimental defaults are supplied by code.
    return TemporalConfig(delay_ms=100, match_radius_m=1.0,
                          match_weighting="inverse_distance", residual_hidden_dim=8,
                          quality_hidden_dim=8, activation="relu", time_encoding="linear")


class TinyTemporalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(3, 128)
        self.tarc = TARC(synthetic_config())
        self.control_head = nn.Linear(128, 3)
        self.temporal_aux = []
        self.seen_temporal = None

    def forward(self, ego, speed, other, transform, temporal_batch=None):
        self.seen_temporal = temporal_batch
        self.temporal_aux = []
        predictions = []
        for item in range(ego.shape[0]):
            history = temporal_batch["history_lidar"][item, 0]
            features = self.encoder(history)
            output = self.tarc(features, history,
                               temporal_batch["history_age_s"][item, 0],
                               temporal_batch["history_valid"][item, 0])
            predictions.append(self.control_head(output["features"].mean(0)))
            if self.training and temporal_batch["reference_valid"][item, 0]:
                reference = temporal_batch["reference_lidar"][item, 0]
                self.temporal_aux.append({"output": output,
                                          "reference_features": self.encoder(reference),
                                          "reference_xyz": reference})
        predictions = torch.stack(predictions)
        return tuple(predictions[:, index] for index in range(3))


def synthetic_batch(batch_size=2):
    history = torch.tensor([[0.0, 0.0, -1.0], [0.1, 0.0, -1.0]])
    history = history.repeat(batch_size, 1, 4, 1, 1)
    history[:, :, 0, :, 0] -= 0.1
    temporal = {
        "history_lidar": history,
        "history_transform": torch.eye(4).repeat(batch_size, 1, 4, 1, 1),
        "history_age_s": torch.tensor([0.25, 0.2, 0.15, 0.1]).repeat(batch_size, 1, 1),
        "history_valid": torch.ones(batch_size, 1, 4, dtype=torch.bool),
        "reference_lidar": history[:, :, -1].clone(),
        "reference_transform": torch.eye(4).repeat(batch_size, 1, 1, 1),
        "reference_valid": torch.ones(batch_size, 1, dtype=torch.bool),
        "ego_speed_mps": torch.full((batch_size,), 12.0),
    }
    return (torch.zeros(batch_size, 3, 2, 2), torch.zeros(batch_size, 2, 3),
            torch.full((batch_size,), 0.4), torch.zeros(batch_size),
            torch.ones(batch_size), torch.zeros(batch_size),
            torch.ones(batch_size, 3), torch.zeros(batch_size, 1, 2, 3),
            torch.zeros(batch_size, 1), torch.eye(4).repeat(batch_size, 1, 1, 1),
            torch.eye(4).repeat(batch_size, 1, 1), torch.ones(batch_size), temporal)


class TemporalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.settings = TrainingSettings(beta=1.0, sigma=0.5, weight_decay=0.01,
                                         epochs=1, batch_size=2, seed=7)

    def test_joint_groups_and_nonzero_finite_gradients(self):
        model = TinyTemporalModel()
        optimizer = build_optimizer(model, self.settings)
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-4, 1e-5])
        self.assertTrue(all(group["weight_decay"] == 0.01 for group in optimizer.param_groups))
        grouped = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(grouped), {id(parameter) for parameter in model.parameters()})
        loss, metrics = forward_loss(model, synthetic_batch(), self.settings, "cpu", True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["quality_messages"], 2)
        self.assertEqual(metrics["quality_matched_points"], 4)
        loss.backward()
        for module in (model.encoder, model.tarc.residual, model.tarc.quality_head):
            gradients = [parameter.grad for parameter in module.parameters()]
            self.assertTrue(all(gradient is not None and torch.isfinite(gradient).all()
                                for gradient in gradients))
            self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0)
        optimizer.step()

    def test_validation_uses_same_temporal_forward_without_reference_supervision(self):
        model = TinyTemporalModel()
        batch = synthetic_batch()
        baseline = evaluate(model, [batch], self.settings, "cpu")
        self.assertTrue(model.training)
        self.assertEqual(model.temporal_aux, [])
        self.assertIsNotNone(model.seen_temporal)
        batch[12]["reference_lidar"].fill_(float("nan"))
        self.assertEqual(evaluate(model, [batch], self.settings, "cpu"), baseline)
        self.assertEqual(model.temporal_aux, [])

    def test_missing_reference_skips_quality_loss_without_fabricated_target(self):
        batch = synthetic_batch()
        batch[12]["reference_valid"].zero_()
        model = TinyTemporalModel()
        loss, metrics = forward_loss(model, batch, self.settings, "cpu", True)
        self.assertEqual(metrics["quality_messages"], 0)
        self.assertEqual(metrics["quality_mse"], 0.0)
        loss.backward()
        self.assertTrue(torch.isfinite(model.encoder.weight.grad).all())

    def test_no_supervision_in_eval_mode(self):
        model = TinyTemporalModel().eval()
        with self.assertRaisesRegex(ValueError, "only during training"):
            forward_loss(model, synthetic_batch(), self.settings, "cpu", True)

    def test_legacy_batch_is_rejected_by_training_driver(self):
        with self.assertRaisesRegex(ValueError, "thirteen"):
            forward_loss(TinyTemporalModel(), synthetic_batch()[:12], self.settings, "cpu", True)

    def test_training_settings_are_explicit_and_validated(self):
        for keyword, value in (("sigma", 0), ("beta", -1), ("weight_decay", float("nan")),
                               ("epochs", True), ("batch_size", 0), ("seed", -1)):
            settings = dict(beta=1, sigma=1, weight_decay=0, epochs=1, batch_size=1, seed=7)
            settings[keyword] = value
            with self.assertRaises(ValueError):
                TrainingSettings(**settings)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            argument_parser().parse_args(["--config", "model.json"])

    def test_actual_publication_manifest_cli_is_required(self):
        arguments = ["--config", "model.json", "--data", "train", "--output-dir", "new-run",
                     "--pretrained-checkpoint", "base.th", "--beta", "1", "--sigma", "1",
                     "--weight-decay", "0", "--epochs", "1", "--batch-size", "2", "--seed", "7",
                     "--device", "cpu", "--num-dataloader-workers", "0"]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            argument_parser().parse_args(arguments)
        args = argument_parser().parse_args(arguments + ["--publication-manifest", "publications.json"])
        self.assertEqual(args.publication_manifest, "publications.json")

    def test_config_unwraps_legacy_wandb_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps({"npoints": {"value": 2048, "desc": None},
                                        "temporal": {"value": {"enabled": False}},
                                        "wandb_version": 1, "_wandb": {"value": {}}}),
                            encoding="utf-8")
            config = load_config(path)
        self.assertEqual(config.npoints, 2048)
        self.assertEqual(config.temporal, {"enabled": False})
        self.assertFalse(hasattr(config, "_wandb"))

    def test_explicit_training_values_recorded_and_conflicts_rejected(self):
        config = SimpleNamespace(temporal=synthetic_config().as_dict(), npoints=2048,
                                 nblocks=2, input_dim=3, cpt=True, seed=None)
        bound = bind_training_config(config, self.settings)
        self.assertEqual(TemporalConfig.from_config(bound).require_training(),
                         {key: value for key, value in vars(self.settings).items() if key != "seed"})
        self.assertEqual(bound.batch_size, 2)
        self.assertEqual(bound.seed, 7)
        self.assertNotIn("beta", config.temporal)
        config.temporal["beta"] = 9.0
        with self.assertRaisesRegex(ValueError, "conflicts"):
            bind_training_config(config, self.settings)

    def test_rr_gate_cannot_change_joint_finetuning_weights(self):
        config = SimpleNamespace(temporal=synthetic_config().as_dict(), npoints=2048,
                                 nblocks=2, input_dim=3, cpt=True)
        config.temporal["rr_gate"] = True
        with self.assertRaisesRegex(ValueError, "reuses the same TARC weights"):
            bind_training_config(config, self.settings)

    def test_explicit_seed_controls_python_numpy_and_torch(self):
        seed_everything(13)
        first = random.random(), np.random.rand(), torch.rand(2)
        seed_everything(13)
        second = random.random(), np.random.rand(), torch.rand(2)
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))

    def test_online_agent_compatible_wandb_config_round_trip(self):
        config = SimpleNamespace(temporal=synthetic_config().as_dict(), npoints=2048,
                                 nblocks=2, input_dim=3, cpt=True)
        config = bind_training_config(config, self.settings)
        with tempfile.TemporaryDirectory() as temporary:
            save_runtime_config(config, temporary)
            reloaded = load_config(Path(temporary) / "config.yaml")
        self.assertEqual(reloaded.temporal, config.temporal)
        self.assertEqual(reloaded.seed, 7)


if __name__ == "__main__":
    unittest.main()
