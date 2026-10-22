"""Dedicated TARC joint fine-tuning; legacy BC/DAgger drivers stay unchanged.

Run with ``python -m training.train_tarc --help``.  All experimental training
settings are required explicitly.  This file does not start a CARLA process.
"""

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from cooperative.config import TemporalConfig


@dataclass(frozen=True)
class TrainingSettings:
    beta: float
    sigma: float
    weight_decay: float
    epochs: int
    batch_size: int
    seed: int

    def __post_init__(self):
        for name in ("beta", "sigma", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("{} must be a finite number".format(name))
        if self.beta < 0 or self.weight_decay < 0 or self.sigma <= 0:
            raise ValueError("beta and weight_decay must be >= 0; sigma must be > 0")
        for name in ("epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError("{} must be an explicit positive integer".format(name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an explicit integer in [0, 2**32)")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    """Read plain or original W&B-export YAML without inventing model settings."""
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        if path.suffix.lower() == ".json":
            values = json.load(handle)
        else:
            import yaml
            values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a mapping")
    return SimpleNamespace(**{
        key: value["value"] if isinstance(value, dict) and "value" in value else value
        for key, value in values.items() if not key.startswith("_") and key != "wandb_version"
    })


def build_optimizer(model, settings):
    """Joint AdamW with the two fixed paper learning rates."""
    if not hasattr(model, "tarc") or model.tarc is None:
        raise ValueError("Enabled TARC model required")
    temporal = [parameter for parameter in model.tarc.parameters() if parameter.requires_grad]
    temporal_ids = {id(parameter) for parameter in temporal}
    base = [parameter for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in temporal_ids]
    if not temporal or not base:
        raise ValueError("Joint fine-tuning requires trainable TARC and base parameters")
    return torch.optim.AdamW([
        {"params": temporal, "lr": 1e-4, "name": "tarc"},
        {"params": base, "lr": 1e-5, "name": "coopernaut"},
    ], weight_decay=settings.weight_decay)


def bind_training_config(config, settings):
    """Record explicit CLI training values, rejecting two conflicting sources."""
    config = deepcopy(config)
    temporal = TemporalConfig.from_config(config)
    if temporal is None:
        raise ValueError("train_tarc requires explicit temporal.enabled: true")
    if temporal.rr_gate:
        raise ValueError("train_tarc requires temporal.rr_gate=false; RR-Gate reuses the same TARC weights")
    temporal_training = {key: getattr(settings, key)
                         for key in ("beta", "sigma", "weight_decay", "epochs", "batch_size")}
    for key, value in temporal_training.items():
        previous = getattr(temporal, key)
        if previous is not None and previous != value:
            raise ValueError("Explicit {} conflicts with temporal.{} in config".format(key, key))
    prior_seed = getattr(config, "seed", None)
    if prior_seed is not None and prior_seed != settings.seed:
        raise ValueError("Explicit seed conflicts with seed in config")
    config.temporal = dict(temporal.as_dict(), **temporal_training)
    config.batch_size = settings.batch_size
    config.num_epochs = settings.epochs
    config.seed = settings.seed
    TemporalConfig.from_config(config).require_training()
    return config


def save_runtime_config(config, output):
    """Write the W&B-style layout consumed by the original online Agent."""
    import yaml
    values = {key: {"value": value} for key, value in vars(config).items()}
    values["wandb_version"] = 1
    (Path(output) / "config.yaml").write_text(yaml.safe_dump(values, sort_keys=True), encoding="utf-8")


def move_temporal_batch(batch, device):
    return {key: value.to(device=device, dtype=torch.float32) if value.is_floating_point()
            else value.to(device=device) for key, value in batch.items()}


def quality_loss(model, connected_zero, sigma):
    """Average message-level quality MSE over messages with valid matches."""
    losses = []
    matched_points = 0
    for record in model.temporal_aux:
        loss, count = model.tarc.reliability_loss(
            record["output"], record["reference_features"], record["reference_xyz"], sigma)
        count = int(count)
        if count:
            losses.append(loss)
            matched_points += count
    if not losses:
        return connected_zero, 0, 0
    return torch.stack(losses).mean(), len(losses), matched_points


def forward_loss(model, batch, settings, device, supervise_quality):
    """Use the same cooperative forward in training and offline validation."""
    if len(batch) != 13:
        raise ValueError("TARC training requires the enabled thirteen-item temporal batch")
    ego = batch[1].to(device=device, dtype=torch.float32)
    speed = batch[2].to(device=device, dtype=torch.float32)
    target = batch[6].to(device=device, dtype=torch.float32)
    other = batch[7].to(device=device, dtype=torch.float32)
    transform = batch[9].to(device=device, dtype=torch.float32)
    temporal = move_temporal_batch(batch[12], device)
    predictions = model(ego, speed, other, transform, temporal_batch=temporal)
    if len(predictions) != 3:
        raise ValueError("Coopernaut must return throttle, brake, steer")
    # Preserve the original sum of the three control L1 terms.
    control = sum(F.l1_loss(prediction.reshape_as(target[:, index]), target[:, index])
                  for index, prediction in enumerate(predictions))
    reliability = control * 0
    messages = matched_points = 0
    if supervise_quality:
        if not model.training:
            raise ValueError("Reference quality supervision is permitted only during training")
        reliability, messages, matched_points = quality_loss(model, reliability, settings.sigma)
    loss = control + settings.beta * reliability
    return loss, {"loss": float(loss.detach()), "control_l1": float(control.detach()),
                  "quality_mse": float(reliability.detach()),
                  "quality_messages": messages, "quality_matched_points": matched_points}


def evaluate(model, loader, settings, device):
    """Validate causal control prediction without reading quality references."""
    prior_mode = model.training
    model.eval()
    weighted_control = 0.0
    examples = 0
    try:
        with torch.no_grad():
            for batch in loader:
                _, metrics = forward_loss(model, batch, settings, device, supervise_quality=False)
                size = batch[1].shape[0]
                weighted_control += metrics["control_l1"] * size
                examples += size
    finally:
        model.train(prior_mode)
    if not examples:
        raise ValueError("Validation loader produced no examples")
    return weighted_control / examples


def train(config, settings, checkpoint, output_dir, allow_pretrained=False, eval_data=None):
    config = bind_training_config(config, settings)
    if not getattr(config, "publication_manifest", None):
        raise ValueError("train_tarc requires an explicit publication_manifest from actual online decisions")
    config.eval_data = str(eval_data) if eval_data is not None else None
    seed_everything(settings.seed)
    # Import legacy/native dependencies only after validating user settings.
    from models import CooperativePointTransformer
    from cooperative.checkpoint import load_checkpoint
    from utils.temporal_point_transformer_loader import get_data_loader

    if not getattr(config, "cpt", False):
        raise ValueError("train_tarc requires cpt: true")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    model = CooperativePointTransformer(config).to(config.device)
    load_checkpoint(model, checkpoint, allow_pretrained=allow_pretrained)
    optimizer = build_optimizer(model, settings)
    loader = get_data_loader(config, shuffle=True, drop_last=True, include_reference=True)
    if len(loader) == 0:
        raise ValueError("Training dataset is smaller than batch_size")
    validation = None
    if eval_data is not None:
        validation_config = deepcopy(config)
        validation_config.data = eval_data
        validation_config.daggerdata = None
        validation = get_data_loader(validation_config, shuffle=False, drop_last=False,
                                     include_reference=False)
    run = {"model": vars(config), "training": vars(settings),
           "checkpoint": str(checkpoint), "allow_pretrained": allow_pretrained,
           "eval_data": str(eval_data) if eval_data is not None else None,
           "learning_rates": {"tarc": 1e-4, "coopernaut": 1e-5}}
    (output / "run_config.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    save_runtime_config(config, output)
    with (output / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(settings.epochs):
            model.train()
            count = 0
            totals = {"loss": 0.0, "control_l1": 0.0, "quality_mse": 0.0,
                      "quality_messages": 0, "quality_matched_points": 0}
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = forward_loss(model, batch, settings, config.device, True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss at epoch {}".format(epoch))
                loss.backward()
                optimizer.step()
                count += 1
                for key in totals:
                    totals[key] += metrics[key]
            record = {"epoch": epoch, "batches": count}
            record.update({key: value / count if key in ("loss", "control_l1", "quality_mse")
                           else value for key, value in totals.items()})
            if validation is not None:
                record["validation_control_l1"] = evaluate(model, validation, settings, config.device)
            torch.save(model.state_dict(), output / "model-{}.th".format(epoch))
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record), flush=True)
    return model


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Explicit model/temporal YAML or JSON")
    parser.add_argument("--data", required=True)
    parser.add_argument("--publication-manifest", required=True,
                        help="Explicit actual-publication trace manifest mapping all train/validation episodes")
    parser.add_argument("--eval-data", help="Optional separate validation dataset; never inherited from old config")
    parser.add_argument("--output-dir", required=True, help="New directory, never overwrite an existing run")
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument("--pretrained-checkpoint", help="Original Coopernaut weights; only new TARC keys may be missing")
    checkpoint.add_argument("--tarc-checkpoint", help="Complete TARC weights, strict loading; starts a new optimizer")
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--num-dataloader-workers", type=int, required=True)
    return parser


def main(argv=None):
    args = argument_parser().parse_args(argv)
    settings = TrainingSettings(args.beta, args.sigma, args.weight_decay, args.epochs,
                                args.batch_size, args.seed)
    config = load_config(args.config)
    config.data = args.data
    config.publication_manifest = str(Path(args.publication_manifest).resolve())
    config.device = args.device
    config.num_dataloader_workers = args.num_dataloader_workers
    if args.num_dataloader_workers < 0:
        raise ValueError("num_dataloader_workers must be non-negative")
    # Old checkpoint config paths cannot silently add a DAgger dataset.
    config.daggerdata = None
    train(config, settings, args.pretrained_checkpoint or args.tarc_checkpoint,
          args.output_dir, allow_pretrained=args.pretrained_checkpoint is not None,
          eval_data=args.eval_data)


if __name__ == "__main__":
    main()
