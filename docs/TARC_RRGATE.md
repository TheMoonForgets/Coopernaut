# TARC and RR-Gate integration

The optional path adds sender-specific temporal feature compensation after the
remote Point Transformer encoder and before cross-vehicle voxel pooling.
The original BC/DAgger entry points remain unchanged. An absent `temporal`
configuration, or `temporal.enabled: false`, calls the original forward body;
no TARC parameters or temporal random draws are created in that mode.

## Feature and coordinate flow

```text
local LiDAR -> original encoder -> 128 points x 128 features per message
           -> four distinct delivered messages from the same sender
           -> historical sender-to-current-ego coordinate transform, once
           -> TARC -> optional RR-Gate -> original native voxel pooling
           -> original ROI, sampling, cooperative aggregation and control head
```

TARC uses the newest delivered message as the query/coordinate anchor. Earlier
points are soft matched within the configured radius in current ego coordinates.
Features are projected from 128 to 64 dimensions, receive relative time encoding,
and enter four-head temporal attention. The residual MLP consumes
`[anchor_feature(128), context(64), absolute_anchor_age_seconds(1)]` and predicts
a 128-dimensional feature residual. A one-message history returns the exact
anchor features. Coordinates are never extrapolated or corrected by the MLP.

The message quality head uses **66 inputs**: mean context (64), normalized
matched historical/anchor feature disagreement (1), and historical local-match
fraction (1). The last two exclude the anchor self-match and invalid frames;
both are zero without older-frame support. A sigmoid produces one `q` per
message. Its input contains neither undelayed reference features nor absolute
anchor age. Relative history timing can influence the attention context.

`cooperative/tarc.py` accepts coordinates that are already aligned. Both model
paths apply `ego_inverse(now) @ sender_world(capture)` once and send the returned
anchor coordinates directly to fusion. Feature channels are not rotated. The
loader retains the original offline flip-Z, roof compensation, car alignment,
sparse quantization and ground-plane pose conventions; this change does not
attempt to reconcile historical offline/online preprocessing differences.

## Delivery, validity and empty inputs

`cooperative/messages.py` stores detached payload snapshots for online inference.
History belongs to `(episode, sender_id)`, never a neighbor slot. Consumption
requires `capture <= arrival <= now`, with arrival determined by configured
delay and the original capture tick. Repeated sensor frames do not become new
history. The buffer retains up to four distinct messages, sorted oldest first.
Reset temporal state at an episode/reconnection boundary.

The simulator tick is 50 ms. `delay_ms` must be a non-negative multiple of 50;
actual age is `(now_tick - capture_tick) * 0.05`, not a fixed nominal delay.
The online context builder records capture-frame poses. When a matching capture
pose is unavailable it skips publishing that sensor frame and may still consume
previously delivered history; it does not substitute the current sender pose.

`utils/temporal_point_transformer_loader.py` independently builds differentiable
training inputs from frame-indexed files **and actual publication records**.
Each target requires its exact recorded decision row; recorded neighbor slots
and ordering replace a fresh random neighbor selection. History is restricted
to same-episode, same-sender publications with both `publish_tick <= target`
and `capture_tick + delay_ticks <= target`. An archived frame that was never
published cannot become history. Missing history is masked, not cloned.
Missing manifest mappings, decisions, or chosen published payloads/capture
poses are errors. The four-record window is selected before reading raw data;
irrelevant older archive files are not required. The wrapper returns the
original 12-item tuple when disabled and adds
a thirteenth tensor dictionary when enabled:

| Field | Batched shape |
| --- | --- |
| `history_lidar` | `[B,M,4,P,3]` |
| `history_transform` | `[B,M,4,4,4]` |
| `history_age_s`, `history_valid` | `[B,M,4]` |
| `reference_lidar` | `[B,M,P,3]` |
| `reference_transform`, `reference_valid` | `[B,M,4,4]`, `[B,M]` |
| `ego_speed_mps` | `[B]`, physical m/s |
| `history_capture_tick`, `sender_id` | `[B,M,4]`, `[B,M]`, audit only |

Invalid messages are excluded from fusion. If every remote message is invalid,
or the original ROI removes every remote point, ego features alone enter the
existing cooperative aggregation/head. This explicit enabled-mode behavior
does not alter the legacy zero-cloud placeholder path.

## Fixed RR-Gate

RR-Gate has no learned parameters and runs on each compensated message before
voxel pooling. With age `a` in seconds and risk `h`:

```text
r = q * exp(-a / 0.5)
g = 0.2 + 0.8*r*(1 - 0.5*h*(1-r))
F_hat = F_tarc + 0.1*log(g)
```

Valid TTC gives `h=1` at or below 1.5 s, `h=0` at or above 4.0 s, and linear
interpolation between them. Missing, negative or non-finite TTC uses
`clip((ego_speed_mps - 3)/7, 0, 1)`. Online Agent integration currently supplies
physical speed, so it uses this proxy unless a caller provides optional TTC.
Normalized backbone speed (`speed/30`) must not be used here. Invalid numeric
quality, age or speed fails explicitly. Absent messages do not receive a
minimum gate. Attention logits and the native pooling operation are unchanged.

The native MAX forward has a zero floor. A real-operator check found that its
backward can still select a negative candidate when the output is zero. In the
new training path only, `cooperative/pooling.py` preserves the exact native
forward values and blocks gradients at the zero floor (including a valid zero
subgradient at the boundary). Positive-output gradients pass through. Legacy
and evaluation paths retain their original behavior. This correction is
covered by finite-difference tests against the real native CPU operator.

## Configuration and training

Offline training requires `--publication-manifest` from an actual online
capture; a directory of archived continuous LiDAR alone is insufficient. The
explicit manifest may map both training and validation episodes:

```json
{"episodes": {"/absolute/dataset/episode_001": "/absolute/logs/episode_001.jsonl"}}
```

Relative episode/JSONL paths resolve against the manifest's directory. Episode
directories must resolve exactly to the directory containing `measurements/`
(or, for a flat synthetic fixture, the measurement file's parent). There is no
automatic trace search or inference from archived frame presence. Each JSONL
decision has this schema:

```json
{"schema_version":1,"episode_id":"episode_001","decision_tick":15,"slots":[2,null,null],"published":[{"sender_id":2,"capture_tick":15,"sensor_frame":15,"publish_tick":15}]}
```

There must be a row for every target decision, including rows with an empty
`published` list. Slot count must match `max_num_neighbors`. Decision ticks must
increase strictly, `publish_tick` must equal its decision tick, and schema 1
requires `capture_tick == sensor_frame` in absolute world ticks, no later than
publication. Duplicate successful capture/sensor identities are errors; repeated
sensor inputs must not be recorded as new publications. Optional header rows
must declare integer schema version 1, `tick_ms: 50`, and the same episode ID.
Mixed episodes or impossible publication timing are rejected. Current reference data remains
optional supervision and does not create a publication record.

To capture records, explicitly set the online Agent's top-level
`publication_log_dir` to a chosen directory. The Agent prints the unique JSONL
path. Bind that file to its actual dataset episode directory in the manifest;
the recorder never guesses the mapping. Recording does not encode additional
neighbor messages. Keep this setting absent when recording is unnecessary.

For the first TARC training run, collect this trace with the original CPT
checkpoint and temporal mode disabled. Recording still follows the actual
selected neighbor encoders, adds no extra encoder calls, and does not create
TARC weights or use its random initialization. This avoids requiring an
already-trained TARC model to prepare its first training dataset. In the
original Agent's W&B-style YAML, the optional setting is:

```yaml
publication_log_dir:
  value: /explicit/path/to/new-publication-logs
```

The recordings and archived dataset must come from the same episode. A manifest
is an explicit association, not evidence that unrelated trajectories match.
The existing external `FeatureDelay` bridge remains a separate baseline tool;
do not apply its delay hook again around this model's enabled temporal path.

Copy [`configs/tarc.example.yaml`](../configs/tarc.example.yaml) and fill every
required temporal choice. The template deliberately leaves experimental values
`null`, so it cannot silently start an experiment. Choose match weighting,
matching radius, residual/quality MLP widths, activation (`relu` or `gelu`), and
time encoding (`linear` or `sinusoidal`). Squared-distance softmax additionally
requires a positive temperature. Architecture dimensions and RR constants above
are fixed; unsupported overrides are rejected.

Use the dedicated driver after choosing the explicit training values below:

```sh
python -m training.train_tarc \
  --config /path/to/chosen-config.yaml \
  --data /path/to/train --eval-data /path/to/validation \
  --publication-manifest /path/to/explicit-publications.json \
  --pretrained-checkpoint /path/to/original/model-105.th \
  --output-dir /path/to/new-run \
  --beta YOUR_BETA --sigma YOUR_SIGMA --weight-decay YOUR_WEIGHT_DECAY \
  --epochs YOUR_EPOCHS --batch-size YOUR_BATCH_SIZE --seed YOUR_SEED \
  --device cuda --num-dataloader-workers YOUR_WORKER_COUNT
```

`YOUR_*` tokens are placeholders, not numerical recommendations. `--eval-data`
is optional; an old config's validation path is not reused implicitly. CLI
training values are required and conflicting non-null temporal values are
rejected. Python, NumPy, Torch and CUDA RNGs are seeded from the explicit seed.
`frame_stack` does not create TARC history.

The driver requires `rr_gate: false`: train TARC once, then compare the same
checkpoint with RR-Gate disabled/enabled. Joint AdamW uses fixed learning rates
`1e-4` for TARC and `1e-5` for the original trainable model, with explicit weight
decay. The loss is the original sum of throttle/brake/steer L1 losses plus
`beta * quality_MSE`. The detached quality target is
`exp(-normalized_feature_L1/sigma)` using soft-matched undelayed features.
Messages without reference matches are omitted from that loss.

Current references are supervision only. Their encoding is isolated from
causal feature encoding and does not update encoder BatchNorm/RNG state or
receive gradients. Validation uses `include_reference=False`, so the loader
does not load or sample current remote references; the model's eval path does
not encode references or compute the quality target. An undelayed current
message can still legitimately be history when the configured delay is zero.

Original checkpoints bootstrap training only with
`--pretrained-checkpoint`: all original keys must match and only the entire
new TARC parameter set may be absent. Alternatively `--tarc-checkpoint` loads
complete weights strictly and starts a **fresh optimizer**; optimizer resume
is not implemented. Online inference always requires a complete TARC checkpoint.

Each epoch exports `model-{epoch}.th`, alongside the original Agent-compatible
W&B-style `config.yaml`, `run_config.json`, and per-epoch `metrics.jsonl` in a
new output directory. Point the existing Agent to that `config.yaml` and use
the corresponding checkpoint number. Change only `temporal.rr_gate` to enable
rules for the same trained weights. Do not expect a base-only checkpoint to
perform learned temporal compensation.

## Verification boundaries

```sh
python -m unittest discover -s tests -v
```

Verification includes core, loader/trainer, model/online and zero-floor
derivative tests in the Torch 2.7.1 runtime. Model-flow tests use explicit test
doubles for legacy dependencies; original-vs-disabled output/RNG equality is
also checked on CUDA. Three additional tests use the real Open3D native CPU
operator in a compatible Torch 2.2 runtime: zero-floor forward behavior,
positive unique-max and TARC-to-native gradients, and finite differences for
the training-only zero-floor correction. Native tests skip when a compatible
operator is unavailable.
These tests are not a full Coopernaut training run or a CARLA closed-loop run.
Loader regressions also distinguish a sparse actual publication schedule from
continuous archived sensor data and require explicit failure on missing records.

For 0 ms regression, compare the original path with modules disabled and the
same weights/input/RNG state; separately test zero-residual, neutral-gate
integration. A trained, enabled method at age zero need not equal the original
checkpoint: compensation may change features, and `q<1` can still reduce the
gate. Native max pooling retains its original zero floor, including for
all-negative channels. The tests cover unique maxima and the zero floor;
they do not claim validation of every tie case or every Open3D build.
