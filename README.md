# lerobot-wandb

[日本語](./README.ja.md)

Move LeRobot datasets, trained policies, and rollout results through Weights & Biases Artifacts without replacing or patching upstream [LeRobot](https://github.com/huggingface/lerobot).

![LeRobot and W&B workflow overview](./assets/wandb-workflow-overview-en.jpg)

The diagram shows the full LeRobot and W&B workflow. `lerobot-wandb` handles the Artifact steps around recording, training, and rollout; the robot-facing commands remain standard LeRobot commands.

## What this package does

`lerobot-wandb` is a companion CLI for an existing LeRobot installation. It gives local LeRobot directories a versioned W&B lifecycle:

- upload and download datasets, models, and rollout datasets as W&B Artifacts;
- validate local directories before upload and after download;
- create browser-playable dataset and rollout previews without changing the canonical video files;
- record requested and resolved Artifact references for reproducibility;
- connect rollout results to the exact model version that produced them; and
- promote an evaluated model version with an alias or a W&B Registry link.

It is a separate distribution, not a native LeRobot plugin:

- distribution: `lerobot-wandb`
- Python package: `lerobot_wandb`
- command: `lerobot-wandb`

## How it fits with LeRobot

| Step | Command | Owner |
| --- | --- | --- |
| Record demonstrations | `lerobot-record` | upstream LeRobot |
| Upload or download a dataset | `lerobot-wandb dataset ...` | this companion |
| Train a policy | `lerobot-train` | upstream LeRobot |
| Upload, download, or promote a policy | `lerobot-wandb model ...` | this companion |
| Run a policy on the robot | `lerobot-rollout` | upstream LeRobot |
| Publish rollout results | `lerobot-wandb rollout upload` | this companion |

The handoff is always a local directory. W&B stores finished Artifacts; it is not part of the robot control loop.

## Requirements

- Python 3.10 or later
- a W&B account and network access for Artifact operations
- upstream LeRobot `>=0.6.1,<0.6.2` for commands that inspect LeRobot datasets or videos
- the normal robot, camera, and video dependencies required by your LeRobot setup

The current package version is `0.1.0`. It has not been published to PyPI yet, so install it from GitHub.

## Install

Install the companion beside an existing LeRobot environment:

```bash
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login
lerobot-wandb --help
```

For a fresh environment, install the tested LeRobot extra as well:

```bash
pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot-wandb.git"
```

The base package does not declare LeRobot as a hard dependency. This avoids replacing an existing upstream installation during dependency resolution. At runtime, LeRobot-dependent commands check the installed version and explain how to proceed when it is missing or unsupported. `--allow-unsupported-lerobot` is available for experiments, but it is not a compatibility guarantee.

Set defaults for the examples below:

```bash
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="so101-pick-cube"
```

The examples use Linux and a Bash-compatible shell. On Windows, use the matching PowerShell activation command and replace `/dev/ttyACM*` with your `COM` ports.

## Quick start: train from a W&B dataset

This is the shortest path when a dataset already exists in W&B:

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube:raw" \
  --root ./datasets/pick-cube

lerobot-train \
  --dataset.repo_id=local/pick-cube \
  --dataset.root=./datasets/pick-cube \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=./outputs/train/act_pick_cube \
  --steps=100000 \
  --policy.push_to_hub=false

lerobot-wandb model upload \
  --root ./outputs/train/act_pick_cube/checkpoints/last/pretrained_model \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube-policy \
  --alias candidate
```

The checkpoint path depends on the policy and training configuration. Confirm the path in the LeRobot training output before uploading it.

## End-to-end SO-101 workflow

The following example covers the full path from demonstration recording to model promotion. Replace the ports, camera, task, episode counts, names, and policy settings with values for your setup.

### 1. Record demonstrations locally

Recording is a standard upstream LeRobot operation. W&B is not involved yet.

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id=local/pick-cube \
  --dataset.root=./data/pick-cube \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=30 \
  --dataset.push_to_hub=false
```

`repo_id` is LeRobot's local label. `root` is the directory that the companion will validate and upload.

### 2. Upload the dataset

```bash
lerobot-wandb dataset upload \
  --root ./data/pick-cube \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube \
  --alias raw
```

The command validates the dataset before creating the W&B Run, then prints an immutable resolved reference such as:

```text
your-wandb-entity/so101-pick-cube/pick-cube:v0
```

Save that `vN` reference when reproducibility matters. An alias such as `raw` can later point to another version.

#### Video previews

Canonical video files stay inside the Artifact unchanged. Preview media is a separate browser-playable derivative for review.

- Default: one deterministic representative episode per camera
- Exact episodes: repeat `--preview-episode INDEX`
- All episodes and cameras: `--preview-all`
- Raise the default 50-episode limit: `--preview-max-episodes NUMBER`
- Disable previews: `--no-preview`

The aggregate preview budget is the smaller of 250 MiB and 20% of the canonical dataset directory. Preview files are for inspection, not training.

Current v3 datasets are supported. Canonical v2.1 datasets can be uploaded, downloaded, and materialized. For v2.1, support is limited to dataset transfer; `rollout upload` requires v3.

### 3. Download and materialize the dataset

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube:raw" \
  --root ./datasets/pick-cube
```

The command resolves the requested reference, downloads the Artifact transactionally, validates the result, and writes the dataset to `./datasets/pick-cube`. Once the download finishes, LeRobot reads that local tree directly and does not need a W&B connection for training.

If you requested an alias, record the resolved `vN` reference printed by the command.

### 4. Train with upstream LeRobot

```bash
lerobot-train \
  --dataset.repo_id=local/pick-cube \
  --dataset.root=./datasets/pick-cube \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir=./outputs/train/act_pick_cube \
  --job_name=act_pick_cube \
  --batch_size=8 \
  --steps=100000 \
  --policy.push_to_hub=false
```

This is an ordinary LeRobot training run. The companion does not wrap the command or publish its final model automatically.

To resume with the saved training configuration:

```bash
lerobot-train --resume=true \
  --config_path=./outputs/train/act_pick_cube/checkpoints/last/pretrained_model/train_config.json
```

Check the actual checkpoint layout before continuing. Adapter-only PEFT or LoRA directories can be stored as Artifacts, but rollout and Registry use require a self-contained policy, such as a merged checkpoint.

### 5. Upload the trained model

```bash
lerobot-wandb model upload \
  --root ./outputs/train/act_pick_cube/checkpoints/last/pretrained_model \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube-policy \
  --alias candidate
```

The pre-upload check verifies the expected configuration and weight files. It does not load or execute the weights, so perform any policy-specific validation separately.

Add `--registry-collection pick-cube-policy` if you also want to link a self-contained model to W&B Registry. A model that is not deployable can still be stored as an Artifact, but it cannot receive a deployable Registry link.

### 6. Download the model and publish a rollout

Download the candidate on the robot machine:

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube-policy:candidate" \
  --root ./policies/pick-cube-candidate
```

Copy the immutable reference printed by the command. Use it for rollout lineage instead of the movable `candidate` alias:

```bash
export MODEL_REF="your-wandb-entity/so101-pick-cube/pick-cube-policy:v0"
```

Run the policy with upstream LeRobot:

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path=./policies/pick-cube-candidate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --dataset.repo_id=local/rollout_pick-cube \
  --dataset.root=./data/rollout-pick-cube \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.push_to_hub=false
```

Count successful episodes during the evaluation. After disconnecting the robot, publish the result:

```bash
export EPISODES_SUCCEEDED="14"

lerobot-wandb rollout upload \
  --root ./data/rollout-pick-cube \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube-rollout \
  --model-ref "$MODEL_REF" \
  --episodes-succeeded "$EPISODES_SUCCEEDED"
```

The operator supplies the success count; the companion does not score the physical task. The rollout is stored as a separate `rollout` Artifact and declares the evaluated model as a lineage input. Canonical rollout videos remain unchanged. A deterministic H.264/yuv420p derivative is logged as Run Media for browser playback when video is available.

### 7. Promote the evaluated model

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection pick-cube-policy
```

Promotion moves aliases and optionally adds a Registry link; it does not upload model bytes. Promote the exact version used for evaluation. Re-uploading a downloaded directory would create a new model version without the rollout lineage edge.

## Resulting W&B objects

| Object | Location |
| --- | --- |
| Teaching data | `dataset` Artifact collection `pick-cube` |
| Training input | materialized local directory `./datasets/pick-cube` |
| Trained policy | local checkpoint, then `model` Artifact `pick-cube-policy` |
| Evaluation episodes | local rollout tree, then `rollout` Artifact `pick-cube-rollout` |
| Dataset-to-policy trace | selected dataset reference plus the local training configuration |
| Policy-to-rollout trace | rollout Run input edge and rollout Artifact metadata |

## What this companion does not do

The historical LeRobot fork included W&B behavior inside training. This repository intentionally does not reproduce those hooks. It does not:

- accept a W&B Artifact reference directly inside `lerobot-train`;
- materialize a dataset from within the training command;
- publish the final model on the same training Run;
- replace upstream `lerobot-record`, `lerobot-train`, or `lerobot-rollout`;
- monkey-patch LeRobot or install files into the `lerobot` package; or
- act as a streaming recorder or deployment controller.

Keeping these boundaries explicit lets the companion run beside an ordinary upstream LeRobot installation.

## Troubleshooting

- **LeRobot is missing or unsupported:** install upstream LeRobot `0.6.1`. Use `--allow-unsupported-lerobot` only after checking compatibility for your environment.
- **Preview encoding is unavailable:** add `--no-preview` to dataset upload. Canonical Artifact files are unaffected.
- **A downloaded alias changed:** use the resolved immutable `vN` reference printed by the command for training records, rollout lineage, and promotion.
- **A model cannot be linked to Registry:** confirm that the directory contains a complete deployable policy rather than only an adapter.
- **A command or option differs from this page:** use `lerobot-wandb --help`, the subcommand's `--help`, and `pyproject.toml` as the command reference for this release.

## Development

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot-dependent tests are skipped when LeRobot is absent. The compatibility CI installs supported upstream releases before running those tests. Build checks also confirm that the wheel owns only `lerobot_wandb/*` and the `lerobot-wandb` console entry point.

## Migration history

This repository is a clean snapshot of the companion Artifact-transfer surface from [guchengwei/lerobot](https://github.com/guchengwei/lerobot), source commit `ebdc227057056e077f90fa10155fd505fa53989d`, created for [issue #46](https://github.com/guchengwei/lerobot/issues/46). See [MIGRATION.md](./MIGRATION.md) for the source boundary and the fork-specific hooks that were left behind.
