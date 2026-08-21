# lerobot-wandb

[日本語](./README.ja.md)

`lerobot-wandb` stores and retrieves LeRobot datasets, trained policies, and rollout results as Weights & Biases Artifacts without modifying upstream [LeRobot](https://github.com/huggingface/lerobot).

![LeRobot and W&B workflow overview](./assets/wandb-workflow-overview-en.jpg)

The diagram shows the full LeRobot and W&B workflow. `lerobot-wandb` handles the Artifact steps around recording, training, and rollout; the robot-facing commands remain standard LeRobot commands.

## What it is

`lerobot-wandb` is a companion CLI for an existing LeRobot installation. It adds Artifact transfer and validation around a normal LeRobot workflow rather than replacing LeRobot itself.

It can:

- upload and download datasets, models, and rollout datasets as W&B Artifacts;
- validate local directories before upload and after download;
- create browser-playable review previews without changing canonical video files;
- record both requested Artifact references and the immutable versions they resolve to;
- attach rollout results to the model version used for evaluation; and
- promote an evaluated model with aliases or a W&B Registry link.

LeRobot and `lerobot-wandb` own different parts of the workflow. `lerobot-wandb` is a separate distribution, not a native LeRobot plugin:

- Python distribution: `lerobot-wandb`
- import package: `lerobot_wandb`
- console command: `lerobot-wandb`

LeRobot continues to own robot control, recording, training, and policy execution.

| Task | Command | Owner |
| --- | --- | --- |
| Record demonstrations | `lerobot-record` | LeRobot |
| Upload or download a dataset | `lerobot-wandb dataset ...` | `lerobot-wandb` |
| Train a policy | `lerobot-train` | LeRobot |
| Upload, download, or promote a model | `lerobot-wandb model ...` | `lerobot-wandb` |
| Run a policy on the robot | `lerobot-rollout` | LeRobot |
| Publish rollout results | `lerobot-wandb rollout upload` | `lerobot-wandb` |

The handoff between the two tools is a local directory. W&B stores completed Artifacts; it is not part of the robot control loop.

## Requirements

- Python 3.12 or later
- a W&B account and network access when reading or writing Artifacts
- upstream LeRobot `>=0.6.1,<0.6.2` for commands that inspect LeRobot datasets or videos
- the system, robot, camera, and video dependencies required by your LeRobot setup

The current package version is `0.1.0`. It has not been published to PyPI yet, so install it from GitHub.

## Install

Install `lerobot-wandb` into the same Python environment as LeRobot. Commands that inspect LeRobot data import LeRobot from that environment.

If you already have a LeRobot checkout with `.venv`:

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login
lerobot-wandb --help
```

For a new workspace, create one environment and install both packages into it:

```bash
mkdir lerobot-workspace
cd lerobot-workspace

uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate

uv pip install "lerobot[core_scripts,training,feetech]==0.6.1"
uv pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"

wandb login
lerobot-info
lerobot-wandb --help
```

`feetech` is included here because the examples below use an SO-101. Choose the LeRobot extra required by your own hardware. For FFmpeg, PyTorch, and other platform-specific setup, follow the upstream [LeRobot installation guide](https://huggingface.co/docs/lerobot/installation).

The base package does not declare LeRobot as a hard dependency, so installing the companion does not ask the resolver to replace an existing LeRobot installation. Commands that need LeRobot check its installed version at runtime. `--allow-unsupported-lerobot` bypasses the supported-version check for experiments, but it does not make an unsupported version compatible.

## Uninstall

Activate the LeRobot environment and remove only the companion package:

```bash
source /path/to/lerobot/.venv/bin/activate
uv pip uninstall lerobot-wandb
```

This removes the `lerobot-wandb` distribution, the `lerobot_wandb` import package, and the `lerobot-wandb` console command. LeRobot remains installed and is not modified. Shared dependencies such as `wandb`, `datasets`, and `pandas` are not automatically removed.

Package uninstall does not delete local datasets, downloaded or materialized Artifacts, models, rollout directories, training outputs, sidecar metadata, or other user data. It also does not delete remote W&B Artifacts, Runs, or Registry objects, and it does not remove W&B authentication or configuration.

## Configure the workflow examples

The examples use the same W&B names and local paths throughout. Set them once for your project:

```bash
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="your-wandb-project"
export DATASET_NAME="your-dataset"
export POLICY_NAME="your-policy"
export ROLLOUT_NAME="your-rollout"

export DATASET_ROOT="./data/$DATASET_NAME"
export TRAIN_DATASET_ROOT="./datasets/$DATASET_NAME"
export TRAIN_OUTPUT="./outputs/train/$POLICY_NAME"
export POLICY_ROOT="$TRAIN_OUTPUT/checkpoints/last/pretrained_model"
export DOWNLOADED_POLICY_ROOT="./policies/$POLICY_NAME-candidate"
export ROLLOUT_ROOT="./data/$ROLLOUT_NAME"
```

These names are examples, not a naming convention enforced by the package.

The shell examples assume Linux and Bash. On Windows, activate the same LeRobot environment with the appropriate PowerShell command and replace `/dev/ttyACM*` with the relevant `COM` ports.

## Quick start: train from a W&B dataset

If the dataset already exists in W&B, the shortest path is download → train → upload:

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"

lerobot-train \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$TRAIN_OUTPUT" \
  --job_name="$POLICY_NAME" \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.entity="$WANDB_ENTITY" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false

lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

LeRobot creates the W&B training Run and logs its training metrics. `--wandb.disable_artifact=true` keeps checkpoint publication out of that Run; the following `lerobot-wandb model upload` remains the model Artifact publication step.

Checkpoint layouts vary by policy and training configuration. Confirm the actual LeRobot output path before uploading a model.

## Full workflow

The W&B steps are the same regardless of robot. The LeRobot commands below use SO-101 as a concrete example.

### 1. Record demonstrations with LeRobot

Recording stays entirely in LeRobot:

```bash
lerobot-record \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$DATASET_ROOT" \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.num_episodes=30 \
  --dataset.push_to_hub=false
```

`repo_id` is LeRobot's local identifier. `root` is the directory that `lerobot-wandb` will validate and upload.

### 2. Publish the dataset to W&B

```bash
lerobot-wandb dataset upload \
  --root "$DATASET_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$DATASET_NAME" \
  --alias raw
```

The command validates the dataset before starting the W&B Run. After upload it prints the resolved immutable reference, for example:

```text
your-wandb-entity/your-wandb-project/your-dataset:v0
```

Keep the `vN` reference when reproducibility matters. An alias such as `raw` can move to a later version.

#### Dataset video previews

Canonical dataset videos are uploaded unchanged. Review previews are separate H.264-compatible derivatives used only for browser playback.

Preview selection:

- default: one deterministic representative episode per camera;
- exact episodes: repeat `--preview-episode INDEX`;
- every episode and camera: `--preview-all`;
- no previews: `--no-preview`.

Selected previews are logged once in a `dataset_previews` W&B Table with `episode`, `camera`, and `camera_key` columns. The CLI rejects selections above 10,000 episode-camera rows.

Preview files are prepared before `wandb.init()`. The CLI reports each selected episode/camera as it is processed, then shows the measured total before upload. The preview budget is `min(250 MiB, 20% of the canonical dataset directory size)`.

If the measured previews exceed that budget, an interactive terminal asks for confirmation and defaults to No. In non-interactive environments the command fails unless `--force-preview-budget` is set. That flag only accepts the measured preview-size overage; it does not bypass row limits, dataset validation, episode bounds, encoding failures, temporary-path checks, or W&B upload errors.

Current v3 datasets are supported. Canonical v2.1 datasets can also be uploaded, downloaded, and materialized; v2.1 support is limited to dataset transfer. `rollout upload` requires v3.

### 3. Download the training dataset

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$DATASET_NAME:raw" \
  --root "$TRAIN_DATASET_ROOT"
```

The download is transactional: the command resolves the requested reference, downloads the Artifact, validates it, and only then places the result at `$TRAIN_DATASET_ROOT`. Once the download finishes, LeRobot reads that local tree directly; W&B is not needed to access the training data. With `--wandb.enable=true` below, the training process still needs W&B connectivity to sync metrics.

If you used an alias, save the resolved `vN` reference printed by the command.

### 4. Train with LeRobot

```bash
lerobot-train \
  --dataset.repo_id="local/$DATASET_NAME" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="$TRAIN_OUTPUT" \
  --job_name="$POLICY_NAME" \
  --batch_size=8 \
  --steps=100000 \
  --wandb.enable=true \
  --wandb.entity="$WANDB_ENTITY" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.disable_artifact=true \
  --policy.push_to_hub=false
```

This is a normal LeRobot training run. The `--wandb.*` flags use LeRobot's built-in W&B integration to log the training Run and metrics to `$WANDB_ENTITY/$WANDB_PROJECT`. `--wandb.disable_artifact=true` prevents LeRobot from also publishing checkpoint Artifacts because this workflow publishes the validated model explicitly in step 5. `lerobot-wandb` does not wrap `lerobot-train` or upload the final checkpoint automatically.

To resume from the saved training configuration:

```bash
lerobot-train --resume=true \
  --config_path="$POLICY_ROOT/train_config.json"
```

Check the actual checkpoint layout before continuing. Adapter-only PEFT/LoRA directories can be stored as Artifacts, but rollout and Registry workflows need a self-contained policy, such as a merged checkpoint.

### 5. Publish the trained model

```bash
lerobot-wandb model upload \
  --root "$POLICY_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$POLICY_NAME" \
  --alias candidate
```

The pre-upload check verifies the expected configuration and weight files. It does not load or execute the weights, so run policy-specific validation separately.

Add `--registry-collection "$POLICY_NAME"` if you also want to link a self-contained model into W&B Registry. An incomplete or adapter-only model can still be stored as an Artifact, but it should not be promoted as a deployable Registry entry.

### 6. Evaluate the model and publish rollouts

Download the candidate onto the robot machine:

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/$POLICY_NAME:candidate" \
  --root "$DOWNLOADED_POLICY_ROOT"
```

Use the immutable reference printed by the command for evaluation lineage. Do not assume the version is `v0`:

```bash
export MODEL_REF="paste-the-resolved-vN-reference-here"
```

Run the policy with LeRobot:

```bash
lerobot-rollout \
  --strategy.type=episodic \
  --policy.path="$DOWNLOADED_POLICY_ROOT" \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --dataset.repo_id="local/$ROLLOUT_NAME" \
  --dataset.root="$ROLLOUT_ROOT" \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick up the cube and place it in the bin" \
  --dataset.push_to_hub=false
```

After evaluation, provide the number of successful episodes and upload the rollout dataset:

```bash
export EPISODES_SUCCEEDED="14"

lerobot-wandb rollout upload \
  --root "$ROLLOUT_ROOT" \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name "$ROLLOUT_NAME" \
  --model-ref "$MODEL_REF" \
  --episodes-succeeded "$EPISODES_SUCCEEDED"
```

You enter the number of successful episodes; `lerobot-wandb` does not decide whether an episode succeeded. The rollout is stored as a separate `rollout` Artifact and records the evaluated model as a lineage input. Canonical rollout videos remain unchanged; when video is present, a deterministic H.264/yuv420p derivative is logged as Run Media for browser playback.

### 7. Promote the evaluated model

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection "$POLICY_NAME"
```

Promotion moves aliases and can add a Registry link. It does not upload model bytes. Promote the exact model version that was evaluated rather than re-uploading a downloaded directory as a new version.

## What gets stored where

| Item | Location |
| --- | --- |
| Demonstration data | local dataset, then a W&B `dataset` Artifact |
| Training input | materialized local directory `$TRAIN_DATASET_ROOT` |
| Training metrics | W&B Run created by `lerobot-train` in `$WANDB_PROJECT` |
| Trained policy | local checkpoint, then a W&B `model` Artifact |
| Evaluation episodes | local rollout directory, then a W&B `rollout` Artifact |
| Dataset → model trace | dataset reference plus local training configuration |
| Model → rollout trace | rollout Run input edge plus rollout Artifact metadata |

## What this companion does not do

The historical LeRobot fork included companion-specific W&B Artifact behavior inside the training path. This repository does not reproduce those hooks. Upstream LeRobot's native `--wandb.*` training logger still works normally. In particular, this companion does not:

- accept a W&B Artifact reference directly inside `lerobot-train`;
- materialize a dataset from inside the training command;
- publish the final model on the same training Run;
- replace `lerobot-record`, `lerobot-train`, or `lerobot-rollout`;
- monkey-patch LeRobot or install files into the `lerobot` package; or
- act as a streaming recorder or deployment controller.

## Troubleshooting

- **LeRobot is missing or unsupported:** activate the environment that contains LeRobot and confirm that upstream LeRobot `0.6.1` is installed. Use `--allow-unsupported-lerobot` only after checking compatibility yourself.
- **`lerobot-wandb` and `lerobot-*` resolve from different environments:** reactivate the LeRobot environment and install the companion there with `uv pip install`.
- **Preview encoding is unavailable:** add `--no-preview` to `dataset upload`. Canonical Artifact files are unaffected.
- **Preview media exceeds the budget:** reduce the preview selection, use `--no-preview`, or use `--force-preview-budget` in a non-interactive environment when the measured size overage is intentional.
- **An alias now points to a different version:** use the resolved immutable `vN` reference for training records, rollout lineage, and promotion.
- **A model cannot be linked to Registry:** check that the directory contains a complete deployable policy rather than only an adapter.
- **A command differs from this README:** treat `lerobot-wandb --help`, the subcommand `--help`, and `pyproject.toml` as the command reference for the installed release.

## Development

From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot-dependent tests are skipped when LeRobot is absent. Compatibility CI installs supported upstream releases before running those tests. Build checks also verify that the wheel owns only `lerobot_wandb/*` and the `lerobot-wandb` console entry point.

## Migration history

This repository was extracted from [guchengwei/lerobot](https://github.com/guchengwei/lerobot) at source commit `ebdc227057056e077f90fa10155fd505fa53989d` as part of [issue #46](https://github.com/guchengwei/lerobot/issues/46). See [MIGRATION.md](./MIGRATION.md) for the exact source boundary and the fork-specific behavior intentionally left behind.
