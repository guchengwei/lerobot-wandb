# W&B companion manual for LeRobot (SO-101 example)

![W&B companion workflow overview](./assets/wandb-workflow-overview-en.svg)

[Manual (日本語)](./MANUAL.ja.md) · [Project README](./README.md)

This manual is for a user who already has ordinary upstream LeRobot installed and wants to move
LeRobot datasets, models, and rollout results through W&B Artifacts. `lerobot-wandb` is a LeRobot
W&B companion integration that runs alongside an existing upstream LeRobot. It provides a companion
CLI; upstream LeRobot does not currently expose a generic plugin contract for this integration, so
this companion is not presented as a native plugin. It does not patch, replace, or install files into
LeRobot.

The examples use version `0.1.0` of the companion and support the LeRobot range `>=0.6.1,<0.6.2`.
There is no PyPI release yet. The source-install command below is the current route; after a
future first release, the command can become `pip install lerobot-wandb`. Do not assume that
future command works today.

The diagram is conceptual. The commands and boundaries below describe the companion interface.
They have not been live-verified against a W&B workspace or robot in this documentation change;
provide your own entity, project, hardware ports, camera settings, and success counts.

## Shortest portable route

The companion composition keeps the upstream training process in charge:

```mermaid
flowchart LR
    A[W&B dataset Artifact] --> B[dataset download/materialize]
    B --> C[local LeRobot dataset tree]
    C --> D[upstream lerobot-train --dataset.root]
    D --> E[local trained model]
    E --> F[model upload/promote]
```

The shortest working route is:

```bash
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login

lerobot-wandb dataset download \
  --ref my-team/so101-pick-cube/pick-cube:raw \
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
  --entity my-team --project so101-pick-cube \
  --name pick-cube-policy --alias candidate

lerobot-wandb model promote \
  --ref my-team/so101-pick-cube/pick-cube-policy:v0 \
  --alias production --registry-collection pick-cube-policy
```

Replace `my-team`, project and Artifact names, the local `repo_id` label if needed, policy settings,
and the resolved model version.
The model path is the upstream LeRobot checkpoint produced under `output_dir`; confirm its exact
location for the policy and training configuration you selected. `model promote` should use the
immutable version that was evaluated, not an alias that may move.

## Companion boundary

This repository deliberately owns Artifact transfer and inspection, not the upstream training
lifecycle. In particular:

- Download a dataset to a local LeRobot tree before calling upstream `lerobot-train`.
- Upload or promote the resulting local model in a separate companion command.
- The training process does not materialize a W&B Artifact or publish a final model on the same
  training run.
- The historical fork's train-time Artifact option and W&B-specific final-model fields are not
  part of this companion interface. Do not copy those fork-only options into an upstream
  `lerobot-train` command.
- There is no import-time patching, wrapper replacement, or hidden dependency on a LeRobot fork.

The companion runs alongside an existing upstream LeRobot without replacing it. Its distribution
leaves LeRobot as a runtime companion rather than a hard resolver dependency, so commands that
inspect LeRobot datasets or videos can use the existing installation. They check its version at
runtime and produce an actionable error when it is missing or unsupported. The global
`--allow-unsupported-lerobot` option is an experimental escape hatch, not a compatibility guarantee.

The Japanese document is a translated mirror of this manual. It does not provide a separate
product, command, or compatibility range.

## 0. Install and prerequisites

Install the companion into the environment that already contains ordinary upstream LeRobot. The
repository is the installable source until the first PyPI release:

```bash
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
wandb login
export WANDB_ENTITY="your-wandb-entity"
export WANDB_PROJECT="so101-pick-cube"
lerobot-wandb --help
```

For a fresh environment, the tested LeRobot extra can be requested from the same source checkout:

```bash
pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot-wandb.git"
```

The base install intentionally leaves an existing LeRobot installation in place. Install and
configure upstream LeRobot using its own documentation, and check that its version is in the
supported range before starting. The companion's upload, download, and promotion commands need
W&B network access and credentials. The recording and rollout commands also need the appropriate
robot, teleoperator, camera, and video dependencies.

The worked commands use Linux and a Bash-compatible shell. In PowerShell, activate the environment
with `.venv\Scripts\Activate.ps1` and replace `/dev/ttyACM*` with the corresponding `COM` ports.
The CLI argument names stay the same.

W&B is a durable store for finished Artifacts, not a filesystem that the robot reaches through.
Local disk remains the recording buffer, materialization destination, and rollout input. No W&B
call is required inside a robot control loop.

## 1. Record a local teaching dataset

This is standard upstream LeRobot recording; W&B is not involved yet. Adjust the robot type,
ports, camera, task, and episode count for your setup.

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

The local `repo_id` is a label used by LeRobot; the `root` is the directory whose bytes will be
validated and uploaded. Keep the root outside any temporary preview directory.

## 2. Upload the dataset Artifact and review media

Validate and publish the local dataset with the companion:

```bash
lerobot-wandb dataset upload \
  --root ./data/pick-cube \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube \
  --alias raw
```

The command validates metadata, schema, Parquet files, indices, and referenced videos before it
creates the W&B Run. It prints the immutable resolved reference, for example
`your-wandb-entity/so101-pick-cube/pick-cube:v0`. Save that reference when reproducibility matters.

Dataset bytes remain canonical Artifact files. If video exists, the default deterministic
representative preview is a separate browser-playable Run Media item. To request exact episodes,
repeat `--preview-episode`; to request all episodes and cameras, use `--preview-all`. All-episode
review is capped at 50 episodes by default and can be raised explicitly with
`--preview-max-episodes`. Use `--no-preview` when review media is not wanted or an H.264 encoder is
not available. Preview media is for inspection only and must not be treated as training data.

The current v3 dataset layout is supported. Canonical v2.1 datasets can be transferred and
materialized, but v2.1 is transfer-only here and cannot be used for rollout evaluation publication.

## 3. Download and materialize the exact dataset locally

Download the immutable or named dataset version into a local tree before training:

```bash
lerobot-wandb dataset download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube:raw" \
  --root ./datasets/pick-cube
```

The companion creates a lineage Run, resolves the requested reference, validates the downloaded
tree, and writes the files under `./datasets/pick-cube`. If an alias is used, record the resolved
`vN` reference printed by the command; the alias can point to another version later.

## 4. Train with upstream LeRobot from the local tree

Run the ordinary upstream `lerobot-train` command. The explicit local root is the seam between the
companion and LeRobot:

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

This training command stays under upstream LeRobot's control and is intentionally separate from
the companion's W&B transfer commands. It reads the materialized LeRobot tree and saves a local
checkpoint. With the shown `output_dir`, the final policy is normally under
`./outputs/train/act_pick_cube/checkpoints/last/pretrained_model`; inspect the training log and
checkpoint layout before uploading it. The companion does not infer or rewrite this path.

To resume, use the saved upstream training configuration rather than an output directory alone:

```bash
lerobot-train --resume=true \
  --config_path=./outputs/train/act_pick_cube/checkpoints/last/pretrained_model/train_config.json
```

If your policy or upstream release uses a different checkpoint layout, pass that verified local
directory to `model upload` in the next step. PEFT/LoRA adapter-only directories may be uploaded for
storage, but they are not self-contained deployable policies; publish a merged checkpoint for
rollout and Registry use.

## 5. Upload and fetch the trained model

Upload the local policy directory as a versioned model Artifact. The pre-upload structural
validation checks expected configuration and weight files; it does not load or execute the
weights. Perform model-specific validation before rollout:

```bash
lerobot-wandb model upload \
  --root ./outputs/train/act_pick_cube/checkpoints/last/pretrained_model \
  --entity "$WANDB_ENTITY" \
  --project "$WANDB_PROJECT" \
  --name pick-cube-policy \
  --alias candidate
```

The companion validates the model manifest before creating its upload Run. This is structural
validation only; it checks expected configuration and weight files, but does not load or execute
the weights. An optional
`--registry-collection pick-cube-policy` can link a self-contained model into the unified W&B
Registry; an adapter-only or otherwise non-deployable directory is still uploaded but is refused a
deployable Registry link.

On the machine that will run the robot, download the candidate into a local policy directory:

```bash
lerobot-wandb model download \
  --ref "$WANDB_ENTITY/$WANDB_PROJECT/pick-cube-policy:candidate" \
  --root ./policies/pick-cube-candidate
```

Copy the full immutable reference printed by the command into `MODEL_REF`. Do not replace it with
the mutable alias when recording rollout lineage:

```bash
export MODEL_REF="your-wandb-entity/so101-pick-cube/pick-cube-policy:v0"
```

The resulting directory is a local upstream policy path. The download is transactional and repeats
structural validation of the expected configuration and weight files; it does not load or execute
the weights. Perform model-specific validation before rollout.

## 6. Roll out on the robot and publish the result

Run the ordinary upstream rollout command against the downloaded local policy. The `rollout_`
prefix is the upstream convention for a local evaluation dataset.

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

Count successful episodes while the rollout runs. Disconnect the robot before publishing the
result; upload is a separate W&B operation:

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

The success count is supplied by the operator; the companion does not score physical task success.
Rollout publication accepts the current v3 layout. The model is declared as an input for lineage
and is not downloaded again. The rollout remains a distinct `rollout` Artifact, and its canonical
video bytes are kept unchanged. When a video exists, a deterministic H.264/yuv420p derivative is
logged as Run Media for browser playback only; it is not part of the Artifact and is not training
data.

Use the immutable `MODEL_REF`, not `candidate`. Otherwise a moved alias could associate the result
with a model that the robot did not use.

## 7. Promote the evaluated version

Promotion changes aliases and, optionally, a Registry link. It uploads no model bytes:

```bash
lerobot-wandb model promote \
  --ref "$MODEL_REF" \
  --alias production \
  --registry-collection pick-cube-policy
```

Promote the exact version evaluated by the rollout. Re-uploading the downloaded directory would
create a new version without the evaluation lineage edge. A non-model Artifact is rejected, and a
model that is not a self-contained policy cannot receive a deployable Registry link. There is no
automatic production decision: inspect the rollout Run and decide whether the evaluated version is
ready.

## Where things live afterwards

| Object | Local or W&B location |
| --- | --- |
| Teaching dataset | `dataset` Artifact collection `pick-cube` |
| Materialized training input | `./datasets/pick-cube` |
| Trained policy | local checkpoint, then `model` Artifact `pick-cube-policy` |
| Rollout episodes | local rollout tree, then `rollout` Artifact `pick-cube-rollout` |
| Dataset-to-policy trace | the chosen Artifact ref and your local training configuration |
| Policy-to-rollout trace | rollout Run input edge and rollout Artifact metadata |

The purpose of this workflow is reproducible handoff: canonical dataset bytes are materialized
locally, upstream LeRobot trains from that tree, and the separately uploaded model and rollout keep
the immutable references needed to explain what was evaluated and promoted.

## Troubleshooting and verification boundaries

- `lerobot-wandb --help`, each subcommand's `--help`, and the source `pyproject.toml` are the
  authoritative CLI surface for this release.
- If runtime compatibility fails, install ordinary upstream LeRobot `0.6.1` or use the documented
  experimental override only after checking compatibility yourself.
- If preview encoding is unavailable, use `--no-preview` for dataset upload. Canonical Artifact
  files are unaffected by that choice.
- Upload, download, and promotion need online W&B credentials. This manual does not claim that a
  live workspace, robot, or physical success result was verified.
- PyPI registration, trusted-publisher binding, and the first release remain future work. Use the
  Git source-install route until the project announces a release.
