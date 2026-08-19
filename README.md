# lerobot-wandb

`lerobot-wandb` is a LeRobot W&B companion integration for an ordinary upstream
[LeRobot](https://github.com/huggingface/lerobot) installation. It runs alongside
an existing upstream LeRobot and moves datasets, models, and rollout datasets
through W&B Artifacts without patching or replacing LeRobot.

The package version is 0.1.0. It is not published to PyPI yet; the
repository checkout is the current installable source while the first release is
prepared.

The names are intentionally stable:

- distribution: `lerobot-wandb`
- import package: `lerobot_wandb`
- console command: `lerobot-wandb`

This is a companion distribution, not a native LeRobot plugin. Upstream LeRobot
does not currently expose a generic dataset-storage or training-lifecycle plugin
contract for this integration.

## Manual

For the reader-facing, SO-101-oriented workflow, start with the [Manual](./MANUAL.md)
or [Manual (日本語)](./MANUAL.ja.md). The shortest route is to materialize a W&B
dataset into a local LeRobot tree, run ordinary upstream `lerobot-train` from that
root, then upload and promote the resulting local model with this companion.

The manual uses the Git source-install route because the first PyPI release is future
work; `pip install lerobot-wandb` is not an available release path yet.

## Install

Install the repository checkout while release ownership and trusted publishing
are completed:

```bash
pip install "lerobot-wandb @ git+https://github.com/guchengwei/lerobot-wandb.git"
```

In a fresh environment, request the tested LeRobot compatibility extra as well:

```bash
pip install "lerobot-wandb[lerobot] @ git+https://github.com/guchengwei/lerobot-wandb.git"
```

The package deliberately leaves LeRobot as a runtime companion rather than a hard
resolver dependency, so it can be installed alongside an existing upstream
installation without making the resolver replace it. Commands that inspect
LeRobot datasets or videos require a supported installed distribution and report
an actionable error when it is absent or outside this release's supported range (`>=0.6.1,<0.6.2`). The only
released LeRobot version currently covered by the blocking compatibility checks
is `0.6.1`; LeRobot `0.6.2` has not been published yet. When a newer release
exists, add it to the compatibility matrix and re-evaluate this specifier before
declaring support.

## Companion workflow

The supported companion composition is explicit and keeps the upstream training
process in charge:

```text
W&B dataset Artifact
        |
        v
lerobot-wandb dataset download/materialize
        |
        v
local LeRobot dataset tree
        |
        v
upstream lerobot-train --dataset.root=...
        |
        v
local trained model
        |
        v
lerobot-wandb model upload/promote
```

For example:

```bash
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

`local/pick-cube` is the local LeRobot dataset label. Replace it consistently in
`--dataset.repo_id` if your materialized dataset uses another label; replace the
example entity, project, Artifact names, and policy settings with your values.

The companion contract covers a W&B-backed remote lifecycle for local directories:

- Validate local dataset/model directories before upload;
- after an Artifact download completes, the downloaded contents are materialized
  dataset/model files on local disk and can be read without a W&B network connection;
- Artifact transfer, requested/resolved refs, sidecars, and lineage metadata;
- dataset review preview media that never replaces canonical dataset bytes;
- rollout Artifact publication with the evaluated model declared as lineage input;
- Registry collection links and explicit Promotion of an evaluated model version.

Current v3 and canonical v2.1 dataset transfer are supported; v2.1 is transfer-only
for rollout publication.

Dataset upload previews are bounded and deterministic. Without a selector, one
representative episode is logged under `dataset_video/representative/<camera>`.
Use `--preview-episode INDEX` for exact episodes or `--preview-all` for an explicit
all-episode request (50 episodes by default). The aggregate Run Media budget is
the smaller of 250 MiB and 20% of the canonical dataset directory. Use
`--no-preview` to disable review media.

## Fork-only behavior

The following are deliberately not reproduced here:

- train-time W&B Artifact references in the fork's `lerobot-train`;
- training-time dataset materialization inside the train command;
- W&B-specific final-model fields or publication on the same training run;
- monkey-patching, import-time mutation, or replacement of upstream LeRobot CLIs.

Those behaviors depend on lifecycle edits in the historical LeRobot fork. Upstream
LeRobot remains in charge of recording, training, rollout, and any optional W&B
logging. This companion does not automatically take over those lifecycles or
publish a final model on the same training run.

## Development

From this repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e '.[test]'
.venv/bin/python -m pytest -q
```

LeRobot-dependent tests are skipped when LeRobot is absent. The CI compatibility
matrix installs the supported upstream releases before running them. Build checks
also verify that a wheel owns only `lerobot_wandb/*` and one
`lerobot-wandb` console entry point, with no `lerobot/*` files.

## Migration provenance

This repository is a clean snapshot of the companion Artifact-transfer surface
from [guchengwei/lerobot](https://github.com/guchengwei/lerobot), source commit
`ebdc227057056e077f90fa10155fd505fa53989d`, completed for
[issue #46](https://github.com/guchengwei/lerobot/issues/46). The snapshot is
intentionally easier to audit than a path-filtered fork history; the source,
boundary decisions, and excluded fork hooks are recorded in [MIGRATION.md](MIGRATION.md).

PyPI registration, trusted-publisher binding, and the first release are future
work. This repository contains the release workflow structure but no token or
publisher credential.
