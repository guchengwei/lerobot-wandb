# lerobot-wandb

`lerobot-wandb` is a standalone companion integration for an ordinary upstream
[LeRobot](https://github.com/huggingface/lerobot) installation. It moves datasets,
models, and rollout datasets through W&B Artifacts without patching or replacing
LeRobot.

The standalone package version is 0.1.0. It is not published to PyPI yet; the
repository checkout is the installable source for this migration PR.

The names are intentionally stable:

- distribution: `lerobot-wandb`
- import package: `lerobot_wandb`
- console command: `lerobot-wandb`

This is a companion distribution, not a native LeRobot plugin. Upstream LeRobot
does not currently expose a generic dataset-storage or training-lifecycle plugin
contract for this integration.

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

The base package deliberately does not hard-depend on `lerobot`. It can therefore
be installed alongside an existing upstream installation without making the
resolver replace it. Commands that inspect LeRobot datasets or videos validate
the installed distribution at runtime and report an actionable error when it is
absent or outside this release's supported range (`>=0.6.1,<0.6.2`). The only
released LeRobot version currently covered by the blocking compatibility checks
is `0.6.1`; LeRobot `0.6.2` has not been published yet. When a newer release
exists, add it to the compatibility matrix and re-evaluate this specifier before
declaring support.

## Portable workflow

The supported standalone composition is explicit and keeps the upstream training
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
  --dataset.root=./datasets/pick-cube \
  --output_dir=./outputs/pick-cube

lerobot-wandb model upload \
  --root ./outputs/pick-cube/pretrained_model \
  --entity my-team --project so101-pick-cube \
  --name pick-cube-policy --alias candidate

lerobot-wandb model promote \
  --ref my-team/so101-pick-cube/pick-cube-policy:v0 \
  --alias production --registry-collection pick-cube-policy
```

The companion owns dataset/model/rollout Artifact transfer, requested and
resolved refs, sidecars, lineage, inspection/validation, model promotion and
Registry links, and browser-playable review previews. Canonical dataset bytes are
never replaced by preview derivatives. Current v3 and canonical v2.1 dataset
transfer are supported; v2.1 is transfer-only for rollout publication.

Dataset upload previews are bounded and deterministic. Without a selector, one
representative episode is logged under `dataset_video/representative/<camera>`.
Use `--preview-episode INDEX` for exact episodes or `--preview-all` for an explicit
all-episode request (50 episodes by default). The aggregate Run Media budget is
the smaller of 250 MiB and 20% of the canonical dataset directory. Use
`--no-preview` to disable review media.

## Fork-only behavior

The following are deliberately not reproduced here:

- `DatasetConfig.artifact_ref` in `lerobot-train`;
- training-time dataset materialization inside the train command;
- W&B-specific final-model fields or publication on the same training run;
- monkey-patching, import-time mutation, or replacement of upstream LeRobot CLIs.

Those behaviors depend on lifecycle edits in the historical LeRobot fork. They
remain legacy fork-only hooks and are not required for this standalone workflow.

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

This repository is a clean standalone snapshot of the portable companion surface
from [guchengwei/lerobot](https://github.com/guchengwei/lerobot), source commit
`ebdc227057056e077f90fa10155fd505fa53989d`, completed for
[issue #46](https://github.com/guchengwei/lerobot/issues/46). The snapshot is
intentionally easier to audit than a path-filtered fork history; the source,
boundary decisions, and excluded fork hooks are recorded in [MIGRATION.md](MIGRATION.md).

PyPI registration, trusted-publisher binding, and the first release are future
work. This repository contains the release workflow structure but no token or
publisher credential.
