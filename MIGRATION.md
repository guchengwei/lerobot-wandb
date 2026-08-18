# Issue #46 migration record

This repository is the standalone owner of the `lerobot-wandb` companion
distribution. It is intentionally a clean snapshot rather than a path-filtered
copy of the complete LeRobot fork.

## Source and scope

- source repository: `https://github.com/guchengwei/lerobot`
- source snapshot: `ebdc227057056e077f90fa10155fd505fa53989d`
- migration issue: `https://github.com/guchengwei/lerobot/issues/46`
- distribution version: `0.1.0`
- supported LeRobot contract: `>=0.6.1,<0.6.2`, validated by the CI compatibility matrix against the existing `0.6.1` release
- LeRobot `0.6.2` is not published yet and is intentionally not a blocking CI target; when it is released, add it to the matrix and re-evaluate this specifier after compatibility testing

The snapshot includes the portable companion surface under the former
`packages/lerobot-wandb` path: Artifact transfer and download/materialization,
dataset inspection and validation, v2.1/v3 transfer handling, bounded review
previews, model upload/download/promotion, rollout publication, refs, sidecars,
lineage and the W&B SDK boundary.

## Explicitly excluded

The historical fork's training lifecycle integration is not portable across an
ordinary upstream LeRobot release and is not copied here. In particular, this
repository does not provide or emulate:

- `DatasetConfig.artifact_ref`;
- train-time Artifact materialization inside `lerobot-train`;
- fork-specific final-model or Registry publication fields;
- monkey patches, import-time mutations, private entry-point tricks, or replacement
  wrappers for upstream `lerobot-train`.

Use the explicit portable composition documented in the README: materialize a
dataset locally, run upstream `lerobot-train --dataset.root=...`, then upload or
promote the resulting model with this companion.

## Two-phase cutover

This migration deliberately stops after the standalone repository and its open
PR are independently buildable. The source fork is still in active use while
the PR is reviewed, so the current fork keeps its embedded
packages/lerobot-wandb copy and editable/path dependency. That copy is a
temporary compatibility surface, not a second standalone release authority.

After this PR is merged and the standalone package has an installable release,
the second phase will update the fork's documentation and dependency wiring,
remove the embedded copy and editable dependency, and rerun the fork's training
extra checks. That later cutover is intentionally separate: deleting the copy
before the standalone package is installable would break the fork's current
training workflow.

## Ownership and release boundary

The wheel installs only `lerobot_wandb/*` plus the single `lerobot-wandb` console
script. LeRobot imports stay behind `lerobot_wandb.lerobot_adapter`, and the base
distribution does not declare LeRobot as a mandatory dependency. Installing or
uninstalling this distribution must leave an existing LeRobot installation's
version, location, files, and commands unchanged.

Tags and releases belong to this repository. The release workflow is prepared for
OIDC trusted publishing but is intentionally not enabled until PyPI registration
and publisher binding are completed by a repository owner.
