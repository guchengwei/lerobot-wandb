# Dataset Preview Table Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish dataset review videos as one filterable `dataset_previews` W&B Table and replace the obsolete 50-episode preview guard with a universal 10,000-row preflight.

**Architecture:** Keep dataset inspection, source selection, encoding, byte-budget policy, canonical Artifact upload, and temporary-file ownership unchanged. Enforce the W&B publication limit in `cmd_dataset_upload()` immediately after source expansion, then construct the complete Table in a small CLI-layer helper while derivatives remain alive through `run.finish()`.

**Tech Stack:** Python 3.12+, W&B Python SDK `>=0.24.1,<0.28.0`, pytest 8, Ruff.

## Global Constraints

- One selected `DatasetPreviewSource` becomes exactly one Table row; accept 10,000 rows and reject 10,001 before encoding or any W&B side effect.
- Use the stable key `dataset_previews` only; never dual-write old `dataset_video/...` media keys.
- Table columns, in order: `episode`, `camera`, `camera_key`, `selection`, `video`, `source_path`, `preview_bytes`, `transcoded`.
- `camera` is the final non-empty component split on both `.` and `/`; `camera_key` remains exact and authoritative.
- Selection is one invocation-wide value: `representative`, `explicit`, or `all`; episode remains the actual integer.
- Preserve the canonical Artifact, encoding/fast path, byte budget, progress, summary facts, and `ExitStack`/`run.finish()` lifetime.
- Remove `--preview-max-episodes`, `DEFAULT_PREVIEW_MAX_EPISODES`, `max_episodes`, `dataset_media_key()`, and all compatibility aliases/callers.
- No `row_id`, optional table-key summary field, batching, truncation, aliases, or new dependency.

---

## File map

- `src/lerobot_wandb/cli.py`: owns the 10,000-row publication preflight, selection label, Table helper, stable log key, and CLI option removal.
- `src/lerobot_wandb/dataset_transfer.py`: source selection only; remove the obsolete episode-count policy.
- `src/lerobot_wandb/dataset_preview.py`: encoding/preparation only; remove the dead dynamic media-key formatter.
- `tests/test_dataset_transfer.py`: preserve representative/explicit/all selection while deleting episode-limit expectations.
- `tests/test_dataset_upload_media.py`: mock W&B and prove row schema, derivation, orchestration, side-effect ordering, lifetime, and row boundaries.
- `README.md`, `README.ja.md`: synchronized user contract for the Table and 10,000-row limit.

### Task 1: Replace the episode guard with a row preflight

**Files:**
- Modify: `src/lerobot_wandb/dataset_transfer.py`
- Modify: `src/lerobot_wandb/cli.py`
- Test: `tests/test_dataset_transfer.py`
- Test: `tests/test_dataset_upload_media.py`

**Interfaces:**
- Consumes: `select_dataset_preview_sources(dataset, *, episodes=(), preview_all=False)`.
- Produces: `DATASET_PREVIEW_TABLE_MAX_ROWS = 10_000` and a `cmd_dataset_upload()` preflight over the expanded `sources` list.

- [ ] **Step 1: Rewrite selection tests for the clean interface**

Remove tests that expect `--preview-all` to fail based only on episode count. Update all direct calls to omit `max_episodes`. Retain tests proving default representative selection, repeatable explicit selectors, all selection, index validation, duplicate removal, and explicit/all mutual exclusion.

Add or retain this observable contract:

```python
sources = select_dataset_preview_sources(dataset, preview_all=True)
assert len(sources) == dataset.metadata.total_episodes * len(dataset.video_keys)
```

- [ ] **Step 2: Run transfer tests and observe the old API expectations**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_transfer.py -q
```

Expected before implementation: failures from the removed `max_episodes` expectations or the current 50-episode refusal.

- [ ] **Step 3: Remove the episode-count policy from selection**

Change the signature to:

```python
def select_dataset_preview_sources(
    dataset: TransferDataset,
    *,
    episodes: Sequence[int] = (),
    preview_all: bool = False,
) -> list[DatasetPreviewSource]:
```

Delete `DEFAULT_PREVIEW_MAX_EPISODES`, `max_episodes` validation, and the `dataset.metadata.total_episodes > max_episodes` branch. Preserve every other selection and validation branch.

- [ ] **Step 4: Add failing upload preflight tests**

In `tests/test_dataset_upload_media.py`, use synthetic source lists and mocks rather than real videos. Prove:

```python
# 10_000 sources proceeds to prepare_dataset_previews.
# 10_001 sources raises DatasetDirectoryError.
# On rejection, prepare_dataset_previews, wandb.init, upload_directory, and run.log are untouched.
```

Cover the argument labels representative, explicit, and all through parametrized namespaces. Also assert 60 explicit selected sources are accepted, proving no separate 50-episode guard remains.

- [ ] **Step 4a: Add a parser regression for the removed option**

Replace `test_dataset_preview_all_has_a_positive_configurable_default_limit` with:

```python
def test_preview_max_episodes_is_rejected(capsys):
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "dataset",
                "upload",
                "--root",
                "dataset",
                "--project",
                "project",
                "--name",
                "name",
                "--preview-all",
                "--preview-max-episodes",
                "51",
            ]
        )
    assert "unrecognized arguments: --preview-max-episodes 51" in capsys.readouterr().err
```

Before removing the option, this test fails because parsing succeeds.

- [ ] **Step 5: Run the focused boundary tests and observe failure**

Run the exact new node IDs with:

```bash
.venv/bin/python -m pytest tests/test_dataset_upload_media.py -q -k "preview_row_limit or former_episode_limit or preview_max_episodes_is_rejected"
```

Expected before implementation: 10,001 reaches preparation or no row-limit error is raised.

- [ ] **Step 6: Implement the CLI preflight and parser clean cutover**

In `cli.py`, define:

```python
DATASET_PREVIEW_TABLE_MAX_ROWS = 10_000
```

Select without `max_episodes`, then before `ExitStack`/preparation:

```python
if len(sources) > DATASET_PREVIEW_TABLE_MAX_ROWS:
    raise DatasetDirectoryError(
        f"Dataset preview selection produced {len(sources):,} episode-camera rows, "
        f"exceeding the {DATASET_PREVIEW_TABLE_MAX_ROWS:,}-row W&B Table limit. "
        "Select fewer episodes or use --no-preview."
    )
```

Delete the parser option `--preview-max-episodes`, its default/import/plumbing, and `preview_max_episodes` from test namespaces. Do not add an override: W&B cannot publish an oversized single run Table without violating the chosen contract.

- [ ] **Step 7: Run focused selection and boundary tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_transfer.py tests/test_dataset_upload_media.py -q -k "preview_all or preview_row_limit or former_episode_limit or preview_max_episodes_is_rejected"
```

Expected: PASS.

- [ ] **Step 8: Commit the row-policy cutover**

Commit only Task 1 files:

```bash
git add src/lerobot_wandb/cli.py src/lerobot_wandb/dataset_transfer.py tests/test_dataset_transfer.py tests/test_dataset_upload_media.py
git commit -m "fix: limit dataset previews by table rows"
```

### Task 2: Publish previews as one W&B Table

**Files:**
- Modify: `src/lerobot_wandb/cli.py`
- Modify: `src/lerobot_wandb/dataset_preview.py`
- Test: `tests/test_dataset_upload_media.py`

**Interfaces:**
- Consumes: `PreparedPreviewBatch.previews`, each `PreparedDatasetPreview(source, path, bytes, used_source)`.
- Produces: `_dataset_preview_table(preview_batch: PreparedPreviewBatch, *, selection: str) -> wandb.Table` and `run.log({"dataset_previews": table})`.

- [ ] **Step 1: Add a capturing W&B Table fake**

Extend `_patch_dataset_upload()` so successful upload tests never use a live W&B service:

```python
@dataclass
class CapturedTable:
    columns: list[str]
    data: list[list[object]]

monkeypatch.setattr(cli.wandb, "Table", CapturedTable)
```

Keep the existing `wandb.Video` path recorder. Return or expose captured Table calls where assertions need them.

- [ ] **Step 2: Rewrite the old media-key tests as failing Table-contract tests**

Replace `test_default_representative_media_key_is_schema_neutral` and dynamic-key assertions with focused tests for:

```python
assert table.columns == [
    "episode", "camera", "camera_key", "selection", "video",
    "source_path", "preview_bytes", "transcoded",
]
assert list(run.log.call_args.args[0]) == ["dataset_previews"]
```

Rows must prove:

- representative keeps its actual integer episode and label `representative`;
- explicit and all invocations use the corresponding label;
- multiple episodes × cameras preserve batch order and exact cardinality;
- `observation.images.front` and `observation/images/front` both label as `front`;
- distinct exact keys that collapse to `front` remain distinguishable in `camera_key`;
- punctuation/slashes are not normalized in `camera_key`;
- `source.relative_path.as_posix()`, `bytes`, and `not used_source` populate the scalar fields;
- every video value came from `wandb.Video(str(path), format="mp4")`.

- [ ] **Step 3: Run the new Table tests and observe failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_upload_media.py -q -k "table or camera_identity or playable_preview"
```

Expected before implementation: missing `wandb.Table` construction and dynamic-key payloads.

- [ ] **Step 4: Implement the minimal CLI Table helper**

Add `import re` and implement the publication helper in `cli.py`:

```python
_DATASET_PREVIEW_TABLE_COLUMNS = [
    "episode",
    "camera",
    "camera_key",
    "selection",
    "video",
    "source_path",
    "preview_bytes",
    "transcoded",
]


def _dataset_preview_camera(video_key: str) -> str:
    parts = [part for part in re.split(r"[./]", video_key) if part]
    if not parts:
        raise DatasetDirectoryError("Dataset preview camera key has no non-empty component")
    return parts[-1]


def _dataset_preview_table(
    preview_batch: PreparedPreviewBatch,
    *,
    selection: str,
) -> wandb.Table:
    rows = []
    for prepared in preview_batch.previews:
        episode = prepared.source.episode
        if episode is None:
            raise DatasetDirectoryError("Dataset preview source is missing an episode index")
        camera_key = prepared.source.video_key
        rows.append(
            [
                episode,
                _dataset_preview_camera(camera_key),
                camera_key,
                selection,
                wandb.Video(str(prepared.path), format="mp4"),
                prepared.source.relative_path.as_posix(),
                prepared.bytes,
                not prepared.used_source,
            ]
        )
    return wandb.Table(columns=_DATASET_PREVIEW_TABLE_COLUMNS, data=rows)
```

If existing repo naming/type-check conventions prefer an immutable tuple for the module constant, follow that convention while passing a list to W&B/fakes as required by tests.

- [ ] **Step 5: Replace dynamic publication with the stable key**

Derive once:

```python
selection = (
    "all"
    if args.preview_all
    else "explicit"
    if args.preview_episodes
    else "representative"
)
```

Then publish only:

```python
if preview_batch.previews:
    run.log(
        {
            "dataset_previews": _dataset_preview_table(
                preview_batch,
                selection=selection,
            )
        }
    )
```

Keep this inside the existing `try/finally` and `ExitStack` so `run.finish()` precedes derivative cleanup.

- [ ] **Step 6: Remove the dead key formatter cleanly**

Delete `dataset_media_key()` from `dataset_preview.py`, its `urllib.parse.quote` import, the CLI import/private alias, used-key set, and obsolete tests. Do not leave a deprecated alias or dual-write path.

- [ ] **Step 7: Prove lifetime, no-preview, and summary contracts**

Update existing tests to assert:

```python
assert prepared_path.exists()  # inside run.finish side effect
assert not prepared_path.exists()  # after command returns
```

Also prove `--no-preview` neither constructs `wandb.Table` nor calls `run.log()`, reused canonical files remain, and the existing five preview summary values are unchanged.

- [ ] **Step 8: Run the complete focused media file**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_upload_media.py -q
```

Expected: PASS.

- [ ] **Step 9: Run import/static feedback for changed Python files**

Run:

```bash
.venv/bin/ruff check src/lerobot_wandb/cli.py src/lerobot_wandb/dataset_preview.py tests/test_dataset_upload_media.py tests/test_dataset_transfer.py
```

Expected: PASS. Apply only fixes directly caused by Tasks 1–2.

- [ ] **Step 10: Commit Table publication**

Commit only Task 2 files:

```bash
git add src/lerobot_wandb/cli.py src/lerobot_wandb/dataset_preview.py tests/test_dataset_upload_media.py
git commit -m "feat: publish dataset previews as a table"
```

### Task 3: Synchronize user documentation and verify the repository

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Verify: all changed implementation and tests

**Interfaces:**
- Consumes: the completed CLI behavior and exact Table schema from Tasks 1–2.
- Produces: equivalent English/Japanese user guidance and final repository evidence.

- [ ] **Step 1: Update the English preview documentation**

In the existing Video previews section, state that selected review media appears once in the Run's `dataset_previews` Table; it supports filtering, numeric sorting, and grouping by `episode`, `camera`, and exact `camera_key`. Include:

```text
episode = 12
camera = front
camera_key = observation.images.front
```

Document refusal above 10,000 episode-camera rows and the remedies of fewer explicit selectors or `--no-preview`. Remove all `--preview-max-episodes` and 50-episode language. Keep byte-budget and `--force-preview-budget` guidance unchanged.

- [ ] **Step 2: Apply equivalent Japanese documentation**

Update the matching `README.ja.md` section with the same facts, examples, limits, and remedies. Do not add behavior to one language only.

- [ ] **Step 3: Check stale user-facing contracts**

Use repository search to confirm no current docs or help text still advertise:

```text
--preview-max-episodes
DEFAULT_PREVIEW_MAX_EPISODES
dataset_video/episode_
```

Any source/test references must either be intentionally proving removal or be deleted.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_dataset_upload_media.py tests/test_dataset_transfer.py -q
```

Expected: PASS.

- [ ] **Step 5: Run the full suite once**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS with no new skips or failures attributable to this change.

- [ ] **Step 6: Run repository Ruff gates**

Run:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected: both PASS.

- [ ] **Step 7: Smoke-test the real W&B data types offline**

Without a live service, construct one real `wandb.Table` with a tiny local MP4 fixture or existing test fixture, place a real `wandb.Video(path, format="mp4")` in the `video` column, and use W&B offline/disabled mode to exercise serialization while the file exists. Confirm the Table contains the eight columns and one row. Do not claim browser playback or live UI filtering without a live acceptance dataset/service.

- [ ] **Step 7a: Run the live two-episode/two-camera acceptance**

With authenticated W&B access and a real LeRobot dataset containing at least `front` and `wrist`, run:

```bash
.venv/bin/lerobot-wandb dataset upload \
  --root ./dataset \
  --project test \
  --name filter-test \
  --preview-episode 12 \
  --preview-episode 13
```

Open the resulting Run and verify one `dataset_previews` Table with four rows: `(12, front)`, `(12, wrist)`, `(13, front)`, `(13, wrist)`. Play every video; sort `episode` numerically; filter independently and jointly by `episode = 12` and `camera = front`; group by camera; confirm exact `camera_key` values remain visible. Compare the canonical Artifact manifest/checksums with a `--no-preview` upload of the same dataset. If credentials or a suitable dataset are unavailable, record that exact external prerequisite as an acceptance blocker rather than claiming the live behavior passed.

- [ ] **Step 8: Inspect the complete diff and commit docs**

Confirm the diff contains only the approved source, tests, plan/spec, and synchronized README changes. Commit:

```bash
git add README.md README.ja.md
git commit -m "docs: describe dataset preview tables"
```

- [ ] **Step 9: Request two-stage code review**

Dispatch a spec-compliance reviewer first, then a code-quality reviewer after any spec fixes. Review against issue #5 plus `docs/superpowers/specs/2026-08-20-dataset-preview-table-design.md`. Resolve every confirmed finding, rerun the narrow affected check, then rerun the full suite/Ruff gates if production code changed.
