# Dataset Preview Table Design

**Status:** Approved for implementation  
**Issue:** [#5](https://github.com/guchengwei/lerobot-wandb/issues/5)

## Problem

Dataset uploads currently log every prepared preview under a dynamic key such as `dataset_video/episode_000012/observation.images.front`. That preserves identity but hides episode and camera dimensions inside key names, so W&B cannot naturally filter, sort, or group the review set by those fields. The existing `--preview-all` safety limit is episode-based, while W&B Tables have a 10,000-row limit and each selected episode can contribute multiple camera rows.

The canonical dataset Artifact, preview selection and encoding, byte-budget policy, summary metadata, and temporary-file lifetime already have the required semantics. This change replaces only the run-media publication shape and its mismatched episode-count guard.

## Decisions

- Publish all selected dataset previews once under the stable run key `dataset_previews` as one `wandb.Table`.
- Create one row for each prepared episode × camera preview. Do not also publish the videos under the old dynamic keys.
- Use exactly the eight columns below, in the listed order. Do not add `row_id` or `dataset_preview_table_key` summary metadata.
- Keep W&B-specific Table and `wandb.Video` construction in a directly testable CLI helper named `_dataset_preview_table(preview_batch, *, selection)`. The selection and encoder modules remain W&B-agnostic.
- Replace the `--preview-all` 50-episode limit with one universal pre-encoding limit of 10,000 selected preview sources (therefore 10,000 prospective Table rows).
- Remove `--preview-max-episodes` without a compatibility alias or replacement override. Explicit `--preview-episode` selectors have no selection-count limit of their own; like representative and `--preview-all` selection, they are restricted only by the universal 10,000-row Table guard. Existing selector validity, range checking, mutual exclusion, and duplicate removal remain unchanged.

## Architecture and data flow

`cmd_dataset_upload()` keeps the following order:

1. Inspect and validate the transfer dataset.
2. Unless `--no-preview` is set, call `select_dataset_preview_sources()` for representative, explicit, or all selection.
3. Immediately compare `len(sources)` with `DATASET_PREVIEW_TABLE_MAX_ROWS = 10_000`. A count of 10,000 is accepted; 10,001 or more raises `DatasetDirectoryError` reporting the selected row count and limit. This happens before entering preview preparation, `wandb.init()`, canonical Artifact upload, or any Table/media construction.
4. Prepare the accepted sources in the existing sibling temporary directory and apply the existing measured-byte budget confirmation or `--force-preview-budget` behavior.
5. Initialize W&B and upload the canonical dataset directory unchanged.
6. If prepared previews exist, derive the invocation selection label once, build one Table through the CLI helper, and call `run.log({"dataset_previews": table})` once.
7. Update the existing dataset and preview summary fields unchanged.
8. Call `run.finish()` in the existing `finally` block, then allow the outer `ExitStack` to clean generated derivatives.

`--no-preview` continues to skip selection, preparation, Table construction, and `run.log()` while uploading the canonical Artifact and recording zero-valued preview summary facts.

## Table contract

Construct the Table as:

```python
wandb.Table(
    columns=[
        "episode",
        "camera",
        "camera_key",
        "selection",
        "video",
        "source_path",
        "preview_bytes",
        "transcoded",
    ],
    data=rows,
)
```

Each row is derived from one `PreparedDatasetPreview`:

| Column | Type | Derivation |
|---|---|---|
| `episode` | integer | `prepared.source.episode`, preserving the actual episode index. A representative row uses its real numeric episode, never the string `representative`. Selected preview sources are expected to carry an episode; a missing value is an error rather than a substitute value. |
| `camera` | string | The final non-empty component of `camera_key` after splitting on both `.` and `/`: `[part for part in re.split(r"[./]", camera_key) if part][-1]`. For example, `observation.images.front` becomes `front`, and `observation/images/front` also becomes `front`. A key with no non-empty component is rejected rather than assigned an alias. |
| `camera_key` | string | Exact `prepared.source.video_key`, unchanged and authoritative. Short-label collisions are allowed and remain distinguishable here. |
| `selection` | string | One invocation-wide value: `representative` when neither explicit selectors nor `--preview-all` were supplied, `explicit` when at least one `--preview-episode` was supplied, or `all` for `--preview-all`. |
| `video` | `wandb.Video` | `wandb.Video(str(prepared.path), format="mp4")`. |
| `source_path` | string | Canonical dataset-relative `prepared.source.relative_path.as_posix()`, not the derivative temporary path. |
| `preview_bytes` | integer | `prepared.bytes`. |
| `transcoded` | boolean | `not prepared.used_source`: `False` for an exact compatible canonical source reused by the fast path; `True` for a generated derivative. |

The helper preserves `PreparedPreviewBatch.previews` order. It neither reselects sources nor recomputes encoding or budget values.

## CLI clean cutover and row preflight

Define `DATASET_PREVIEW_TABLE_MAX_ROWS = 10_000` in the CLI/publication layer, next to the preflight that enforces it. The limit counts selected sources rather than episodes because every source becomes exactly one Table row and camera multiplicity matters. Exceeding it raises `DatasetDirectoryError` with the selected row count, the 10,000-row limit, and the remedies of selecting fewer episodes or using `--no-preview`.

Remove all parts of the previous episode-limit contract:

- the `--preview-max-episodes` parser option and its help text;
- `args.preview_max_episodes` plumbing;
- `DEFAULT_PREVIEW_MAX_EPISODES`;
- the `max_episodes` parameter and 50-episode refusal in `select_dataset_preview_sources()`;
- tests and README text describing a configurable `--preview-all` episode limit.

The universal guard applies identically after representative, explicit, and all selection. It does not truncate, batch, sample, or partially publish a selection, and neither `--force-preview-budget` nor any other flag bypasses it. At 10,001 rows, derivative generation, W&B Run creation, Artifact upload, and preview upload have not started. Users must select fewer rows or use `--no-preview`.

Remove the obsolete dynamic-key implementation completely: `dataset_media_key()`, its `urllib.parse.quote` import, the CLI import and `_dataset_media_key` alias, used-key tracking, and their tests. There are no deprecated aliases or dual-write transition.

## Error and lifetime behavior

Selection and row-limit failures occur before preview preparation and before any W&B side effect. Encoding and measured-budget failures retain their current pre-`wandb.init()` behavior. Once a Run exists, the existing `try`/`finally` guarantees `run.finish()` for Artifact, Table-construction, logging, or summary failures.

The Table holds `wandb.Video` values whose paths must remain readable until `run.finish()` returns. Generated derivatives remain in the existing sibling temporary directory, outside the canonical dataset root, and the `ExitStack` removes them only afterward. Compatible canonical source files reused by the fast path are not owned by that temporary directory and are never deleted. The canonical Artifact continues to contain only original dataset bytes; Table publication does not alter, copy into, or replace canonical video files. Existing encoding profile, fast-path eligibility, aggregate byte budget, progress reporting, and budget confirmation remain unchanged.

## Test contracts

Tests mock W&B completely and require no live service. Add a small fake or capturing `wandb.Table` implementation that records `columns` and `data`; keep `wandb.Video`, `wandb.init`, and Artifact upload mocked. Focused tests must establish:

- preview publication makes one `run.log()` call whose only key is `dataset_previews`, with no `dataset_video/...` keys;
- the Table uses the exact ordered eight-column schema;
- one prepared preview produces one row, and multiple episodes × multiple cameras produce the matching row count in stable order;
- `episode` is an integer containing the actual episode for representative, explicit, and all modes;
- selection labels are exactly `representative`, `explicit`, and `all` for their corresponding CLI inputs;
- camera labels use the final non-empty dot-or-slash component;
- `camera_key` preserves punctuation and slashes exactly, and two exact keys that collapse to one short label remain distinguishable;
- `source_path`, `preview_bytes`, and `transcoded` come from the prepared preview as specified, covering both reused-source and transcoded rows;
- every video cell is the result of `wandb.Video(path, format="mp4")`;
- generated files exist while the Table is logged and through `run.finish()`, then are removed; reused canonical files remain;
- `--no-preview` constructs and logs no Table while preserving the canonical upload and zero preview summary values;
- all existing preview summary fields and byte-budget behavior remain correct;
- 10,000 selected sources reach preparation, while 10,001 fail before `prepare_dataset_previews`, `wandb.init`, `upload_directory`, or `run.log`; parameterize the rejection across representative, explicit, and all argument modes;
- explicit selection above the former 50-episode threshold is accepted when its episode × camera row count is at most 10,000;
- the parser no longer accepts `--preview-max-episodes`, while `--preview-all` and `--preview-episode` remain mutually exclusive;
- transfer-selection tests call `select_dataset_preview_sources()` without `max_episodes`, retain representative/explicit/all behavior, and no longer expect an episode-count refusal.

Prefer direct helper tests for row derivation and upload orchestration tests for ordering, preflight side effects, and lifetime. Avoid constructing or encoding 10,000 real videos: use synthetic selected-source sequences and mocks that prove whether preparation was reached.

## README requirements

Update `README.md` and `README.ja.md` together with equivalent user-facing content:

- selected previews appear in the Run's `dataset_previews` Table;
- users can filter, sort, and group by numeric `episode`, short `camera`, and exact `camera_key`;
- include a concrete example such as `episode = 12`, `camera = front`, `camera_key = observation.images.front`;
- remove `--preview-max-episodes`, the 50-episode limit, and wording that `--force-preview-budget` interacts with that old limit;
- describe the universal 10,000-row Table refusal and that explicit selectors have no separate limit;
- do not describe the old `dataset_video/episode_.../...` key layout as a current or compatibility surface.

## Non-goals

- Changing which representative episode, explicit episodes, or cameras are selected.
- Changing selector validation apart from removing the episode-count limit.
- Changing preview encoding, fast-path reuse, byte-budget calculation or approval, progress rendering, or source ordering.
- Changing canonical Artifact contents, dataset schemas, or upload/download behavior.
- Adding camera aliases, collision avoidance for short labels, a custom W&B dashboard/report, Table batching, truncation, sampling, or a second summary/Table key.
- Changing rollout preview publication.

## Verification gates

Implementation is complete only after all of these pass:

1. Focused tests: `.venv/bin/python -m pytest -q tests/test_dataset_upload_media.py tests/test_dataset_transfer.py`
2. Full suite: `.venv/bin/python -m pytest -q`
3. Static checks: `.venv/bin/ruff check .` and `.venv/bin/ruff format --check .`
4. English/Japanese README review confirms equivalent behavior and no remaining `--preview-max-episodes` or dynamic-key contract.
5. Manual acceptance with two episodes and at least `front` and `wrist` cameras confirms one four-row `dataset_previews` Table, playable video cells, numeric episode sorting, independent and combined episode/camera filtering, grouping by camera, and visible exact `camera_key` values. The canonical Artifact contents must match a preview-disabled upload.
