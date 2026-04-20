# Processed v1 Manifests

## Purpose

Processed v1 manifests are the canonical split-level modeling interface for locating processed
sample payloads and reading text, split, schema, and audit metadata.

## Artifact Role

`manifest`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/processed/v1/manifests/<split>.jsonl`
- Archive member path: `data/processed/v1/manifests/<split>.jsonl`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/text-to-sign-production/data/processed/v1/manifests/train.jsonl`
- `/content/text-to-sign-production/data/processed/v1/manifests/val.jsonl`
- `/content/text-to-sign-production/data/processed/v1/manifests/test.jsonl`

## Produced By

Section 3.3, Dataset Build run/publish, writes processed manifests from filtered manifests.

## Consumed By

Section 4.1 materializes baseline configs that point to `train.jsonl` and `val.jsonl`. Section 4.4
reads those manifests for training and validation. Section 5.3 reads the validation manifest for
qualitative export.

## Lifecycle

Processed manifests are regenerated Dataset Build outputs and are packaged into the Dataset Build
manifests and reports archive.

## Structure / Contents

Each JSONL record carries `processed_schema_version: t2sp-processed-v1`. Required fields include
`sample_id`, `processed_schema_version`, `text`, `split`, `fps`, `num_frames`, `sample_path`,
`source_video_id`, `source_sentence_id`, `source_sentence_name`, `selected_person_index`,
`multi_person_frame_count`, and `max_people_per_frame`. Audit fields may include source paths,
video metadata, frame-valid counts, face-missing counts, core-channel counts, and issue summaries.

## Validation

Consumers must reject records whose `processed_schema_version` is not `t2sp-processed-v1`.
Manifest `sample_path` values must resolve to processed v1 sample `.npz` files for the same split.

## Related Artifacts

- [Filtered manifests](../../interim/filtered-manifests.md)
- [Processed samples](samples.md)
- [Processed reports](reports.md)
- [Manifests and reports archive](../../artifacts/dataset-build/manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Dataset Build execution](../../../execution/dataset-build.md)
- [Baseline Modeling execution](../../../execution/baseline-modeling.md)
