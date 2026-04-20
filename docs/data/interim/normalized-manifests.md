# Normalized Manifests

## Purpose

Normalized manifests carry parsed OpenPose and normalized coordinate audit data before the active
filter policy decides which samples remain.

## Artifact Role

`intermediate`, `manifest`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/interim/normalized_manifests/normalized_<split>.jsonl`
- Archive member path: `data/interim/normalized_manifests/normalized_<split>.jsonl`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/text-to-sign-production/data/interim/normalized_manifests/normalized_train.jsonl`

## Produced By

Section 3.3, Dataset Build run/publish, writes normalized manifests from raw manifests and extracted
OpenPose frame JSONs.

## Consumed By

Section 3.3, Dataset Build run/publish, consumes normalized manifests when applying the active
filter policy and exporting processed v1 artifacts.

## Lifecycle

Normalized manifests are regenerated Dataset Build intermediates and are packaged into the Dataset
Build manifests and reports archive.

## Structure / Contents

Each JSONL record carries normalized sample metadata, source path references, frame counts, video
metadata, parser audit fields, selected-person audit fields, and core-channel usability metrics.
Coordinates are normalized against the observed BFH canvas, `x / 1280` and `y / 720`.

## Validation

Dataset Build validates that parsed OpenPose channels can be represented consistently and that
sample-level parsing failures remain explicit for filtering.

## Related Artifacts

- [Raw manifests](raw-manifests.md)
- [Filtered manifests](filtered-manifests.md)
- [Processed samples](../processed/v1/samples.md)
- [Manifests and reports archive](../artifacts/dataset-build/manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution/dataset-build.md)
