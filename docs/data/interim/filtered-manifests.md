# Filtered Manifests

## Purpose

Filtered manifests identify normalized samples that pass the active Dataset Build filter policy and
are eligible for processed v1 export.

## Artifact Role

`intermediate`, `manifest`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/interim/filtered_manifests/filtered_<split>.jsonl`
- Archive member path: `data/interim/filtered_manifests/filtered_<split>.jsonl`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/text-to-sign-production/data/interim/filtered_manifests/filtered_train.jsonl`

## Produced By

Section 3.3, Dataset Build run/publish, writes filtered manifests after applying the active filter
policy to normalized manifests.

## Consumed By

Section 3.3, Dataset Build run/publish, consumes filtered manifests when writing processed v1
manifests and samples.

## Lifecycle

Filtered manifests are regenerated Dataset Build intermediates and are packaged into the Dataset
Build manifests and reports archive.

## Structure / Contents

Each JSONL record represents a retained sample and carries the normalized metadata needed to write
the processed manifest entry and `.npz` sample payload.

## Validation

The active filter policy keeps samples with usable body data and at least one usable hand. Samples
with missing text, invalid time ranges, missing keypoints, zero frames, parse failure, all-invalid
frames, unusable body, or both hands unusable are excluded from this surface.

## Related Artifacts

- [Normalized manifests](normalized-manifests.md)
- [Interim reports](reports.md)
- [Processed manifests](../processed/v1/manifests.md)
- [Processed samples](../processed/v1/samples.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution/dataset-build.md)
