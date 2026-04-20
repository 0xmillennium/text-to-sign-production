# Processed v1 Samples

## Purpose

Processed v1 samples are compressed NumPy payloads containing normalized pose arrays referenced by
processed v1 manifests.

## Artifact Role

`output`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/processed/v1/samples/<split>/<sample_id>.npz`
- Sample archive member path: `data/processed/v1/samples/<split>/<sample_id>.npz`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/text-to-sign-production/data/processed/v1/samples/train/--7E2sU6zP4_10-5-rgb_front.npz`
- `/content/text-to-sign-production/data/processed/v1/samples/val/-d5dN54tH2E_0-1-rgb_front.npz`
- `/content/text-to-sign-production/data/processed/v1/samples/test/-fZc293MpJk_0-1-rgb_front.npz`

## Produced By

Section 3.3, Dataset Build run/publish, writes processed sample `.npz` files from filtered
manifests and normalized OpenPose arrays.

## Consumed By

Section 4.4 reads train and validation samples through processed manifests during baseline
training. Section 5.3 reads validation samples for qualitative reference and prediction export.

## Lifecycle

Processed samples are regenerated Dataset Build outputs and are packaged by split into
`dataset_build_samples_<split>.tar.zst`.

## Structure / Contents

This surface belongs to the same processed schema as processed manifests:
`processed_schema_version: t2sp-processed-v1`. Each `.npz` payload contains
`processed_schema_version`, `body`, `body_confidence`, `left_hand`, `left_hand_confidence`,
`right_hand`, `right_hand_confidence`, `face`, `face_confidence`, `people_per_frame`,
`selected_person_index`, and `frame_valid_mask`.

## Validation

Consumers must reject sample payloads whose scalar `processed_schema_version` is not
`t2sp-processed-v1`. Array shapes must match the processed v1 contract: body `(frames, 25, 2)`,
hands `(frames, 21, 2)`, face `(frames, 70, 2)`, matching confidence arrays, and frame-level masks
or audit arrays with `frames` entries.

## Related Artifacts

- [Processed manifests](manifests.md)
- [Sample archives](../../artifacts/dataset-build/sample-archives.md)
- [Qualitative sample artifacts](../../artifacts/baseline-modeling/qualitative-sample-artifacts.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Dataset Build execution](../../../execution.md)
- [Baseline Modeling execution](../../../execution.md)
