# Dataset Build Sample Archives

## Purpose

Sample archives publish processed v1 sample `.npz` payloads by split so sample-heavy outputs can be
restored independently from manifests and reports.

## Artifact Role

`archive`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Drive-published archive path: `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_<split>.tar.zst`
- `<split>` is one of `train`, `val`, or `test`.

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_train.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_val.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_samples_test.tar.zst`

## Produced By

Section 3.3, Dataset Build run/publish, packages and publishes one archive per processed split.

## Consumed By

Section 3.2, Dataset Build archive extraction, restores sample archives for reuse. Section 7.1,
final artifact inspection, checks the published archive set.

## Lifecycle

These are Drive-published Dataset Build archives for processed sample payloads. They are regenerated
from processed v1 samples and restored by split.

## Structure / Contents

Each archive contains:

- `data/processed/v1/samples/<split>/*.npz`

## Validation

Each split archive must exist at the fixed Dataset Build artifact root and contain only the matching
processed v1 sample split.

## Related Artifacts

- [Processed samples](../../processed/v1/samples.md)
- [Processed manifests](../../processed/v1/manifests.md)
- [Manifests and reports archive](manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Dataset Build execution](../../../execution/dataset-build.md)
