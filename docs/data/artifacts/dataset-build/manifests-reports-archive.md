# Dataset Build Manifests And Reports Archive

## Purpose

The manifests and reports archive publishes the non-sample Dataset Build outputs needed to inspect,
restore, or cite the processed v1 dataset surface.

## Artifact Role

`archive`

## Artifact Unit

`archive bundle`

## Canonical Path Pattern

- Drive-published archive path: `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_manifests_reports.tar.zst`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/dataset_build_manifests_reports.tar.zst`

## Produced By

Section 3.3, Dataset Build run/publish, packages and publishes this archive.

## Consumed By

Section 3.2, Dataset Build archive extraction, restores this archive for reuse. Section 7.1, final
artifact inspection, checks the published archive.

## Lifecycle

This is the Drive-published Dataset Build archive for manifests and report artifacts. It is
restorable without rebuilding samples.

## Structure / Contents

The archive contains these member directories:

- `data/interim/raw_manifests/`
- `data/interim/normalized_manifests/`
- `data/interim/filtered_manifests/`
- `data/interim/reports/`
- `data/processed/v1/manifests/`
- `data/processed/v1/reports/`

## Validation

The archive must exist at the fixed Dataset Build artifact root and include all listed member
directories. Reports are first-class artifacts only in `data/interim/reports/*.json`,
`data/processed/v1/reports/*`, and this archive.

## Related Artifacts

- [Raw manifests](../../interim/raw-manifests.md)
- [Normalized manifests](../../interim/normalized-manifests.md)
- [Filtered manifests](../../interim/filtered-manifests.md)
- [Interim reports](../../interim/reports.md)
- [Processed manifests](../../processed/v1/manifests.md)
- [Processed reports](../../processed/v1/reports.md)
- [Sample archives](sample-archives.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Dataset Build execution](../../../execution/dataset-build.md)
