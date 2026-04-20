# Processed v1 Reports

## Purpose

Processed v1 reports document data-quality and split-integrity outcomes for the processed dataset.

## Artifact Role

`output`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/processed/v1/reports/*`
- Archive member path: `data/processed/v1/reports/*`

## Verified Example Path

- `/content/text-to-sign-production/data/processed/v1/reports/data-quality-report.json`
- `/content/text-to-sign-production/data/processed/v1/reports/data-quality-report.md`
- `/content/text-to-sign-production/data/processed/v1/reports/split-report.json`
- `/content/text-to-sign-production/data/processed/v1/reports/split-report.md`
- `/content/text-to-sign-production/data/processed/v1/reports/assumption-report.md`

## Produced By

Section 3.3, Dataset Build run/publish, writes processed reports after processed manifests and
samples are exported.

## Consumed By

Section 3.3 packages processed reports into the Dataset Build manifests and reports archive.
Section 7.1, final artifact inspection, checks the published surface.

## Lifecycle

Processed reports are regenerated with each Dataset Build run and are first-class Dataset Build
report artifacts.

## Structure / Contents

`data-quality-report.*` summarizes processed sample quality checks. `split-report.*` summarizes
split membership and integrity checks. `assumption-report.md` publishes human-readable Dataset
Build assumptions for the processed v1 output.

## Validation

The JSON files must be valid JSON and the Markdown files must be readable documentation artifacts.
All processed report files remain under `data/processed/v1/reports/` and are included in
`dataset_build_manifests_reports.tar.zst`.

## Related Artifacts

- [Processed manifests](manifests.md)
- [Processed samples](samples.md)
- [Interim reports](../../interim/reports.md)
- [Manifests and reports archive](../../artifacts/dataset-build/manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Dataset Build execution](../../../execution/dataset-build.md)
