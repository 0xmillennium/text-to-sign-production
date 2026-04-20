# Interim Reports

## Purpose

Interim reports summarize Dataset Build assumptions and filtering outcomes before processed v1
publication.

## Artifact Role

`output`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Worktree path: `/content/text-to-sign-production/data/interim/reports/*.json`
- Archive member path: `data/interim/reports/*.json`

## Verified Example Path

- `/content/text-to-sign-production/data/interim/reports/assumption-report.json`
- `/content/text-to-sign-production/data/interim/reports/filter-report.json`

## Produced By

Section 3.3, Dataset Build run/publish, writes interim reports while building and filtering the
dataset.

## Consumed By

Section 3.3, Dataset Build run/publish, packages interim reports into the Dataset Build manifests
and reports archive. Section 7.1, final artifact inspection, checks the published surface.

## Lifecycle

Interim reports are regenerated with each Dataset Build run and are first-class Dataset Build report
artifacts.

## Structure / Contents

`assumption-report.json` records Dataset Build assumptions and observed raw-data conditions.
`filter-report.json` records filtering counts and exclusion reasons for the active policy.

## Validation

The reports are valid JSON files written during Dataset Build. They must remain in
`data/interim/reports/` and must be included in `dataset_build_manifests_reports.tar.zst`.

## Related Artifacts

- [Filtered manifests](filtered-manifests.md)
- [Processed reports](../processed/v1/reports.md)
- [Manifests and reports archive](../artifacts/dataset-build/manifests-reports-archive.md)

## Related Docs

- [Data leaf template](../template.md)
- [Dataset Build execution](../../execution/dataset-build.md)
