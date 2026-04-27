# Baseline Qualitative Metadata

## Purpose

Qualitative metadata defines the validation panel, records per-sample qualitative artifact paths,
and summarizes the exported panel.

## Artifact Role

`manifest`, `summary`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Panel definition: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/panel_definition.json`
- Per-sample records: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/records.jsonl`
- Panel summary: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/panel_summary.json`

## Verified Example Path

- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/panel_definition.json`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/records.jsonl`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/panel_summary.json`

## Produced By

Section 5.3, Qualitative export/publish, writes qualitative metadata. Section 5.2, Qualitative
archive extraction, can restore it from an existing qualitative archive.

## Consumed By

Section 6.3, Record/package assembly and publish, requires qualitative metadata before packaging.
Section 7.1 checks the final artifact surface.

## Lifecycle

Qualitative metadata is a runtime output under the run root and is included in the qualitative
archive.

## Structure / Contents

`panel_definition.json` uses schema version `t2sp-qualitative-panel-v1` and keys `sample_ids`,
`schema_version`, `selection_rule`, and `split`. `records.jsonl` lines include `fps`,
`frame_valid_count`, `num_frames`, `prediction_artifact`, `reference_artifact`, `sample_id`,
`source_processed_sample_path`, `split`, `target_channels`, and `text`. `panel_summary.json` uses
schema version `t2sp-qualitative-export-v1` and keys `artifacts`, `checkpoint_path`,
`panel_definition_path`, `records_path`, `sample_count`, `sample_ids`, `schema_version`,
`selection_rule`, `split`, and `target_channels`.

## Validation

Panel definitions must use validation split `val`, unique sample IDs, and schema version
`t2sp-qualitative-panel-v1`. Panel summaries must use schema version
`t2sp-qualitative-export-v1` and refer to the written records and artifact paths.

## Related Artifacts

- [Processed manifests](../../processed/v1/manifests.md)
- [Processed samples](../../processed/v1/samples.md)
- [Qualitative sample artifacts](qualitative-sample-artifacts.md)
- [Evidence bundles](evidence-bundles.md)
- [Qualitative archive](qualitative-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
