# Baseline Qualitative Sample Artifacts

## Purpose

Qualitative sample artifacts store reference and prediction pose arrays for the exported validation
panel.

## Artifact Role

`output`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Prediction artifact: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/predictions/<zero_padded_index>__<sanitized_sample_id>.npz`
- Reference artifact: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/references/<zero_padded_index>__<sanitized_sample_id>.npz`
- `<zero_padded_index>` is the zero-based panel order padded to four digits, such as `0000`.
- `<sanitized_sample_id>` replaces characters outside `[A-Za-z0-9_.-]` with `_`, then trims
  leading and trailing `.` or `_`; an empty sanitized value becomes `sample`.

## Verified Example Path

- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/predictions/`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/references/`

Only the prediction and reference directory roots are verified in the current artifact inventory.
Concrete `.npz` filenames follow the canonical path pattern above.

## Produced By

Section 5.3, Qualitative export/publish, writes prediction and reference `.npz` files. Section 5.2,
Qualitative archive extraction, can restore them from an existing qualitative archive.

## Consumed By

Section 5.3 records these paths in `qualitative/records.jsonl` and `qualitative/panel_summary.json`.
Section 7.1 checks the final artifact surface.

## Lifecycle

Qualitative sample artifacts are runtime outputs under `qualitative/` and are included in the
qualitative archive.

## Structure / Contents

Each prediction and reference `.npz` contains `body`, `left_hand`, `right_hand`, and
`frame_valid_mask`. This is not the same schema as processed v1 sample `.npz` files.

## Validation

Qualitative export writes one reference and one prediction file for each selected panel sample.
Records and panel summaries must point to the written files.

## Related Artifacts

- [Processed samples](../../processed/v1/samples.md)
- [Qualitative metadata](qualitative-metadata.md)
- [Qualitative archive](qualitative-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
