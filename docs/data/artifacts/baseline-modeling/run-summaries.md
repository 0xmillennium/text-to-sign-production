# Baseline Modeling Run Summaries

## Purpose

Run summaries store training provenance, scalar outcomes, history, and checkpoint references for a
baseline run.

## Artifact Role

`summary`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Checkpoint summary: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/checkpoints/run_summary.json`
- Metrics summary: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/metrics/run_summary.json`
- Record-side copied summary: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/record/run_summary.json`

## Verified Example Path

- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/checkpoints/run_summary.json`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/metrics/run_summary.json`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/record/run_summary.json`

## Produced By

Section 4.4, Training run/publish, writes `checkpoints/run_summary.json` and copies it to
`metrics/run_summary.json`. Section 6.3 copies the metrics summary to `record/run_summary.json`.

## Consumed By

Section 5.3 uses the metrics summary when writing baseline evidence. Section 6.3 uses the metrics
summary when assembling the runtime package. Section 7.1 checks the final artifact surface.

## Lifecycle

The checkpoint and metrics summaries are training-side outputs. `record/run_summary.json` is a
copied record-side summary surface.

## Structure / Contents

Keys include `backbone_name`, `best_checkpoint_path`, `best_epoch`, `best_validation_loss`,
`checkpoint_output_path`, `config`, `config_path`, `final_train_loss`, `final_validation_loss`,
`final_validation_metric`, `history`, `last_checkpoint_path`, `seed`, and `target_channels`.

## Validation

The JSON root must be a mapping. Downstream evidence and package assembly require the metrics
summary path and checkpoint references to exist.

## Related Artifacts

- [Configs](configs.md)
- [Checkpoints](checkpoints.md)
- [Evidence bundles](evidence-bundles.md)
- [Package manifest](package-manifest.md)
- [Training archive](training-archive.md)
- [Record archive](record-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
