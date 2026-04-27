# Baseline Modeling Configs

## Purpose

Baseline modeling configs capture both the original operator-facing source config and the effective
run config used by a specific baseline run.

## Artifact Role

`config`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Drive-published source config: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/config/source_baseline.yaml`
- Drive-published effective config: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/config/baseline.yaml`

## Verified Example Path

- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/config/source_baseline.yaml`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/config/baseline.yaml`

## Produced By

Section 4.1, Baseline run-root preparation and config materialization, writes both config files.

## Consumed By

Section 4.2 checks whether training outputs can be reused. Section 4.4 uses the effective config for
training. Section 5.3 uses it for qualitative export. Section 6.3 includes both config paths in the
runtime package.

## Lifecycle

`source_baseline.yaml` preserves the original operator-facing config. `baseline.yaml` is the
effective run config written into the run root and archived with training outputs.

## Structure / Contents

The effective config points training to `data/processed/v1/manifests/train.jsonl` and validation to
`data/processed/v1/manifests/val.jsonl`. It uses backbone `google/flan-t5-base`, target split usage
`train` and `val`, and writes the run-root checkpoint output directory into the effective config.

## Validation

Baseline training loads the effective config before running. The materialized config must preserve
the source config path distinction and must point only to the train and validation processed
manifests for the baseline run.

## Related Artifacts

- [Processed manifests](../../processed/v1/manifests.md)
- [Checkpoints](checkpoints.md)
- [Run summaries](run-summaries.md)
- [Training archive](training-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
