# Baseline Training Archive

## Purpose

The training archive publishes baseline training outputs for reuse or restoration under the same
run root.

## Artifact Role

`archive`

## Artifact Unit

`archive bundle`

## Canonical Path Pattern

- Drive-published archive path: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/archives/baseline_training_outputs.tar.zst`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/archives/baseline_training_outputs.tar.zst`

## Produced By

Section 4.4, Training run/publish, packages and publishes the training archive.

## Consumed By

Section 4.3, Training archive extraction, restores training outputs. Section 4.2 checks whether
training outputs can be reused. Section 7.1 checks the final artifact surface.

## Lifecycle

This archive is created from training outputs and can restore the training-side run-root contents.

## Structure / Contents

The archive contains:

- `config/`
- `checkpoints/`
- `metrics/`

## Validation

The archive must contain the config directory, checkpoint files, and metrics summary required by
baseline reuse and downstream packaging.

## Related Artifacts

- [Configs](configs.md)
- [Checkpoints](checkpoints.md)
- [Run summaries](run-summaries.md)
- [Qualitative archive](qualitative-archive.md)
- [Record archive](record-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
