# Baseline Modeling Checkpoints

## Purpose

Baseline modeling checkpoints persist model and optimizer state for the best and last training
epochs of a run.

## Artifact Role

`checkpoint`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Drive-published best checkpoint: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/checkpoints/best.pt`
- Drive-published last checkpoint: `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/checkpoints/last.pt`

## Verified Example Path

- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/checkpoints/best.pt`
- `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/190420261845/checkpoints/last.pt`

## Produced By

Section 4.4, Training run/publish, writes checkpoints. Section 4.3, Training archive extraction,
can restore them from an existing training archive.

## Consumed By

Section 4.2 checks checkpoint presence for reuse. Section 5.3 uses a checkpoint for qualitative
export. Section 6.3 records checkpoint paths in the runtime package.

## Lifecycle

Checkpoints are runtime training outputs under the run root and are included in the training
archive.

## Structure / Contents

Checkpoint schema version is `t2sp-baseline-checkpoint-v1`. Required fields include
`backbone_name`, `config`, `epoch`, `model_state_dict`, `optimizer_state_dict`, `role`,
`schema_version`, `seed`, `target_channels`, `train_loss`, `validation_loss`, and
`validation_metric`.

## Validation

Checkpoint loading validates the schema version and required fields before prediction or evidence
assembly uses a checkpoint payload.

## Related Artifacts

- [Configs](configs.md)
- [Run summaries](run-summaries.md)
- [Qualitative metadata](qualitative-metadata.md)
- [Training archive](training-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution/baseline-modeling.md)
