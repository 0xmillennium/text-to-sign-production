# Data Artifact Dictionary

`docs/data` is the canonical dictionary for input data, intermediate data, processed data,
published archives, and baseline modeling artifacts. Use these pages for artifact paths,
roles, producers, consumers, and contents.

## Standard

- [Template](template.md): Canonical structure and vocabulary for every `docs/data` leaf page.

## Raw Inputs

- [How2Sign translations](raw/how2sign-translations.md): Raw split translation files used to create Dataset Build manifests.
- [How2Sign keypoint archives](raw/how2sign-keypoint-archives.md): Raw split OpenPose keypoint archives restored before Dataset Build processing.

## Interim Dataset Build Artifacts

- [Raw manifests](interim/raw-manifests.md): JSONL records that preserve translation rows and keypoint alignment status.
- [Normalized manifests](interim/normalized-manifests.md): JSONL records after OpenPose parsing, normalization, and audit enrichment.
- [Filtered manifests](interim/filtered-manifests.md): JSONL records that pass the active Dataset Build filter policy.
- [Interim reports](interim/reports.md): JSON summaries for Dataset Build assumptions and filtering outcomes.

## Processed v1 Dataset Artifacts

- [Processed manifests](processed/v1/manifests.md): Split JSONL manifests for the processed v1 modeling interface.
- [Processed samples](processed/v1/samples.md): Split `.npz` sample payloads referenced by processed manifests.
- [Processed reports](processed/v1/reports.md): JSON and Markdown outputs for processed data quality and split checks.

## Dataset Build Published Archives

- [Manifests and reports archive](artifacts/dataset-build/manifests-reports-archive.md): Published archive containing interim manifests, interim reports, processed manifests, and processed reports.
- [Sample archives](artifacts/dataset-build/sample-archives.md): Published split archives containing processed `.npz` samples.

## Baseline Modeling Artifacts

- [Configs](artifacts/baseline-modeling/configs.md): Source and effective run configuration files for a baseline run.
- [Checkpoints](artifacts/baseline-modeling/checkpoints.md): Best and last model checkpoint files written by baseline training.
- [Run summaries](artifacts/baseline-modeling/run-summaries.md): Training summary JSON copies under checkpoints, metrics, and record roots.
- [Qualitative metadata](artifacts/baseline-modeling/qualitative-metadata.md): Panel definition, per-sample metadata records, and panel summary.
- [Qualitative sample artifacts](artifacts/baseline-modeling/qualitative-sample-artifacts.md): Prediction and reference `.npz` files for qualitative review.
- [Evidence bundles](artifacts/baseline-modeling/evidence-bundles.md): Runtime evidence JSON files tying config, checkpoint, summary, and qualitative panel outputs together.
- [Package manifest](artifacts/baseline-modeling/package-manifest.md): Runtime-side baseline modeling package manifest.
- [Training archive](artifacts/baseline-modeling/training-archive.md): Published archive for config, checkpoint, and metrics outputs.
- [Qualitative archive](artifacts/baseline-modeling/qualitative-archive.md): Published archive for qualitative outputs.
- [Record archive](artifacts/baseline-modeling/record-archive.md): Published archive for runtime record package outputs.
