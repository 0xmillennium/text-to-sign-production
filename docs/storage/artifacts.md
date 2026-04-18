# Artifact Storage Strategy

Dataset Build produces large outputs that are important for reproducibility but are not appropriate
for GitHub storage. This repository keeps the workflow explicit: code and documentation stay in
Git, while heavy generated artifacts stay in the fixed Google Drive location used by the supported
Colab workflow.

## What Stays Outside GitHub

The following should not be committed to GitHub:

- raw How2Sign downloads
- processed `.npz` samples
- large interim manifests and reports
- packaged Dataset Build archives
- Sprint 3 baseline checkpoints, qualitative outputs, runtime package records, and archives

Git ignore rules cover the canonical generated locations, including `data/archives/` and
`outputs/`.

## Fixed Google Drive Pattern

The supported Colab workflow uses exactly one Drive layout:

- Inputs live under `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Published archives live under
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`
- Sprint 3 baseline runs live under
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

There is no storage-provider switch, no local-vs-Drive branch in the notebook, and no user-edited
artifact destination configuration. The notebook does not support public download URLs or `gdown`.

## Archive Packaging

Local packaging is part of the primary CLI by default:

```bash
python scripts/dataset_build.py
```

To skip packaging:

```bash
python scripts/dataset_build.py --no-package
```

The archive set is:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Subset packaging removes stale unrequested sample archives from the local archive directory so the
local packaging output reflects the requested split selection.

Split sample archives are manifest-driven. Each `dataset_build_samples_<split>.tar.zst` contains
exactly the `.npz` files referenced by `sample_path` values in
`data/processed/v1/manifests/<split>.jsonl`; it is not a blind snapshot of
`data/processed/v1/samples/<split>/`.

## Sprint 3 Baseline Run Layout

Sprint 3 baseline Colab runs use this root:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

Each run uses this stable structure:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

`config/baseline.yaml` is the effective config used by the workflow and points
`checkpoint.output_dir` at the run's `checkpoints/` directory.
`config/source_baseline.yaml` preserves the original operator-provided config for provenance.

The deterministic archive names are:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

For training, qualitative export, and record packaging, resume behavior is:

1. Reuse already-extracted outputs in the expected run directory.
2. Otherwise extract the corresponding archive from `archives/`.
3. Otherwise run the step and write both extracted outputs and the archive under the run root.

The `record/` directory is a runtime-side package surface. It is not a formal experiment record.

## Future Stage Reuse

Later thesis stages can reuse stored Dataset Build artifacts by restoring the packaged outputs into
a working environment instead of rerunning the full raw-data pipeline every time. They can also use
Sprint 3 baseline run roots as baseline evidence, while keeping formal experiment-record authoring
for later phases.

The boundary remains unchanged:

- GitHub Pages may describe the workflow.
- GitHub Pages must not expose private artifact locations.
- The repository remains the source of truth for code, docs, and reproducible commands.
