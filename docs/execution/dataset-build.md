# Dataset Build Execution

Dataset Build supports exactly two primary public execution interfaces:

- Colab notebook: `notebooks/colab/dataset_build_colab.ipynb`
- CLI script: `python scripts/dataset_build.py`

Both call the same reusable stage workflow:

`text_to_sign_production.workflows.dataset_build.run_dataset_build`

## Colab Notebook

The Colab notebook is orchestration-only. It mounts Google Drive, clones the repository, imports
the Dataset Build workflow from `src/`, and calls it with the fixed Colab policy.

The notebook exposes only `PIPELINE_SPLITS`. It does not expose public URL downloads, `gdown`,
mounted extracted keypoint directories, alternate archive formats, storage-provider switches, or
custom input/output roots.

## CLI Script

For local terminal execution against the canonical raw dataset layout, run:

```bash
python scripts/dataset_build.py
```

For a build without local archive packaging, run:

```bash
python scripts/dataset_build.py --no-package
```

Subset runs remain split-aware:

```bash
python scripts/dataset_build.py --splits train val
```

The CLI is intentionally thin: it parses arguments, calls `run_dataset_build`, and prints concise
status output.

## Fixed Colab Inputs

The Colab workflow reads translations only from these fixed Google Drive paths:

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_train.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_val.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_test.csv`

It reads keypoints only from `.tar.zst` archives at:

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/train_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/val_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/test_2D_keypoints.tar.zst`

Archives are copied into `/content/how2sign_downloads`, extracted there, staged into the canonical
repo raw layout, and then temporary local extraction artifacts are cleaned.

## Fixed Outputs

The supported Colab output destination is:

`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

The archive set is:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Subset runs publish only the selected split archives plus the shared manifests/reports archive.

## Developer Utilities

The only scripts kept outside the primary execution interface are developer utilities:

- `python scripts/validate_manifest.py --manifest ... --kind raw|normalized|processed`
- `python scripts/view_sample.py --split train --sample-id ...`

They are for inspection and debugging, not for running the Dataset Build stage.

## Manual Limits

Local CI does not live-validate actual Colab Drive mounting, large archive transfer speed, or real
Google Drive artifact publishing. Those remain manual operational checks.
