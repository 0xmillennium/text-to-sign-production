# Dataset Build Execution

Dataset Build keeps its public execution surface narrow:

- Colab notebook: `notebooks/colab/dataset_build_colab.ipynb`
- CLI script: `python scripts/dataset_build.py`
- reusable workflow entrypoint:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`

The operator-facing interfaces both call the reusable stage workflow.

The active default filter policy is `configs/data/filter-v2.yaml`. The legacy strict policy
`configs/data/filter-v1.yaml` remains available only when intentionally selected for
reproducibility or comparison.

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

The CLI is intentionally thin: it parses arguments, uses the active default filter config unless
`--config` is provided, calls `run_dataset_build`, and prints concise status output.

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

## Outputs

Local packaging writes Dataset Build archives under `data/archives/`. The supported Colab workflow
packages locally first, then publishes the resulting archives only to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

The archive set is:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Equivalently, split sample archives use `dataset_build_samples_<split>.tar.zst`. Subset runs
create or publish only the selected split archives plus the shared manifests/reports archive.

Split sample archives are manifest-driven. Each `dataset_build_samples_<split>.tar.zst` contains
exactly the `.npz` files referenced by `data/processed/v1/manifests/<split>.jsonl`; it is built
from manifest sample paths rather than the entire `data/processed/v1/samples/<split>/` directory.

## Developer Utilities

The only scripts kept outside the primary execution interface are developer utilities:

- `python scripts/validate_manifest.py --manifest ... --kind raw|normalized|processed`
- `python scripts/view_sample.py --split train --sample-id ...`

They are for inspection and debugging, not for running the Dataset Build stage.

## Manual Limits

Local CI does not live-validate actual Colab Drive mounting, large archive transfer speed, real
Google Drive artifact publishing, or real How2Sign runtime behavior. Those remain manual
operational checks documented under `tests/operational/`.
