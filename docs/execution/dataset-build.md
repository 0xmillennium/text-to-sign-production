# Dataset Build Execution

Current public stage: Dataset Build.

Dataset Build turns fixed How2Sign/BFH raw inputs into processed Dataset Build outputs:
manifest-driven `.npz` samples, processed manifests, reports, and stable `.tar.zst` archives.

## Public Surface

- Main Colab notebook: `notebooks/colab/text_to_sign_production_colab.ipynb`
- CLI script: `python scripts/dataset_build.py`
- Reusable workflow entrypoint:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`

The notebook and CLI both call the reusable stage workflow.

## Local CLI

Run all splits against the canonical local raw layout:

```bash
python scripts/dataset_build.py
```

Run selected splits:

```bash
python scripts/dataset_build.py --splits train val
```

Run without local archive publication:

```bash
python scripts/dataset_build.py --no-package
```

The active default filter policy is `configs/data/filter-v2.yaml`: usable body plus at least one
usable hand. `configs/data/filter-v1.yaml` remains only for intentional reproducibility or
comparison runs.

## Colab Notebook

The single project-wide notebook is:

`notebooks/colab/text_to_sign_production_colab.ipynb`

The notebook is orchestration-only. It mounts Drive, acquires the repository, installs
dependencies, and operates the stage workflow from `src/`.

For Dataset Build outputs, the notebook provides separate cells to:

- reuse extracted outputs already present in the worktree
- extract archived outputs from the fixed Drive artifact root
- run Dataset Build and publish archives when neither extracted nor archived outputs are available

The notebook does not support public URL downloads, `gdown`, alternate archive formats, mounted
extracted keypoint directories, storage-provider switches, or custom input/output roots.

## Fixed Colab Inputs

Translations are read only from:

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_train.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_val.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_test.csv`

Keypoint archives are read only from:

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/train_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/val_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/test_2D_keypoints.tar.zst`

Archives are copied into `/content/how2sign_downloads`, extracted locally, staged into the
canonical repo raw layout, and cleaned after staging.

## Extracted Outputs

Dataset Build extracted outputs are the working directories under the repo:

- `data/interim/raw_manifests/`
- `data/interim/normalized_manifests/`
- `data/interim/filtered_manifests/`
- `data/interim/reports/`
- `data/processed/v1/manifests/`
- `data/processed/v1/reports/`
- `data/processed/v1/samples/<split>/`

These are local working artifacts and are not committed.

## Archived Outputs

Local archived outputs are written under:

`data/archives/`

Colab publishes Dataset Build archived outputs to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

Archive names are stable:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Split archives use the pattern `dataset_build_samples_<split>.tar.zst`.

Each split sample archive is manifest-driven. It contains exactly the `.npz` files referenced by
`data/processed/v1/manifests/<split>.jsonl`, not a blind copy of the full sample directory.

## Manual Limits

Local CI does not live-validate real Google Drive mounting, large archive transfer speed, real
Drive publishing, or real How2Sign runtime behavior. Those checks are documented under
`tests/operational/`.
