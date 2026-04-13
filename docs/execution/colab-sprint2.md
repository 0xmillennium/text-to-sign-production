# Colab Sprint 2 Execution

Sprint 2 supports exactly one Google Colab execution path. The goal is a repeatable, thesis-grade
workflow with fewer branches and no notebook-owned operational logic.

## Primary Notebook

The main Colab entry point is:

`notebooks/colab/sprint2_pipeline_colab.ipynb`

The notebook is orchestration-only. It mounts Google Drive, clones the repository, calls thin
repository scripts, and shows their progress-bearing output.

## Fixed Inputs

The notebook reads only from these fixed Google Drive paths:

- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_train.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_val.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/translations/how2sign_realigned_test.csv`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/train_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/val_2D_keypoints.tar.zst`
- `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/archives/test_2D_keypoints.tar.zst`

Translations are CSV files only. Keypoints are `.tar.zst` archives only.

## Execution Flow

1. Open the notebook in Google Colab.
2. Mount Google Drive at `/content/drive`.
3. Edit `PIPELINE_SPLITS` only if you want a subset such as `["val"]`.
4. Run `python scripts/stage_colab_inputs.py --splits ...`.
   This copies the fixed Drive archives into `/content/how2sign_downloads`, extracts them there,
   stages the canonical raw layout under the repository, and cleans temporary local artifacts.
5. Run the existing Sprint 2 scripts directly for the selected splits:
   - `python scripts/prepare_raw.py --splits ...`
   - `python scripts/normalize_keypoints.py --splits ...`
   - `python scripts/filter_samples.py --config configs/data/filter-v1.yaml --splits ...`
   - `python scripts/export_training_manifest.py --splits ...`
6. Run `python scripts/publish_colab_outputs.py --splits ...`.
   This creates `.tar.zst` output archives and copies them to the fixed Drive artifact location.

## Fixed Outputs

The only supported output destination is:

- `/content/drive/MyDrive/text-to-sign-production/artifacts/sprint2/processed-v1/`

The publish step writes:

- `sprint2_manifests_reports.tar.zst`
- `sprint2_samples_train.tar.zst`
- `sprint2_samples_val.tar.zst`
- `sprint2_samples_test.tar.zst`

Subset runs publish only the selected split archives plus the shared manifests/reports archive.

## Why This Is Script-Based

DVC remains the reproducibility standard for the implemented pipeline definition, but the Colab
workflow calls the repository scripts directly because the heavy raw-data runtime does not benefit
from extra notebook logic or extra operational modes.

The important boundary is fixed:

- notebooks orchestrate
- scripts coordinate
- reusable Python modules implement the long-running copy, extract, archive, and publish logic

## Manual Limits

Local CI does not live-validate actual Colab Drive mounting, large archive transfer speed, or real
Google Drive artifact publishing. Those remain manual operational checks.
