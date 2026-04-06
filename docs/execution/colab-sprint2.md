# Colab Sprint 2 Execution

Heavy Sprint 2 runs are supported through Google Colab because the repository needs a practical
way to execute the existing How2Sign pipeline over large real-data trees without treating the
local development machine as the only runtime.

## Why Colab Is Used

- Local development remains the preferred place for code, tests, linting, and docs validation.
- Full real-data Sprint 2 execution is operationally expensive because the raw dataset contains a
  very large number of small files.
- Google Colab provides a browser-based runtime that can stage raw data, run the existing
  repository scripts, and package outputs without committing large artifacts to GitHub.

## Primary Notebook

The main Colab entry point is:

`notebooks/colab/sprint2_pipeline_colab.ipynb`

The notebook is runner-only. It does not own business logic. It clones the repository, installs
dependencies, stages raw inputs into the canonical layout, calls the existing Sprint 2 scripts,
packages outputs, and optionally copies archives to a private/shared Google Drive location.

## Execution Flow

1. Open the notebook in the Google Colab website.
2. Edit the repository ref if you need something other than `master`.
3. Choose a raw-input mode:
   - public source URLs that you provide in the notebook
   - already available mounted/local paths that you provide in the notebook
4. Stage raw data into:
   - `data/raw/how2sign/translations/`
   - `data/raw/how2sign/bfh_keypoints/`
5. Run the existing Sprint 2 scripts directly:
   - `python scripts/prepare_raw.py`
   - `python scripts/normalize_keypoints.py`
   - `python scripts/filter_samples.py --config configs/data/filter-v1.yaml`
   - `python scripts/export_training_manifest.py`
6. Run `python scripts/package_sprint2_outputs.py`.
7. Download selected archives locally or copy them to private/shared storage.

## Why Script-Based Execution Is Preferred In Colab

DVC remains part of the repository and remains the reproducibility standard for the Sprint 2
pipeline definition. The Colab notebook does not force `dvc repro` for heavy real-data runs
because hashing and traversing very large raw trees is an unnecessary operational cost in that
runtime.

The notebook therefore uses the same implemented script entry points that DVC already references.
This keeps the processing logic consistent while making the heavy execution path more practical.

## Raw Input Configuration

The notebook intentionally keeps raw source locations user-editable.

- Public How2Sign URLs are not hardcoded unless you supply them in the notebook.
- Public Google Drive-hosted files in `public_urls` mode use `gdown` instead of a plain direct
  download call so common public-share links are handled more reliably.
- Mounted or copied raw inputs can come from Google Drive or another user-controlled location.
- The canonical in-repo layout must still be produced before the pipeline runs.

## Outputs

The notebook packages outputs into explicit `.tar.zst` archives:

- `sprint2_manifests_reports.tar.zst`
- `sprint2_samples_train.tar.zst`
- `sprint2_samples_val.tar.zst`
- `sprint2_samples_test.tar.zst`

These archives are meant for manual download, private/shared storage, or future reuse in later
thesis work. They are not meant to be committed to GitHub.

## Manual Steps And Limits

- You still need to provide the public raw URLs or mounted raw paths.
- If you want Google Drive copy behavior beyond the example defaults, create a local
  `configs/storage.local.yaml` inside the Colab runtime.
- Live validation of Colab runtime behavior, public download stability, and Drive upload is not
  performed in local CI. Those steps remain manual operational checks.
