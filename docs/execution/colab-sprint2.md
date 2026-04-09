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
packages outputs, and optionally copies archives to a private/shared Google Drive location. In
`public_urls` mode it stages split-scoped keypoint archives with visible in-notebook extraction
progress before moving the extracted split into the canonical raw layout. In `mounted_paths` mode
it either copies mounted keypoint archives into local Colab runtime storage before extraction or
copies already extracted split trees into the canonical raw layout.

## Execution Flow

1. Open the notebook in the Google Colab website.
2. Edit the repository ref if you need something other than `master`.
3. Edit `PIPELINE_SPLITS` in the notebook if you only want a subset such as `["train"]`.
4. Choose a raw-input mode:
   - `RAW_INPUT_MODE = "public_urls"` for public/shared translation URLs plus public/shared
     keypoint archive URLs
   - `RAW_INPUT_MODE = "mounted_paths"` with `MOUNTED_KEYPOINT_SOURCE_KIND = "archive"` for
     translation files plus keypoint archives already available in mounted Drive or another local
     path
   - `RAW_INPUT_MODE = "mounted_paths"` with `MOUNTED_KEYPOINT_SOURCE_KIND = "extracted_dir"` for
     translation files plus already extracted keypoint split roots or parent directories
5. Stage the selected split raw data into:
   - `data/raw/how2sign/translations/`
   - `data/raw/how2sign/bfh_keypoints/`
6. Run the existing Sprint 2 scripts directly for the selected splits:
   - `python scripts/prepare_raw.py --splits ...`
   - `python scripts/normalize_keypoints.py --splits ...`
   - `python scripts/filter_samples.py --config configs/data/filter-v1.yaml --splits ...`
   - `python scripts/export_training_manifest.py --splits ...`
7. Run `python scripts/package_sprint2_outputs.py --splits ...`.
8. Download selected archives locally or copy them to private/shared storage.

## Why Script-Based Execution Is Preferred In Colab

DVC remains part of the repository and remains the reproducibility standard for the Sprint 2
pipeline definition. The Colab notebook does not force `dvc repro` for heavy real-data runs
because hashing and traversing very large raw trees is an unnecessary operational cost in that
runtime.

The notebook therefore uses the same implemented script entry points that DVC already references.
This keeps the processing logic consistent while making the heavy execution path more practical.

## Raw Input Configuration

The notebook intentionally keeps raw source locations user-editable and explicit.

- `RAW_INPUT_MODE` chooses between public/shared URLs and mounted/local paths.
- `MOUNTED_KEYPOINT_SOURCE_KIND` is used only with `RAW_INPUT_MODE = "mounted_paths"` and applies
  to all selected `PIPELINE_SPLITS` in that notebook run.
- The notebook keeps separate per-split mappings for:
  - `TRANSLATION_URLS`
  - `KEYPOINT_ARCHIVE_URLS`
  - `MOUNTED_TRANSLATION_FILES`
  - `MOUNTED_KEYPOINT_ARCHIVE_FILES`
  - `MOUNTED_KEYPOINT_EXTRACTED_SPLIT_ROOTS`

- Public How2Sign URLs are not hardcoded unless you supply them in the notebook.
- Public Google Drive-hosted files in `public_urls` mode use `gdown` instead of a plain direct
  download call so common public-share links are handled more reliably.
- Translation files can therefore come from either:
  - direct URL download into `data/raw/how2sign/translations/`
  - mounted/local file copy into `data/raw/how2sign/translations/`
- Keypoint data can therefore come from either:
  - public/shared archive URLs
  - mounted/local archive files
  - mounted/local already extracted split roots
- The notebook validates only the selected `PIPELINE_SPLITS`, so subset runs do not require you to
  stage every official split.
- Archive-based keypoint staging explicitly supports both `.tar.gz` and `.tar.zst`.
- Archive-based keypoint staging streams archives through native `tar`, shows extraction progress in
  the notebook output, and moves the extracted split tree into canonical layout instead of copying
  it.
- Mounted extracted keypoint inputs can point either at the split root itself or a parent directory
  containing the split tree. The notebook keeps the previously fixed split-root detection behavior
  for `openpose_output/json`.
- Large keypoint archives should still be copied or downloaded into local Colab runtime storage
  under `/content/...` before extraction. That keeps extraction progress visible and avoids reading
  very large keypoint trees directly from mounted Google Drive during heavy processing.
- Mounted or copied raw inputs can come from Google Drive or another user-controlled location.
- The canonical in-repo layout must still be produced before the pipeline runs.

## Outputs

The notebook packages outputs into explicit `.tar.zst` archives:

- `sprint2_manifests_reports.tar.zst`
- `sprint2_samples_train.tar.zst`
- `sprint2_samples_val.tar.zst`
- `sprint2_samples_test.tar.zst`

When `PIPELINE_SPLITS` is a subset, the notebook passes that subset to the packaging helper and
only the requested split archives are regenerated.

These archives are meant for manual download, private/shared storage, or future reuse in later
thesis work. They are not meant to be committed to GitHub.

## Manual Steps And Limits

- You still need to provide the public raw URLs or mounted raw paths.
- If you want Google Drive copy behavior beyond the example defaults, create a local
  `configs/storage.local.yaml` inside the Colab runtime.
- Live validation of Colab runtime behavior, public download stability, and Drive upload is not
  performed in local CI. Those steps remain manual operational checks.
