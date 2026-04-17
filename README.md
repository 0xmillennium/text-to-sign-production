# text-to-sign-production

`text-to-sign-production` is a reproducible research repository for a graduation thesis on
text-to-sign production. The long-term system direction is:

`English text -> pose tokens -> keypoints -> skeleton/avatar`

The current implemented ML/data-pipeline stage is **Dataset Build**. It builds training-ready
How2Sign/BFH dataset artifacts from fixed translation files and OpenPose keypoint inputs while
explicitly deferring tokenizer, modeling, and rendering work.

## Dataset Build Includes

- Split-preserving ingestion of How2Sign translation TSVs and BFH OpenPose JSON clips
- Raw manifests under `data/interim/raw_manifests/`
- Normalized `.npz` sample export under `data/processed/v1/samples/`
- Manifest-driven processed dataset access under `data/processed/v1/manifests/`
- Runtime assumption validation, split-integrity reporting, and data-quality reporting
- One reusable Python workflow entrypoint:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`
- Two operator-facing execution interfaces:
  - Colab: `notebooks/colab/dataset_build_colab.ipynb`
  - CLI: `python scripts/dataset_build.py`

## Dataset Build Explicitly Does Not Include

- A tokenizer or retrieval system
- Pose-token generation or tokenizer training
- Text-to-pose model training or inference
- Skeleton rendering or avatar animation
- Model-weight publishing or finalized weight licensing

This boundary keeps the repository honest about its current maturity: it provides a reproducible
dataset layer for later thesis stages instead of placeholder ML code.

## Repository Structure

```text
.
├── configs/                  # Dataset Build filter policies
├── data/                     # Raw, interim, processed, and archive dataset roots
├── docs/                     # MkDocs source, ADRs, experiment records, and ops docs
├── notebooks/
│   └── colab/                # Primary Dataset Build Colab notebook
├── scripts/                  # Primary CLI plus optional developer utilities
├── src/text_to_sign_production/
│   ├── data/                 # Reusable Dataset Build data pipeline package
│   ├── ops/                  # Reusable Colab/archive operations layer
│   └── workflows/            # Stage-level workflow orchestration
├── tests/
│   ├── unit/                 # Isolated deterministic logic tests
│   ├── integration/          # CI-safe stage, CLI, and ops collaboration tests
│   ├── e2e/                  # Local tiny end-to-end Dataset Build happy paths
│   ├── operational/          # Manual/external-runtime validation guidance
│   ├── fixtures/             # Small static test fixtures and golden snippets
│   └── support/              # Reusable test-only builders, scenarios, and assertions
├── Makefile                  # Common local developer commands
├── mkdocs.yml                # MkDocs configuration
├── pyproject.toml            # Primary Python project configuration
└── requirements-colab.txt    # Minimal Colab install requirements
```

## Local Setup

The repository targets Python 3.11+ for development.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

You can also use the Makefile:

```bash
make install
```

## Common Commands

```bash
make lint
make test
make docs
make check
make ci-local
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m e2e
python scripts/dataset_build.py
python scripts/dataset_build.py --no-package
python scripts/validate_manifest.py --manifest data/interim/raw_manifests/raw_train.jsonl --kind raw
python scripts/view_sample.py --split train --sample-id <sample_id>
```

- `python scripts/dataset_build.py` runs the full Dataset Build stage against the canonical local
  raw dataset layout, uses the active `configs/data/filter-v2.yaml` policy by default, and creates
  local `.tar.zst` archives under `data/archives/`.
- `python scripts/dataset_build.py --no-package` runs the stage without local archive packaging.
- `validate_manifest.py` and `view_sample.py` are optional developer utilities, not primary
  workflow entrypoints.

## Testing Architecture

The test suite has four layers:

- `unit`: isolated logic such as OpenPose parsing, validation helpers, filtering policy, progress,
  archive internals, and package smoke checks.
- `integration`: CI-safe stage, CLI, and ops checks on tiny fixture or builder-backed workspaces.
- `e2e`: local happy paths that cross Dataset Build stage boundaries with tiny fake data.
- `operational`: real Colab, Google Drive, large archive, publish, and real How2Sign checks that do
  not run in normal CI.

`tests/fixtures/` stores small static examples when a checked-in file is clearer. `tests/support/`
stores reusable fake-data builders, scenario builders, archive helpers, path helpers, and shared
assertions. `tests/conftest.py` is kept narrow for pytest-only wiring.

Normal CI-safe testing is:

```bash
python -m pytest
```

Operational checks are documented under `tests/operational/` and must be run manually or by an
explicit release-time procedure.

## Colab Usage Philosophy

Notebooks are runner-only interfaces. They may install the repository, import code from `src/`,
and execute orchestrated workflows, but they must not become the home of core project logic.

The Dataset Build Colab notebook is:

`notebooks/colab/dataset_build_colab.ipynb`

It exposes only `PIPELINE_SPLITS`. It does not support public URLs, `gdown`, alternate archive
formats, mounted extracted keypoint directories, storage-provider switches, or user-defined
input/output roots.

## Dataset Build Workflow

The stage-level orchestration entrypoint is:

`text_to_sign_production.workflows.dataset_build.run_dataset_build`

It orchestrates the existing reusable functions for raw manifest creation, normalization,
filtering, final manifest/report export, and output packaging or publishing. Pipeline logic stays
in `src/text_to_sign_production/data/`; scripts and notebooks remain thin.

The active default filter policy is `configs/data/filter-v2.yaml`: body must be usable and at least
one hand must be usable. The legacy strict `configs/data/filter-v1.yaml` policy remains available for
intentional reproducibility or comparison runs, but it is not the current default.

## Artifact Storage And Git Hygiene

- Large raw, interim, processed, and archive artifacts are kept out of GitHub.
- The supported Colab workflow reads only from the fixed Drive raw paths under
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`.
- Keypoint inputs are `.tar.zst` archives only; translations use the canonical `.csv` filenames.
- The supported Colab workflow publishes only to
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`.
- Local packaging writes only to `data/archives/`.
- Packaged outputs are `dataset_build_manifests_reports.tar.zst` plus one manifest-driven
  `dataset_build_samples_<split>.tar.zst` archive for each selected split.
- GitHub Pages documents the workflow but must not expose private storage links or folder IDs.

## Verified Raw Data Facts

The repository currently targets the manually downloaded raw data under `data/raw/how2sign/`.
Local inspection confirmed:

- Translation files are tab-separated even though the canonical filenames use the `.csv`
  extension.
- Translation columns are exactly `VIDEO_ID`, `VIDEO_NAME`, `SENTENCE_ID`, `SENTENCE_NAME`,
  `START_REALIGNED`, `END_REALIGNED`, and `SENTENCE`.
- The direct alignment rule is `SENTENCE_NAME == openpose_output/json/<clip_dir_name>`.
- Observed unmatched translation rows without a keypoint directory are: train `118`, val `2`,
  test `14`.
- BFH MP4 metadata is mostly `1280x720`, with a small `1920x1080` minority in the train split.
  The pipeline keeps the fixed `1280x720` OpenPose normalization basis and reports metadata
  deviations explicitly.
- FPS varies across clips, so it is stored per sample instead of assumed globally.

## Processed Dataset Contract

- Processed schema version: `t2sp-processed-v1`
- Processed sample format: compressed `.npz`
- Processed samples include body, left-hand, and right-hand arrays
- Active filtering keeps samples with usable body and at least one usable hand
- Face is retained in schema, export, and reporting but is optional for v1 completion
- Future models must read processed manifests only

Future thesis stages can build tokenization, retrieval, modeling, evaluation, and downstream
rendering on top of this versioned, auditable dataset layer instead of reading raw files directly.
