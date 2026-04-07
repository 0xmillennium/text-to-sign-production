# text-to-sign-production

`text-to-sign-production` is a reproducible research repository for a graduation thesis on
text-to-sign production. The long-term system direction is:

`English text -> pose tokens -> keypoints -> skeleton/avatar`

Sprint 2 turns the Sprint 1 scaffold into a working, reproducible data pipeline for How2Sign BFH
keypoints plus aligned translations. The repository now builds a manifest-driven, training-ready
v1 dataset while still explicitly deferring modeling work. This operational update adds Colab
support for heavy real-data execution and a documented artifact-storage workflow that keeps large
outputs outside GitHub.

## Sprint 2 Includes

- Split-preserving ingestion of How2Sign translation TSVs and BFH OpenPose JSON clips
- Raw manifests at:
  - `data/interim/raw_manifests/raw_train.jsonl`
  - `data/interim/raw_manifests/raw_val.jsonl`
  - `data/interim/raw_manifests/raw_test.jsonl`
- Processed `.npz` sample export under `data/processed/v1/samples/`
- Manifest-driven processed dataset access at:
  - `data/processed/v1/manifests/train.jsonl`
  - `data/processed/v1/manifests/val.jsonl`
  - `data/processed/v1/manifests/test.jsonl`
- Runtime assumption validation and auditable data-quality reporting
- DVC stages for `prepare_raw`, `normalize_keypoints`, `filter_samples`, and
  `export_training_manifest`
- A Colab notebook for heavy script-based Sprint 2 execution
- Explicit output packaging via `python scripts/package_sprint2_outputs.py`
- A documented private/shared artifact-storage pattern using example Google Drive configuration
- Sprint 1 quality gates, docs, ADRs, and notebook philosophy carried forward unchanged

## Sprint 2 Explicitly Does Not Include

- A tokenizer or retrieval system
- Pose-token generation or tokenizer training
- Text-to-pose model training or inference
- Skeleton rendering or avatar animation
- Retrieval implementation
- Model-weight publishing or finalized weight licensing

Sprint 2 keeps the repository honest about its current maturity by focusing on dataset quality and
reproducibility instead of placeholder ML code.

## Repository Structure

```text
.
├── configs/                  # Data filters and example storage configuration
├── data/                     # Raw, interim, and processed dataset roots
├── docs/                     # MkDocs source, ADRs, experiment records, and ops docs
├── notebooks/                # Thin runner notebooks only
│   └── colab/                # Primary Colab heavy-execution notebook
├── scripts/                  # Thin CLI entrypoints and operational helpers
├── src/text_to_sign_production/
│   └── data/                 # Reusable data pipeline package
├── tests/                    # Fast deterministic tests plus fixture-backed pipeline tests
├── dvc.yaml                  # Reproducible Sprint 2 data pipeline stages
├── Makefile                  # Common local developer commands
├── mkdocs.yml                # MkDocs configuration
├── pyproject.toml            # Primary Python project configuration
└── requirements-colab.txt    # Minimal Colab install requirements
```

## Local Setup

Sprint 1 targets Python 3.11 for development.

Full contributor setup:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

You can also use the Makefile:

```bash
make install
```

Targeted installs stay available when you only need one extra set:

```bash
make install-dev
make install-docs
```

- `make install` installs both `dev` and `docs` extras because `make ci-local` and strict docs
  builds both depend on them.
- `make install-dev` installs only contributor tooling such as `mypy`, `ruff`, `pytest`,
  `pre-commit`, `pip-audit`, and type stubs.
- `make install-docs` installs only the MkDocs toolchain.

## Common Commands

```bash
make lint
make test
make docs
make check
make ci-local
python scripts/prepare_raw.py
python scripts/validate_manifest.py --manifest data/interim/raw_manifests/raw_train.jsonl --kind raw
python scripts/normalize_keypoints.py
python scripts/filter_samples.py --config configs/data/filter-v1.yaml
python scripts/export_training_manifest.py
python scripts/package_sprint2_outputs.py
dvc repro
```

- `make lint` runs `ruff` checks, `ruff format --check`, and `mypy`.
- `make test` runs the fast pytest suite with coverage output.
- `make docs` builds the MkDocs site in strict mode.
- `make check` runs linting, tests, and docs checks together.
- `make ci-local` runs the main local quality checks together, including `pre-commit` and
  `pip-audit`.
- `dvc repro` executes the full Sprint 2 data pipeline against the canonical raw dataset layout.
- `python scripts/package_sprint2_outputs.py` creates explicit `.tar.zst` archives for transfer or
  private/shared storage.
- Split-aware script runs are supported throughout Sprint 2 via `--splits`, including export and
  packaging for subset Colab workflows.

## Colab Usage Philosophy

Notebooks are runner-only interfaces. They may install the repository, import code from `src/`,
and execute smoke checks or future orchestrated workflows, but they must not become the home of
core project logic. All important logic belongs in the package and supporting modules.

Sprint 2 keeps that rule intact and now ships:

- `notebooks/colab/sprint2_pipeline_colab.ipynb` for heavy real-data execution in Google Colab
- `notebooks/colab_smoke_test.ipynb` for minimal repository smoke checks

## DVC Role

DVC now drives the reproducible Sprint 2 data pipeline through these implemented stages:

- `prepare_raw`
- `normalize_keypoints`
- `filter_samples`
- `export_training_manifest`

The canonical model-facing access layer is the processed JSONL manifests, not the raw files.

For heavy Colab execution, the repository uses the same implemented scripts directly instead of
forcing `dvc repro` over very large raw trees. DVC remains the pipeline-definition standard.

## Artifact Storage And Git Hygiene

- Large raw, interim, processed, archive, and DVC-cache artifacts are kept out of GitHub.
- `configs/storage.example.yaml` documents the storage contract without embedding private values.
- `configs/storage.local.yaml` is intentionally local-only and ignored by Git.
- GitHub Pages documents the workflow but must not expose private storage links or folder IDs.
- Future sprints can reuse packaged Sprint 2 outputs from private/shared storage instead of
  republishing large artifacts.

## Verified Raw Data Facts

The repository currently targets the manually downloaded raw data under `data/raw/how2sign/`.
Local inspection confirmed:

- Translation files are tab-separated even though the canonical filenames use the `.csv`
  extension.
- Translation columns are exactly:
  - `VIDEO_ID`
  - `VIDEO_NAME`
  - `SENTENCE_ID`
  - `SENTENCE_NAME`
  - `START_REALIGNED`
  - `END_REALIGNED`
  - `SENTENCE`
- The direct alignment rule is `SENTENCE_NAME == openpose_output/json/<clip_dir_name>`.
- Observed unmatched translation rows without a keypoint directory are:
  - train: `118`
  - val: `2`
  - test: `14`
- Runtime assumption validation found BFH MP4 metadata is mostly `1280x720`, with a small
  `1920x1080` minority in the train split. Sprint 2 keeps the fixed `1280x720` OpenPose
  normalization basis and reports metadata deviations explicitly.
- FPS varies across clips, so it is stored per sample instead of assumed globally.

## Processed Dataset Contract

- Processed schema version: `t2sp-processed-v1`
- Processed sample format: compressed `.npz`
- Required core channels: body, left hand, right hand
- Face is retained in schema, export, and reporting but is optional for v1 completion
- Future models must read processed manifests only

## Quality And Reproducibility Foundation

- Local quality checks are available through `pre-commit`, `ruff`, `mypy`, `pytest`, and
  strict MkDocs builds.
- Runtime assumption checks and generated reports make raw-data assumptions executable instead of
  informal.
- The repository is structured so GitHub-based automation can continue without redesigning the
  package, docs, notebook, or DVC foundations established in Sprint 1.
- Release and deployment automation are intentionally treated as repository-level concerns that
  can be layered on top of this baseline.

## How Future Sprints Build On This Foundation

Future sprints can now build tokenization, retrieval, modeling, evaluation, and downstream
rendering on top of a versioned, auditable dataset layer instead of reading raw files directly.
