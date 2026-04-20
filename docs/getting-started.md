# Getting Started

## Prerequisites

- Python 3.11 or newer
- `git`
- `zstd` and `tar` for local archive creation or extraction

## Install Paths

Base package only:

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Developer tooling:

```bash
python -m pip install -e ".[dev]"
```

Documentation tooling:

```bash
python -m pip install -e ".[docs]"
```

Baseline Modeling tooling:

```bash
python -m pip install -e ".[modeling]"
```

Full contributor path:

```bash
python -m pip install -e ".[dev,docs,modeling]"
```

The Makefile exposes the same setup paths:

```bash
make install-dev
make install-docs
make install-modeling
make install
```

`make install` installs `dev` and `docs`. Run `make install-modeling` as well before baseline
training or qualitative export.

## Smoke Check

```bash
python -c "from text_to_sign_production import smoke_check; print(smoke_check())"
```

Expected output:

```text
t2sp-smoke-ok
```

## Dataset Build Quick Start

Current public stage: Dataset Build.

For local Dataset Build runs, use the stage CLI against the canonical raw layout under
`data/raw/how2sign/`:

```bash
python scripts/dataset_build.py
```

This writes extracted outputs under `data/interim/` and `data/processed/v1/`, then publishes
archived outputs under `data/archives/`.

To skip local archive publication:

```bash
python scripts/dataset_build.py --no-package
```

## Baseline Modeling Quick Start

Implemented internal downstream surface: Baseline Modeling.

Baseline Modeling requires processed Dataset Build outputs and the `modeling` extra:

```bash
python scripts/baseline_modeling.py all --run-name baseline-default
```

Local run roots default to:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

The workflow uses archive-aware resume for training outputs, qualitative outputs, and
record/package outputs.

## Colab Heavy-Run Path

Use the single project-wide notebook:

`notebooks/colab/text_to_sign_production_colab.ipynb`

The notebook uses fixed Google Drive paths, supports reuse of extracted outputs, extraction from
archived outputs, and run/publish cells for Dataset Build, baseline training, qualitative panel
export, and record/package outputs.

## Next Steps

- Read [Development Setup](development-setup.md) before contributing.
- Read [Repository Structure](repository-structure.md) before changing boundaries.
- Read [Execution](execution.md) before running the supported Colab notebook.
- Read [Experiments](experiments.md) before writing Baseline Modeling records.
