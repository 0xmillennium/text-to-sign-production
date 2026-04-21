# text-to-sign-production

[![CI](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml)
[![Docs](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](pyproject.toml)

`text-to-sign-production` is a reproducible research repository for a graduation thesis on
English text-to-sign-pose production. The long-term direction is:

`English text -> pose tokens -> keypoints -> skeleton/avatar`

## What This Repository Is

Current public stage: Dataset Build.

Dataset Build creates training-ready How2Sign/BFH artifacts from fixed translation files and
OpenPose keypoint inputs. It exports versioned processed manifests, compressed `.npz` pose samples,
reports, and manifest-driven archives that later modeling stages consume.

Implemented internal downstream surface: Baseline Modeling.

Baseline Modeling is the implemented internal downstream baseline surface. It consumes processed
Dataset Build outputs, runs train/validation baseline jobs, writes checkpoints and
`run_summary.json`, exports a qualitative validation panel, and packages runtime evidence for formal
experiment records.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Current Public Workflow Surfaces

- Dataset Build CLI: `python scripts/dataset_build.py`
- Dataset Build workflow entrypoint:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`
- Baseline Modeling CLI: `python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]`
- Baseline Modeling workflow entrypoint:
  `text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`
- Main Colab notebook: `notebooks/colab/text_to_sign_production_colab.ipynb`

Lower-level scripts such as `scripts/train_baseline.py`, `scripts/export_qualitative_panel.py`,
`scripts/validate_manifest.py`, and `scripts/view_sample.py` are developer utilities, not the main
public workflow surface. `scripts/evaluate_baseline.py` remains a placeholder until broader
evaluation work starts.

## Current Maturity And Boundaries

- Dataset Build is the implemented public stage.
- Baseline Modeling is implemented as an internal downstream baseline surface, not as the main
  thesis contribution.
- Processed dataset access is manifest-driven.
- The active Dataset Build filter policy is `configs/data/filter-v2.yaml`: usable body plus at
  least one usable hand.
- Large generated artifacts stay outside GitHub.
- Notebooks are orchestration-only; business logic belongs in `src/`.
- DVC is not part of the active workflow.
- Broader evaluation, contribution modeling, playback/demo, and final thesis packaging are later
  roadmap items.

## Installation Matrix

| Path | Command | Use When |
| --- | --- | --- |
| Base package | `python -m pip install -e .` | Import package code and run lightweight utilities. |
| Dev | `python -m pip install -e ".[dev]"` | Run tests, lint, type checks, and local quality tooling. |
| Docs | `python -m pip install -e ".[docs]"` | Build the MkDocs site. |
| Modeling | `python -m pip install -e ".[modeling]"` | Run Baseline Modeling training or qualitative export. |
| Contributor/full | `python -m pip install -e ".[dev,docs,modeling]"` | Work across docs, tests, and baseline modeling. |

The Makefile exposes the common setup paths:

```bash
make install-dev
make install-docs
make install-modeling
make install
```

`make install` installs `dev` and `docs`. Install the `modeling` extra as well when running baseline
training or qualitative export.

## Quick Starts

### Dataset Build

Dataset Build expects the canonical local raw layout under `data/raw/how2sign/`.

```bash
python scripts/dataset_build.py
```

The default command runs all splits, uses `configs/data/filter-v2.yaml`, writes extracted outputs
under `data/interim/` and `data/processed/v1/`, and publishes local archived outputs under
`data/archives/`.

To run selected splits:

```bash
python scripts/dataset_build.py --splits train val
```

To skip local archive publication:

```bash
python scripts/dataset_build.py --no-package
```

### Baseline Modeling

Baseline Modeling consumes processed Dataset Build manifests and `.npz` samples.

```bash
python scripts/baseline_modeling.py all --run-name baseline-default
```

Local run roots default to:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

Each run root contains extracted outputs and archived outputs:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

The deterministic archives are:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

### Colab Heavy-Run Path

Use the single project-wide notebook:

`notebooks/colab/text_to_sign_production_colab.ipynb`

The notebook is stage-oriented and operator-facing. From Dataset Build outputs onward, each major
step has separate cells to reuse extracted outputs, extract archived outputs, or run and publish
outputs.

The fixed Drive input root is:

`/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`

Dataset Build archives publish to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

Baseline Modeling run roots publish to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

## Artifact And Storage Policy

Git stores source, docs, tests, configs, ADRs, and experiment-record templates. Git does not store
raw How2Sign data, processed `.npz` samples, checkpoints, qualitative panel exports, runtime record
packages, or `.tar.zst` archives.

Dataset Build archived outputs use stable names:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Baseline Modeling archived outputs use stable names under each run root:

- `baseline_training_outputs.tar.zst`
- `baseline_qualitative_outputs.tar.zst`
- `baseline_record_package.tar.zst`

The runtime record package is baseline evidence. The formal Baseline Modeling experiment record is a
Markdown record that cites `run_summary.json`, checkpoints, qualitative outputs, the evidence
bundle, packaged runtime artifacts, and Dataset Build provenance.

## Testing And Quality Summary

The automated test suite is layered:

- `unit`: isolated logic and contract tests
- `integration`: CI-safe multi-module workflow, CLI, notebook, and ops tests
- `e2e`: tiny local happy paths across stage boundaries
- `operational`: manual Colab, Drive, large archive, and real How2Sign checks

Common commands:

```bash
make lint
make test
make docs
make check
python -m pytest
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m e2e
```

Operational checks are documented under `tests/operational/` and are excluded from normal pytest
runs.

## Docs Map

- [Home](docs/index.md)
- [Getting Started](docs/getting-started.md)
- [Development Setup](docs/development-setup.md)
- [Repository Structure](docs/repository-structure.md)
- [Execution](docs/execution.md)
- [Data Overview](docs/data/index.md)
- [Testing](docs/testing/index.md)
- [Decisions](docs/decisions/index.md)
- [Experiments](docs/experiments/index.md)
- [Roadmap](docs/roadmap.md)
- [Literature Positioning](docs/literature-positioning.md)

## Roadmap Pointer

The thesis roadmap is documented in [docs/roadmap.md](docs/roadmap.md). Sprint 3 establishes the
baseline. Sprint 4 adds broader evaluation and error analysis. Sprint 5 and Sprint 6 are the main
planned contribution stages.

## Contribution And Development Pointer

Start with [docs/development-setup.md](docs/development-setup.md), then read
[docs/repository-structure.md](docs/repository-structure.md) and [docs/testing/index.md](docs/testing/index.md).
Record long-lived architecture decisions under `docs/decisions/` and empirical baseline records
under `docs/experiments/`.
