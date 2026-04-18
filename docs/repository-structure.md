# Repository Structure

## Top-Level Layout

```text
.
├── .github/workflows/
├── configs/
│   └── data/
├── data/
├── docs/
│   ├── execution/
│   └── storage/
├── notebooks/
│   └── colab/
├── scripts/
├── src/text_to_sign_production/
│   ├── data/
│   ├── modeling/
│   ├── ops/
│   └── workflows/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── operational/
│   ├── fixtures/
│   └── support/
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Purpose Of Each Area

- `.github/workflows/` contains CI, documentation deployment, and release automation.
- `configs/` contains Dataset Build filter policies, including active `filter-v2.yaml` and legacy
  `filter-v1.yaml`.
- `data/` contains the canonical raw, interim, processed, and archive dataset roots.
- `docs/` contains the MkDocs site, ADRs, experiment logging templates, and operational workflow
  guidance.
- `notebooks/` contains runner-only notebooks with no critical project logic.
- `scripts/` contains the primary Dataset Build CLI, the Sprint 3 baseline-modeling stage CLI,
  optional developer utilities, lower-level Sprint 3 baseline utilities, and later Sprint 3
  placeholder commands.
- `src/text_to_sign_production/data/` contains reusable data-pipeline logic.
- `src/text_to_sign_production/modeling/` contains the Sprint 3 baseline modeling scaffold for
  future backbones, processed-data loading, models, training, inference, and configuration code.
- `src/text_to_sign_production/ops/` contains reusable Colab, archive, and progress operations.
- `src/text_to_sign_production/workflows/` contains stage-level orchestration.
- `tests/unit/` holds isolated deterministic logic tests.
- `tests/integration/` holds CI-safe stage, CLI, and ops collaboration tests.
- `tests/e2e/` holds local happy paths that cross Dataset Build stage boundaries on tiny data.
- `tests/operational/` holds manual/external-runtime validation guidance for Colab, Drive, large
  archives, and real How2Sign smoke checks.
- `tests/fixtures/` holds small static fixtures and golden snippets.
- `tests/support/` holds reusable test-only builders, scenarios, archive helpers, path helpers, and
  shared assertions.

## Dataset Build Boundaries

The `data` package stays narrow: constants, schemas, JSONL I/O, MP4 metadata, OpenPose parsing,
raw manifest creation, normalization, filtering, processed-manifest export, report rendering,
validation, and small generic utilities are kept in separate modules.

The `ops` package owns long-running copy, extract, archive, publish, and shared progress helpers.
The `workflows` package composes reusable functions into stage-level workflows.

## Modeling Scaffold Boundaries

The `modeling` package is reserved for Sprint 3 Baseline Modeling. It now contains the Phase 2
processed-manifest, processed-`.npz`, and variable-length collation contracts for baseline modeling
inputs, the Phase 3 backbone wrapper and conservative model forward surface, Phase 4 reusable
mask-aware loss and validation-metric utilities, the Phase 5 config-driven training,
validation-loop, checkpointing, and runtime-provenance surface, and a Phase 6 qualitative panel
export/evidence surface for inspecting baseline outputs. Phase 7A adds public workflow
operationalization around those existing baseline pieces. It does not implement experiment-record
authoring, broad evaluation behavior, playback/rendering, or thesis-contribution model capability.

## Structural Principles

- Critical logic belongs in `src/`, not notebooks.
- Dataset Build has two operator-facing execution interfaces: the primary Colab notebook and one
  CLI script.
- Sprint 3 baseline modeling has one public stage-oriented CLI,
  `scripts/baseline_modeling.py`, plus lower-level training and qualitative-export utilities.
- The reusable Python workflow entrypoint is
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`.
- The reusable Sprint 3 baseline workflow entrypoint is
  `text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`.
- The active Dataset Build default filter policy is `configs/data/filter-v2.yaml`; legacy
  `configs/data/filter-v1.yaml` is retained for reproducibility-oriented runs.
- The Colab notebook supports only fixed mounted-Drive roots and stage workflow calls.
- Optional scripts are developer utilities, not stage execution entrypoints.
- Baseline evaluation remains a placeholder until later modeling phases land.
- Documentation is versioned with the codebase.
- `tests/conftest.py` remains narrow and only contains pytest wiring, not reusable domain helpers.
- Models must read processed manifests, not raw files.
- Large generated artifacts remain outside GitHub even when the workflow is documented publicly.
