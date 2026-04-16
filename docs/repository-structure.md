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
│   ├── ops/
│   └── workflows/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   ├── operational/
│   ├── fixtures/
│   └── support/
├── dvc.yaml
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Purpose Of Each Area

- `.github/workflows/` contains CI, documentation deployment, and release automation.
- `configs/` contains the Dataset Build filtering policy.
- `data/` contains the canonical raw, interim, processed, and archive dataset roots.
- `docs/` contains the MkDocs site, ADRs, experiment logging templates, and operational workflow
  guidance.
- `notebooks/` contains runner-only notebooks with no critical project logic.
- `scripts/` contains the primary Dataset Build CLI plus optional developer utilities.
- `src/text_to_sign_production/data/` contains reusable data-pipeline logic.
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
- `dvc.yaml` defines the reproducible Dataset Build stage.

## Dataset Build Boundaries

The `data` package stays narrow: constants, schemas, JSONL I/O, MP4 metadata, OpenPose parsing,
raw manifest creation, normalization, filtering, processed-manifest export, report rendering,
validation, and small generic utilities are kept in separate modules.

The `ops` package owns long-running copy, extract, archive, publish, and shared progress helpers.
The `workflows` package composes those reusable functions into the public Dataset Build stage.

## Structural Principles

- Critical logic belongs in `src/`, not notebooks.
- Dataset Build has exactly two primary public execution interfaces: one Colab notebook and one
  CLI script.
- The Colab notebook supports only the fixed mounted-Drive workflow and exposes only
  `PIPELINE_SPLITS`.
- Optional scripts are developer utilities, not stage execution entrypoints.
- Documentation is versioned with the codebase.
- `tests/conftest.py` remains narrow and only contains pytest wiring, not reusable domain helpers.
- Models must read processed manifests, not raw files.
- Large generated artifacts remain outside GitHub even when the workflow is documented publicly.
