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
- `tests/` holds deterministic tests for the package and fixture-backed Dataset Build checks.
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
- Models must read processed manifests, not raw files.
- Large generated artifacts remain outside GitHub even when the workflow is documented publicly.
