# Repository Structure

## Top-Level Layout

```text
.
├── .github/workflows/
├── configs/
│   └── storage.example.yaml
├── data/
├── docs/
│   ├── execution/
│   └── storage/
├── notebooks/
│   └── colab/
├── scripts/
├── src/text_to_sign_production/
│   └── data/
├── tests/
├── dvc.yaml
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Purpose Of Each Area

- `.github/workflows/` contains CI, documentation deployment, and release automation.
- `configs/` contains the Sprint 2 filtering policy and example operational storage configuration.
- `data/` contains the canonical raw, interim, and processed dataset roots.
- `docs/` contains the MkDocs site, ADRs, experiment logging templates, and operational workflow
  guidance.
- `notebooks/` contains runner-only notebooks with no critical project logic, including Colab
  execution support.
- `scripts/` contains thin CLI entrypoints for the data pipeline plus small operational helpers.
- `src/text_to_sign_production/` is the Python package root, including the reusable Sprint 2 data
  modules.
- `tests/` holds deterministic tests for the package and fixture-backed Sprint 2 pipeline checks.
- `dvc.yaml` defines the implemented Sprint 2 stages.

## Structural Principles

- Critical logic belongs in `src/`, not notebooks.
- Documentation is versioned with the codebase.
- Research process artifacts such as ADRs and experiment logs are first-class repository assets.
- Models must read processed manifests, not raw files.
- DVC stages now implement the dataset build while preserving the Sprint 1 repository conventions.
- Large generated artifacts remain outside GitHub even when the workflow is documented publicly.
