# Repository Structure

## Top-Level Layout

```text
.
├── .github/workflows/
├── configs/
├── docs/
├── notebooks/
├── scripts/
├── src/text_to_sign_production/
├── tests/
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Purpose Of Each Area

- `.github/workflows/` contains CI, documentation deployment, and release automation.
- `configs/` is reserved for future explicit configuration files.
- `docs/` contains the MkDocs site, ADRs, and experiment logging templates.
- `notebooks/` contains runner-only notebooks with no critical project logic.
- `scripts/` is reserved for small repo-support utilities.
- `src/text_to_sign_production/` is the Python package root.
- `tests/` holds deterministic tests for the package and future infrastructure.

## Structural Principles

- Critical logic belongs in `src/`, not notebooks.
- Documentation is versioned with the codebase.
- Research process artifacts such as ADRs and experiment logs are first-class repository assets.
- DVC support is present for future pipeline work, but no actual data stages are implemented yet.
