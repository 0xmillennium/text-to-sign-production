# Development Setup

## Purpose

This page explains how to set up a local development environment to contribute to the repository.

It is contributor-focused. The supported runtime workflow lives in [Execution](execution.md).

## Install Tooling

Use the smallest install path that matches the work.

```bash
make install-dev
make install-docs
make install-modeling
make install
```

- `make install-dev` installs code-quality and test tooling.
- `make install-docs` installs the MkDocs toolchain.
- `make install-modeling` installs the optional PyTorch and Hugging Face modeling stack used by
  Baseline Modeling.
- `make install` installs `dev` and `docs` and registers local hooks.

Equivalent direct installs:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
python -m pip install -e ".[modeling]"
python -m pip install -e ".[dev,docs,modeling]"
```

Use the `modeling` extra only when your contribution work needs the optional modeling stack.

## Local Quality Commands

Run the smallest useful command set for the change you are making.

```bash
make lint
make test
make docs
make check
make ci-local
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m e2e
```

## Quality Gates

- `ruff` handles linting and formatting checks.
- `mypy` provides static type verification for `src` and `tests`.
- `pytest` validates unit, integration, and local e2e tests.
- `mkdocs build --strict` validates documentation content and navigation.
- `pre-commit` enforces file hygiene, branch protection, secret scanning, and commit message rules.
- `pip-audit` checks installed dependencies for known vulnerabilities during CI-style runs.

Manual Colab, Drive, large-archive, and real How2Sign checks live under `tests/operational/` and
are excluded from normal pytest runs.

## Branch And Commit Discipline

- The default branch is `master`.
- Local hooks block direct commits to `master`.
- Conventional commit structure is enforced through `commitlint`; length limits are reported as
  warnings.
