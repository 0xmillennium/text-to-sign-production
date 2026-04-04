# Development Setup

## Install Tooling

```bash
make install
```

This installs the package in editable mode, developer tooling, documentation dependencies, and
registers `pre-commit` plus `commit-msg` hooks.

## Local Quality Commands

```bash
make lint
make test
make docs
make check
make ci-local
```

## Quality Gates

- `ruff` handles linting and formatting checks.
- `mypy` provides lightweight static type verification for the scaffold.
- `pytest` validates package importability and smoke behavior.
- `pre-commit` enforces file hygiene, branch protection, secret scanning, and commit message rules.
- `pip-audit` checks installed dependencies for known vulnerabilities during CI-style runs.

## Branch And Commit Discipline

- The default branch is `master`.
- Local hooks block direct commits to `master`.
- Conventional commits are enforced through `commitlint`.
