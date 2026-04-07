# Development Setup

## Install Tooling

```bash
make install
```

This installs the package in editable mode, both optional extra sets (`dev` and `docs`), and
registers `pre-commit` plus `commit-msg` hooks.

If you only need one side of the toolchain, use:

```bash
make install-dev
make install-docs
```

- `make install-dev` installs code-quality and test tooling only.
- `make install-docs` installs the MkDocs toolchain only.
- `make install` remains the default because the local CI-style workflow also builds docs.

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
- Conventional commit structure is enforced through `commitlint`; length limits are reported as
  warnings.
