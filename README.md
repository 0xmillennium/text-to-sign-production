# text-to-sign-production

`text-to-sign-production` is a reproducible research repository for a graduation thesis on
text-to-sign production. The long-term system direction is:

`English text -> pose tokens -> keypoints -> skeleton/avatar`

Sprint 1 establishes the infrastructure needed to build that system responsibly. This sprint is
intentionally focused on repository quality, documentation, and reproducibility rather than on
machine learning implementation.

## Sprint 1 Includes

- A minimal importable Python package under `src/`
- Deterministic smoke-test functionality and pytest coverage
- `ruff`, `mypy`, `pytest`, `pre-commit`, `commitlint`, and `gitleaks` quality gates
- MkDocs documentation with the Material theme
- ADR and experiment log templates
- A minimal DVC bootstrap for future data and model pipeline work
- A thin Colab notebook that installs the repository and runs a smoke check

## Sprint 1 Explicitly Does Not Include

- A tokenizer or retrieval system
- Pose-token generation
- Keypoint prediction or normalization logic
- Skeleton rendering or avatar animation
- Dataset processing stages beyond documented future plans
- Model-weight publishing or finalized weight licensing

Sprint 1 avoids placeholder ML code that could misrepresent the current maturity of the project.

## Repository Structure

```text
.
├── configs/                  # Future runtime and experiment configuration files
├── docs/                     # MkDocs source, ADRs, and experiment records
├── notebooks/                # Thin runner notebooks only
├── scripts/                  # Utility scripts that support the repo workflow
├── src/text_to_sign_production/
│   └── ...                   # Minimal package scaffold for Sprint 1
├── tests/                    # Fast deterministic tests
├── Makefile                  # Common local developer commands
├── mkdocs.yml                # MkDocs configuration
├── pyproject.toml            # Primary Python project configuration
└── requirements-colab.txt    # Minimal Colab install requirements
```

## Local Setup

Sprint 1 targets Python 3.11 for development.

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

You can also use the Makefile:

```bash
make install
```

## Common Commands

```bash
make lint
make test
make docs
make check
make ci-local
```

- `make lint` runs `ruff` checks, `ruff format --check`, and `mypy`.
- `make test` runs the fast pytest suite with coverage output.
- `make docs` builds the MkDocs site in strict mode.
- `make check` runs linting, tests, and docs checks together.
- `make ci-local` runs the main local quality checks together, including `pre-commit` and
  `pip-audit`.

## Colab Usage Philosophy

Notebooks are runner-only interfaces. They may install the repository, import code from `src/`,
and execute smoke checks or future orchestrated workflows, but they must not become the home of
core project logic. All important logic belongs in the package and supporting modules.

Sprint 1 therefore includes exactly one thin notebook: `notebooks/colab_smoke_test.ipynb`.

## DVC Role

DVC is initialized in a minimal skeleton form so the thesis repository is ready for future
data- and model-oriented stages without pretending those stages already exist. The planned future
pipeline stages are documented as:

- `prepare_raw`
- `normalize_keypoints`
- `filter_samples`
- `build_splits`
- `export_training_manifest`

No real DVC pipeline is implemented in Sprint 1.

## Quality And Reproducibility Foundation

- Local quality checks are available through `pre-commit`, `ruff`, `mypy`, `pytest`, and
  strict MkDocs builds.
- The repository is structured so GitHub-based automation can be added cleanly without changing
  the package, docs, notebook, or DVC foundations established in Sprint 1.
- Release and deployment automation are intentionally treated as repository-level concerns that
  can be layered on top of this baseline.

## How Future Sprints Build On This Foundation

Future sprints can add configuration schemas, dataset handling, DVC stages, experiments,
evaluation, and the actual text-to-sign modeling pipeline on top of this baseline without needing
to revisit the repository fundamentals. Sprint 1 makes that later work easier to review,
reproduce, and publish.
