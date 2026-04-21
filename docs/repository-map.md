# Repository Map

## Purpose

This page is a map of the repository. It helps readers find the right surface quickly.

It is not the execution runbook or the artifact reference. Detailed behavior lives in the
canonical docs pages for execution, data, testing, decisions, and experiments.

## Top-Level Layout

```text
.
├── .github/workflows/       # CI, docs publishing, and release automation
├── configs/                 # Versioned project configuration files
├── data/                    # Local raw, interim, and processed artifact roots
├── docs/                    # Canonical documentation surfaces and reference pages
├── notebooks/               # Notebook surfaces, including the supported Colab workflow
├── notes/                   # Internal handoff and project context material
├── scripts/                 # Thin developer-facing wrappers and utilities
├── src/                     # Project source code
├── tests/                   # Automated tests, fixtures, and operational checklists
├── Makefile                 # Common local automation entry points
├── mkdocs.yml               # Documentation navigation and MkDocs configuration
├── pyproject.toml           # Python project metadata and tool configuration
└── requirements-colab.txt   # Colab-focused dependency set
```

## Source Code Layout

- `src/text_to_sign_production/data/`: dataset ingestion, manifest, normalization, filtering, and reporting logic.
- `src/text_to_sign_production/modeling/`: baseline modeling data, training, inference, and model components.
- `src/text_to_sign_production/ops/`: reusable operational helpers for archives, Colab workflow support, and progress reporting.
- `src/text_to_sign_production/workflows/`: top-level workflow orchestration for dataset build and baseline modeling.

## Documentation Map

- [`docs/execution.md`](execution.md): operator-facing guide for the supported single Colab notebook workflow.
- [`docs/data/`](data/index.md): canonical artifact and data dictionary.
- [`docs/testing/`](testing/index.md): canonical testing surface and test-structure reference.
- [`docs/decisions/`](decisions/index.md): canonical ADR surface for long-lived decisions.
- [`docs/experiments/`](experiments/index.md): canonical experiment-record surface.

## Data And Artifact Layout

- `data/raw/`: local raw inputs restored or placed for processing.
- `data/interim/`: generated manifests and reports between raw intake and processed outputs.
- `data/processed/`: generated model-facing manifests, reports, and processed samples.
- Large generated artifacts, archives, checkpoints, and runtime outputs stay outside Git.

For artifact paths, ownership, and contents, use [`docs/data/`](data/index.md).

## Testing Layout

- `tests/unit/`: isolated logic and contract tests.
- `tests/integration/`: CI-safe multi-module and workflow-adjacent tests.
- `tests/e2e/`: small happy-path tests across larger boundaries.
- `tests/operational/`: manual operational and external-runtime checklists.
- `tests/fixtures/`: static sample inputs and expected outputs.
- `tests/support/`: reusable test helpers, builders, and assertions.

For testing semantics, layer definitions, and surface details, use [`docs/testing/`](testing/index.md).

## Notes And Handoff

`notes/` holds internal project context and handoff material. It is not part of the main public
documentation surface.

## Structural Principles

- Business logic belongs in `src/`.
- The single supported operator workflow is the Colab notebook.
- Each docs surface owns its own domain.
- Large generated artifacts stay outside Git.
- This page is a map, not a runbook.
