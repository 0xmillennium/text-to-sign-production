# Repository Structure

This page summarizes the current repository architecture. It is intended to help contributors find
the right layer without turning notebooks, scripts, or tests into homes for production logic.

## Top-Level Layout

```text
.
├── .github/workflows/       # CI, docs publishing, and release automation
├── configs/                 # Versioned configuration files
├── data/                    # Local raw/interim/processed/archive roots, ignored where generated
├── docs/                    # MkDocs source, ADRs, experiment records, and reference docs
├── notebooks/               # Runner-only notebooks
├── notes/                   # Project handoff and sprint briefing artifacts
├── scripts/                 # Thin CLI wrappers and developer utilities
├── src/text_to_sign_production/
│   ├── data/                # Dataset Build data pipeline logic
│   ├── modeling/            # Baseline Modeling surface
│   ├── ops/                 # Archive, Colab, progress, and publish helpers
│   └── workflows/           # Stage-oriented orchestration
├── tests/                   # Unit, integration, e2e, operational docs, fixtures, support
├── Makefile
├── mkdocs.yml
├── pyproject.toml
└── requirements-colab.txt
```

## Current Architecture

Current public stage: Dataset Build.

Dataset Build lives primarily in `src/text_to_sign_production/data/` and is orchestrated by
`src/text_to_sign_production/workflows/dataset_build.py`. It owns raw manifest creation,
normalization, filtering, processed manifest export, reports, and versioned `.npz` sample output.

Implemented internal downstream surface: Baseline Modeling.

Baseline Modeling lives in `src/text_to_sign_production/modeling/` and is orchestrated by
`src/text_to_sign_production/workflows/baseline_modeling.py`. It owns processed-data contracts,
baseline model surface, mask-aware losses and metrics, train/validation runtime, checkpointing,
qualitative panel export, evidence bundle writing, and archive-aware run packaging.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

The `ops` package owns reusable long-running operational primitives such as copy, extract, archive,
publish, and progress behavior. Notebooks and scripts call those primitives instead of
reimplementing core behavior.

## Public Interfaces

- Main Colab notebook: `notebooks/colab/text_to_sign_production_colab.ipynb`
- Dataset Build CLI: `python scripts/dataset_build.py`
- Dataset Build workflow:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`
- Baseline Modeling CLI:
  `python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]`
- Baseline Modeling workflow:
  `text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`

Developer utilities remain available for inspection or lower-level development, but they are not
the main public workflow surface.

## Data And Artifact Roots

- `data/raw/how2sign/`: canonical local raw input layout
- `data/interim/`: generated raw, normalized, filtered manifests, and interim reports
- `data/processed/v1/`: generated processed manifests, reports, and `.npz` samples
- `data/archives/`: local Dataset Build archived outputs
- `outputs/modeling/baseline-modeling/runs/<run_name>/`: local Baseline Modeling run roots

Generated data, checkpoints, qualitative outputs, runtime packages, and archives remain outside
Git.

## Documentation And Records

- `docs/decisions/`: ADRs for long-lived architecture and workflow decisions
- `docs/experiments/`: experiment record inventory, canonical template, and real experiment records
- `docs/execution.md`: operator-facing notebook execution guide
- `docs/data/`: artifact dictionary, report surfaces, and archive references
- `notes/LLM_PROJECT_HANDOFF.md`: project-wide handoff
- `notes/SPRINT3_BRIEFING.md`: Baseline Modeling handoff companion

## Testing Layout

- `tests/unit/`: isolated deterministic logic and contract tests
- `tests/integration/`: CI-safe workflow, CLI, notebook, and multi-module tests
- `tests/e2e/`: tiny local happy paths across stage boundaries
- `tests/operational/`: manual Colab, Drive, large archive, and real data checklists
- `tests/fixtures/`: small static fixtures and golden snippets
- `tests/support/`: reusable test-only builders, fake collaborators, and assertions

`tests/conftest.py` stays narrow and provides pytest wiring only.

## Structural Principles

- Business logic belongs in `src/`.
- Notebooks are orchestration-only documentation artifacts.
- Public workflow language is stage-oriented.
- Dataset Build remains the current public stage.
- Baseline Modeling remains a baseline-only internal downstream surface.
- Models read processed manifests and processed `.npz` samples, not raw files.
- Archive/resume behavior uses extracted outputs first, then archived outputs, then run/publish.
- Large generated artifacts remain outside GitHub.
