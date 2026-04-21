# ADR-0014: Adopt Sprint 3 Stage-Oriented Baseline Workflow And Notebook Extension

- Status: Superseded
- Date: 2026-04-18

Historical status note: this ADR captures the earlier workflow framing that presented public CLI
and reusable workflow entrypoints alongside the project notebook during Sprint 3. The repository
now uses a notebook-only supported workflow, so this ADR remains a historical record only.

## Context

Dataset Build remains the current public stage. Sprint 3 adds Baseline Modeling as a baseline-only
downstream surface on top of processed Dataset Build outputs. The repository already uses
stage-oriented public workflow language and thin notebooks, so Sprint 3 needs to expose baseline
operation without fragmenting the public surface or implying that Baseline Modeling is the main
thesis contribution.

The project also uses one main Colab notebook. Keeping a Dataset-Build-specific notebook name would
make the notebook surface misleading once it orchestrates both Dataset Build and baseline outputs.

## Decision

Adopt a stage-oriented Baseline Modeling workflow surface for Sprint 3:

- Public CLI: `python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]`
- Reusable workflow entrypoint:
  `text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`
- Single project-wide Colab notebook:
  `notebooks/colab/text_to_sign_production_colab.ipynb`

The notebook remains orchestration-only. It may call reusable workflow and archive helpers from
`src/`, but production logic stays out of notebook cells.

Public wording remains:

- Current public stage: Dataset Build
- Implemented internal downstream surface: Baseline Modeling
- Not yet implemented: broader evaluation, contribution modeling, playback/demo

## Consequences

- Sprint 3 has a stable operator-facing workflow without becoming the current public stage.
- The notebook name is stage-agnostic and can continue to serve future stages without another
  rename.
- Dataset Build and Baseline Modeling share one Colab surface while preserving stage boundaries.
- Lower-level scripts remain developer utilities rather than the main public workflow.
- Future docs must avoid presenting Sprint 3 as the thesis contribution or as a replacement for
  Dataset Build.

## Alternatives Considered

- Create a separate Sprint 3 notebook. Rejected because the project standard was one main Colab
  notebook, not one notebook per stage.
- Keep the Dataset-Build-specific notebook name. Rejected because the notebook now covered Dataset
  Build and baseline orchestration, so the old name would mislead operators and documentation.
- Expose only lower-level training scripts. Rejected because that would fragment the public
  workflow and make archive/resume behavior harder to document consistently.

## Superseded By

- [ADR-0016: Adopt Notebook-Only Supported Workflow And Single Execution Guide](adr-0016-adopt-notebook-only-supported-workflow-and-single-execution-guide.md)
