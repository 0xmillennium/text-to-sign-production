# ADR-0019: Split Notebook Architecture as Target Workflow

- Status: Accepted
- Date: 2026-04-25

## Context

The current single project-wide Colab notebook remains the present supported execution surface, but
the roadmap has expanded beyond one notebook's natural responsibility boundary. A single broad
notebook makes it harder to keep dataset construction, model training, testing, visual inspection,
and release workflows distinct.

The repository already follows a notebook thin-driver principle: notebooks should orchestrate
shared project code rather than own core implementation logic.

## Decision

The current single project-wide Colab notebook may remain the present supported execution surface
during transition. The target workflow is a split notebook architecture.

Notebook surfaces should be separated by responsibility:

- dataset build/export notebook
- base model training/release notebook
- standard model testing notebook
- visual testing/visual inspection notebook
- C1 training/release notebook
- C2 training/release notebook
- final combined model training/release notebook
- final visual interface/demo surface when notebook-backed demonstration is appropriate

Notebooks are thin drivers. Shared dataset, model, training, evaluation, artifact-loading, and
visualization logic belongs in reusable project code, primarily under `src/`. Notebook cells should
orchestrate shared code rather than own core logic.

This decision extends the notebook thin-driver principle and does not invalidate the current
supported Colab workflow during transition.

## Consequences

- Existing notebook workflow may remain valid until split surfaces are created.
- Sprint closure for notebook-owning sprints requires the relevant dedicated notebook surface.
- Future notebooks should not duplicate core implementation logic.
- Documentation should distinguish current single-notebook reality from target split-notebook
  architecture.
