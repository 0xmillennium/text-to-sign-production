# ADR-0020: Published Model Artifacts as Downstream Interface

- Status: Accepted
- Date: 2026-04-25

## Context

Downstream stages need stable, reproducible model inputs for testing, visual inspection,
comparison, demos, and thesis packaging. Local checkpoints are fragile and environment-specific,
and Drive run roots or runtime archives are better understood as execution evidence than as durable
released interfaces.

The project therefore needs an artifact discipline that separates informal or local execution
outputs from released model artifacts intended for downstream use.

## Decision

Released model artifacts are part of the project's reproducibility contract.

`M0`, `M1`, `M2`, and `M3` should be published through Hugging Face or an equivalent documented
artifact surface when they are accepted for downstream use. Local checkpoints, Drive run roots,
runtime archives, and temporary outputs are execution evidence, not released model artifacts.

Downstream notebooks, tests, demos, and comparison surfaces should consume published artifacts
whenever practical. Each released model artifact should include enough metadata for reproducible
loading and use:

- model identity
- configuration reference
- dataset/version reference where applicable
- usage instructions
- release or artifact identifier

Informal local checkpoints must not become the default downstream interface.

## Consequences

- Sprint 4 cannot be considered closed merely because a baseline checkpoint exists.
- `M0` must be released or equivalently published before it becomes the trusted downstream
  reference.
- Later `M1`, `M2`, and `M3` releases should follow the same artifact discipline.
- Test and demo surfaces should prefer artifact references over local checkpoint paths.
