# ADR-0015: Define Sprint 3 Runtime Artifact, Archive/Resume, And Evidence Strategy

- Status: Accepted
- Date: 2026-04-18

## Context

Sprint 3 Baseline Modeling produces more than a checkpoint. A reproducible baseline run needs
configuration provenance, train/validation summaries, checkpoints, qualitative inspection outputs,
runtime evidence, and a transfer/resume surface that works locally and in Colab.

The project keeps large generated artifacts outside GitHub. Therefore, Sprint 3 needs stable run
roots and deterministic archive names so later phases can restore or compare baseline evidence
without rerunning expensive steps every time.

## Decision

Define each Sprint 3 baseline run root as:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

Define deterministic archive names:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

Define resume priority for training outputs, qualitative panel outputs, and record/package outputs:

1. Reuse already-extracted outputs in the expected run root.
2. Otherwise extract archived outputs from `archives/`.
3. Otherwise run the step and publish both extracted outputs and archived outputs.

Define the runtime record package as baseline evidence and an input to the formal Markdown
experiment record. It is not a database, tracking server, or replacement for experiment-record
authoring.

## Consequences

- Local and Colab runs share one artifact vocabulary.
- Operators can recover from interrupted sessions by reusing extracted outputs or extracting
  archived outputs.
- Later phases can cite formal experiment records while using runtime artifacts for reproduction.
- Storage docs, execution docs, notebook cells, and operational checklists must use the same
  artifact names and resume vocabulary.

## Alternatives Considered

### Store Only Checkpoints

Rejected. Checkpoints alone do not preserve run summaries, qualitative evidence, or package
provenance.

### Introduce A Tracking Server

Rejected. Sprint 3 needs a narrow, repository-native record surface, not new infrastructure.

### Use Non-Deterministic Archive Names

Rejected. Stable names are necessary for simple Colab resume cells and later sprint comparisons.
