# text-to-sign-production Docs

## What These Docs Cover

These docs are the canonical documentation surface for the repository. They cover:

- execution and supported workflow behavior
- data and artifact reference surfaces
- testing and validation structure
- experiment and evidence records
- architectural decisions
- research planning, literature framing, and contribution-audit outcomes

Use the README for a higher-level repository introduction. Use these docs for the operational,
reference, and research-navigation surfaces that support the work. Generated artifacts and runtime
outputs are documented here, but they are not stored in these pages themselves.

## Current Repository State

- Dataset Build is completed.
- Baseline Modeling is completed as the implemented baseline milestone.
- The Contribution Audit is completed.
- The current selected pair is fixed as `C1 = Dynamic VQ Pose Tokens` and
  `C2 = Channel-Aware Loss Reweighting`.
- Broader selected-pair implementation, minimal visual demo work, and final thesis integration
  remain downstream work under the current roadmap.

## Start Here

- New to the repository: start with [Getting Started](getting-started.md).
- Running the supported workflow: use [Execution](execution.md).

## By Intent

- Running the supported workflow: [Execution](execution.md)
- Understanding data and artifacts: [Data Artifact Dictionary](data/index.md)
- Understanding validation and test structure: [Testing](testing/index.md)
- Reading empirical evidence: [Experiment Records](experiments/index.md)
- Understanding thesis planning and contribution selection:
  [Research Context](research/index.md) and
  [Contribution Audit](research/contribution-audit/index.md)
- Contributing locally:
  [Development Setup](development-setup.md),
  [Repository Map](repository-map.md), and
  [Decisions](decisions/index.md)

## Current Research Direction

The current thesis-facing direction recorded in the completed Contribution Audit is:

- `C1 = Dynamic VQ Pose Tokens`
- `C2 = Channel-Aware Loss Reweighting`
- fallback = `Articulator-Partitioned Latent Structure`
- deferred = `Motion Primitives Representation`

Audit detail, candidate evidence, scorecards, and selection records live under
[docs/research/](research/index.md).

## Documentation Map

- [Getting Started](getting-started.md): first-path routing for new readers.
- [Execution](execution.md): supported notebook-first workflow.
- [Data Artifact Dictionary](data/index.md): canonical reference for data and published artifacts.
- [Testing](testing/index.md): current test layers, validation surfaces, and contract boundaries.
- [Experiment Records](experiments/index.md): durable records for completed empirical work.
- [Research Context](research/index.md): roadmap, literature positioning, bibliography, and audit
  navigation.
- [Contribution Audit](research/contribution-audit/index.md): completed audit framework and final
  current outcome.
- [Development Setup](development-setup.md): local contributor setup and quality tooling.
- [Repository Map](repository-map.md): repository structure and documentation ownership.
- [Decisions](decisions/index.md): ADRs and long-lived architecture/process decisions.
