# text-to-sign-production

[![CI](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml)
[![Docs](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](pyproject.toml)

## Concise Project Summary

`text-to-sign-production` is a reproducible thesis repository for English text-to-sign-pose
production. The repository currently covers Dataset Build, baseline modeling, experiment/evidence
recording, and research planning for post-baseline contribution work.

The long-term modeling direction is still:

`English text -> pose tokens -> keypoints -> skeleton/avatar`

That direction is thesis-facing roadmap context, not a claim that every downstream stage is already
implemented in the current repository.

## Current Status

- Dataset Build is completed.
- Baseline Modeling is completed as the implemented baseline milestone.
- The Contribution Audit is completed.
- The current thesis-facing direction is fixed as `C1 = Dynamic VQ Pose Tokens` and
  `C2 = Channel-Aware Loss Reweighting`.
- Downstream selected-pair implementation, broader post-baseline evaluation hardening, minimal
  visual demo work, and final thesis packaging remain future work under the current roadmap.

## Implemented Surfaces

The repository supports the following implemented surfaces today:

- Dataset Build: manifest-driven processed dataset creation from fixed raw translation files and
  OpenPose keypoint inputs.
- Baseline Modeling: reproducible English text-to-continuous-pose baseline training, qualitative
  export, and runtime evidence packaging on top of processed Dataset Build outputs.
- Experiment and evidence records: durable Markdown records for completed dataset and baseline work.
- Research planning and audit surfaces: roadmap, literature framing, and the completed
  contribution-audit record that fixes the current selected pair.

This does not mean Sprint 5 through Sprint 8 are already implemented. The selected-pair modeling
program remains downstream work.

## Selected Current Research Direction

The completed Contribution Audit fixes the current thesis-facing direction as:

- `C1 = Dynamic VQ Pose Tokens`
- `C2 = Channel-Aware Loss Reweighting`
- fallback = `Articulator-Partitioned Latent Structure`
- deferred = `Motion Primitives Representation`

Detailed audit reasoning, candidate evidence, scorecards, and final audit outcomes live under
[docs/research/](docs/research/index.md). This README intentionally stays at the repository-entry
level.

## Supported Workflow

The single supported operational workflow is the project-wide Colab notebook:

`notebooks/colab/text_to_sign_production_colab.ipynb`

That notebook is documented in [Execution](docs/execution.md). It is the supported operational path
for the currently implemented workflow surfaces:

- Dataset Build
- Baseline Modeling

Lower-level scripts and helpers exist in the repository, but they are not the primary documented
workflow surface.

## Artifact / Publication / Reproducibility

- Large generated artifacts stay outside GitHub.
- Dataset Build, Baseline Modeling, and experiment/runtime evidence are described through the
  documented artifact and record surfaces under [docs/data/](docs/data/index.md) and
  [docs/experiments/](docs/experiments/index.md).
- The repository remains reproducibility-oriented: manifests, configs, commands, archives, and
  experiment records are expected to stay traceable.
- The post-audit roadmap adopts a publication-oriented artifact discipline: downstream consumers
  should use published artifacts whenever practical rather than informal local checkpoints.

## Scope Boundaries

- Minimal visual demo work is planned, not implemented.
- Selected-pair contribution implementation is downstream roadmap work, not a completed repository
  surface.
- This README is not the execution runbook, not the audit ledger, and not the full roadmap.
- Detailed workflow behavior belongs in [docs/execution.md](docs/execution.md), and detailed
  thesis-path reasoning belongs in [docs/research/](docs/research/index.md).

## Documentation Map

- [Docs Home](docs/index.md): canonical documentation entrypoint.
- [Execution](docs/execution.md): supported notebook-first workflow.
- [Data Artifact Dictionary](docs/data/index.md): artifact paths, roles, producers, consumers, and
  contents.
- [Testing](docs/testing/index.md): test structure, validation layers, and coverage boundaries.
- [Experiment Records](docs/experiments/index.md): durable empirical evidence records.
- [Research Context](docs/research/index.md): roadmap, literature positioning, bibliography, and
  contribution-audit surfaces.
- [Development Setup](docs/development-setup.md): contributor environment setup and local checks.

## Getting Started

- If you want to run the supported workflow, start with [Docs Home](docs/index.md) and then read
  [Execution](docs/execution.md).
- If you want to contribute locally, start with
  [Development Setup](docs/development-setup.md) and [Repository Map](docs/repository-map.md).
