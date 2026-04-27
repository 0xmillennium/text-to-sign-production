# text-to-sign-production

[![CI](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/ci.yml)
[![Docs](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml/badge.svg)](https://github.com/0xmillennium/text-to-sign-production/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-%3E%3D3.11-blue.svg)](pyproject.toml)

`text-to-sign-production` is a reproducible research-engineering repository for gloss-free,
How2Sign-compatible English text-to-sign pose production. It keeps dataset, artifact, experiment,
testing, execution, and research-planning records close to the code so claims remain traceable.

Canonical details live in the published GitHub Pages documentation site when deployed and in the
local `docs/` tree. Use this README as the stable repository landing page, then follow the docs that
match your task.

## Documentation

- [Docs Home](docs/index.md): top-level documentation overview and routing.
- [Research Context](docs/research/index.md): research-chain portal and current research frame.
- [Research Roadmap](docs/research/roadmap.md): Phase 0-12 research roadmap.
- [Source Corpus](docs/research/source-corpus.md): canonical source registry.
- [Contribution Audit Result](docs/research/contribution-audit/audit-result.md): audit synthesis
  for current research planning.
- [Execution](docs/execution.md): supported operational workflow guidance.
- [Data](docs/data/index.md): artifact and data documentation.
- [Experiments](docs/experiments/index.md): durable experiment and evidence records.
- [Testing](docs/testing/index.md): test layers, contracts, and validation guidance.
- [Decisions](docs/decisions/index.md): ADR history and documentation standards.
- [Repository Map](docs/repository-map.md): repository structure and documentation ownership.
- [Development Setup](docs/development-setup.md): local contributor setup and checks.

## Current Research Boundary

- No final implementation model is selected.
- No final candidate ranking is assigned.
- No experimental proof of sign-language intelligibility is claimed.
- No released public model artifact is claimed unless a documented release artifact exists.

## Scope

In scope:

- reproducible text-to-sign pose-production research pipeline
- artifact and data documentation
- evaluation and reporting surfaces
- experiment records
- thesis-oriented documentation

Out of scope near term:

- public gloss release
- near-term full avatar implementation
- treating automatic metrics or visual plausibility as proof of sign intelligibility

## Development

For local setup, quality commands, branch discipline, and contributor checks, use
[Development Setup](docs/development-setup.md). For workflow operation, use
[Execution](docs/execution.md) instead of duplicating run instructions here.

## License

This repository is licensed under the [MIT License](LICENSE).
