# Architecture Decision Records

This repository uses Architecture Decision Records (ADRs) to document project-level technical and
process decisions that affect implementation, collaboration, or reproducibility.

## How To Read These Records

ADRs are historical design records. They explain why a decision was accepted at the time it was
made, but they are not a complete replacement for current implementation guidance.

Use the execution, storage, testing, data, and experiment docs for active behavior. Current public
stage: Dataset Build. Implemented internal downstream surface: Baseline Modeling. Not yet
implemented: broader evaluation, contribution modeling, playback/demo.

## Why ADRs Matter Here

- A thesis project benefits from explicit historical context.
- Reproducible research requires explainable workflow choices.
- ADRs make tradeoffs visible before the repository grows more complex.

## Current ADR Set

- ADR-0001: repository-centered workflow
- ADR-0002: notebooks as thin drivers
- ADR-0003: DVC inclusion
- ADR-0004: English as the project language
- ADR-0005: Sprint 1 excludes ML implementation
- ADR-0006: Sprint 2 is data-first
- ADR-0007: preserve official How2Sign splits
- ADR-0008: manifest-driven processed dataset
- ADR-0009: `.npz` processed sample format
- ADR-0010: face is optional in v1
- ADR-0011: adopt Filter V2 one-hand policy for Dataset Build
- ADR-0012: remove DVC from the active workflow
- ADR-0013: define the post-Dataset-Build research roadmap
- ADR-0014: adopt Sprint 3 stage-oriented baseline workflow and notebook extension
- ADR-0015: define Sprint 3 runtime artifact, archive/resume, and evidence strategy

Use the template in this section for future records.
