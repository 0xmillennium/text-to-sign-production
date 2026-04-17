# Architecture Decision Records

This repository uses Architecture Decision Records (ADRs) to document project-level technical and
process decisions that affect implementation, collaboration, or reproducibility.

## How To Read These Records

ADRs are historical design records. They explain why a decision was accepted at the time it was
made, but they are not a complete replacement for current implementation guidance.

Use the data, execution, testing, and storage docs for active Dataset Build behavior. Later
implementation hardening may narrow or extend earlier ADR assumptions without rewriting the
historical record. For example, the current Dataset Build uses `configs/data/filter-v2.yaml`,
retains samples with usable `body` plus at least one usable hand, keeps the processed schema at
`t2sp-processed-v1`, and packages split sample archives from processed manifests rather than from
whole sample directories.

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

Use the template in this section for future records.
