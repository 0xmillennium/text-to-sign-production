# ADR-0003: DVC Inclusion

- Status: Superseded
- Date: 2026-04-04

Historical status note: this ADR records the Sprint 1 decision to include DVC as a reproducibility
foundation before real data workflow stages existed. Dataset Build has since been implemented, and
ADR-0012 removes DVC from the active workflow. Use the current data and execution documentation for
active Dataset Build behavior.

## Context

The long-term system will eventually depend on datasets, derived artifacts, and model-adjacent
assets that require provenance and reproducible orchestration.

## Decision

DVC is included in Sprint 1 as a minimal bootstrap so future data and model workflow stages can
grow from an established reproducibility foundation.

## Consequences

- The repository becomes ready for future staged pipelines without faking present capability.
- Contributors can document future DVC stage boundaries early.
- Sprint 1 still avoids implementing any real data pipeline or model workflow.

## Superseded By

- [ADR-0012: Remove DVC from the Active Workflow](adr-0012-remove-dvc-active-workflow.md)
