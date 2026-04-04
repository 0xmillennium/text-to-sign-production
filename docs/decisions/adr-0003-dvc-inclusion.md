# ADR-0003: DVC Inclusion

- Status: Accepted
- Date: 2026-04-04

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
