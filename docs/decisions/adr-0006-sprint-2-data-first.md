# ADR-0006: Sprint 2 Is Data-First

- Status: Accepted
- Date: 2026-04-06

Historical status note: this ADR records the Sprint 2 data-first decision at the time DVC
reproduction was still part of the repository workflow. ADR-0012 later removes DVC from the active
workflow. Use the current data and execution documentation for active Dataset Build behavior.

## Context

The repository already had Sprint 1 infrastructure in place, but it did not yet expose a
reproducible, versioned dataset interface for downstream research. Moving directly into modeling
would have forced future experiments to depend on ad hoc raw-file access and undocumented data
assumptions.

## Decision

Sprint 2 focuses on the data pipeline rather than model implementation. The sprint adds raw
inspection, raw manifests, validation, normalized `.npz` exports, processed JSONL manifests,
reporting, and DVC reproduction without adding training or inference code.

## Consequences

- Future experiments start from a documented and auditable dataset contract.
- Modeling remains explicitly deferred instead of being represented by placeholder code.
- The repository can validate data assumptions before training begins.
