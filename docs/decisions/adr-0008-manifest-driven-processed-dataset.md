# ADR-0008: Manifest-Driven Processed Dataset

- Status: Accepted
- Date: 2026-04-06

## Context

Downstream models need a stable, explicit way to consume data without reimplementing raw How2Sign
directory traversal or join logic. Reading raw files directly would couple experiments to
undocumented paths and schema assumptions.

## Decision

Sprint 2 defines processed JSONL manifests as the canonical training-facing access layer. Future
models must read the processed manifests and the `.npz` sample paths they declare instead of
walking the raw dataset layout directly.

## Consequences

- Dataset access becomes explicit, versioned, and easier to validate.
- Raw ingestion logic remains centralized in the data pipeline.
- Future schema changes can be introduced through manifest and schema-version updates.
