# ADR-0002: Notebooks As Thin Drivers

- Status: Accepted
- Date: 2026-04-04

## Context

Notebook-heavy research repositories often accumulate hidden logic, which makes testing,
review, and reuse harder over time.

## Decision

Notebooks in this project will remain thin drivers that install the repository, import code from
`src/`, and execute orchestrated steps without owning critical project logic.

## Consequences

- Reusable logic stays testable inside the Python package.
- Colab and local notebooks remain lightweight entry points rather than implementation silos.
- Sprint 1 notebooks can stay minimal without implying missing ML components already exist.
