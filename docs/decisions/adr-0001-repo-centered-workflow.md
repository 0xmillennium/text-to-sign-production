# ADR-0001: Repository-Centered Workflow

- Status: Accepted
- Date: 2026-04-04

## Context

This thesis project needs reproducibility, reviewability, and a clear history of engineering
choices. Allowing important artifacts to remain outside the repository would weaken those goals.

## Decision

The repository will be the primary source of truth for code, documentation, ADRs, experiment
records, CI/CD configuration, and reproducibility instructions.

## Consequences

- Contributors have one canonical place to inspect project state.
- Research-supporting artifacts remain versioned with the code.
- Informal side channels should not become the sole home of critical knowledge.
