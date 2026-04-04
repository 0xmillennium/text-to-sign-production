# ADR-0005: Sprint 1 Excludes ML Implementation

- Status: Accepted
- Date: 2026-04-04

## Context

The thesis roadmap includes substantial machine learning and animation work, but Sprint 1 is
explicitly limited to infrastructure. Adding speculative ML abstractions now would create false
signals about project readiness.

## Decision

Sprint 1 will not implement the tokenizer, retrieval system, pose generation model, keypoint
pipeline, or avatar animation logic.

## Consequences

- The repository stays honest about what exists today.
- Future sprints can design ML components with clearer requirements and less rework.
- Current validation remains fast, deterministic, and infrastructure-focused.
