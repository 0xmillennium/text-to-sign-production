# ADR-0016: Adopt Notebook-Only Supported Workflow And Single Execution Guide

- Status: Accepted
- Date: 2026-04-21

## Context

The repository now has one supported operator-facing workflow: the single project-wide Colab
notebook. Earlier workflow language still presented public CLI/script and reusable workflow
entrypoints as peer public surfaces, which split the operator story across multiple docs and made
the canonical workflow harder to identify.

The repository still contains lower-level package code, scripts, and workflow helpers, but those
surfaces exist to implement the supported workflow and support development. They are not the main
public operator interface.

## Decision

Adopt the single Colab notebook as the supported operator workflow for this repository.
`docs/execution.md` is the single execution guide for that workflow.

Public CLI/script workflow is no longer a supported public operator surface. Lower-level code,
scripts, and workflow helpers remain in the repository as implementation detail and development
support, not as the main public workflow.

This ADR supersedes ADR-0014 as the governing workflow decision.

## Consequences

- Operators have one canonical workflow and one canonical execution guide.
- Workflow docs should describe notebook execution through `docs/execution.md` instead of
  presenting CLI/script entrypoints as peer public surfaces.
- Lower-level code can continue to evolve as implementation detail without redefining the public
  operator interface.
- Historical ADRs that describe older workflow framing remain useful context but no longer govern
  current workflow guidance.
