# ADR-0017: Standardize Structured Documentation Surfaces With Index And Template

- Status: Accepted
- Date: 2026-04-21

## Context

The repository already uses structured documentation surfaces where overview/navigation,
canonical-standard, and leaf-record roles are distinct. `docs/data` and `docs/experiments` already
follow this pattern, but `docs/decisions` still exposed a separate top-level overview file and a
lighter ADR template, which left parallel decision sources in place.

The repository needs one documentation architecture for these structured surfaces so readers can
predict where navigation lives, where standards live, and where real records live.

## Decision

Structured documentation surfaces in this repository use:

- `index.md` for navigation and overview
- `template.md` for the canonical standard
- leaf files for real records or leaf docs

This pattern now governs at least `docs/data`, `docs/experiments`, and `docs/decisions`.
`docs/data` acts as the canonical artifact dictionary.

This is a documentation architecture decision for the repository, not only a local refactor of the
decisions directory.

## Consequences

- Structured documentation surfaces should not create parallel overview or standard files outside
  their own directory pattern.
- Readers can expect the same roles for `index.md`, `template.md`, and leaf pages across the
  governed documentation areas.
- `docs/decisions` must keep only its overview/index, canonical template, and real ADR files.
- Future structured documentation additions should follow the same pattern unless a later ADR
  changes the documentation architecture.
