# ADR-0023: Phase-Based Research Governance And Public Documentation Boundaries

- Status: Accepted
- Date: 2026-04-27

## Context

The repository has moved from sprint-era planning and selected-pair contribution framing to a
Phase 0-12 research frame. Public entry surfaces and accepted ADRs still carried older wording that
could be read as current governance: fixed C1/C2 selected-pair planning, M0/M1/M2/M3 final
model-matrix sequencing, and implementation-status snapshots in README or documentation index
pages.

The project needs a single governing decision that preserves ADR history while making the current
planning authority and public-documentation boundaries explicit.

## Decision

Current planning authority is the Phase 0-12 research roadmap and the supporting research chain
under `docs/research/`.

The current research authority chain is:

- `docs/research/source-selection-criteria.md`
- `docs/research/source-corpus.md`
- `docs/research/contribution-audit/`
- `docs/research/contribution-audit/audit-result.md`
- `docs/research/literature-positioning.md`
- `docs/research/roadmap.md`

README is a stable repository landing page, not an implementation snapshot. `docs/index.md` is a
top-level documentation navigation and overview page, not an implementation snapshot. Detailed
operational, data, experiment, testing, and research status belongs in the appropriate canonical
documentation surfaces rather than being duplicated in README or `docs/index.md`.

Old sprint-era, C1/C2 selected-pair, and M0/M1/M2/M3 final-model-matrix planning is superseded as
current governance. ADR history remains preserved, but superseded ADRs must not be used as current
planning authority.

Phase 6+ model-candidate work remains planned until Phase 5 can evaluate M0 outputs reproducibly.
Public documentation must preserve the current boundaries: no final implementation model is
selected, no final candidate ranking is assigned, no experimental proof of sign-language
intelligibility is claimed, and no released public model artifact is claimed unless a documented
release artifact exists.

## Consequences

- Public entry surfaces should route readers to canonical docs instead of repeating volatile status.
- Superseded ADRs remain available as historical records, but ADR-0023 governs current planning and
  public-documentation boundaries.
- Research, experiment, data, testing, and execution details should be maintained in their existing
  index/template/leaf documentation architecture.
- Future Phase 6+ model work must not be presented as selected, ranked, implemented, or released
  until the required Phase 5 evaluation and release evidence exists.
