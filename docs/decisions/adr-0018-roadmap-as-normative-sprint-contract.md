# ADR-0018: Roadmap as Normative Sprint Contract

- Status: Superseded
- Date: 2026-04-25

Status note: this ADR is retained as historical context only. ADR-0023 is the current governing
decision for phase-based research governance and public documentation boundaries. This ADR must not
be used to reintroduce C1/C2 selected-pair framing, sprint-era roadmap sequencing as current
planning, M0/M1/M2/M3 final model-matrix framing, or released/loadable M0 claims without documented
release evidence.

## Context

The project's historical sprint plan changed several times. Treating old sprint names, stale
planning history, or embedded implementation-status notes as authoritative created confusion about
what the project should build next and how completion should be assessed.

The repository needs a stable planning contract for sprint responsibilities, scope boundaries,
inputs, outputs, acceptance criteria, and risks, while still allowing implementation status to be
audited separately against that contract.

## Decision

`docs/research/roadmap.md` is the normative ideal project roadmap.

The roadmap defines the ideal sprint sequence, sprint responsibilities, inputs, outputs, scope
boundaries, acceptance criteria, and risks. It is not a historical log of old sprint numbering.
Old sprint numbering and stale planning history are not authoritative.

Sprint implementation status must be determined by separate repository-status audits, not by the
roadmap itself. The roadmap must not contain `Status` or `Current Implementation Assessment`
sections unless a future ADR explicitly changes this policy.

Handoff and status documents may summarize current state, but they do not replace the roadmap. If
the roadmap changes, dependent surfaces such as handoff and status summaries should be reviewed.

## Consequences

- Future roadmap edits must preserve the distinction between ideal plan and implementation status.
- Status labels such as `completed`, `mostly completed`, `partially completed`, and `planned`
  belong in status audits or handoff/status summaries unless formally adopted into the roadmap by a
  later decision.
- New implementation work should be planned against the normative sprint contract.
