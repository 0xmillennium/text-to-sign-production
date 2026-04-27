# ADR-0022: Standard Testing Surfaces Before Contribution Implementation

- Status: Superseded
- Date: 2026-04-25

Status note: this ADR is retained as historical context only. ADR-0023 is the current governing
decision for phase-based research governance and public documentation boundaries. This ADR must not
be used to reintroduce C1/C2 selected-pair framing, sprint-era roadmap sequencing as current
planning, M0/M1/M2/M3 final model-matrix framing, or released/loadable M0 claims without documented
release evidence.

## Context

Later contribution models need a stable base reference and comparable evaluation procedure.
Without a released `M0`, later contribution models do not have a trusted reference. Without
reusable testing, models risk being evaluated through model-specific or non-comparable procedures.

The project also needs to keep evaluation-support visual inspection distinct from final visual
interface work so demo work does not displace evaluation discipline.

## Decision

Contribution implementation should not proceed to full C1/C2 model work before:

- `M0` is trained, evaluated, accepted, and released or equivalently published
- reusable testing surfaces are defined

Sprint 5 owns reusable model testing surfaces:

- terminal-based standard model testing interface
- standard model testing notebook
- visual testing / visual inspection notebook

The standard model testing surface should produce comparable reports and capture
model/dataset/config/split/run metadata. The visual testing / visual inspection notebook is an
evaluation-support surface, not the final user-facing interface.

Final visual interface work belongs later and should not replace the standard testing surfaces.
Every released model should be testable through the same evaluation path.

## Consequences

- Active work should prioritize Sprint 4 closure before C1/C2 implementation.
- Sprint 5 should follow Sprint 4 to establish reusable testing and visual inspection surfaces.
- C1/C2 model work should rely on those surfaces rather than inventing model-specific test
  shortcuts.
- Final visual interface work remains downstream of released artifacts and reusable
  inference/visualization code.
