# ADR-0021: Four-Model 2x2 Comparison Architecture

- Status: Accepted
- Date: 2026-04-25

## Context

The thesis needs comparison evidence that can separate the base model behavior from the effects of
two selected contribution roles. A single final-model comparison would make it difficult to
interpret whether observed changes came from the base model, one contribution, the other
contribution, or their combination.

The repository also needs to keep the comparison architecture stable without coupling ADRs and the
roadmap to concrete candidate method names that belong in the research-planning and
contribution-selection surfaces.

## Decision

Thesis-facing model comparison is organized around the four-model structure:

- `M0 = Base`
- `M1 = Base + C1`
- `M2 = Base + C2`
- `M3 = Base + C1 + C2`

`C1` and `C2` are contribution roles selected by the research-planning/contribution-selection
process. This ADR does not hard-code concrete method names for `C1` or `C2`. Concrete method
choices belong in the research/contribution-audit surfaces.

Model implementation sprints should preserve clear boundaries between:

- base functionality
- C1 implementation delta
- C2 implementation delta
- combined C1+C2 integration

The 2x2 comparison is the basis for interpreting base performance, C1 effect, C2 effect, and
combined C1+C2 effect.

## Consequences

- C1/C2 implementation should not blur unrelated baseline changes into contribution effects.
- `M3` should combine C1 and C2 without introducing unrelated changes that invalidate
  interpretation.
- If C1 or C2 selection changes, research/contribution-audit docs must be updated and the impact
  on the four-model structure reviewed.
- Sprint 9 comparative analysis should compare all four released models under the same protocol.
