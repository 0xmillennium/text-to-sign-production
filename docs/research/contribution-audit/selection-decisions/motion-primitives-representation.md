# Motion Primitives Representation

## Candidate Under Decision

`DISCRETE-A2 — Motion Primitives Representation`

## Decision Context

Current selection-decision record after completed scorecard review and current veto synthesis.

## Weighted Score Status

`61.25 / 100`

## Confidence Status

`medium`

## Hard Veto Checks

### Independent Additivity Veto

- `Status`: `not yet determined`
- `Rationale`: The current audit record supports an independent `Base + candidate` path only conditionally, because a clean gloss-free adaptation must first be defined.

### Problem Drift Veto

- `Status`: `pass`
- `Rationale`: The candidate remains within the core text-conditioned sign-pose generation problem and proposes a learned intermediate representation rather than a problem-definition shift.

### Data / Workflow Compatibility Veto

- `Status`: `not yet determined`
- `Rationale`: The candidate is conceptually compatible with the current pose surface, but the current record still treats repository/workflow fit as adaptation-heavy relative to Dynamic VQ.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: The ablation plan is meaningful and usable: baseline comparison, internal primitive variants, family-level comparison against Dynamic VQ, and later combined-model comparison if needed.

### Evidence Sufficiency Veto

- `Status`: `pass`
- `Rationale`: The candidate has a visible evidence surface with a strong direct source, a strong supporting source, and explicit limitation notes. The evidence is mixed but not absent.

### Pairwise Compatibility Veto

- `Status`: `not yet determined`
- `Rationale`: Current pairwise compatibility with a selected structure-aware candidate has not been closed in the audit record.

## Veto Summary

No current veto gate clearly fails, but the candidate is not veto-clean because additivity, workflow fit, and pairwise compatibility remain open.

## Contribution-Eligibility Status

Eligibility requires all of the following:

- weighted score at least `75/100`
- evidence confidence of `medium` or `high`
- all veto gates passed

`Not currently established`

## Pairwise Compatibility Note

Current pairwise compatibility with a selected structure-aware `C2` candidate remains open and is not needed for current non-selection.

## Final Status

`deferred`

## Decision Rationale

This candidate is deferred rather than selected because its current score is below the eligibility threshold and because the current record still leaves its gloss-free additivity and workflow fit conditional. It remains scientifically plausible, but it is not currently competitive with `Dynamic VQ Pose Tokens` as the leading discrete/data-driven path.

## Record Boundary Note

- This document records decision-level reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and
  [Scorecards](../scorecards/index.md).
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome
  surface.
