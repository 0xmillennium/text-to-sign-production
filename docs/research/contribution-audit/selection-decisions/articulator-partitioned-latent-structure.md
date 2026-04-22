# Articulator-Partitioned Latent Structure

## Candidate Under Decision

`STRUCTURE-B2 — Articulator-Partitioned Latent Structure`

## Decision Context

Current selection-decision record after completed scorecard review and current veto synthesis.

## Weighted Score Status

`76.25 / 100`

## Confidence Status

`medium`

## Hard Veto Checks

### Independent Additivity Veto

- `Status`: `not yet determined`
- `Rationale`: The current audit record supports an independent `Base + C2` path conceptually, but leaves that path conditional on defining a sufficiently clean independent implementation. Architectural entanglement risk remains higher than in the selected simpler structure-aware alternative.

### Problem Drift Veto

- `Status`: `pass`
- `Rationale`: The candidate remains within the core text-conditioned sign-pose generation problem and strengthens internal articulator structure rather than changing the project’s problem definition.

### Data / Workflow Compatibility Veto

- `Status`: `not yet determined`
- `Rationale`: The current audit record supports conceptual compatibility with the current pose surface, but architecture, training, and artifact-packaging complexity remain materially higher than in the selected `C2` alternative.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: The ablation plan is meaningful and usable: `Base vs Base+ArticulatorStructure`, simpler vs stronger partitioning variants, and later combined-model comparison if needed.

### Evidence Sufficiency Veto

- `Status`: `pass`
- `Rationale`: The candidate has a strong direct source and two strong near-direct supporting sources. The evidence surface is scientifically strong even though it does not remove implementation-risk concerns.

### Pairwise Compatibility Veto

- `Status`: `not yet determined`
- `Rationale`: Joint-model compatibility with a discrete/data-driven `C1` candidate remains plausible, but the current record still treats that compatibility as conditional rather than fully closed.

## Veto Summary

No veto gate currently fails, but the candidate is not provisionally veto-clean because independent additivity, workflow compatibility, and pairwise compatibility remain open.

## Contribution-Eligibility Status

Eligibility requires all of the following:

- weighted score at least `75/100`
- evidence confidence of `medium` or `high`
- all veto gates passed

`Not currently established`

## Pairwise Compatibility Note

Current pairwise compatibility with a selected discrete/data-driven `C1` candidate remains plausible but not yet fully closed in the audit record.

## Final Status

`kept_as_fallback`

## Decision Rationale

This candidate is retained as the principal fallback within the structure-aware family because it is
scientifically strong in the local literature, above the score threshold, and better supported than
a simple deferral would suggest. It is not selected as `C2` because the current audit record still
leaves more implementation uncertainty and higher complexity burden than the selected
`Channel-Aware Loss Reweighting` path for this repository.

## Record Boundary Note

- This document records decision-level reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and
  [Scorecards](../scorecards/index.md).
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome
  surface.
