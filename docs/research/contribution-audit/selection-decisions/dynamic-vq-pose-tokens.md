# Dynamic VQ Pose Tokens

## Candidate Under Decision

`DISCRETE-A1 — Dynamic VQ Pose Tokens`

## Decision Context

Current selection-decision record after completed scorecard review and current veto synthesis.

## Weighted Score Status

`87.50 / 100`

## Confidence Status

`medium`

## Hard Veto Checks

### Independent Additivity Veto

- `Status`: `pass`
- `Rationale`: The current audit record supports an independent `Base + candidate` path, does not require `C2`, and supports the existence of an independent variant.

### Problem Drift Veto

- `Status`: `pass`
- `Rationale`: The candidate stays within the core text-conditioned sign-pose generation problem and changes the representation strategy rather than the project’s problem definition.

### Data / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: The candidate is compatible with the current processed pose surface and current workflow, and heavy new preprocessing is not currently indicated. Remaining packaging and artifact-extension work does not currently rise to veto level.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: The ablation plan is clear and directly usable: `Base vs Base+DynamicVQ`, internal comparison against a fixed-VQ variant, and later combined-model comparison if needed.

### Evidence Sufficiency Veto

- `Status`: `pass`
- `Rationale`: The candidate has a visible and meaningful evidence surface: one strong direct source, one strong family-level supporting source, and one explicit limitation source. The current record supports `medium` confidence rather than evidentiary failure.

### Pairwise Compatibility Veto

- `Status`: `pass`
- `Rationale`: The current audit record supports a representation-level `C1` role for this candidate and does not indicate factorial-design break. Combined use with the selected structure-aware `C2` candidate is sufficiently supported for current selection purposes.

## Veto Summary

All current veto gates pass.

## Contribution-Eligibility Status

Eligibility requires all of the following:

- weighted score at least `75/100`
- evidence confidence of `medium` or `high`
- all veto gates passed

`Eligible`

## Pairwise Compatibility Note

Current pairwise review supports compatibility with `Channel-Aware Loss Reweighting` as the selected `C2` candidate. The current record supports this pairing as a representation-level contribution combined with a loss-level structure-aware contribution, without currently indicated factorial-design break.

## Final Status

`selected_for_C1`

## Decision Rationale

This candidate is selected as `C1` because the current audit interprets it as the strongest current
discrete/data-driven candidate on the combined evidence surface: it has the highest score in its
family, a strong additivity profile in the audit record, a meaningful but manageable limitation
boundary, and a sufficiently supported path to joint use with the selected `C2` candidate.
Relative to Motion Primitives Representation, the current audit treats it as the cleaner current
gloss-free fit with lower adaptation burden for this repository, while keeping visible that those
comparative fit judgments are audit-level conclusions rather than direct paper findings.

## Record Boundary Note

- This document records decision-level reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and
  [Scorecards](../scorecards/index.md).
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome
  surface.
