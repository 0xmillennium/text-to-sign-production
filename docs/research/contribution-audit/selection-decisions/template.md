# Selection Decision Template

This template records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: Not yet assigned.
- `Candidate card link`: Not yet assigned.
- `Scorecard link`: Not yet assigned.
- `Candidate record type`: Not yet assigned.
- `Originating family ID`: Not yet assigned.
- `Originating family link`: Not yet assigned.

## Decision Inputs

- `Candidate card reviewed`: Not yet established.
- `Scorecard reviewed`: Not yet established.
- `Source-corpus evidence reviewed`: Not yet established.
- `Boundary evidence reviewed`: Not yet established.
- `Evaluation constraints reviewed`: Not yet established.
- `Open scorecard issues reviewed`: Not yet established.

## Scorecard Summary

- `Weighted total score`: Not yet recorded.
- `Scorecard type`: Not yet recorded.
- `Evidence-confidence label`: Not yet assigned.
- `Score interpretation`: Not yet documented.
- `Blocking caveats from scorecard`: Not yet documented.

## Evidence-Confidence Status

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: Not yet assigned.
- `Confidence rationale`: Not yet documented.

## Hard Veto Checks

Use status values:

- `pass`
- `fail`
- `not_yet_determined`

### Evidence Chain Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

### Dataset / Supervision Compatibility Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

### Repository / Workflow Compatibility Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

### Evaluability Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

### Scope-Control Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

### Additivity / Isolation Veto

- `Status`: Not yet recorded.
- `Rationale`: Not yet recorded.

## Veto Summary

- `All vetoes passed`: Not yet established.
- `Failed vetoes`: Not yet documented.
- `Veto summary rationale`: Not yet documented.

## Decision Eligibility Status

Eligibility for primary advancement requires all of the following:

- weighted score at least `75/100`,
- evidence confidence of `medium` or `high`,
- all hard veto gates passed,
- no unresolved blocking caveat from the scorecard.

The primary-advancement threshold applies only to `advance_for_audit_result_consideration`.
Baseline, counter-alternative, auxiliary/additive, and future-work decisions may use role-specific
decision statuses when the scorecard supports that non-primary role.

- `Eligibility status`: Not yet recorded.
- `Eligibility rationale`: Not yet documented.

## Decision Status

Allowed values:

- `advance_for_audit_result_consideration`
- `keep_as_counter_alternative`
- `keep_as_baseline_or_ablation`
- `keep_as_auxiliary_or_additive_objective`
- `defer_pending_evidence`
- `defer_as_future_work`
- `reject_for_current_audit`

- `Decision status`: Not yet assigned.

## Decision Rationale

Not yet recorded.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Not yet documented.
- `Evaluation follow-up`: Not yet documented.
- `Implementation follow-up`: Not yet documented.
- `Boundary-condition follow-up`: Not yet documented.

## Audit-Result Handoff Note

- `Handoff summary`: Not yet documented.
- `How this decision should be used by audit-result.md`: Not yet documented.
- `Limitations to preserve in final audit result`: Not yet documented.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
