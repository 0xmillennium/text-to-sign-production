# Direct Text-to-Pose Baseline Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-M0-DIRECT-TEXT-TO-POSE-BASELINE`
- `Candidate card link`: [Direct Text-to-Pose Baseline](../candidate-cards/direct-text-to-pose-baseline.md)
- `Scorecard link`: [Direct Text-to-Pose Baseline Scorecard](../scorecards/direct-text-to-pose-baseline-scorecard.md)
- `Candidate record type`: `baseline_candidate`
- `Originating family ID`: `FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES`
- `Originating family link`: [Foundational Text-to-Pose Baselines](../candidate-universe/foundational-text-to-pose-baselines.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `86.3`
- `Scorecard type`: `baseline readiness`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Baseline readiness only; suitable as the M0 comparison floor, not a model contribution.
- `Blocking caveats from scorecard`: Exact baseline architecture, artifact schema, split policy, and evaluation protocol remain to be fixed.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The baseline role is well supported, but implementation details and expected baseline quality remain conditional.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: Baseline-family, dataset-boundary, and evaluation evidence are present and sufficient for an M0 comparison-floor role.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: The candidate is gloss-free and aligned with text/transcript-to-pose/keypoint supervision, with artifact availability still conditional.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: The baseline is the lowest-complexity comparison surface and is compatible with a minimal experiment workflow once artifact schema and splits are fixed.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: The candidate gives later candidates a clear comparison floor and can be evaluated through the project evaluation protocol.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: The record does not claim model contribution status and does not expand scope beyond direct text-to-pose baseline readiness.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The isolated role is clean because the candidate is the M0 baseline itself.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes all gates for baseline-retention, not for primary model advancement.

## Decision Eligibility Status

- `Eligibility status`: Role-specific retention; not a primary model advancement decision.
- `Eligibility rationale`: The primary-advancement threshold does not apply because this is a baseline-readiness scorecard. The score supports keeping the candidate as the M0 baseline and ablation floor.

## Decision Status

- `Decision status`: `keep_as_baseline_or_ablation`

## Decision Rationale

The high score supports baseline readiness only. The candidate should be retained as the M0 comparison floor for later model candidates, not interpreted as a final model contribution or task-solving direction.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve baseline evidence as comparison context rather than model-contribution evidence.
- `Evaluation follow-up`: Fix metric set, qualitative inspection protocol, and metric limitation documentation.
- `Implementation follow-up`: Define exact baseline architecture, preprocessing, splits, batching, and output schema.
- `Boundary-condition follow-up`: Prevent later audit stages from reading baseline readiness as contribution strength.

## Audit-Result Handoff Note

- `Handoff summary`: Direct Text-to-Pose Baseline is retained as the M0 baseline and ablation floor.
- `How this decision should be used by audit-result.md`: Use it as the comparison baseline against which advanced model candidates are interpreted.
- `Limitations to preserve in final audit result`: Baseline readiness does not imply strong task performance or model contribution value.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
