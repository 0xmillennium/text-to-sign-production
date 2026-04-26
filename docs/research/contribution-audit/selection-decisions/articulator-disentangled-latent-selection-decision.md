# Articulator-Disentangled Latent Modeling Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-ARTICULATOR-DISENTANGLED-LATENT`
- `Candidate card link`: [Articulator-Disentangled Latent Modeling](../candidate-cards/articulator-disentangled-latent.md)
- `Scorecard link`: [Articulator-Disentangled Latent Modeling Scorecard](../scorecards/articulator-disentangled-latent-scorecard.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING`
- `Originating family link`: [Structure-Aware and Articulator-Aware Modeling](../candidate-universe/structure-aware-articulator-modeling.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `78.8`
- `Scorecard type`: `full model`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Viable structure-preservation candidate with conditional schema, mask, loss-weighting, and channel-aware evaluation fit.
- `Blocking caveats from scorecard`: No blocking caveat; artifact schema, partitions, masks, channel weights, and evaluation sensitivity remain open.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The scientific case is coherent and source-supported, but readiness depends on artifact schema, stable partitions, loss weighting, and evaluation sensitivity.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: The scorecard reviews structure-aware and articulator-aware evidence including DARSLP, MCST, A²V-SLP, ILRSLP, M3T, and multi-channel baseline context.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: The candidate can remain gloss-free if articulator groups are derived from pose/keypoint schema, but body/hand/face partitions and masks must be stable.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: Partition utilities, masks, channel-aware losses, and diagnostics can fit the reproducible workflow if implemented modularly.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: Channel-stratified evaluation and ablations are possible, although sign-level interpretation of channel gains remains difficult.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: Scope remains controlled if variational, iterative, non-manual-heavy, or 3D extensions are not added prematurely.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The isolated variable is articulator/channel-aware latent grouping or regularization relative to the direct baseline.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes all primary-advancement gates, with more conditional workflow and evaluation fit than the top two candidates.

## Decision Eligibility Status

- `Eligibility status`: Eligible for primary advancement.
- `Eligibility rationale`: The weighted score is above `75/100`, evidence confidence is `medium`, all veto gates pass, and no unresolved blocking caveat prevents advancement.

## Decision Status

- `Decision status`: `advance_for_audit_result_consideration`

## Decision Rationale

The candidate advances as a viable primary model direction focused on preserving body, hand, face, channel, and articulator structure. It is scientifically coherent, but its readiness is more conditional than learned-token or latent-diffusion routes because it depends on stable partitions, masks, loss weighting, and channel-aware evaluation.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve limiting evidence from gloss-to-pose skeletal structure, latent diffusion alternatives, and pose-evaluation constraints.
- `Evaluation follow-up`: Define channel-stratified metrics, sign-level interpretation checks, qualitative inspection, and metric limitation handling.
- `Implementation follow-up`: Specify body/hand/face partition scheme, masks, channel weights, and loss design.
- `Boundary-condition follow-up`: Avoid expanding into variational, iterative, non-manual-heavy, or 3D/avatar systems unless later authorized.

## Audit-Result Handoff Note

- `Handoff summary`: Articulator-Disentangled Latent Modeling advances as a viable primary model candidate.
- `How this decision should be used by audit-result.md`: Treat it as a structure-preservation route that remains behind the top two candidates in workflow/evaluation certainty.
- `Limitations to preserve in final audit result`: Readiness depends on artifact schema, masks, channel partitions, loss balancing, and evaluation sensitivity.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
