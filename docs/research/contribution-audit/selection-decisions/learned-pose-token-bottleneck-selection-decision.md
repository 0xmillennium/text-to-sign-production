# Learned Pose-Token Bottleneck Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-LEARNED-POSE-TOKEN-BOTTLENECK`
- `Candidate card link`: [Learned Pose-Token Bottleneck](../candidate-cards/learned-pose-token-bottleneck.md)
- `Scorecard link`: [Learned Pose-Token Bottleneck Scorecard](../scorecards/learned-pose-token-bottleneck-scorecard.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-LEARNED-POSE-MOTION-REPRESENTATIONS`
- `Originating family link`: [Learned Pose/Motion Representations](../candidate-universe/learned-pose-motion-representations.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `85.0`
- `Scorecard type`: `full model`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Strong near-term model candidate; tokenizer, reconstruction, codebook, and evaluation risk remain.
- `Blocking caveats from scorecard`: No blocking caveat; tokenizer design, reconstruction quality, codebook stability, and evaluation protocol remain open issues.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The evidence base is strong for a near-term model candidate, but project-specific feasibility depends on tokenizer design, reconstruction quality, codebook stability, and evaluation protocol.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: The scorecard reviews direct and transferable learned-representation evidence including SignVQNet, Data-Driven Representation, T2S-GPT, UniGloR, and related limiting evidence.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: Learned pose tokens can be derived from pose/keypoint data without manual gloss supervision, subject to stable artifact schema and masks.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: Tokenizer training, code assignment, and decoding require new modules but can fit the current reproducible workflow.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: Direct text-to-pose versus text-to-token-to-pose provides a clear ablation frame, with reconstruction and downstream motion-quality checks available.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: Scope remains controlled if dynamic duration, modality-specific token stacks, or heavy frontier tokenization are not added prematurely.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The isolated variable is the learned pose/motion token bottleneck relative to the direct baseline.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes all primary-advancement gates with manageable open implementation and evaluation issues.

## Decision Eligibility Status

- `Eligibility status`: Eligible for primary advancement.
- `Eligibility rationale`: The weighted score is above `75/100`, evidence confidence is `medium`, all veto gates pass, and no unresolved blocking caveat prevents advancement.

## Decision Status

- `Decision status`: `advance_for_audit_result_consideration`

## Decision Rationale

The candidate is one of the strongest near-term primary model directions. It is comparatively well isolated against the direct baseline and has strong source-corpus support for gloss-free learned pose/motion representations. The decision advances it for audit-result consideration while preserving tokenizer, reconstruction, codebook stability, and evaluation caveats.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve limiting evidence from latent diffusion, gloss-to-pose discrete diffusion, and pose-evaluation constraints.
- `Evaluation follow-up`: Separate tokenizer reconstruction quality from generated motion naturalness and semantic adequacy.
- `Implementation follow-up`: Specify tokenizer type, codebook size, temporal granularity, reconstruction target, mask handling, and decoder design.
- `Boundary-condition follow-up`: Avoid expanding this decision into dynamic duration or modality-specific tokenization unless a later audit explicitly authorizes that scope.

## Audit-Result Handoff Note

- `Handoff summary`: Learned Pose-Token Bottleneck advances as a strong primary model candidate.
- `How this decision should be used by audit-result.md`: Treat it as a leading near-term candidate for final synthesis, not as an already selected outcome.
- `Limitations to preserve in final audit result`: Readiness depends on tokenizer reconstruction quality, codebook stability, and evaluation that separates representation quality from sign adequacy.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
