# Text-Pose Semantic Consistency Objective Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-TEXT-POSE-SEMANTIC-CONSISTENCY`
- `Candidate card link`: [Text-Pose Semantic Consistency Objective](../candidate-cards/text-pose-semantic-consistency.md)
- `Scorecard link`: [Text-Pose Semantic Consistency Objective Scorecard](../scorecards/text-pose-semantic-consistency-scorecard.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-TEXT-POSE-SEMANTIC-ALIGNMENT`
- `Originating family link`: [Text-Pose Semantic Alignment and Conditioning](../candidate-universe/text-pose-semantic-alignment.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `69.1`
- `Scorecard type`: `model/additive-objective`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Important semantic-alignment candidate, but likely auxiliary unless standalone evaluation is explicitly justified.
- `Blocking caveats from scorecard`: Standalone versus auxiliary role, semantic metric validity, encoder choice, negative sampling, and interaction with other candidates remain unresolved.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The semantic-alignment concern is well supported, but standalone readiness is conditional because evidence often appears as a component inside larger systems.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: The scorecard reviews semantic-alignment evidence and limiting evidence, while recognizing that much support appears inside larger diffusion, multimodal, token, or variational systems.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: Paired text/transcript and pose/keypoint artifacts can support gloss-free alignment or embedding-consistency signals without manual gloss annotation.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: Encoder extraction, feature caching, and auxiliary objective implementation can fit the workflow if kept modular and split-safe.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: With/without-objective ablation is possible, but semantic metric validity and automatic evaluation remain major caveats.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: Scope remains controlled only if the candidate is retained as an auxiliary or additive mechanism rather than inflated into a standalone model direction without further evidence.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The candidate is not cleanly isolated as a standalone primary model, but it can be retained as an additive objective with explicit ablation requirements.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes gates for auxiliary/additive retention, not for primary model advancement.

## Decision Eligibility Status

- `Eligibility status`: Not eligible for primary advancement; eligible for role-specific auxiliary/additive retention.
- `Eligibility rationale`: The score is below the primary-advancement threshold, and the scorecard identifies unresolved standalone-versus-auxiliary ambiguity. The refreshed template permits role-specific auxiliary/additive retention when the scorecard supports that non-primary role.

## Decision Status

- `Decision status`: `keep_as_auxiliary_or_additive_objective`

## Decision Rationale

The candidate should not advance as a standalone primary model direction at this stage. It should be retained as an auxiliary or additive objective that may strengthen another model candidate if a later ablation explicitly defines where it attaches and how semantic gains are evaluated.

This decision does not advance the candidate as a standalone primary model direction.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve limiting evidence from gloss-pose alignment, representation bottlenecks, gloss-to-pose constraints, and pose-evaluation limitations.
- `Evaluation follow-up`: Define semantic metrics, qualitative inspection, and how embedding improvements will be interpreted against sign adequacy.
- `Implementation follow-up`: Choose text encoder, pose encoder, alignment loss, negative sampling, feature caching, and auxiliary loss weighting.
- `Boundary-condition follow-up`: Decide explicitly whether this objective attaches to the baseline, learned-token candidate, latent-diffusion candidate, or structure-aware candidate.

## Audit-Result Handoff Note

- `Handoff summary`: Text-Pose Semantic Consistency is retained as an auxiliary/additive objective, not as a standalone primary model candidate.
- `How this decision should be used by audit-result.md`: Use it as a possible strengthening mechanism for another candidate if a later ablation defines the attachment point.
- `Limitations to preserve in final audit result`: Semantic alignment evidence is important but currently not cleanly isolated as a standalone candidate.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
