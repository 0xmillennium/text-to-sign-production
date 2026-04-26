# Gloss-Free Latent Diffusion Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-GLOSS-FREE-LATENT-DIFFUSION`
- `Candidate card link`: [Gloss-Free Latent Diffusion](../candidate-cards/gloss-free-latent-diffusion.md)
- `Scorecard link`: [Gloss-Free Latent Diffusion Scorecard](../scorecards/gloss-free-latent-diffusion-scorecard.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-LATENT-GENERATIVE-PRODUCTION`
- `Originating family link`: [Latent Generative Production](../candidate-universe/latent-generative-production.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `84.1`
- `Scorecard type`: `full model`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Strong scientific support and high potential, but implementation and evaluation risk are high.
- `Blocking caveats from scorecard`: No blocking caveat; latent setup, stochastic training, compute, denoising, reconstruction, and evaluation sensitivity remain major open issues.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: Method and source-corpus support are very strong, but readiness is moderated by latent reconstruction, stochastic training, compute cost, and evaluation risk.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: The scorecard reviews direct and near-direct latent/generative evidence, including Text2SignDiff and MS2SL, plus relevant boundary and counter-evidence.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: The candidate is framed as gloss-free and has How2Sign-direct or How2Sign-compatible evidence.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: The candidate is compatible with the workflow in principle, but requires heavier latent, scheduler, seed-control, and evaluation infrastructure.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: Baseline versus latent generation comparison is possible, although generative diversity, semantic adequacy, and motion realism are metric-sensitive.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: Scope remains acceptable if multimodal audio expansion, sparse-keyframe frontier methods, and 3D/avatar systems remain outside the current decision.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The primary isolated variable is text-conditioned latent generative production relative to the direct baseline.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes all primary-advancement gates, but risk and evaluation caveats must remain explicit.

## Decision Eligibility Status

- `Eligibility status`: Eligible for primary advancement.
- `Eligibility rationale`: The weighted score is above `75/100`, evidence confidence is `medium`, all veto gates pass, and no unresolved blocking caveat prevents advancement.

## Decision Status

- `Decision status`: `advance_for_audit_result_consideration`

## Decision Rationale

The candidate advances because it has very strong latent/generative source support and clear relevance to gloss-free text-conditioned pose generation. It should be carried forward as a high-potential primary model candidate, but its implementation and evaluation risk are materially higher than the learned-token candidate.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve counter-evidence from learned-token alternatives, gloss-to-pose discrete diffusion, avatar/frontier scope pressure, sparse-keyframe alternatives, and pose-evaluation limits.
- `Evaluation follow-up`: Define generative-quality, semantic-adequacy, qualitative-inspection, and baseline-comparison protocols.
- `Implementation follow-up`: Specify latent representation, reconstruction target, denoising schedule, compute assumptions, seed control, and inference-cost expectations.
- `Boundary-condition follow-up`: Prevent the decision from expanding into 3D/avatar, sparse-keyframe, CFM, or audio-conditioned multimodal scope unless separately authorized.

## Audit-Result Handoff Note

- `Handoff summary`: Gloss-Free Latent Diffusion advances as a strong but risk-heavy primary model candidate.
- `How this decision should be used by audit-result.md`: Treat it as a leading candidate with stronger method evidence but higher implementation/evaluation risk.
- `Limitations to preserve in final audit result`: High scientific support must not be read as low implementation risk.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
