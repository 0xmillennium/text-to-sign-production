# Retrieval-Augmented Pose Comparator Selection Decision

This document records candidate-level decision reasoning after a populated candidate card and scorecard have been reviewed. It is downstream of candidate cards and scorecards, and upstream of the whole-audit result. It does not replace the audit result.

## Candidate Under Decision

- `Candidate ID`: `CAND-RETRIEVAL-AUGMENTED-POSE-COMPARATOR`
- `Candidate card link`: [Retrieval-Augmented Pose Comparator](../candidate-cards/retrieval-augmented-pose-comparator.md)
- `Scorecard link`: [Retrieval-Augmented Pose Comparator Scorecard](../scorecards/retrieval-augmented-pose-comparator-scorecard.md)
- `Candidate record type`: `counter_alternative_candidate`
- `Originating family ID`: `FAM-RETRIEVAL-STITCHING-PRIMITIVES`
- `Originating family link`: [Retrieval, Stitching, and Motion-Primitives Alternatives](../candidate-universe/retrieval-stitching-primitives.md)

## Decision Inputs

- `Candidate card reviewed`: Established.
- `Scorecard reviewed`: Established.
- `Source-corpus evidence reviewed`: Established.
- `Boundary evidence reviewed`: Established.
- `Evaluation constraints reviewed`: Established.
- `Open scorecard issues reviewed`: Established.

## Scorecard Summary

- `Weighted total score`: `65.9`
- `Scorecard type`: `comparator`
- `Evidence-confidence label`: `medium`
- `Score interpretation`: Useful no-public-gloss comparator, not a primary neural generation candidate.
- `Blocking caveats from scorecard`: No-public-gloss adaptation, retrieval leakage, semantic adequacy, and gloss/dictionary dependency risks remain unresolved.

## Evidence-Confidence Status

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: Retrieval and stitching evidence is technically meaningful, but direct no-public-gloss compatibility is conditional and much support remains boundary or counter-alternative evidence.

## Hard Veto Checks

### Evidence Chain Veto

- `Status`: `pass`
- `Rationale`: The scorecard reviews retrieval, stitching, staged-production, motion-primitives, sparse-keyframe, evaluation, and latent-generation counter-evidence.

### Dataset / Supervision Compatibility Veto

- `Status`: `pass`
- `Rationale`: A no-public-gloss retrieval comparator is possible from paired text/pose examples, but gloss-dependent stitching or dictionary assumptions must remain outside scope.

### Repository / Workflow Compatibility Veto

- `Status`: `pass`
- `Rationale`: Retrieval indices, feature caches, tie-breaking, and split-safe lookup can fit the workflow if explicitly documented.

### Evaluability Veto

- `Status`: `pass`
- `Rationale`: Retrieval can be evaluated as a comparator, but leakage-safe splits and semantic adequacy checks are mandatory caveats.

### Scope-Control Veto

- `Status`: `pass`
- `Rationale`: Scope remains controlled if the candidate stays a comparator and does not become a gloss-dependent stitching system or primary model contribution.

### Additivity / Isolation Veto

- `Status`: `pass`
- `Rationale`: The candidate is not an additive model mechanism, but it is isolated as a comparator/counter-alternative.

## Veto Summary

- `All vetoes passed`: Established.
- `Failed vetoes`: None.
- `Veto summary rationale`: The candidate passes gates for comparator retention, not for primary model advancement.

## Decision Eligibility Status

- `Eligibility status`: Not eligible for primary advancement; eligible for role-specific comparator retention.
- `Eligibility rationale`: The score is below the primary-advancement threshold and the scorecard type is comparator. The refreshed template permits role-specific counter-alternative retention when the scorecard supports that non-primary role.

## Decision Status

- `Decision status`: `keep_as_counter_alternative`

## Decision Rationale

The candidate should be retained as a no-public-gloss comparator and sanity-check counter-alternative. It should not be advanced as a primary neural generation mechanism because key evidence is gloss-dependent, dictionary-dependent, transferable, boundary, or frontier-heavy, and because retrieval leakage and semantic mismatch are serious risks.

## Conditions Or Follow-Up Required

- `Evidence follow-up`: Preserve gloss-dependency and primitive/dictionary assumptions as boundary evidence.
- `Evaluation follow-up`: Define leakage-safe retrieval splits, retrieval metrics, semantic adequacy checks, and qualitative inspection.
- `Implementation follow-up`: Specify retrieval feature type, index construction, tie-breaking, unmatched-query handling, transition handling, and split-safe caching.
- `Boundary-condition follow-up`: Prevent this comparator from becoming a full gloss-dependent stitching or dictionary-based production system.

## Audit-Result Handoff Note

- `Handoff summary`: Retrieval-Augmented Pose Comparator is retained as a counter-alternative and sanity-check comparator.
- `How this decision should be used by audit-result.md`: Use it to test whether learned or generative candidates outperform no-public-gloss retrieval/reuse baselines.
- `Limitations to preserve in final audit result`: Retrieval realism must not be treated as semantic correctness unless leakage-safe and semantic-adequacy checks are satisfied.

## Record Boundary Note

- This document records candidate-level decision reasoning only.
- It is downstream of populated [Candidate Cards](../candidate-cards/index.md) and [Scorecards](../scorecards/index.md).
- It does not create or modify candidate cards or scorecards.
- It does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome surface.
- It does not assign legacy C1/C2 outcomes.
