# Candidate Card Template

This template is for concrete candidate-level audit records. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

Not yet assigned.

## Candidate Name

Not yet assigned.

## Candidate Record Type

Allowed labels:

- `model_candidate`
- `counter_alternative_candidate`
- `baseline_candidate`
- `evaluation_support_candidate`
- `frontier_future_work_candidate`

- `Candidate record type`: Not yet assigned.

## Originating Candidate-Universe Family

- `Family ID`: Not yet assigned.
- `Family link`: Not yet assigned.
- `Family type/status`: Not yet assigned.
- `Reason this candidate is derived from this family`: Not yet documented.

Allowed family type/status values:

- `primary_candidate_family`
- `counter_alternative_family`
- `boundary_family`
- `evaluation_family`
- `frontier_watch_family`
- `background_context_family`

## Candidate-Level Technical Definition

Not yet documented.

## Candidate Boundary

- `Included mechanism`: Not yet documented.
- `Explicitly excluded mechanisms`: Not yet documented.
- `Not to be confused with`: Not yet documented.

## Research Role In Current Audit

Allowed labels:

- `primary_model_candidate`
- `counter_alternative_candidate`
- `baseline_or_ablation_candidate`
- `auxiliary_additive_objective`
- `fallback_candidate`
- `evaluation_support_candidate`
- `future_work_candidate`

- `Research role label`: Not yet assigned.

## Expected Improvement Mechanism

Not yet documented.

## Targeted Failure Modes

Controlled vocabulary candidates:

- `oversmoothing`
- `temporal_jitter`
- `timing_misalignment`
- `hand_articulation_loss`
- `non_manual_loss`
- `cross_channel_inconsistency`
- `semantic_mismatch`
- `mode_averaging`
- `sequence_drift`
- `unnatural_transitions`
- `token_collapse`
- `reconstruction_loss`
- `retrieval_discontinuity`
- `evaluation_blind_spot`

- `Applicable failure modes`: Not yet documented.

## Data And Supervision Assumptions

- `Dataset compatibility`: Not yet established.
  Allowed labels: `how2sign_direct`, `how2sign_compatible`, `cross_lingual_transferable`, `dataset_private_transferable`, `dataset_unclear`.

- `Gloss dependency`: Not yet established.
  Allowed labels: `gloss_free`, `gloss_optional`, `gloss_adaptable`, `gloss_dependent`, `gloss_unclear`, `notation_dependent`.

- `Extra annotation required`: Not yet established.
- `Segmentation required`: Not yet established.
- `Dictionary/example bank required`: Not yet established.
- `3D/parametric artifacts required`: Not yet established.
- `Current project artifacts sufficient`: Not yet established.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Not yet established.
- `Compatible with current experiment workflow`: Not yet established.
- `Heavy new preprocessing required`: Not yet established.
- `Thin-driver notebook principle affected`: Not yet established.
- `Reproducibility risk`: Not yet documented.

## Evidence Chain

- `Source-selection basis`: Not yet documented.
- `Source-corpus entries used`: Not yet documented.
- `Candidate-universe family used`: Not yet documented.
- `Boundary families consulted`: Not yet documented.
- `Evaluation family consulted`: Not yet documented.

## Supporting Source-Corpus Evidence

Allowed `Support type` values:

- `direct_candidate_support`
- `near_direct_method_support`
- `transferable_method_support`
- `boundary_support`
- `evaluation_support`

### Source 1

- `Source corpus entry`: Not yet assigned.
- `Source link`: Not yet assigned.
- `Support type`: Not yet documented.
- `Claim supported`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 2

- `Source corpus entry`: Not yet assigned.
- `Source link`: Not yet assigned.
- `Support type`: Not yet documented.
- `Claim supported`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 3

- `Source corpus entry`: Optional.
- `Source link`: Optional.
- `Support type`: Optional.
- `Claim supported`: Optional.
- `Evidence note`: Optional.
- `Interpretation boundary`: Optional.

## Limiting Or Contradicting Evidence

Allowed `Limitation type` values:

- `method_limitation`
- `dataset_limitation`
- `supervision_limitation`
- `evaluation_limitation`
- `scope_limitation`
- `counter_evidence`

### Source 1

- `Source corpus entry`: Not yet assigned.
- `Source link`: Not yet assigned.
- `Limitation type`: Not yet documented.
- `Claim limited`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 2

- `Source corpus entry`: Optional.
- `Source link`: Optional.
- `Limitation type`: Optional.
- `Claim limited`: Optional.
- `Evidence note`: Optional.
- `Interpretation boundary`: Optional.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: Not yet documented.
- `Least-secure claim`: Not yet documented.
- `Evidence confidence`: Not yet established.
- `Evidence confidence rationale`: Not yet documented.

## Comparison To Strong Alternatives

Not yet documented.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: Not yet documented.
- `Required ablation`: Not yet documented.
- `Primary metric family`: Not yet documented.
- `Secondary metric family`: Not yet documented.
- `Qualitative inspection needed`: Not yet established.
- `Metric limitation note`: Not yet documented.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Not yet established.
- `Baseline changed`: Not yet documented.
- `Primary isolated variable`: Not yet documented.
- `Requires another candidate`: Not yet established.
- `Independent ablation possible`: Not yet established.
- `Likely interactions with other candidate families`: Not yet documented.

## Complexity / Risk Estimate

- `Implementation complexity`: Not yet documented.
- `Failure cost`: Not yet documented.
- `Hidden dependency risk`: Not yet documented.
- `Scope-creep risk`: Not yet documented.
- `Evaluation risk`: Not yet documented.

## Downstream Scoring Notes

- `Scorecard readiness`: Not yet established.
- `Open questions for scorecard`: Not yet documented.
- `Evidence gaps to resolve before selection decision`: Not yet documented.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
