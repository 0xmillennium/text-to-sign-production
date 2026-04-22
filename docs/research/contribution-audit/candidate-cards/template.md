# Candidate Card Template

This template preserves the candidate-level evidence-record structure for a concrete audit candidate. It is a template only and does not record a real candidate outcome.

## Candidate ID

Not yet assigned.

## Candidate Name

Not yet assigned.

## Family

Allowed labels:

- `discrete_data_driven`
- `structure_multi_channel`
- `retrieval_stitching`
- `diffusion`
- `wildcard`

- `Family label`: Not yet assigned.
- `Family-level universe record`: Not yet linked.

## Initial Status

Allowed labels:

- `primary_candidate`
- `counter_alternative`
- `wildcard`

- `Initial status label`: Not yet assigned.

## One-Sentence Technical Definition

Not yet documented.

## Research Role

Allowed labels:

- `candidate_for_C1`
- `candidate_for_C2`
- `counterfactual_comparator`
- `fallback_path`
- `future_work_candidate`

- `Research role label`: Not yet assigned.

## Base-Model Additivity

Use audit-neutral wording such as `Established: ...`, `Conditional: ...`, or `Not yet established in the current audit record.`

- `additive_over_base`: Not yet established in the current audit record.
- `what_changes`: Not yet documented.
- `requires_other_contribution`: Not yet established in the current audit record.
- `independent_variant_exists`: Not yet established in the current audit record.

## Expected Improvement Mechanism

Not yet documented.

## Targeted Failure Modes

Controlled vocabulary candidates:

- `oversmoothing`
- `temporal_jitter`
- `timing_misalignment`
- `hand_articulation_loss`
- `cross_channel_inconsistency`
- `semantic_mismatch`
- `mode_averaging`
- `sequence_drift`
- `unnatural_transitions`

- `Applicable failure modes`: Not yet documented.

## Required Data / Supervision Assumptions

Use audit-neutral wording such as `Current audit assumption: ...`, `Conditional: ...`, or `Current audit evidence is insufficient to state this confidently.`

- `Current processed manifests sufficient`: Not yet established in the current audit record.
- `Extra annotation required`: Not yet established in the current audit record.
- `Gloss required`: Not yet established in the current audit record.
- `Dictionary/example bank required`: Not yet established in the current audit record.
- `Segmentation required`: Not yet established in the current audit record.

## Repository / Workflow Compatibility

Use audit-neutral wording such as `Current audit assessment: ...`, `Conditional: ...`, or `Not yet established in the current audit record.`

- `Compatible with current .npz schema`: Not yet established in the current audit record.
- `Compatible with current Colab-centered workflow`: Not yet established in the current audit record.
- `Heavy new preprocessing required`: Not yet established in the current audit record.
- `Thin-driver notebook principle affected`: Not yet established in the current audit record.

## Supporting Evidence

Allowed `Support type` values:

- `direct`
- `near-direct`
- `indirect`

### Source 1

- `Citation`: Not yet documented.
- `Support type`: Not yet documented.
- `Claim supported`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Source location`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 2

- `Citation`: Not yet documented.
- `Support type`: Not yet documented.
- `Claim supported`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Source location`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 3

- `Citation`: Optional. Populate only if needed.
- `Support type`: Optional. Populate only if needed.
- `Claim supported`: Optional. Populate only if needed.
- `Evidence note`: Optional. Populate only if needed.
- `Source location`: Optional. Populate only if needed.
- `Interpretation boundary`: Optional. Populate only if needed.

## Contradicting Or Limiting Evidence

### Source 1

- `Citation`: Not yet documented.
- `Claim limited`: Not yet documented.
- `Evidence note`: Not yet documented.
- `Source location`: Not yet documented.
- `Interpretation boundary`: Not yet documented.

### Source 2

- `Citation`: Optional. Populate only if needed.
- `Claim limited`: Optional. Populate only if needed.
- `Evidence note`: Optional. Populate only if needed.
- `Source location`: Optional. Populate only if needed.
- `Interpretation boundary`: Optional. Populate only if needed.

### Candidate-Level Evidence Synthesis

- `Best-supported claim`: Not yet documented.
- `Least-secure claim`: Not yet documented.
- `Evidence confidence rationale`: Not yet documented.

## Comparison To Strong Alternatives

Not yet documented.

## Minimum Ablation Plan

Not yet documented.

## Position In The Four-Model Matrix

Use audit-neutral wording such as `Established: ...`, `Conditional: ...`, or `Not yet established in the current audit record.`

- `fits_as_C1`: Not yet established in the current audit record.
- `fits_as_C2`: Not yet established in the current audit record.
- `fits_joint_model`: Not yet established in the current audit record.
- `breaks_factorial_design`: Not yet established in the current audit record.

## Complexity / Risk Estimate

- `Implementation complexity`: Not yet documented.
- `Failure cost (days or bucket)`: Not yet documented.
- `Hidden dependency risk`: Not yet documented.
- `Scope-creep risk`: Not yet documented.

## Record Boundary Note

- This document records candidate-level evidence only.
- [Scorecards](../scorecards/index.md) and
  [Selection Decisions](../selection-decisions/index.md) are maintained in their dedicated audit
  surfaces.
- No final selection outcome is recorded here.
