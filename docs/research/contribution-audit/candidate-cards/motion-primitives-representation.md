# Motion Primitives Representation

## Candidate ID

`DISCRETE-A2`

## Candidate Name

Motion Primitives Representation

## Family

- `Family label`: `discrete_data_driven`
- `Family-level universe record`: [Discrete / Data-Driven Representation](../candidate-universe/discrete-data-driven-representation.md)

## Initial Status

- `Initial status label`: `primary_candidate`

## One-Sentence Technical Definition

Learn a reusable motion-primitives representation over sign-pose sequences and use those primitives
as the intermediate target for text-conditioned sign-pose generation.

## Research Role

- `Research role label`: `candidate_for_C1`

## Base-Model Additivity

- `additive_over_base`: Conditional: an independent `Base + candidate` path is plausible, but only
  if a gloss-free motion-primitives adaptation can be defined cleanly enough to avoid collapsing
  into a broader redesign.
- `what_changes`: Learn a reusable motion-primitives representation over sign-pose sequences and use
  those primitives as the intermediate target for text-conditioned sign-pose generation.
- `requires_other_contribution`: Established: this candidate does not inherently require `C2`.
- `independent_variant_exists`: Conditional: an independent `Base + candidate` implementation
  remains contingent on defining that variant explicitly.

## Expected Improvement Mechanism

Replace monolithic full-sequence regression with a representation built from reusable motion
components, with the expectation of improving expressiveness, reducing regression-to-the-mean
behavior, and producing more interpretable motion structure.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `mode_averaging`, `unnatural_transitions`,
  `hand_articulation_loss`, `temporal_jitter`

## Required Data / Supervision Assumptions

- `Current processed manifests sufficient`: Conditional: current processed manifests may be
  sufficient for a gloss-free adaptation, but this depends on whether primitive learning can be
  derived cleanly from the current processed pose sequences without new gloss supervision.
- `Extra annotation required`: Current audit assumption: no new gloss corpus is assumed available
  by default, and no explicit segmentation annotation is assumed unless later required by the
  chosen variant.
- `Gloss required`: Current audit assumption: the project should prefer a gloss-free adaptation,
  and no new gloss corpus is assumed available by default.
- `Dictionary/example bank required`: Current audit assumption: no dictionary or example bank is
  required.
- `Segmentation required`: Current audit assumption: no explicit segmentation annotation is assumed
  unless later required by the chosen variant.

## Repository / Workflow Compatibility

- `Compatible with current .npz schema`: Current audit assessment: conceptually compatible with the
  current processed pose surface.
- `Compatible with current Colab-centered workflow`: Conditional: conceptually compatible with the
  current Colab-centered workflow, but more adaptation-heavy than Dynamic VQ and not yet
  established as a clean low-friction fit.
- `Heavy new preprocessing required`: Current audit assessment: heavy raw-data preprocessing is not
  currently indicated, but added representation-learning and artifact-definition steps are likely.
- `Thin-driver notebook principle affected`: Current audit assessment: added
  representation-learning components and artifact-definition steps are likely. Impact on the
  thin-driver notebook principle is not yet established in the current audit record.

## Supporting Evidence

### Source 1

- `Citation`: [3](../../bibliography.md#ref-03)
- `Support type`: `direct`
- `Claim supported`: A motion-primitives-based intermediate representation is a legitimate and empirically competitive sign language production direction.
- `Evidence note`: The paper explicitly formulates sign language production as two jointly trained sub-tasks and proposes a Mixture of Motion Primitives architecture in which distinct motion primitives are learned and temporally combined at inference. It reports stronger user-evaluation performance and an 11% back-translation improvement over competing results.
- `Source location`: Abstract; Introduction; contribution list.
- `Interpretation boundary`: This source directly supports the motion-primitives idea. However, its original formulation uses gloss supervision, so it cannot be cited as evidence for a fully gloss-free drop-in implementation without qualification.

### Source 2

- `Citation`: [4](../../bibliography.md#ref-04)
- `Support type`: `indirect`
- `Claim supported`: Discrete/data-driven representations based on reusable motion units are a serious alternative to direct continuous regression in sign language production.
- `Evidence note`: The paper supports the broader family claim that continuous pose generation can be reframed as discrete sequence generation through learned motion units and a codebook, which strengthens the plausibility of motion-primitives-style intermediate structure.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source does not specifically validate a motion-primitives mixture-of-experts design; it supports the broader discrete/data-driven family into which the motion-primitives candidate fits.

## Contradicting Or Limiting Evidence

### Limitation / Boundary Source 1

- `Citation`: [3](../../bibliography.md#ref-03)
- `Claim limited`: The motion-primitives candidate transfers cleanly to the current gloss-free audit setting.
- `Evidence note`: The paper states that its first translation sub-task uses gloss supervision to learn the latent sign-language representation. That means the current project would be adapting the family rather than reproducing the original setup.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This is a genuine transfer-risk note, not merely a minor detail. It raises the candidate’s adaptation cost and should remain visible in the card.

### Limitation / Boundary Source 2

- `Citation`: [7](../../bibliography.md#ref-07)
- `Claim limited`: Reusable units automatically yield natural continuous sign sequences.
- `Evidence note`: The paper argues that simply concatenating expressive sign units creates robotic and unnatural output and motivates explicit stitching and prosody-aware modifications. This is an important boundary for motion-primitives systems, where composition quality is central.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This does not contradict motion primitives as a family; it limits overclaiming about continuity and transition quality.

### Candidate-Level Evidence Synthesis

- `Best-supported claim`: Motion primitives are a directly supported, expressive, compositional sign language production direction with evidence from both user evaluation and back-translation.
- `Least-secure claim`: A gloss-free motion-primitives adaptation will remain as cleanly additive and low-risk as Dynamic VQ in this repository.
- `Evidence confidence rationale`: The candidate has one strong direct source and one strong family-level supporting source, but also carries a visible transfer-risk because the strongest direct source relies on gloss supervision and the continuity problem remains non-trivial.

## Comparison To Strong Alternatives

Relative to dynamic VQ, this candidate offers a more explicitly compositional motion representation
but may be harder to adapt cleanly to a gloss-free baseline; relative to retrieval/stitching, it
preserves a learned generative contribution rather than shifting the thesis toward example
retrieval; relative to diffusion, it is more structured and likely cheaper to prototype.

## Minimum Ablation Plan

Base vs Base+MotionPrimitives; compare primitive-count or primitive-structure variants internally;
compare against DynamicVQ within the same family before any final C1 selection; if later selected,
compare Base+MotionPrimitives vs Base+MotionPrimitives+C2.

## Position In The Four-Model Matrix

- `fits_as_C1`: Conditional: this candidate can be treated as a plausible `C1`-family fit if an
  independent `Base + candidate` implementation is defined cleanly.
- `fits_as_C2`: Established: no in the current research-role framing. This candidate is documented
  as a `C1`-family discrete/data-driven path rather than as a structure-aware `C2` path.
- `fits_joint_model`: Conditional: joint-model compatibility with a later structure-aware `C2`
  remains plausible if a clean independent `Base + candidate` implementation exists.
- `breaks_factorial_design`: Conditional: not inherently indicated, but there is elevated risk that
  factorial clarity could weaken if the motion-primitives adaptation cannot be defined as a clean
  independent `Base + candidate` path.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium-high implementation complexity.
- `Failure cost (days or bucket)`: Higher failure cost than Dynamic VQ.
- `Hidden dependency risk`: Medium-high because the representation design may depend on stronger
  assumptions.
- `Scope-creep risk`: Medium-high if the design drifts toward segmentation-heavy or gloss-dependent
  formulations.

## Record Boundary Note

- This document records candidate-level evidence only.
- [Scorecards](../scorecards/index.md) and
  [Selection Decisions](../selection-decisions/index.md) are maintained in their dedicated audit
  surfaces.
- No final selection outcome is recorded in this document.
