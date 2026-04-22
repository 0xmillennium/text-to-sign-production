# Channel-Aware Loss Reweighting

## Candidate ID

`STRUCTURE-B1`

## Candidate Name

Channel-Aware Loss Reweighting

## Family

- `Family label`: `structure_multi_channel`
- `Family-level universe record`: [Structure-Aware / Multi-Channel Improvement](../candidate-universe/structure-aware-multi-channel-improvement.md)

## Initial Status

- `Initial status label`: `primary_candidate`

## One-Sentence Technical Definition

Keep the base text-to-pose model architecture broadly intact but apply channel-aware loss
reweighting so that body, left hand, right hand, and other supported pose channels are optimized
with explicitly differentiated importance.

## Research Role

- `Research role label`: `candidate_for_C2`

## Base-Model Additivity

- `additive_over_base`: Established: yes. The current audit record supports an independent
  `Base + C2` path for this candidate, although the contribution remains lighter-weight than more
  structurally invasive alternatives.
- `what_changes`: Keep the base text-to-pose model architecture broadly intact but apply
  channel-aware loss reweighting so that body, left hand, right hand, and other supported pose
  channels are optimized with explicitly differentiated importance.
- `requires_other_contribution`: Established: this candidate does not inherently require `C1`.
- `independent_variant_exists`: Established: the current audit record supports the existence of an
  independent `Base + C2` variant.

## Expected Improvement Mechanism

Attack the tendency of uniform pose regression objectives to underrepresent high-information
articulators, especially the hands, by making the optimization objective more sensitive to
channel-specific importance and error profiles.

## Targeted Failure Modes

- `Applicable failure modes`: `hand_articulation_loss`, `cross_channel_inconsistency`,
  `oversmoothing`, `timing_misalignment`

## Required Data / Supervision Assumptions

- `Current processed manifests sufficient`: Current audit assumption: current processed text-pose
  pairs are sufficient.
- `Extra annotation required`: Current audit assumption: no extra supervision beyond the existing
  pose-channel structure is required.
- `Gloss required`: Current audit assumption: no new gloss annotation is required.
- `Dictionary/example bank required`: Current audit assumption: no dictionary or example bank is
  required.
- `Segmentation required`: Current audit assumption: no segmentation annotation is required.

## Repository / Workflow Compatibility

- `Compatible with current .npz schema`: Current audit assessment: highly compatible with the
  current processed `.npz` pose surface.
- `Compatible with current Colab-centered workflow`: Current audit assessment: highly compatible
  with the current Colab-centered workflow.
- `Heavy new preprocessing required`: Current audit assessment: heavy new preprocessing is not
  currently indicated.
- `Thin-driver notebook principle affected`: Current audit assessment: implementation should remain
  in core project logic with the notebook remaining a thin driver, although training configuration
  and reporting surfaces may need extension.

## Supporting Evidence

### Source 1

- `Citation`: [6](../../bibliography.md#ref-06)
- `Support type`: `near-direct`
- `Claim supported`: DARSLP directly supports channel-aware weighted regularization inside a gloss-free combined method, and therefore partly supports a lighter standalone channel-aware loss candidate through that weighting component.
- `Evidence note`: The paper explicitly states that DARSLP applies channel-aware regularization by aligning predicted latent distributions with articulator-based priors and weighting loss contributions according to articulator regions. It presents that mechanism as part of a gloss-free framework that also includes articulator-based disentanglement rather than as a standalone loss-only intervention.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This is the strongest local source for the channel-aware mechanism relevant to this candidate. However, DARSLP is not identical to a loss-reweighting-only intervention; it combines channel-aware regularization with articulator-based disentanglement.

### Source 2

- `Citation`: [1](../../bibliography.md#ref-01)
- `Support type`: `near-direct`
- `Claim supported`: Sign language production must treat sign as a multi-channel articulation problem rather than a single undifferentiated manual-only target.
- `Evidence note`: The paper explicitly frames sign languages as multi-channel visual languages combining manual and non-manual features and criticizes earlier sign language production work for focusing mainly on manual features, leading to robotic and non-expressive production. It also shows that manual+non-manual representations matter empirically.
- `Source location`: Abstract; Introduction; evaluation discussion.
- `Interpretation boundary`: This source supports the need for explicit channel sensitivity, but it does not directly propose channel-aware loss reweighting as such. It is best used as family and mechanism support, not as a one-to-one method match.

### Source 3

- `Citation`: [5](../../bibliography.md#ref-05)
- `Support type`: `near-direct`
- `Claim supported`: Unified feature representations can ignore structural correlations across channels, motivating more explicit channel-aware handling.
- `Evidence note`: The paper states that many current transformer-based sign language production models map multi-channel sign poses into unified representations and ignore structural channel correlations; MCST-Transformer is proposed specifically to address that.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This strengthens the rationale for this candidate, but it is more directly aligned with structured channel modeling than with reweighting alone.

## Contradicting Or Limiting Evidence

### Limitation / Boundary Note

- `Citation`: [6](../../bibliography.md#ref-06)
- `Claim limited`: Channel-aware loss reweighting by itself is likely to be a sufficiently strong thesis-level contribution.
- `Evidence note`: Even the strongest local source for the relevant mechanism, DARSLP, does not rely on weighting alone; it combines channel-aware regularization with articulator-based disentanglement and a structured latent space. That means a loss-only intervention may be too modest relative to the richer structure used in the source paper.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This is not evidence against the candidate’s validity. It is a scale-of-contribution boundary: the candidate may be useful and additive, but its standalone thesis strength must be tested rather than assumed.

### Candidate-Level Evidence Synthesis

- `Best-supported claim`: Explicit channel-aware treatment is well supported in gloss-free and multi-channel sign language production literature, and DARSLP directly supports channel-aware weighting inside a richer combined method while only partly supporting a lighter standalone weighting intervention.
- `Least-secure claim`: A relatively light-weight channel-aware loss intervention will, by itself, produce a clearly thesis-level gain rather than a modest optimization improvement.
- `Evidence confidence rationale`: The candidate has one strong local source for the relevant channel-aware mechanism inside a combined method and two strong supporting sources for the broader channel-aware rationale. The main remaining uncertainty is not whether explicit channel-aware treatment is defensible, but whether a relatively modest loss-level intervention is strong enough as a standalone thesis contribution.

## Comparison To Strong Alternatives

Relative to articulator-partitioned latent structure, this candidate is simpler and lower-risk but
may be less expressive; relative to discrete-token candidates, it keeps the representation
continuous and therefore may not attack regression-to-the-mean as directly; relative to
retrieval/stitching or diffusion, it is much easier to align with the current baseline and thesis
scope.

## Minimum Ablation Plan

Base vs Base+ChannelAwareLoss; compare alternative channel-weight schedules internally; compare
Base+ChannelAwareLoss vs Base+C1+ChannelAwareLoss if later combined; inspect hand quality and
cross-channel consistency qualitatively in addition to metric changes.

## Position In The Four-Model Matrix

- `fits_as_C1`: Established: no in the current research-role framing. This candidate is documented
  as a `C2`-family structure-aware path rather than as a discrete/data-driven `C1` path.
- `fits_as_C2`: Established: yes. The current audit record supports this candidate as a plausible
  `C2`-family fit.
- `fits_joint_model`: Conditional: joint-model compatibility with a later discrete/data-driven
  `C1` candidate appears plausible but remains to be tested.
- `breaks_factorial_design`: Not currently indicated. The current audit record supports an
  independent `Base + C2` variant and does not identify an inherent requirement for `C1`.

## Complexity / Risk Estimate

- `Implementation complexity`: Low-medium implementation complexity.
- `Failure cost (days or bucket)`: Relatively low failure cost if wrong.
- `Hidden dependency risk`: Medium because the choice of channel weights may interact with training
  stability.
- `Scope-creep risk`: Low.

## Record Boundary Note

- This document records candidate-level evidence only.
- [Scorecards](../scorecards/index.md) and
  [Selection Decisions](../selection-decisions/index.md) are maintained in their dedicated audit
  surfaces.
- No final selection outcome is recorded in this document.
