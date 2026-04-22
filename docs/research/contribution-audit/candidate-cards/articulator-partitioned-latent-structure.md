# Articulator-Partitioned Latent Structure

## Candidate ID

`STRUCTURE-B2`

## Candidate Name

Articulator-Partitioned Latent Structure

## Family

- `Family label`: `structure_multi_channel`
- `Family-level universe record`: [Structure-Aware / Multi-Channel Improvement](../candidate-universe/structure-aware-multi-channel-improvement.md)

## Initial Status

- `Initial status label`: `primary_candidate`

## One-Sentence Technical Definition

Introduce an articulator-aware model structure in which body and hand information are represented
or decoded through explicitly partitioned latent or prediction pathways rather than a single
undifferentiated pose-generation channel.

## Research Role

- `Research role label`: `candidate_for_C2`

## Base-Model Additivity

- `additive_over_base`: Conditional: yes, with medium-confidence implementation risk. The current
  audit record supports an independent `Base + C2` path conceptually, but architectural entanglement
  risk is higher than in Channel-Aware Loss Reweighting.
- `what_changes`: Introduce an articulator-aware model structure in which body and hand information
  are represented or decoded through explicitly partitioned latent or prediction pathways rather
  than a single undifferentiated pose-generation channel.
- `requires_other_contribution`: Established: this candidate does not inherently require `C1`.
- `independent_variant_exists`: Conditional: this candidate fits as a plausible `C2`-family option
  if an independent `Base + C2` implementation can be defined cleanly.

## Expected Improvement Mechanism

Improve structured coordination and channel-specific fidelity by giving different articulators
partially specialized representational capacity, with the expectation of reducing cross-channel
interference and improving hand/body modeling without collapsing everything into one shared
regression path.

## Targeted Failure Modes

- `Applicable failure modes`: `cross_channel_inconsistency`, `hand_articulation_loss`,
  `oversmoothing`, `temporal_jitter`, `sequence_drift`

## Required Data / Supervision Assumptions

- `Current processed manifests sufficient`: Conditional: current processed text-pose pairs may be
  sufficient if the supported schema exposes channel structure clearly enough.
- `Extra annotation required`: Current audit assumption: no extra supervision beyond existing
  channel or group structure is required by default.
- `Gloss required`: Current audit assumption: no new gloss annotation is required.
- `Dictionary/example bank required`: Current audit assumption: no dictionary or example bank is
  required.
- `Segmentation required`: Current audit assumption: no explicit segmentation annotation is
  required.

## Repository / Workflow Compatibility

- `Compatible with current .npz schema`: Current audit assessment: conceptually compatible with the
  current processed pose surface.
- `Compatible with current Colab-centered workflow`: Conditional: conceptually compatible with the
  current Colab-centered workflow, but with added architecture, training, and artifact-packaging
  complexity relative to the baseline workflow.
- `Heavy new preprocessing required`: Current audit assessment: heavy new preprocessing is not
  currently indicated.
- `Thin-driver notebook principle affected`: Current audit assessment: training and packaging
  complexity will increase. Impact on the thin-driver notebook principle is not yet established in
  the current audit record.

## Supporting Evidence

### Source 1

- `Citation`: [6](../../bibliography.md#ref-06)
- `Support type`: `direct`
- `Claim supported`: An articulator-partitioned latent structure is a plausible and high-value gloss-free sign language production intervention.
- `Evidence note`: The paper explicitly states that DARSLP uses an articulator-based disentanglement strategy in which face, right hand, left hand, and body features are modeled separately to promote structured and interpretable representation learning. It also presents this as a gloss-free framework and positions the factorized latent space as part of the mechanism for improving motion diversity and realism.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This is the strongest direct evidence for this candidate. It does not mean the current repository must replicate DARSLP exactly, but it strongly validates the family and mechanism behind an articulator-partitioned latent structure.

### Source 2

- `Citation`: [1](../../bibliography.md#ref-01)
- `Support type`: `near-direct`
- `Claim supported`: Sign language production benefits from explicitly respecting multi-channel articulation and cannot be reduced to manual-only or flattened representations without losing realism and comprehensibility.
- `Evidence note`: The paper emphasizes that understandable sign language production must embody both manual and non-manual features, and that earlier methods focused mainly on manual features, leading to robotic and non-expressive output. It therefore supports the broader rationale for explicit articulator structure.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source strongly supports the broader need for explicit articulator/channel structure, but it is complementary to DARSLP rather than a one-to-one articulator-disentangled latent-space paper.

### Source 3

- `Citation`: [5](../../bibliography.md#ref-05)
- `Support type`: `near-direct`
- `Claim supported`: Channel-level spatial and temporal modeling is a serious alternative to unified feature representations in sign language production.
- `Evidence note`: The paper argues that unified feature representations ignore structural correlations between channels and proposes explicit channel-level spatial and temporal attention to address that issue.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source supports the explicit-structure rationale for this candidate, but it is not itself an articulator-disentangled latent-space paper.

## Contradicting Or Limiting Evidence

### Limitation / Boundary Note

- `Citation`: [6](../../bibliography.md#ref-06)
- `Claim limited`: A more articulated structural factorization automatically yields a better candidate in practice.
- `Evidence note`: DARSLP provides strong direct support for articulator-based disentanglement, but the same line of work also frames sign language production as difficult because many models still regress to mean poses, lose hand detail, or fail to produce natural transitions. That means structured factorization is promising, not automatically sufficient; implementation quality still matters.
- `Source location`: Introduction.
- `Interpretation boundary`: This is not evidence against the candidate. It is a reminder that articulator partitioning is a strong hypothesis, not an automatic success condition. Architectural entanglement and training complexity remain real risks.

### Candidate-Level Evidence Synthesis

- `Best-supported claim`: Explicit articulator-based disentanglement is a directly supported and scientifically defensible structure-aware candidate in gloss-free sign language production.
- `Least-secure claim`: A more structured latent or decoder partition will necessarily outperform simpler channel-aware interventions such as Channel-Aware Loss Reweighting in this repository’s exact setup.
- `Evidence confidence rationale`: The candidate has one strong direct source and two strong supporting sources for the broader multi-channel rationale. The main remaining uncertainty is not whether explicit articulator structure is defensible, but how much added complexity it will introduce relative to simpler structure-aware alternatives.

## Comparison To Strong Alternatives

Relative to channel-aware loss reweighting, this candidate is more expressive but higher-risk;
relative to discrete/data-driven candidates, it attacks structural/channel fidelity rather than
representation discretization; relative to retrieval/stitching or diffusion, it remains closer to
the current baseline and is easier to interpret as a second thesis contribution.

## Minimum Ablation Plan

Base vs Base+ArticulatorStructure; compare simpler vs stronger partitioning variants internally;
compare Base+ArticulatorStructure vs Base+C1+ArticulatorStructure if later combined; inspect
hand/body coordination and cross-channel stability qualitatively in addition to metric changes.

## Position In The Four-Model Matrix

- `fits_as_C1`: Established: no in the current research-role framing. This candidate is documented
  as a `C2`-family structure-aware path rather than as a discrete/data-driven `C1` path.
- `fits_as_C2`: Conditional: yes. The current audit record supports this candidate as a plausible
  `C2`-family fit if an independent `Base + C2` implementation can be defined cleanly.
- `fits_joint_model`: Conditional: joint-model compatibility with a discrete/data-driven `C1`
  candidate appears plausible but remains to be tested.
- `breaks_factorial_design`: Not currently indicated, although the current audit record notes
  higher architectural entanglement risk than in the channel-aware-loss candidate.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium implementation complexity.
- `Failure cost (days or bucket)`: Moderate failure cost if wrong.
- `Hidden dependency risk`: Medium-high because latent partitioning and decoder structure may
  interact with baseline behavior in non-obvious ways.
- `Scope-creep risk`: Medium.

## Record Boundary Note

- This document records candidate-level evidence only.
- [Scorecards](../scorecards/index.md) and
  [Selection Decisions](../selection-decisions/index.md) are maintained in their dedicated audit
  surfaces.
- No final selection outcome is recorded in this document.
