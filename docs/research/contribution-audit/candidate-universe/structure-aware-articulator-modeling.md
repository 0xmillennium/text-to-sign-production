# Structure-Aware and Articulator-Aware Modeling

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING`

## Family Name

Structure-Aware and Articulator-Aware Modeling

## Family Type / Status

`primary_candidate_family`

This is a candidate-generating method family, but this record does not instantiate or select any
concrete candidate.

## Scope Definition

This family covers methods that explicitly model the structure of signing bodies and articulators.
It includes body, hand, face, and non-manual channel separation; articulator-wise latent groups;
channel-aware regularization; multi-channel spatial-temporal attention; bone, relative-pose, or
skeletal-structure-aware representations; non-manual feature and face/body/hand modeling pressure;
and modality-specific body/hand/face tokenization when the relevant issue is articulator structure
rather than tokenization itself.

The family focuses on the structure question: how should the model preserve sign-relevant body,
hand, face, channel, and articulator structure? It does not collapse structure-aware modeling into
learned pose/motion representation, latent generative production, text-pose semantic alignment,
evaluation, retrieval/stitching, or 3D avatar production; those are separate family records.

## Included Directions / Subfamilies

- Body/hand/face channel separation.
- Articulator-wise latent representation.
- Channel-aware regularization or reweighting.
- Multi-channel spatial-temporal attention.
- Skeleton or bone-structure-aware modeling.
- Non-manual feature-aware modeling.
- Modality-specific hand/body/face modeling when used to preserve linguistic structure.
- Foundational multi-channel baseline evidence.
- Gloss-dependent structure-aware sources used only as boundary evidence.

## Explicit Exclusions / Out-of-Scope Directions

- Learned tokenization where the main contribution is representation learning rather than
  articulator structure.
- Pure latent diffusion or generative architecture where the main contribution is generation
  mechanism rather than structure preservation.
- Pure text-pose semantic alignment without explicit articulator/channel structure.
- Pure evaluation methodology.
- Retrieval/stitching systems that preserve source motion but do not model articulator/channel
  structure as a learned family.
- 3D avatar/rendering systems where the main contribution is rendering rather than structure-aware
  generation.
- Gloss-dependent structure-aware methods treated as direct support for the project's
  no-public-gloss data regime.
- Recognition-only or sign-to-text-only systems.

Gloss-dependent structure-aware papers are not excluded categorically. They may be used as
boundary evidence when they clarify skeleton, bone, alignment, or articulator design, but not as
direct compatibility evidence without adaptation rationale.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to sign-pose/keypoint/motion production,
  explicit modeling of articulator/channel/skeleton structure, gloss-free or gloss-adaptable
  supervision, method extractability, dataset compatibility, evaluation relevance, and downstream
  contribution-selection relevance.
- `Evidence classes represented`: `direct_task_evidence`, `representation_evidence`,
  `architecture_evidence`, `evaluation_evidence`, and `background_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct support when the structure can map to the project's pose/keypoint schema.
  `cross_lingual_transferable` sources may support the family when articulator/channel transfer is
  methodologically clear.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  family support. `gloss_dependent` sources are boundary evidence only unless a gloss-free
  adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-DARSLP-2025`
- `Source link`: [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025)
- `Relevance type`: Core articulator-disentanglement and channel-aware regularization evidence.
- `Claim supported`: Articulator-specific latent structure and channel-aware regularization are
  credible structure-aware SLP directions.
- `Evidence note`: DARSLP is recorded as a core source for articulator-disentangled latent spaces
  and channel-aware regularization.
- `Interpretation boundary`: This family records structure-aware evidence only; it does not
  select a concrete disentanglement or regularization candidate.

### Source 2

- `Source corpus entry`: `SRC-MCST-2024`
- `Source link`: [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024)
- `Relevance type`: Multi-channel spatial-temporal modeling evidence.
- `Claim supported`: Channel-aware spatial-temporal modeling is a credible architecture-level
  response to the structure of sign pose.
- `Evidence note`: MCST is recorded as supporting channel-aware architectural modeling across
  manual and non-manual articulators.
- `Interpretation boundary`: Dataset transfer must be argued because the source is recorded as
  `cross_lingual_transferable`, not How2Sign-direct.

### Source 3

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Relevance type`: Variational articulator-wise latent modeling evidence.
- `Claim supported`: Articulator-wise latent modeling can be extended with distributional or
  variational structure.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: This is recent frontier evidence and should not automatically outrank
  earlier reviewed structure-aware sources.

### Source 4

- `Source corpus entry`: `SRC-ILRSLP-2025`
- `Source link`: [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025)
- `Relevance type`: Iterative latent refinement over structured articulator-aware spaces.
- `Claim supported`: Structure-aware latent spaces can be refined iteratively rather than
  predicted once.
- `Evidence note`: ILRSLP is recorded as an important continuation of articulator-disentangled
  latent modeling.
- `Interpretation boundary`: It should be grouped with DARSLP/A²V-style structure-aware latent
  modeling rather than treated as an entirely separate family.

### Source 5

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Relevance type`: Frontier non-manual and modality-specific structure evidence.
- `Claim supported`: Recent work pressures structure-aware modeling toward body, hands, and
  face/non-manual separation.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a heavy 3D or parametric representation.

### Source 6

- `Source corpus entry`: `SRC-SIGNIDD-2025`
- `Source link`: [SRC-SIGNIDD-2025](../../source-corpus.md#src-signidd-2025)
- `Relevance type`: Gloss-dependent bone/relative-pose boundary evidence.
- `Claim supported`: Bone direction, bone length, or relative-pose structure may be technically
  relevant to preserving skeletal structure.
- `Evidence note`: Sign-IDD is recorded as useful for skeletal structure and bone-direction/length
  representation, but gloss-to-pose limits direct use.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 7

- `Source corpus entry`: `SRC-MULTICHANNEL-MDN-2021`
- `Source link`: [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Relevance type`: Foundational multi-channel baseline evidence.
- `Claim supported`: Multi-channel 3D pose production is part of the baseline structure-aware
  context.
- `Evidence note`: The source is recorded as an extended multi-channel baseline.
- `Interpretation boundary`: It is foundational baseline evidence, not direct evidence for a
  modern structure-aware contribution candidate by itself.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Structure-aware modeling is directly compatible when body, hand,
  face, channel, skeleton, or articulator structure is derived from pose/keypoint/motion data and
  does not require manual gloss annotations.
- `Gloss-dependent risks`: Gloss-dependent structure-aware methods may inform skeleton, bone,
  alignment, or articulator design, but they must not be treated as direct support without a
  gloss-free adaptation path.
- `Intermediate representation type`: This family may use articulator-specific latent groups,
  channel-specific features, skeleton partitions, bone/relative-pose structures, or
  body/hand/face-specific tokens when the primary issue is preserving structure.
- `Pose/keypoint/skeleton/3D compatibility`: 2D/3D keypoints and skeleton partitions are directly
  relevant. Heavy parametric body/face models may be frontier pressure when they exceed the
  project's immediate pose/keypoint regime.

## Method-Family Rationale

This family exists because sign production is not a flat coordinate-regression problem. It
prevents three common errors: treating all joints or keypoints as an undifferentiated vector,
reducing structure-aware modeling to a minor loss-weighting detail, and ignoring non-manual or
face/body/hand structure when evaluating production quality.

## Counter-Evidence And Limitations

- `Known limitations`: Structure-aware modeling may require reliable body/hand/face partitioning,
  stable keypoint schemas, additional losses, or more complex training objectives.
- `Conflicting evidence`: Some structure-aware evidence is `cross_lingual_transferable`, some is
  frontier/preprint evidence, and some technically relevant skeletal-structure sources are
  `gloss_dependent`.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if the available
  keypoints do not support reliable articulator partitions, if the added structure does not
  improve evaluation, or if complexity exceeds thesis scope.

## Potential Candidate Directions

This family may later produce candidate-level records such as articulator-wise latent grouping,
channel-aware regularization, multi-channel spatial-temporal attention, body/hand/face-specific
modeling, skeleton or bone-structure-aware auxiliary loss, lightweight non-manual feature-aware
modeling, or structure-preserving pose normalization or partitioning. Candidate cards must not be
instantiated here.

## Relation To The Overall Audit

This family is a primary candidate-family source for structure-level contributions. It should help
later audit records distinguish representation choice, structure-aware modeling, generation
architecture, semantic alignment, evaluation design, and frontier 3D/avatar modeling.

## Family-Level Evidence Boundary

This family does not select a concrete structure-aware candidate. It does not rank articulator
disentanglement, channel attention, bone representation, or non-manual feature modeling. It does
not claim that every structure-aware method is compatible with How2Sign, and it does not convert
gloss-dependent structure-aware sources into direct support. It only defines the structure-aware
and articulator-aware modeling universe for later candidate records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish structure-aware evidence from learned-representation evidence,
distinguish structure-aware evidence from latent-generation evidence, state whether the structure
source is `how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`, explicitly
classify `gloss_free`, `gloss_adaptable`, or `gloss_dependent` assumptions, provide adaptation
rationale before using gloss-dependent or non-How2Sign structure-aware sources as direct support,
and specify whether the proposed structure can be represented with the project's available
pose/keypoint artifacts.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
