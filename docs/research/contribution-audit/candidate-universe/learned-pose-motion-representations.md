# Learned Pose/Motion Representations

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-LEARNED-POSE-MOTION-REPRESENTATIONS`

## Family Name

Learned Pose/Motion Representations

## Family Type / Status

`primary_candidate_family`

This is a candidate-generating method family, but this record does not instantiate or select any
concrete candidate.

## Scope Definition

This family covers methods that learn intermediate representations from sign pose, keypoints,
skeleton, motion, or latent signing structure. It includes learned discrete pose tokens,
vector-quantized pose or motion codes, dynamic vector quantization, code-duration or
variable-length token representations, data-driven learned codebooks, gloss-alternative learned
spatio-temporal representations, modality-specific body/hand/face motion tokens, and learned
pose/motion units that are not manually annotated gloss.

The family focuses on the representation question: what intermediate representation should the
model generate or condition on? It does not collapse representation design into latent generative
architecture, structure-aware articulator modeling, semantic alignment, or 3D avatar production;
those are separate family records.

## Included Directions / Subfamilies

- Fixed-length learned pose tokens.
- Dynamic or variable-length pose/motion tokenization.
- VQ or VQ-like codebook representations.
- Finite scalar quantization or modality-specific motion tokens.
- Data-driven gloss-alternative representations.
- Code-duration generation.
- Learned non-gloss intermediate representations derived from keypoints or motion.
- Gloss-dependent discrete representation sources used only as boundary evidence.

## Explicit Exclusions / Out-of-Scope Directions

- Manually annotated gloss as the representation itself.
- HamNoSys or notation-dependent representations as direct project-compatible representation
  choices.
- Pure latent diffusion or generative architecture where learned representation is not the family
  focus.
- Structure-aware articulator/channel modeling when the primary contribution is channel
  separation, disentanglement, or regularization rather than learned token representation.
- Pure evaluation methodology.
- Retrieval/stitching systems that reuse stored signs rather than learning a pose/motion
  representation.
- 3D avatar/rendering systems where the main contribution is rendering or parametric avatar
  generation rather than learned pose/motion representation.
- Recognition-only or sign-to-text-only representations.

Gloss-dependent representation papers are not excluded categorically. They may be used as boundary
evidence when they clarify representation design, but not as direct compatibility evidence without
adaptation rationale.

## Source-Selection Basis

- `Relevant source-selection criteria`: Representation relevance, relation to
  sign-pose/keypoint/motion production, gloss-free or gloss-adaptable supervision, method
  extractability, dataset compatibility, and downstream contribution-selection relevance.
- `Evidence classes represented`: `direct_task_evidence`, `representation_evidence`,
  `architecture_evidence`, `dataset_evidence`, `evaluation_evidence`, and
  `reproducibility_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` sources provide strongest direct support.
  `how2sign_compatible` and `cross_lingual_transferable` sources may support the family when
  representation transfer is methodologically clear.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  family support. `gloss_dependent` sources are boundary evidence only unless a gloss-free
  adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Relevance type`: Core gloss-free discrete representation evidence.
- `Claim supported`: Learned discrete pose-token production is a credible gloss-free
  representation family.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This family records representation-level evidence only; it does not
  select a concrete tokenization candidate.

### Source 2

- `Source corpus entry`: `SRC-DATA-DRIVEN-REP-2024`
- `Source link`: [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024)
- `Relevance type`: Data-driven learned representation evidence.
- `Claim supported`: Learned pose/motion representations can reduce dependence on scarce
  linguistic annotation.
- `Evidence note`: The source is recorded as core evidence that learned pose/motion
  representations can replace scarce linguistic annotation.
- `Interpretation boundary`: Transfer to the project still requires compatibility with the
  available pose/keypoint schema.

### Source 3

- `Source corpus entry`: `SRC-T2S-GPT-2024`
- `Source link`: [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Relevance type`: Dynamic VQ and text-to-code generation evidence.
- `Claim supported`: Variable-length or dynamic sign tokenization is a serious representation
  direction.
- `Evidence note`: T2S-GPT is recorded as core evidence for dynamic variable-length sign
  tokenization and autoregressive text-to-token generation.
- `Interpretation boundary`: Dataset transfer must be argued because the source is recorded as
  `cross_lingual_transferable`, not How2Sign-direct.

### Source 4

- `Source corpus entry`: `SRC-UNIGLOR-2024`
- `Source link`: [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024)
- `Relevance type`: Gloss-alternative learned representation evidence.
- `Claim supported`: Learned keypoint-derived representations can serve as non-gloss intermediate
  representations.
- `Evidence note`: UniGloR is recorded as central evidence for non-gloss learned intermediate
  representations from keypoints.
- `Interpretation boundary`: Learned intermediate representation must not be conflated with
  manually annotated gloss.

### Source 5

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Relevance type`: Frontier modality-specific motion-token evidence.
- `Claim supported`: Recent work pressures representation design toward separate body, hands, and
  face/non-manual motion tokenization.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a heavy 3D or parametric representation.

### Source 6

- `Source corpus entry`: `SRC-G2PDDM-2024`
- `Source link`: [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Relevance type`: Gloss-dependent discrete diffusion boundary evidence.
- `Claim supported`: Discrete pose representations and diffusion over codes are technically
  relevant, but gloss-to-pose assumptions limit direct compatibility.
- `Evidence note`: G2P-DDM is recorded as valuable for VQ/discrete diffusion design but not
  directly compatible with gloss-free How2Sign assumptions.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Learned representations are directly compatible when they are
  derived from pose, keypoints, skeleton, motion, or latent signing structure and do not require
  manual gloss annotation.
- `Gloss-dependent risks`: Gloss-dependent token or code-generation methods may inform
  representation design but must not be treated as direct support without a gloss-free adaptation
  path.
- `Intermediate representation type`: This family includes learned discrete tokens, VQ codes,
  dynamic token sequences, code-duration representations, finite scalar quantization, learned
  keypoint-derived representations, and modality-specific motion tokens.
- `Pose/keypoint/skeleton/3D compatibility`: 2D/3D pose, keypoint, skeleton, and motion-token
  representations are compatible when they can be derived from or mapped to the project's
  pose/keypoint regime. Heavy 3D parametric representations are frontier pressure, not automatic
  implementation scope.

## Method-Family Rationale

This family exists because modern SLP is no longer limited to direct regression from text to raw
continuous pose. It prevents three common errors: treating gloss-free production as requiring
direct text-to-raw-pose regression, treating learned pose/motion tokens as if they were linguistic
gloss, and ignoring the difference between representation design and generation architecture.

## Counter-Evidence And Limitations

- `Known limitations`: Learned representations may introduce reconstruction loss, token collapse,
  temporal discontinuity, codebook instability, or mismatch with the project's available keypoint
  schema.
- `Conflicting evidence`: Some strong representation sources are `cross_lingual_transferable`
  rather than `how2sign_direct`, and some technically relevant discrete sources are
  `gloss_dependent`.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if learned
  tokens are too expensive to train, reconstruction quality is poor, tokenization does not preserve
  sign-relevant detail, or evaluation cannot isolate representation-level gains.

## Potential Candidate Directions

This family may later produce candidate-level records such as fixed learned pose-token
representation, dynamic VQ pose/motion tokens, code-duration sign token generation,
gloss-alternative keypoint-derived representation, modality-specific body/hand/face tokenization,
or lightweight learned motion-code representation. Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is a primary candidate-family source for representation-level contributions. It should
help later audit records distinguish representation choice, generation architecture,
structure-aware modeling, semantic alignment, evaluation design, and downstream 3D/avatar
rendering.

## Family-Level Evidence Boundary

This family does not select a concrete representation candidate. It does not rank fixed VQ,
dynamic VQ, FSQ, learned gloss alternatives, or modality-specific tokens. It does not claim that
every learned representation is compatible with How2Sign, and it does not convert gloss-dependent
representation sources into direct support. It only defines the learned pose/motion representation
universe for later candidate records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish representation evidence from generation-architecture evidence, state
whether the representation is `how2sign_direct`, `how2sign_compatible`, or
`cross_lingual_transferable`, explicitly classify `gloss_free`, `gloss_adaptable`, or
`gloss_dependent` assumptions, provide adaptation rationale before using gloss-dependent or
non-How2Sign representation sources as direct support, and specify whether a proposed
representation can be trained and evaluated with the project's available pose/keypoint artifacts.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
