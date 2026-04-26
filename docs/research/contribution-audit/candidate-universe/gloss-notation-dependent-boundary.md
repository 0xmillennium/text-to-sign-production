# Gloss/Notation-Dependent Pose Generation Boundary

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY`

## Family Name

Gloss/Notation-Dependent Pose Generation Boundary

## Family Type / Status

`boundary_family`

This is not a candidate-generating method family. It records technically relevant but
supervision-constrained sources so later audit records do not treat gloss-dependent or
notation-dependent evidence as directly compatible by default.

## Scope Definition

This family covers sign language production methods whose core assumptions depend on glosses,
HamNoSys, notation, dictionary labels, isolated-sign units, gloss-to-pose input, or gloss-pose
semantic alignment. It includes text-to-gloss-to-pose or gloss-intermediate historical pipelines,
gloss-to-pose generation, discrete diffusion or VQ methods whose input is gloss, HamNoSys or
notation-to-pose systems, gloss-pose monotonic or semantic alignment methods, gloss-supervised
motion-primitives systems, gloss-dependent retrieval, stitching, dictionary, or isolated-sign
systems, and structure-aware or bone-aware methods that are technically useful but
gloss-conditioned.

The family focuses on the supervision-boundary question: which technically valuable methods rely
on symbolic linguistic or notation supervision that the current project cannot assume as public
training data? It does not collapse gloss/notation-dependent boundary evidence into learned
pose/motion representation, latent generative production, text-pose semantic alignment,
retrieval/stitching, evaluation methodology, or dataset-supervision boundary; those are separate
family records.

## Included Directions / Subfamilies

- Gloss-dependent historical SLP pipelines.
- Gloss-to-pose production.
- Gloss-conditioned discrete diffusion.
- HamNoSys or notation-to-pose production.
- Gloss-pose semantic alignment.
- Gloss-supervised motion primitives.
- Gloss-dependent sign stitching or retrieval.
- Bone/relative-pose or skeletal-structure ideas that are technically useful but
  gloss-conditioned.
- Boundary evidence for adaptation risk.

## Explicit Exclusions / Out-of-Scope Directions

- Gloss-free text-to-pose sources that do not require manual gloss annotations.
- Learned pose/motion representations derived from keypoints or motion rather than manual gloss.
- Latent generative methods whose core training path is gloss-free.
- Structure-aware modeling that can be derived from pose/keypoint partitions without gloss.
- Semantic alignment methods that operate without gloss-pose supervision.
- Pure evaluation methodology.
- Recognition-only or sign-to-text-only systems.
- Treating notation-dependent methods as directly compatible with a text/transcript-to-pose
  project without a notation-generation or adaptation path.

These sources are not excluded from the corpus. Their purpose here is to define limits, adaptation
risks, and transferable technical ideas.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to sign-pose/keypoint/motion production,
  boundary relevance, extractable method or representation detail, gloss or notation dependency,
  dataset compatibility, evaluation relevance, and downstream audit relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `evaluation_evidence`, `negative_evidence`, and
  `background_evidence`.
- `Dataset compatibility boundary`: These sources are usually `cross_lingual_transferable` or
  supervision-boundary evidence. They may inform method design, but direct compatibility requires
  explicit mapping to the project's text/video/pose/keypoint regime.
- `Gloss dependency boundary`: `gloss_dependent` and notation-dependent sources are boundary
  evidence only unless a no-gloss or notation-free adaptation path is documented. HamNoSys or
  notation dependence should be treated separately from gloss dependence but with similar
  compatibility caution.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGN-2020`
- `Source link`: [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020)
- `Relevance type`: Historical gloss-dependent SLP boundary evidence.
- `Claim supported`: Historical SLP pipelines often used gloss or staged text-to-pose/video
  assumptions.
- `Evidence note`: Text2Sign is recorded as a foundational gloss-dependent text-to-pose/video
  pipeline.
- `Interpretation boundary`: It must not be treated as direct support for the project's
  no-public-gloss data regime.

### Source 2

- `Source corpus entry`: `SRC-G2PDDM-2024`
- `Source link`: [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Relevance type`: Gloss-to-pose discrete diffusion boundary evidence.
- `Claim supported`: Discrete pose representations and diffusion over pose codes can be
  technically relevant while remaining gloss-to-pose constrained.
- `Evidence note`: G2P-DDM is recorded as valuable for VQ/discrete diffusion design but not
  directly compatible with gloss-free How2Sign assumptions.
- `Interpretation boundary`: This source requires gloss input and must remain boundary evidence
  unless a gloss-free adaptation path is documented.

### Source 3

- `Source corpus entry`: `SRC-HAM2POSE-2025`
- `Source link`: [SRC-HAM2POSE-2025](../../source-corpus.md#src-ham2pose-2025)
- `Relevance type`: Notation-to-pose boundary evidence.
- `Claim supported`: Notation systems can encode pose-relevant production detail, but notation
  input is outside the project's current text-to-pose data regime.
- `Evidence note`: Ham2Pose is recorded as useful for notation systems and pose-distance
  evaluation, but not as a text-to-pose path for this project.
- `Interpretation boundary`: HamNoSys dependence is not the same as gloss dependence, but it is
  still a symbolic-supervision boundary requiring an explicit notation-generation or adaptation
  path.

### Source 4

- `Source corpus entry`: `SRC-LVMCN-2024`
- `Source link`: [SRC-LVMCN-2024](../../source-corpus.md#src-lvmcn-2024)
- `Relevance type`: Gloss-pose semantic alignment boundary evidence.
- `Claim supported`: Fine-grained linguistic-visual alignment is important, but gloss-pose
  alignment solutions may be supervision-incompatible.
- `Evidence note`: LVMCN is recorded as showing that alignment matters, but its solution depends
  on gloss-pose semantic alignment.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 5

- `Source corpus entry`: `SRC-SIGNIDD-2025`
- `Source link`: [SRC-SIGNIDD-2025](../../source-corpus.md#src-signidd-2025)
- `Relevance type`: Gloss-dependent skeletal-structure boundary evidence.
- `Claim supported`: Bone direction, bone length, or relative-pose structure can be technically
  relevant while remaining gloss-to-pose constrained.
- `Evidence note`: Sign-IDD is recorded as useful for skeletal structure and bone-direction/length
  representation, but gloss-to-pose limits direct use.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 6

- `Source corpus entry`: `SRC-MIXED-SIGNALS-2021`
- `Source link`: [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021)
- `Relevance type`: Gloss-supervised motion-primitives boundary evidence.
- `Claim supported`: Motion-primitives approaches are technically relevant but may rely on gloss
  supervision.
- `Evidence note`: Mixed SIGNals is recorded as an important alternative representation family
  but not direct support for no-gloss implementation.
- `Interpretation boundary`: It should remain counter-alternative or boundary evidence unless a
  gloss-free adaptation path is documented.

### Source 7

- `Source corpus entry`: `SRC-SIGN-STITCHING-2024`
- `Source link`: [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024)
- `Relevance type`: Gloss-dependent retrieval/stitching boundary evidence.
- `Claim supported`: Retrieval and stitching can be serious production alternatives, but gloss or
  dictionary assumptions can limit compatibility.
- `Evidence note`: Sign Stitching is recorded as a serious alternative to direct regression and
  useful for explaining retrieval/stitching boundaries.
- `Interpretation boundary`: It is gloss-dependent and must not be treated as direct support for a
  no-public-gloss How2Sign-compatible implementation without adaptation rationale.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: This family is not directly gloss-free. It can support the project
  only by identifying transferable technical ideas or adaptation risks.
- `Gloss-dependent risks`: Manual gloss, gloss-to-pose input, gloss-pose alignment, dictionary
  labels, isolated-sign labels, or manually specified intermediate linguistic units are not
  assumed available in the project's public data regime.
- `Intermediate representation type`: This family may include gloss, HamNoSys, notation,
  dictionary units, gloss-conditioned pose codes, gloss-supervised motion primitives, or
  gloss-pose alignment states. These are symbolic or linguistic/notation intermediates, not
  automatically compatible learned pose/motion representations.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton outputs may be technically
  relevant, but output compatibility alone does not overcome incompatible input-supervision
  assumptions.

## Method-Family Rationale

This boundary family exists because technically useful papers may depend on supervision signals
the project cannot assume. It prevents three common errors: treating gloss-dependent methods as
directly compatible with a no-public-gloss project, treating notation-dependent methods as
text-to-pose methods without a notation-generation path, and discarding technically useful ideas
merely because their original supervision regime is incompatible.

## Counter-Evidence And Limitations

- `Known limitations`: These sources may require gloss annotations, notation inputs,
  dictionaries, isolated-sign units, manually specified symbolic intermediates, or supervised
  alignment signals unavailable in the project.
- `Conflicting evidence`: Some sources offer strong technical ideas but are `gloss_dependent`,
  notation-dependent, `cross_lingual_transferable`, or tied to datasets and supervision regimes
  that differ from How2Sign-style text/video/keypoint data.
- `Reasons this family may not become a candidate`: This is a boundary family. It may inform
  adaptation or negative evidence, but it should not become a direct candidate family unless a
  concrete gloss-free or notation-free adaptation is proposed.

## Potential Candidate Directions

This family does not instantiate candidate directions. It may constrain or inspire later
directions such as gloss-free adaptation of discrete diffusion ideas, notation-inspired pose
constraints without notation input, gloss-free semantic alignment bias, skeleton or
bone-structure constraints adapted from gloss-to-pose sources, or retrieval/stitching adaptation
without gloss dictionaries. Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is upstream boundary evidence for several other families. It should help later audit
records distinguish direct support, boundary evidence, counter-alternative evidence,
adaptation-risk evidence, transferable technical idea, and incompatible supervision assumption.

## Family-Level Evidence Boundary

This family does not select a gloss-dependent or notation-dependent candidate. It does not rank
gloss, HamNoSys, notation, gloss-to-pose, or dictionary-based approaches. It does not claim that
symbolic-supervision methods are compatible with How2Sign, and it does not convert
gloss-dependent or notation-dependent sources into direct support. It only defines the
gloss/notation-dependent boundary for later audit records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, explicitly state whether a source is gloss-free, gloss-adaptable, gloss-dependent, or
notation-dependent, distinguish output compatibility from input-supervision compatibility,
provide adaptation rationale before using gloss-dependent or notation-dependent sources as direct
support, identify whether a technical idea is being used as direct evidence, boundary evidence,
counter-alternative evidence, or transferable inspiration, and avoid treating symbolic
intermediate representations as learned pose/motion representations unless the distinction is
explicitly justified.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
