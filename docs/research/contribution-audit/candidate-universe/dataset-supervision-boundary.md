# Dataset and Supervision Boundary

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-DATASET-SUPERVISION-BOUNDARY`

## Family Name

Dataset and Supervision Boundary

## Family Type / Status

`boundary_family`

This is not a candidate-generating method family. It defines the supervision, dataset, modality,
and compatibility boundary that later candidate-universe and candidate-card records must respect.

## Scope Definition

This family defines the boundary for the project's practical research data regime: English
text/transcript input, sign video, and extracted pose or keypoint supervision. How2Sign-style
compatibility is the primary practical anchor, but it is not an absolute exclusion rule. The
project does not assume access to public manual gloss annotations and is gloss-free at the
linguistic-supervision level.

Non-gloss intermediate representations may still be compatible when they are derived from pose,
motion, keypoints, articulator structure, or learned latent structure. Sources from PHOENIX14T,
CSL-Daily, DGS, CSL, or other datasets may remain relevant when their method, representation,
architecture, evaluation, or supervision evidence transfers clearly to the project's text, video,
pose, or keypoint regime.

Speech/audio-conditioned sources may inform this boundary when they also support or clarify
text-conditioned production, but they do not change the project's primary text-to-pose framing.

## Included Directions / Subfamilies

- How2Sign-direct dataset and modeling evidence.
- How2Sign-compatible text/video/pose/keypoint supervision.
- Gloss-free text-to-pose or text-to-sign-pose production.
- Gloss-adaptable methods where the transferable component does not require manual gloss
  annotations.
- Cross-lingual or cross-dataset methods that can transfer at the representation, architecture,
  evaluation, or supervision level.
- Sources that clarify the difference between linguistic gloss and pose-derived intermediate
  representation.

## Explicit Exclusions / Out-of-Scope Directions

- Recognition-only sources.
- Sign-to-text translation-only sources with no production or pose-generation relevance.
- Methods requiring manual gloss annotations with no plausible gloss-free adaptation path.
- Avatar/rendering-only sources with no relevance to the text/video/pose supervision regime.
- Private-dataset-only claims with no transferable method detail.
- Generic human-pose or co-speech gesture sources without sign-language production transfer.

Gloss-based sources are not excluded categorically. They must be classified as boundary evidence,
adaptation-risk evidence, or indirect transferable evidence rather than treated as directly
compatible by default.

## Source-Selection Basis

- `Relevant source-selection criteria`: Task relevance, accessible full technical source,
  extractable method/dataset/evaluation details, relation to the text/video/pose/keypoint regime,
  gloss-free or gloss-adaptable supervision, and downstream research decision relevance.
- `Evidence classes represented`: `dataset_evidence`, `direct_task_evidence`,
  `representation_evidence`, `architecture_evidence`, `evaluation_evidence`, and
  `reproducibility_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` sources are strongest for this family.
  `how2sign_compatible` and `cross_lingual_transferable` sources may be included when method
  transfer is explicit.
- `Gloss dependency boundary`: `gloss_free` and `gloss_optional` sources are directly compatible.
  `gloss_adaptable` sources require adaptation rationale. `gloss_dependent` sources are boundary
  evidence only unless a gloss-free adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-HOW2SIGN-2021`
- `Source link`: [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Relevance type`: Dataset anchor.
- `Claim supported`: How2Sign defines the project's primary ASL text/video/keypoint data regime.
- `Evidence note`: How2Sign is recorded as the dataset anchor and practical ASL
  text/video/keypoint data-regime source.
- `Interpretation boundary`: The project must not assume public manual gloss annotations merely
  because a dataset source discusses gloss-related annotations.

### Source 2

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Relevance type`: How2Sign-direct modeling evidence.
- `Claim supported`: How2Sign can support recent text/audio-to-keypoint production models under a
  gloss-free setting.
- `Evidence note`: MS2SL is recorded as a core How2Sign-direct multimodal production source.
- `Interpretation boundary`: Audio-conditioning is relevant but does not change the current
  project's primary text-to-pose focus.

### Source 3

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Relevance type`: Gloss-free How2Sign-compatible representation evidence.
- `Claim supported`: Learned discrete pose-token production can be represented as gloss-free and
  How2Sign-direct.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This family records data/supervision compatibility only; it does not
  select a discrete-token contribution.

### Source 4

- `Source corpus entry`: `SRC-UNIGLOR-2024`
- `Source link`: [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024)
- `Relevance type`: Gloss-alternative intermediate representation evidence.
- `Claim supported`: Non-gloss learned intermediate representations from keypoints can be
  compatible with the project's supervision boundary.
- `Evidence note`: UniGloR is recorded as central evidence for non-gloss learned intermediate
  representations from keypoints.
- `Interpretation boundary`: Learned intermediate representation must not be conflated with
  manually annotated gloss.

### Source 5

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Relevance type`: How2Sign-direct gloss-free generative modeling evidence.
- `Claim supported`: Gloss-free latent generative production may be compatible with How2Sign.
- `Evidence note`: Text2SignDiff is recorded as a core source directly evaluating on How2Sign.
- `Interpretation boundary`: This family only records compatibility; architecture selection
  belongs to later candidate records.

### Source 6

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Relevance type`: Frontier How2Sign-direct representation boundary.
- `Claim supported`: The project's dataset boundary should account for body, hands, and
  face/non-manual representation pressure in recent work.
- `Evidence note`: M3T is recorded as frontier/core evidence for body, hands, and face
  tokenization.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a heavy 3D parametric pipeline.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: This family treats gloss-free sources as directly compatible when
  they map spoken-language text/transcripts to sign pose, keypoints, skeleton, motion tokens, or
  learned pose/motion representations without requiring manual gloss annotations.
- `Gloss-dependent risks`: Gloss-dependent sources may inform boundary or adaptation risk, but
  they must not be treated as directly compatible unless a gloss-free adaptation path is
  documented.
- `Intermediate representation type`: Pose-derived tokens, motion-derived latents,
  keypoint-derived representations, articulator channels, and other non-linguistic learned
  representations may be compatible; manually annotated gloss is not assumed.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton outputs are directly aligned
  with the project. 3D parametric/avatar systems may be relevant as frontier or downstream
  evidence but can exceed immediate implementation scope.

## Method-Family Rationale

This boundary family exists because method families cannot be evaluated fairly unless their data
and supervision assumptions are first bounded. It prevents three common errors: treating
gloss-dependent methods as directly compatible with a no-public-gloss project, treating
non-How2Sign sources as irrelevant when their method transfers, and treating learned pose/motion
intermediate representations as linguistic gloss.

## Counter-Evidence And Limitations

- `Known limitations`: How2Sign-direct sources do not cover every sign language, and some strong
  sources use PHOENIX14T, CSL-Daily, DGS, CSL, or other datasets.
- `Conflicting evidence`: Some technically strong methods depend on gloss, notation, private data,
  3D parametric reconstruction, or cross-lingual assumptions.
- `Reasons this family may not become a candidate`: This is a boundary family, not a
  candidate-generating family.

## Potential Candidate Directions

This family does not instantiate candidate directions. It may constrain later directions such as
gloss-free learned pose/motion representations, gloss-free latent generation, structure-aware or
articulator-aware modeling, text-pose semantic alignment, and evaluation design. Candidate cards
must not be instantiated here.

## Relation To The Overall Audit

This family is upstream of all model-family records. It defines whether later candidate families
are compatible with the project's available supervision and dataset regime before those families
are evaluated as concrete contribution possibilities.

## Family-Level Evidence Boundary

This family does not prove that any modeling family is selected. It does not rank candidate
families, claim How2Sign is the only valid dataset, or convert gloss-dependent sources into direct
support. It only defines dataset and supervision compatibility boundaries for later audit records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, state whether their representative sources are `how2sign_direct`,
`how2sign_compatible`, or `cross_lingual_transferable`, explicitly label `gloss_free`,
`gloss_adaptable`, or `gloss_dependent` assumptions, and provide adaptation rationale before using
gloss-dependent or non-How2Sign sources as direct support.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
