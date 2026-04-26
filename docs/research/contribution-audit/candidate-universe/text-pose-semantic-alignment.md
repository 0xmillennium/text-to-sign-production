# Text-Pose Semantic Alignment and Conditioning

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-TEXT-POSE-SEMANTIC-ALIGNMENT`

## Family Name

Text-Pose Semantic Alignment and Conditioning

## Family Type / Status

`primary_candidate_family`

This is a candidate-generating method family, but this record does not instantiate or select any
concrete candidate.

## Scope Definition

This family covers methods that explicitly improve semantic coupling between spoken-language input
and produced sign pose or motion. It includes cross-modal text-pose latent alignment, pose-text or
sign-text contrastive alignment, embedding consistency across text, audio, and sign
representations, text-conditioned pose or latent conditioning mechanisms, auxiliary translation
objectives used to ground motion tokens or latent representations, monotonic or fine-grained
linguistic-visual alignment as boundary evidence when it depends on gloss, gloss attention or
alignment biases adapted to gloss-free settings, and semantic grounding mechanisms that reduce
drift between input sentence meaning and generated motion.

The family focuses on the semantic-conditioning question: how should the model keep generated sign
motion semantically aligned with the source text? It does not collapse semantic alignment into
learned pose/motion representation, latent generative production, structure-aware articulator
modeling, evaluation methodology, retrieval/stitching, or 3D avatar production; those are separate
family records.

## Included Directions / Subfamilies

- Cross-modal text-pose aligners.
- Contrastive pose/text latent-space alignment.
- Embedding consistency learning across text, audio, and sign.
- Auxiliary translation objectives for semantic grounding.
- Text-conditioned latent or token generation where the alignment mechanism is the family focus.
- Gloss-free alignment biases.
- Gloss-pose or monotonic alignment sources used only as boundary evidence.
- Multimodal conditioning sources used when they clarify semantic grounding rather than audio
  modeling itself.

## Explicit Exclusions / Out-of-Scope Directions

- Learned tokenization when the main contribution is representation design rather than semantic
  alignment.
- Latent diffusion or generative architecture when the main contribution is the generation
  mechanism rather than alignment or conditioning.
- Structure-aware articulator/channel modeling when the main contribution is preserving
  body/hand/face structure rather than semantic grounding.
- Pure evaluation methodology.
- Retrieval/stitching systems where semantic matching is not a learned alignment or conditioning
  mechanism.
- Gloss-dependent alignment methods treated as direct support for the project's no-public-gloss
  data regime.
- 3D avatar/rendering systems where the main contribution is rendering rather than semantic
  alignment.
- Recognition-only or sign-to-text-only systems.

Gloss-dependent alignment papers are not excluded categorically. They may be used as boundary
evidence when they clarify the alignment problem, but not as direct compatibility evidence without
adaptation rationale.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to text-to-sign-pose production, semantic
  conditioning or alignment relevance, gloss-free or gloss-adaptable supervision, method
  extractability, dataset compatibility, evaluation relevance, and downstream
  contribution-selection relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `evaluation_evidence`, `dataset_evidence`, and `negative_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct support when their semantic alignment method maps to the project's
  text/pose/keypoint regime. `cross_lingual_transferable` sources may support the family when
  alignment-mechanism transfer is methodologically clear.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  family support. `gloss_dependent` sources are boundary evidence only unless a gloss-free
  adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Relevance type`: Core gloss-free cross-modal alignment evidence.
- `Claim supported`: Cross-modal text-pose alignment can support gloss-free text-to-sign-pose
  generation.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: Its diffusion mechanism belongs to the latent generative family;
  this record uses it only for semantic alignment and conditioning relevance.

### Source 2

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Relevance type`: Multimodal embedding-consistency evidence.
- `Claim supported`: Text, audio, and sign representations can provide useful cross-modal
  conditioning signal for production.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Audio-conditioning is relevant only when it clarifies semantic
  conditioning for text-to-pose; it does not change the project's primary text-to-pose framing.

### Source 3

- `Source corpus entry`: `SRC-LVMCN-2024`
- `Source link`: [SRC-LVMCN-2024](../../source-corpus.md#src-lvmcn-2024)
- `Relevance type`: Gloss-dependent monotonic alignment boundary evidence.
- `Claim supported`: Fine-grained linguistic-visual alignment is a recognized problem in sign
  language production.
- `Evidence note`: LVMCN is recorded as showing that alignment matters, but its solution depends
  on gloss-pose semantic alignment.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 4

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Relevance type`: Alignment-aware variational modeling evidence.
- `Claim supported`: Semantic alignment can be integrated with variational articulator-wise latent
  modeling.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: Its variational and articulator-wise aspects belong to
  latent-generative and structure-aware families; this record uses it only for alignment
  relevance.

### Source 5

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Relevance type`: Auxiliary translation / semantic grounding frontier evidence.
- `Claim supported`: Motion-token spaces may require auxiliary semantic grounding to preserve
  linguistic meaning.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: Its tokenization and 3D/parametric aspects belong to
  learned-representation and frontier families; this record uses it only for semantic grounding
  pressure.

### Source 6

- `Source corpus entry`: `SRC-T2S-GPT-2024`
- `Source link`: [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Relevance type`: Text-to-code conditioning evidence.
- `Claim supported`: Text-conditioned code or token generation requires semantic conditioning
  between input sentence and generated sign representation.
- `Evidence note`: T2S-GPT is recorded as core evidence for dynamic variable-length sign
  tokenization and autoregressive text-to-token generation.
- `Interpretation boundary`: Its primary family role remains learned representation; this record
  uses it only for text-conditioned generation relevance.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Semantic alignment methods are directly compatible when they align
  text/transcripts with pose, keypoints, skeletons, motion tokens, or learned latent sign
  representations without requiring manual gloss annotations.
- `Gloss-dependent risks`: Gloss-pose or gloss-conditioned alignment sources may clarify
  alignment problems but must not be treated as direct support without a gloss-free adaptation
  path.
- `Intermediate representation type`: This family may use text embeddings, pose embeddings,
  cross-modal latent spaces, contrastive text-pose representations, embedding-consistency
  objectives, or semantic grounding objectives when the primary issue is alignment.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton representations are directly
  relevant when they can be aligned with text or transcript embeddings. Heavy 3D parametric
  systems may be frontier evidence when they exceed the project's immediate pose/keypoint regime.

## Method-Family Rationale

This family exists because generation quality is not only a motion-quality problem; generated
motion must remain semantically aligned with the source text. It prevents three common errors:
treating semantic conditioning as an incidental implementation detail, treating learned
representation or latent generation as sufficient without checking text-motion alignment, and
treating gloss-dependent alignment methods as direct support for a no-public-gloss project.

## Counter-Evidence And Limitations

- `Known limitations`: Semantic alignment methods may require reliable paired text-pose data,
  strong pretrained encoders, contrastive objectives, auxiliary losses, or additional evaluation
  signals.
- `Conflicting evidence`: Some alignment evidence is `how2sign_direct`, some is
  `how2sign_compatible`, some is `cross_lingual_transferable`, some is frontier/preprint
  evidence, and some technically relevant alignment sources are `gloss_dependent`.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if alignment
  gains cannot be isolated, if pretrained embeddings do not transfer to signing semantics, if
  semantic grounding does not improve pose quality, or if evaluation cannot measure alignment
  reliably.

## Potential Candidate Directions

This family may later produce candidate-level records such as cross-modal text-pose latent
alignment, contrastive pose/text representation learning, embedding-consistency regularization,
auxiliary translation objective for motion tokens, gloss-free alignment bias for latent
generation, semantic drift reduction objective, or lightweight text-conditioning adapter.
Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is a primary candidate-family source for semantic-alignment or conditioning
contributions. It should help later audit records distinguish representation choice, latent
generative architecture, structure-aware modeling, semantic conditioning, evaluation design,
retrieval/stitching alternatives, and frontier 3D/avatar modeling.

## Family-Level Evidence Boundary

This family does not select a concrete semantic-alignment candidate. It does not rank contrastive
alignment, embedding consistency, auxiliary translation objectives, or gloss-free alignment bias.
It does not claim that every alignment method is compatible with How2Sign, and it does not convert
gloss-dependent alignment sources into direct support. It only defines the text-pose semantic
alignment and conditioning universe for later candidate records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish semantic-alignment evidence from representation evidence, distinguish
semantic-alignment evidence from latent-generation evidence, distinguish semantic-alignment
evidence from structure-aware modeling evidence, state whether the alignment source is
`how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`, explicitly classify
`gloss_free`, `gloss_adaptable`, or `gloss_dependent` assumptions, provide adaptation rationale
before using gloss-dependent or non-How2Sign alignment sources as direct support, and specify
whether the proposed alignment method can be trained and evaluated with the project's available
text/pose/keypoint artifacts.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
