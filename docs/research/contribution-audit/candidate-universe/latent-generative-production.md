# Latent Generative Production

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-LATENT-GENERATIVE-PRODUCTION`

## Family Name

Latent Generative Production

## Family Type / Status

`primary_candidate_family`

This is a candidate-generating method family, but this record does not instantiate or select any
concrete candidate.

## Scope Definition

This family covers methods that generate sign pose, keypoints, skeletons, motion representations,
or avatar-compatible motion through latent-space or generative modeling rather than direct
frame-by-frame raw-pose regression alone. It includes latent diffusion for text-to-sign-pose or
text-to-pose production, diffusion conditioned on text, audio, or multimodal embeddings, VAE or
SignVAE-style latent spaces, non-autoregressive latent prediction, iterative latent refinement,
variational articulator-wise latent modeling when the primary issue is generative latent
prediction, conditional flow matching or sparse-keyframe generative production as frontier
evidence, and 3D diffusion or avatar-compatible latent generation as frontier or boundary
evidence.

The family focuses on the generation question: how should the model generate sign pose or motion
from text-conditioned latent structure? It does not collapse latent generative production into
learned pose/motion representation, structure-aware articulator modeling, text-pose semantic
alignment, retrieval/stitching, evaluation, or 3D avatar production; those are separate family
records.

## Included Directions / Subfamilies

- Gloss-free latent diffusion.
- Text-conditioned latent pose generation.
- Multimodal text/audio-conditioned diffusion.
- VAE or SignVAE latent-space production.
- Non-autoregressive latent prediction.
- Iterative latent refinement.
- Variational latent modeling.
- Conditional flow matching or sparse-keyframe generative production as frontier evidence.
- 3D/avatar-compatible diffusion as frontier evidence.
- Older Gaussian latent-space baseline evidence.

## Explicit Exclusions / Out-of-Scope Directions

- Learned tokenization when the main contribution is representation design rather than generation
  architecture.
- Structure-aware articulator/channel modeling when the main contribution is structure
  preservation rather than generation mechanism.
- Pure text-pose semantic alignment without latent generative modeling.
- Retrieval/stitching systems that generate sequences by reusing or stitching stored signs rather
  than learning latent generative dynamics.
- Pure evaluation methodology.
- Gloss-dependent methods treated as direct support for the project's no-public-gloss data regime.
- 3D avatar/rendering systems where the main contribution is rendering rather than latent
  generation.
- Recognition-only or sign-to-text-only systems.

Frontier or avatar-compatible latent generation is not excluded categorically. It may be used as
frontier or boundary evidence when it clarifies latent generative design or future scope.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to sign-pose/keypoint/motion production, latent
  or generative method relevance, gloss-free or gloss-adaptable supervision, method
  extractability, dataset compatibility, evaluation relevance, and downstream
  contribution-selection relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `evaluation_evidence`, `dataset_evidence`, and
  `reproducibility_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct support when their latent or generative method maps to the project's
  pose/keypoint regime. `cross_lingual_transferable` sources may support the family when
  generation-mechanism transfer is methodologically clear.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  family support. `gloss_dependent` sources are boundary evidence only unless a gloss-free
  adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Relevance type`: Core gloss-free latent diffusion evidence.
- `Claim supported`: Latent diffusion is a credible gloss-free text-to-sign-pose production
  family.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This family records latent generative evidence only; it does not
  select a concrete diffusion candidate.

### Source 2

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Relevance type`: How2Sign-direct multimodal diffusion evidence.
- `Claim supported`: How2Sign can support recent text/audio-conditioned keypoint generation under
  a gloss-free setting.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Audio-conditioning is relevant evidence but does not change the
  project's primary text-to-pose framing.

### Source 3

- `Source corpus entry`: `SRC-DARSLP-2025`
- `Source link`: [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025)
- `Relevance type`: Gloss-free articulator-disentangled latent modeling evidence.
- `Claim supported`: Latent-space sign production can be structured around
  articulator-disentangled representations.
- `Evidence note`: DARSLP is recorded as a core source for articulator-disentangled latent spaces
  and channel-aware regularization.
- `Interpretation boundary`: Structure-aware aspects belong to the structure-aware family; this
  record uses DARSLP only for latent-production relevance.

### Source 4

- `Source corpus entry`: `SRC-ILRSLP-2025`
- `Source link`: [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025)
- `Relevance type`: Iterative latent refinement evidence.
- `Claim supported`: Latent sign production can be refined iteratively instead of predicted once.
- `Evidence note`: ILRSLP is recorded as an important continuation of articulator-disentangled
  latent modeling.
- `Interpretation boundary`: It should be grouped with latent refinement and structure-aware
  latent modeling rather than treated as a standalone selected candidate.

### Source 5

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Relevance type`: Variational latent modeling evidence.
- `Claim supported`: Latent sign production can model distributional uncertainty through
  variational articulator-wise prediction.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: This is recent frontier evidence and should not automatically outrank
  earlier reviewed latent or structure-aware sources.

### Source 6

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Relevance type`: Older non-autoregressive Gaussian latent baseline.
- `Claim supported`: Latent-space non-autoregressive SLP is part of the foundational generative
  baseline landscape.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source that
  critiques back-translation reliability.
- `Interpretation boundary`: It is baseline evidence and should not be treated as sufficient
  modern latent-generation support by itself.

### Source 7

- `Source corpus entry`: `SRC-NEURAL-SIGN-ACTORS-2024`
- `Source link`: [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024)
- `Relevance type`: Frontier text-to-3D avatar diffusion evidence.
- `Claim supported`: Latent/diffusion production can extend toward 3D avatar-compatible signing.
- `Evidence note`: Neural Sign Actors is recorded as important for SMPL-X / 3D avatar-compatible
  production, with heavier-than-current-keypoint-pipeline caution.
- `Interpretation boundary`: This is frontier/avatar-compatible evidence and should not force
  immediate adoption of a 3D parametric pipeline.

### Source 8

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Relevance type`: Frontier sparse-keyframe / conditional-flow evidence.
- `Claim supported`: Sparse-keyframe or flow-based generative production is a frontier alternative
  to direct latent prediction.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and may exceed immediate thesis
  implementation scope.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Latent generative methods are directly compatible when they map
  spoken-language text/transcripts to sign pose, keypoints, skeletons, motion representations, or
  learned latent sign codes without requiring manual gloss annotations.
- `Gloss-dependent risks`: Gloss-dependent generative methods may inform architecture or latent
  design but must not be treated as direct support without a gloss-free adaptation path.
- `Intermediate representation type`: This family may use VAE latents, SignVAE latents, Gaussian
  latent spaces, diffusion latents, iterative latent states, variational distributions, or
  sparse-keyframe latent structures when the primary issue is generation.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton outputs are directly aligned
  with the project. 3D avatar-compatible latent generation is frontier evidence when it exceeds
  the project's immediate pose/keypoint regime.

## Method-Family Rationale

This family exists because generation architecture is separable from representation design. It
prevents three common errors: treating every learned representation paper as a generative
architecture candidate, treating diffusion, VAE, iterative refinement, and flow-based systems as
the same method family without noting their generation differences, and treating frontier 3D
generative systems as immediate implementation requirements.

## Counter-Evidence And Limitations

- `Known limitations`: Latent generative systems may be computationally heavier, harder to train,
  harder to evaluate, sensitive to latent reconstruction quality, or too complex for a minimal
  thesis baseline.
- `Conflicting evidence`: Some strong latent/generative sources are `cross_lingual_transferable`,
  some are `how2sign_direct`, some are frontier/preprint evidence, and some require heavier 3D or
  multimodal pipelines.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if the latent
  model is too expensive, if evaluation cannot isolate generative gains, if generated motion
  quality is poor, or if the method exceeds available data/artifact constraints.

## Potential Candidate Directions

This family may later produce candidate-level records such as gloss-free latent diffusion,
lightweight text-conditioned latent generation, non-autoregressive latent prediction, iterative
latent refinement, variational articulator-wise latent generation, multimodal
text/audio-conditioned diffusion as an extension, or sparse-keyframe or flow-based generation as
frontier/future work. Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is a primary candidate-family source for generation-architecture contributions. It
should help later audit records distinguish representation choice, latent generative architecture,
structure-aware modeling, semantic alignment, retrieval/stitching alternatives, evaluation design,
and frontier 3D/avatar modeling.

## Family-Level Evidence Boundary

This family does not select a concrete latent generation candidate. It does not rank diffusion,
VAE, Gaussian latent space, iterative refinement, variational modeling, or flow-based generation.
It does not claim that every latent generative method is compatible with How2Sign, convert
frontier 3D/avatar systems into immediate implementation requirements, or convert gloss-dependent
generative sources into direct support. It only defines the latent generative production universe
for later candidate records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish latent-generation evidence from learned-representation evidence,
distinguish latent-generation evidence from structure-aware modeling evidence, state whether the
generative source is `how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`,
explicitly classify `gloss_free`, `gloss_adaptable`, or `gloss_dependent` assumptions, provide
adaptation rationale before using gloss-dependent or non-How2Sign generative sources as direct
support, and specify whether the proposed latent generative method can be trained and evaluated
with the project's available pose/keypoint artifacts.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
