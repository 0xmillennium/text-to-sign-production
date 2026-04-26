# Foundational Text-to-Pose Baselines

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES`

## Family Name

Foundational Text-to-Pose Baselines

## Family Type / Status

`boundary_family`

This is not a candidate-generating method family. It defines the historical and technical
baseline floor for later candidate families and contribution comparisons.

## Scope Definition

This family defines the baseline and historical text-to-pose sign language production context. It
covers early and foundational text-to-pose or text-to-sign-pose systems, Progressive
Transformer-style direct text-to-pose production, multi-channel 3D pose baselines,
non-autoregressive latent baseline framing, text-to-pose-to-video or pose-to-video boundary
systems, and historical gloss-dependent SLP systems used to explain why direct text-to-pose or
gloss-free approaches matter.

The family also records baseline evaluation precedent, including back-translation, pose-level
metrics, and user or perceptual evaluation where relevant. It is not meant to define the newest
contribution direction. It defines what later contribution candidates must improve upon or
position against.

## Included Directions / Subfamilies

- Direct text-to-pose baseline production.
- Text-to-gloss-to-pose and direct text-to-pose comparison context.
- Multi-channel 3D pose production.
- Non-autoregressive latent-space baseline production.
- Pose-to-video or text-to-pose-to-video boundary systems.
- Baseline evaluation precedent and regression-to-mean framing.

## Explicit Exclusions / Out-of-Scope Directions

- New learned pose/motion tokenization families as primary contributions.
- Recent latent diffusion or variational generative families as primary contributions.
- Structure-aware articulator modeling as a primary contribution family.
- Pure evaluation-methodology sources that do not define a production baseline.
- Recognition-only or sign-to-text-only systems.
- Avatar/rendering-only systems with no text-to-pose or pose-production relevance.
- Gloss-dependent systems treated as direct support for the project's no-public-gloss data regime.

Historical gloss-dependent systems are not excluded categorically. They may be used as boundary or
historical baseline context, but not as direct compatibility evidence without adaptation rationale.

## Source-Selection Basis

- `Relevant source-selection criteria`: Direct task relevance, extractable
  method/dataset/evaluation details, relation to text-to-pose or text-to-sign-pose production,
  evaluation precedent, and downstream baseline/comparison relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `evaluation_evidence`, `background_evidence`, and
  `reproducibility_evidence`.
- `Dataset compatibility boundary`: Most foundational sources are `cross_lingual_transferable`.
  `how2sign_direct` sources may update or strengthen baseline relevance but should not erase older
  baseline context.
- `Gloss dependency boundary`: `gloss_optional` sources may support comparison between
  gloss-intermediate and direct text-to-pose routes. `gloss_dependent` sources are historical or
  boundary evidence only.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-PROGTRANS-2020`
- `Source link`: [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020)
- `Relevance type`: Foundational direct text-to-pose baseline.
- `Claim supported`: Direct text-to-pose SLP is an established baseline task and comparison point.
- `Evidence note`: Progressive Transformers is recorded as a foundational text-to-pose baseline.
- `Interpretation boundary`: This source is foundational, not a sufficient modern contribution
  direction by itself.

### Source 2

- `Source corpus entry`: `SRC-MULTICHANNEL-MDN-2021`
- `Source link`: [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Relevance type`: Extended multi-channel baseline.
- `Claim supported`: Multi-channel 3D pose production and user-evaluation precedent are part of
  the baseline context.
- `Evidence note`: The source is recorded as an extended multi-channel baseline.
- `Interpretation boundary`: It strengthens baseline framing but does not by itself define the
  current candidate universe.

### Source 3

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Relevance type`: Non-autoregressive latent baseline.
- `Claim supported`: Non-autoregressive latent-space SLP is part of the foundational baseline
  landscape.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source that
  critiques back-translation reliability.
- `Interpretation boundary`: Later latent or generative families should not be collapsed into this
  older baseline family.

### Source 4

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Relevance type`: Recent How2Sign-direct baseline pressure.
- `Claim supported`: Recent How2Sign-direct production work raises the expected baseline standard
  beyond early PHOENIX-only systems.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production.
- `Interpretation boundary`: Its diffusion and multimodal details belong to later architecture
  families; this record uses it only to frame baseline expectations.

### Source 5

- `Source corpus entry`: `SRC-TEXT2SIGN-2020`
- `Source link`: [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020)
- `Relevance type`: Historical gloss-dependent SLP pipeline.
- `Claim supported`: Historical SLP pipelines often used gloss or staged text-to-pose/video
  assumptions.
- `Evidence note`: Text2Sign is recorded as a foundational gloss-dependent text-to-pose/video
  pipeline.
- `Interpretation boundary`: It must not be treated as direct support for the project's
  no-public-gloss data regime.

### Source 6

- `Source corpus entry`: `SRC-EVERYBODY-SIGN-NOW-2020`
- `Source link`: [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020)
- `Relevance type`: Pose-to-video and downstream rendering boundary.
- `Claim supported`: Text-to-pose-to-video systems help separate pose production from downstream
  visual rendering.
- `Evidence note`: Everybody Sign Now is recorded as a pose-to-video/rendering boundary source.
- `Interpretation boundary`: It is not central to near-term pose modeling and should not expand
  the immediate baseline scope into full photorealistic video generation.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Direct text-to-pose and gloss-optional baselines can support the
  project's gloss-free framing when they do not require manual gloss annotations.
- `Gloss-dependent risks`: Historical gloss-dependent pipelines may explain prior assumptions and
  limitations, but they must not be treated as directly compatible without adaptation rationale.
- `Intermediate representation type`: This family may include continuous pose, 3D pose, latent
  pose spaces, gloss-intermediate historical systems, and pose-to-video boundaries, but it does not
  select a modern learned-token or latent-generative contribution.
- `Pose/keypoint/skeleton/3D compatibility`: Pose, keypoint, skeleton, and 3D pose baselines are
  relevant when they define baseline production or evaluation expectations. Photorealistic video
  or avatar output is boundary evidence only.

## Method-Family Rationale

This boundary family exists because later contribution candidates need a stable baseline floor. It
prevents three common errors: treating early baselines as obsolete and ignoring their task or
evaluation definitions, treating historical gloss-dependent systems as directly compatible with
the project's no-public-gloss data regime, and comparing new candidate families without a clear
foundational text-to-pose baseline context.

## Counter-Evidence And Limitations

- `Known limitations`: Many foundational baselines are PHOENIX14T/DGS-centered, older than recent
  How2Sign-direct work, and often rely on back-translation evaluation.
- `Conflicting evidence`: Some newer sources use learned tokens, diffusion, multimodal
  conditioning, or articulator-aware latent spaces that exceed this family's baseline role.
- `Reasons this family may not become a candidate`: This is a boundary/baseline family. It defines
  comparison context, not a new contribution family.

## Potential Candidate Directions

This family does not instantiate candidate directions. It may constrain later candidate directions
by defining the minimum baseline to compare against, whether M0 should be framed as direct
text-to-pose or another minimal baseline, which historical metrics or baseline claims should be
treated cautiously, and which older systems are boundary context rather than candidate sources.
Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is upstream of candidate selection because it defines the baseline and historical
context against which contribution families will later be evaluated. It should help later records
distinguish baseline implementation, candidate contribution, counter-alternative, evaluation
method, and downstream rendering boundary.

## Family-Level Evidence Boundary

This family does not select a model contribution. It does not rank candidate families, imply old
baselines are sufficient for the thesis contribution, or convert gloss-dependent historical
systems into direct support. It only defines baseline and historical text-to-pose context for later
audit records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite this family when they claim
improvement over foundational text-to-pose baselines, cite `source-corpus.md` entries by `SRC-*`
ID, distinguish baseline evidence from candidate-level evidence, state whether a comparison
baseline is `how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`, and avoid
treating boundary/historical sources as direct implementation support without adaptation rationale.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
