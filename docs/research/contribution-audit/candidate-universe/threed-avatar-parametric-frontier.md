# 3D Avatar and Parametric Motion Frontier

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-THREED-AVATAR-PARAMETRIC-FRONTIER`

## Family Name

3D Avatar and Parametric Motion Frontier

## Family Type / Status

`frontier_watch_family`

This is a frontier-watch family, not an immediate candidate-generating family by default. It may
inform positioning, future work, and long-term interface or visualization direction, but this
record does not instantiate or select any concrete candidate.

## Scope Definition

This family covers sources that extend sign language production beyond ordinary pose/keypoint
output into 3D avatar-compatible, parametric, or rendered motion representations. It includes
SMPL-X or SMPL-like signing avatars, FLAME, MANO, SMPL-FX, or richer body/hand/face parametric
representations, 3D avatar-compatible motion generation, 3D diffusion or avatar-compatible
generative production, 3D Gaussian Splatting or rendering-oriented sign production,
photorealistic pose-to-video or signer synthesis as downstream boundary evidence,
non-manual-feature-rich 3D or facial modeling pressure, and frontier multilingual or large-scale
3D SLP systems.

The family focuses on the frontier representation and output-surface question: when should the
project acknowledge or eventually move beyond keypoints into avatar-compatible or parametric
signing? It does not collapse 3D avatar and parametric frontier evidence into learned pose/motion
representation, latent generative production, structure-aware articulator modeling, text-pose
semantic alignment, retrieval/stitching, evaluation methodology, or dataset-supervision boundary;
those are separate family records.

## Included Directions / Subfamilies

- Text-to-3D signing avatar generation.
- SMPL-X or SMPL-style parametric motion.
- SMPL-FX / FLAME / MANO-style body, face, and hand modeling.
- 3D avatar-compatible diffusion.
- Sparse-keyframe or multilingual 3D SLP frontier systems.
- Motion-token systems with explicit body/hand/face or non-manual-feature pressure.
- Photorealistic pose-to-video or signer rendering boundary systems.
- Future interface and visualization pressure beyond keypoints.

## Explicit Exclusions / Out-of-Scope Directions

- Ordinary 2D/3D keypoint text-to-pose production where the main contribution is not avatar or
  parametric representation.
- Learned tokenization where the main contribution is representation learning rather than
  avatar/parametric output.
- Latent generative modeling where the main contribution is generation architecture rather than
  avatar-compatible output.
- Structure-aware modeling where the main contribution is articulator/channel preservation but
  not parametric/avatar output.
- Pure semantic alignment or text-conditioning.
- Pure evaluation methodology.
- Retrieval/stitching systems without avatar/parametric relevance.
- Recognition-only or sign-to-text-only systems.
- Treating frontier avatar systems as immediate implementation requirements.

These sources are not excluded from the audit universe. Their purpose here is to define frontier
pressure, future-work implications, and possible long-term visualization or interface directions.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to sign-pose/keypoint/motion production, avatar
  or parametric-output relevance, method extractability, dataset compatibility, non-manual or
  body/hand/face representation relevance, evaluation relevance, and downstream positioning or
  future-work relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `evaluation_evidence`, `background_evidence`, and
  `reproducibility_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct support when their avatar or parametric motion pipeline maps to the project's
  data regime. `cross_lingual_transferable` sources may support frontier positioning when the
  output representation or visualization strategy transfers methodologically.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  frontier evidence. `gloss_dependent` sources are boundary evidence only unless a gloss-free
  adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-NEURAL-SIGN-ACTORS-2024`
- `Source link`: [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024)
- `Relevance type`: Text-to-3D signing avatar frontier evidence.
- `Claim supported`: Text-conditioned SLP can extend toward 3D avatar-compatible motion rather
  than only keypoint-level pose output.
- `Evidence note`: Neural Sign Actors is recorded as important for SMPL-X / 3D
  avatar-compatible production, with heavier-than-current-keypoint-pipeline caution.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a 3D parametric pipeline.

### Source 2

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Relevance type`: Sparse-keyframe and multilingual 3D SLP frontier evidence.
- `Claim supported`: Sparse-keyframe learning and 3D avatar-compatible production are frontier
  directions for scalable sign language production.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and may exceed immediate thesis
  implementation scope.

### Source 3

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Relevance type`: Body/hand/face and non-manual parametric motion-token frontier evidence.
- `Claim supported`: Recent work pressures representation design toward richer body, hands, face,
  and non-manual modeling beyond ordinary keypoints.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: Its tokenization and semantic-grounding aspects belong to
  learned-representation and semantic-alignment families; this record uses it only for
  avatar/parametric frontier pressure.

### Source 4

- `Source corpus entry`: `SRC-EVERYBODY-SIGN-NOW-2020`
- `Source link`: [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020)
- `Relevance type`: Photorealistic pose-to-video rendering boundary evidence.
- `Claim supported`: Pose production can be separated from downstream signer visualization or
  photorealistic video synthesis.
- `Evidence note`: Everybody Sign Now is recorded as a pose-to-video/rendering boundary source.
- `Interpretation boundary`: This source clarifies downstream visualization and rendering
  boundaries; it is not central to near-term pose modeling.

### Source 5

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Relevance type`: Keypoint-level contrast source for frontier boundary.
- `Claim supported`: Strong keypoint-level latent generative methods remain closer to the
  project's immediate regime than heavy avatar/parametric systems.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This source is included here only to contrast immediate
  pose/keypoint-compatible production with heavier avatar frontier systems.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Avatar/parametric frontier methods are most directly useful when
  they can map spoken-language text or transcripts to avatar-compatible motion without requiring
  manual gloss annotations.
- `Gloss-dependent risks`: Gloss-dependent avatar or rendering systems may inform visualization
  but must not be treated as direct support without a gloss-free adaptation path.
- `Intermediate representation type`: This family may use SMPL-X, SMPL-FX, FLAME, MANO, 3D
  skeletal parameters, mesh-like motion representations, avatar-compatible latent spaces, sparse
  keyframes, or photorealistic pose-conditioned rendering when the primary issue is output surface
  or parametric motion.
- `Pose/keypoint/skeleton/3D compatibility`: Ordinary pose/keypoint/skeleton outputs are the
  project's immediate regime. 3D parametric, avatar, mesh, or rendering outputs are frontier
  pressure unless explicitly scoped as downstream visualization or future work.

## Method-Family Rationale

This family exists because the project's near-term pose/keypoint artifact is not the whole SLP
frontier. It prevents three common errors: treating keypoint-level production as the final form of
user-facing signing, expanding the thesis implementation scope prematurely into heavy
3D/avatar/rendering systems, and ignoring non-manual, face, body, and hand representation pressure
from recent frontier systems.

## Counter-Evidence And Limitations

- `Known limitations`: Avatar/parametric systems may require 3D reconstruction, body/hand/face
  model fitting, rendering infrastructure, signer identity modeling, richer annotations, or
  heavier compute than the current project scope.
- `Conflicting evidence`: Some frontier evidence is `how2sign_direct` or `how2sign_compatible`,
  some is `cross_lingual_transferable`, some is preprint/frontier evidence, and some uses heavy
  3D or rendering assumptions.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if avatar
  fitting, 3D parametric reconstruction, rendering, or non-manual-feature modeling exceeds the
  thesis artifact scope or cannot be evaluated reliably.

## Potential Candidate Directions

This family may later produce candidate-level records such as downstream avatar visualization
bridge, text-to-3D avatar-compatible motion as future work, SMPL-X or SMPL-FX representation
study, non-manual-feature-aware 3D representation, pose-to-video rendering boundary analysis, or
sparse-keyframe avatar control as frontier/future work. Candidate cards must not be instantiated
here.

## Relation To The Overall Audit

This family is a frontier-watch source for future-facing representation and output-surface
decisions. It should help later audit records distinguish immediate pose/keypoint production,
downstream visualization, 3D avatar-compatible motion, parametric body/hand/face modeling,
frontier future work, and implementation scope boundaries.

## Family-Level Evidence Boundary

This family does not select a 3D/avatar/parametric candidate. It does not rank SMPL-X, SMPL-FX,
FLAME, MANO, 3DGS, photorealistic rendering, or sparse-keyframe avatar control. It does not claim
that avatar/parametric systems are immediate implementation requirements, and it does not convert
frontier-heavy systems into near-term thesis scope. It only defines the 3D avatar and parametric
motion frontier for later audit records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish frontier-watch evidence from immediate implementation evidence,
distinguish avatar/parametric output evidence from learned-representation evidence, distinguish
avatar/parametric output evidence from latent-generation evidence, state whether the source is
`how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`, explicitly classify
`gloss_free`, `gloss_adaptable`, or `gloss_dependent` assumptions, provide adaptation rationale
before using non-How2Sign or frontier-heavy sources as direct support, and specify whether the
proposed direction is near-term implementation, downstream visualization, or future work.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
