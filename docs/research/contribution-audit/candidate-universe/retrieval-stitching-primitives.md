# Retrieval, Stitching, and Motion-Primitives Alternatives

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-RETRIEVAL-STITCHING-PRIMITIVES`

## Family Name

Retrieval, Stitching, and Motion-Primitives Alternatives

## Family Type / Status

`counter_alternative_family`

This is a serious competing family included to avoid prematurely narrowing the audit universe, but
this record does not instantiate or select any concrete candidate.

## Scope Definition

This family covers alternatives that avoid or reduce fully generative text-to-raw-pose prediction
by using retrieved, stitched, primitive, dictionary, sparse-keyframe, or reusable motion units. It
includes retrieval-based sign production, dictionary or isolated-sign reuse, sign stitching,
motion primitives or mixture-of-motion-primitives approaches, sparse-keyframe or anchor-based
generation as frontier/counter evidence, hybrid systems that predict intermediate symbolic or
control variables before assembling pose, and historical staged pipelines that separate
text-to-intermediate planning from pose or video synthesis.

The family focuses on the production strategy question: should the system synthesize pose from
scratch, or assemble, stitch, retrieve, blend, or anchor motion from reusable sign/motion units? It
does not collapse retrieval, stitching, and motion-primitives alternatives into learned
pose/motion representation, latent generative production, text-pose semantic alignment,
structure-aware modeling, evaluation methodology, or 3D avatar production; those are separate
family records.

## Included Directions / Subfamilies

- Retrieval-based sign production.
- Dictionary-style sign reuse.
- Sign stitching.
- Motion primitive or mixture-of-primitives production.
- Sparse-keyframe or sparse-anchor generation as frontier evidence.
- Hybrid text-to-control-variable-to-pose pipelines.
- Pose-to-video downstream boundary when used to clarify staged production.
- Gloss-dependent retrieval/stitching or motion-primitive sources used only as boundary evidence.

## Explicit Exclusions / Out-of-Scope Directions

- Pure learned tokenization where the main contribution is representation learning.
- Pure latent diffusion or generative modeling where the main contribution is generation
  architecture.
- Structure-aware articulator/channel modeling where the main contribution is preserving
  articulator structure.
- Text-pose semantic alignment without retrieval, stitching, primitives, or reusable motion units.
- Pure evaluation methodology.
- Rendering-only or avatar-only systems without retrieval/stitching/primitive relevance.
- Recognition-only or sign-to-text-only systems.
- Gloss-dependent retrieval/stitching systems treated as direct support for the project's
  no-public-gloss data regime.

Gloss-dependent retrieval or primitive systems are not excluded categorically. They may be used as
counter-alternative or boundary evidence when they clarify assembly, reuse, or primitive-based
production, but not as direct compatibility evidence without adaptation rationale.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to sign-pose/keypoint/motion production,
  counter-alternative relevance, method extractability, dataset compatibility, supervision
  compatibility, evaluation relevance, and downstream audit relevance.
- `Evidence classes represented`: `direct_task_evidence`, `architecture_evidence`,
  `representation_evidence`, `negative_evidence`, `background_evidence`, and
  `evaluation_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct support when retrieval/stitching or primitive assumptions can map to the
  project's text/pose/keypoint regime. `cross_lingual_transferable` sources may support the
  family when the production strategy transfers methodologically.
- `Gloss dependency boundary`: `gloss_free` and `gloss_adaptable` sources may provide direct
  family support. `gloss_dependent` sources are counter-alternative or boundary evidence only
  unless a gloss-free adaptation path is documented.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-SIGN-STITCHING-2024`
- `Source link`: [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024)
- `Relevance type`: Core retrieval/stitching counter-alternative evidence.
- `Claim supported`: Sign production can be framed as assembling or stitching reusable sign
  examples rather than directly generating every pose frame from scratch.
- `Evidence note`: Sign Stitching is recorded as a serious alternative to direct regression and
  useful for explaining retrieval/stitching boundaries.
- `Interpretation boundary`: It is gloss-dependent and must not be treated as direct support for a
  no-public-gloss How2Sign-compatible implementation without adaptation rationale.

### Source 2

- `Source corpus entry`: `SRC-MIXED-SIGNALS-2021`
- `Source link`: [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021)
- `Relevance type`: Motion-primitives alternative evidence.
- `Claim supported`: Motion primitives or mixture-of-primitives approaches are serious
  alternatives to direct continuous pose regression.
- `Evidence note`: Mixed SIGNals is recorded as an important alternative representation family
  but not direct support for no-gloss implementation.
- `Interpretation boundary`: It is gloss-dependent and should remain counter-alternative or
  boundary evidence unless a gloss-free adaptation path is documented.

### Source 3

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Relevance type`: Frontier sparse-keyframe / sparse-anchor evidence.
- `Claim supported`: Sparse keyframes or anchors may offer a frontier alternative to dense
  frame-by-frame generation.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and may exceed immediate thesis
  implementation scope.

### Source 4

- `Source corpus entry`: `SRC-TEXT2SIGN-2020`
- `Source link`: [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020)
- `Relevance type`: Historical staged-pipeline boundary evidence.
- `Claim supported`: Earlier SLP pipelines used staged planning and intermediate assumptions
  before pose/video synthesis.
- `Evidence note`: Text2Sign is recorded as a foundational gloss-dependent text-to-pose/video
  pipeline.
- `Interpretation boundary`: It must not be treated as direct support for the project's
  no-public-gloss data regime.

### Source 5

- `Source corpus entry`: `SRC-EVERYBODY-SIGN-NOW-2020`
- `Source link`: [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020)
- `Relevance type`: Pose-to-video staged-production boundary evidence.
- `Claim supported`: Staged systems can separate pose production from downstream visual synthesis.
- `Evidence note`: Everybody Sign Now is recorded as a pose-to-video/rendering boundary source.
- `Interpretation boundary`: This source clarifies staged production and rendering boundaries; it
  is not direct evidence for retrieval/stitching as a near-term implementation path.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Retrieval, stitching, or primitive methods are directly compatible
  only when reusable units, anchors, or control variables can be obtained or learned without
  requiring manual gloss annotations.
- `Gloss-dependent risks`: Many retrieval, stitching, dictionary, or primitive systems rely on
  glosses, isolated signs, dictionaries, or manually defined intermediate units. These are
  counter-alternative evidence unless a no-gloss adaptation path is documented.
- `Intermediate representation type`: This family may use retrieved examples, dictionary units,
  motion primitives, sparse keyframes, anchor poses, predicted control variables, or staged
  intermediate plans when the primary issue is assembly or reuse rather than representation
  learning.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton units are relevant when they
  can be retrieved, stitched, or composed within the project's data regime. 3D avatar or rendering
  outputs are boundary/frontier evidence when they exceed immediate pose/keypoint scope.

## Method-Family Rationale

This family exists because direct generation is not the only plausible sign production strategy.
It prevents three common errors: assuming every viable production system must synthesize pose
fully from scratch, ignoring retrieval/stitching or primitive-based systems that may preserve
motion quality or reduce regression-to-mean, and treating gloss-dependent stitching or dictionary
systems as directly compatible with a no-public-gloss project.

## Counter-Evidence And Limitations

- `Known limitations`: Retrieval/stitching and primitive systems may require glosses,
  dictionaries, isolated-sign resources, segment boundaries, reliable retrieval units, transition
  smoothing, or dataset structures unavailable in the project.
- `Conflicting evidence`: Some counter-alternative evidence is technically strong but
  `gloss_dependent` or `cross_lingual_transferable`; some frontier evidence uses sparse
  keyframes, 3D parametric spaces, or heavy generation pipelines.
- `Reasons this family may not become a candidate`: A concrete candidate may fail if suitable
  reusable units cannot be obtained, if stitching causes discontinuities, if gloss/dictionary
  assumptions are unavailable, or if implementation complexity exceeds thesis scope.

## Potential Candidate Directions

This family may later produce candidate-level records such as gloss-free retrieval-augmented pose
generation, pose-nearest-neighbor baseline, lightweight sign-stitching baseline,
motion-primitive-inspired latent regularization, sparse-keyframe control as frontier/future work,
or staged text-to-control-to-pose pipeline as a counter-alternative. Candidate cards must not be
instantiated here.

## Relation To The Overall Audit

This family is a counter-alternative source for production-strategy comparisons. It should help
later audit records distinguish direct generation, learned representation, latent generative
architecture, retrieval/stitching strategy, motion-primitive strategy, downstream rendering, and
evaluation design.

## Family-Level Evidence Boundary

This family does not select a retrieval or stitching candidate. It does not rank retrieval,
stitching, dictionary reuse, motion primitives, or sparse-keyframe control. It does not claim that
retrieval/stitching is compatible with How2Sign without adaptation, and it does not convert
gloss-dependent retrieval or primitive sources into direct support. It only defines the retrieval,
stitching, and motion-primitives counter-alternative universe for later candidate records.

## Downstream Record Policy

Downstream candidate-universe and candidate-card records must cite `source-corpus.md` entries by
`SRC-*` ID, distinguish counter-alternative evidence from primary candidate-family evidence,
distinguish retrieval/stitching evidence from learned-representation evidence, distinguish
retrieval/stitching evidence from latent-generation evidence, state whether the source is
`how2sign_direct`, `how2sign_compatible`, or `cross_lingual_transferable`, explicitly classify
`gloss_free`, `gloss_adaptable`, or `gloss_dependent` assumptions, provide adaptation rationale
before using gloss-dependent or non-How2Sign retrieval/stitching sources as direct support, and
specify whether reusable units, segment boundaries, or primitive structures can be obtained from
the project's available artifacts.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
