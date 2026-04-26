# Retrieval-Augmented Pose Comparator

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-RETRIEVAL-AUGMENTED-POSE-COMPARATOR`

## Candidate Name

Retrieval-Augmented Pose Comparator

## Candidate Record Type

- `Candidate record type`: `counter_alternative_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-RETRIEVAL-STITCHING-PRIMITIVES`
- `Family link`: [FAM-RETRIEVAL-STITCHING-PRIMITIVES](../candidate-universe/retrieval-stitching-primitives.md)
- `Family type/status`: `counter_alternative_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the
  retrieval, stitching, and motion-primitives counter-alternative family as a concrete comparator
  for testing whether generated pose/motion candidates outperform retrieval-based reuse under the
  project's no-public-gloss data regime.

## Candidate-Level Technical Definition

A no-public-gloss retrieval-augmented comparator that retrieves existing pose/keypoint examples or
sentence-level motion references using text/transcript similarity, pose-nearest-neighbor
matching, or compatible embedding features, and uses those retrieved references as a comparison
baseline or weak production alternative. The candidate focuses on the production-strategy
question: can retrieved or reused motion examples challenge direct neural generation as a
comparison point?

## Candidate Boundary

- `Included mechanism`: Sentence-level or segment-level retrieval, text/transcript embedding
  nearest-neighbor retrieval, pose-nearest-neighbor comparator, retrieved reference pose reuse,
  optional lightweight transition or normalization logic, and retrieval-based comparison
  reporting.
- `Explicitly excluded mechanisms`: Gloss dictionaries, isolated-sign lexicons, manual gloss
  segmentation, full sign stitching requiring gloss labels, learned pose-token bottleneck as the
  primary contribution, latent diffusion, structure-aware articulator modeling, semantic
  consistency objective as the primary contribution, 3D avatar, SMPL-X, and rendering.
- `Not to be confused with`: Primary model candidates, learned representation candidates, latent
  generative production candidates, semantic alignment candidates, gloss-dependent sign-stitching
  systems, or frontier avatar systems.

Retrieval/stitching sources can inform this candidate only when their assumptions can be adapted
without requiring public manual gloss annotations.

## Research Role In Current Audit

- `Research role label`: `counter_alternative_candidate`

This record defines a serious comparator and alternative path rather than a primary model
contribution.

## Expected Improvement Mechanism

This candidate does not improve generated motion by learning a new generative model. Its audit
value is comparative: it may preserve real motion quality by reusing examples, reveal whether
neural generators outperform nearest-neighbor or retrieval reuse, expose mode averaging or
oversmoothing in learned models, provide a low-complexity sanity check, and challenge claims that
direct generation is necessary. Any gains are conditional on retrieval quality, dataset coverage,
segment alignment, and avoiding gloss or dictionary dependency.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `mode_averaging`, `unnatural_transitions`,
  `semantic_mismatch`, `timing_misalignment`, `retrieval_discontinuity`,
  `evaluation_blind_spot`; secondary risk: `sequence_drift`

This candidate mainly targets comparator and retrieval-reuse failure modes. It may preserve
motion realism but can create semantic mismatch or transition discontinuity.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_compatible`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed for the minimal comparator;
  retrieval uses text/transcript, segment metadata, pose/keypoint examples, or learned embeddings.
- `Segmentation required`: Sentence/video segment boundaries are required; no gloss segmentation
  is assumed.
- `Dictionary/example bank required`: A retrieved example bank is required, but it should be built
  from available paired text/pose examples, not from a manual gloss dictionary.
- `3D/parametric artifacts required`: Not required for the minimal keypoint-level comparator.
- `Current project artifacts sufficient`: Conditional; sufficient if stable
  text/transcript-to-pose/keypoint pairs, split metadata, and retrieval-safe
  train/validation/test separation are available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if
  pose/keypoint examples can be indexed and retrieved without schema conversion.
- `Compatible with current experiment workflow`: Expected to be compatible as a
  baseline/comparator module and evaluation report.
- `Heavy new preprocessing required`: Low to moderate; requires building retrieval indices,
  feature caches, and leakage-safe retrieval splits.
- `Thin-driver notebook principle affected`: Not expected if retrieval indexing, search, and
  reporting code live outside notebooks.
- `Reproducibility risk`: Medium; depends on deterministic feature extraction, retrieval index
  construction, split leakage prevention, and documented tie-breaking.

## Evidence Chain

- `Source-selection basis`: Counter-alternative relevance, retrieval/stitching/motion-primitives
  relevance, no-public-gloss adaptation risk, dataset compatibility, evaluation relevance, and
  downstream comparison relevance.
- `Supporting source-corpus entries used`: `SRC-SIGN-STITCHING-2024`,
  `SRC-MIXED-SIGNALS-2021`, `SRC-TEXT2SIGN-2020`, `SRC-EVERYBODY-SIGN-NOW-2020`,
  `SRC-SIGNSPARK-2026`
- `Limiting source-corpus entries used`: `SRC-SIGN-STITCHING-2024`,
  `SRC-MIXED-SIGNALS-2021`, `SRC-TEXT2SIGN-2020`, `SRC-POSE-EVAL-2025`,
  `SRC-TEXT2SIGNDIFF-2025`
- `Candidate-universe family used`: [FAM-RETRIEVAL-STITCHING-PRIMITIVES](../candidate-universe/retrieval-stitching-primitives.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGN-STITCHING-2024`
- `Source link`: [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024)
- `Support type`: `transferable_method_support`
- `Claim supported`: Sign production can be framed as assembling or stitching reusable sign
  examples rather than directly generating every pose frame from scratch.
- `Evidence note`: Sign Stitching is recorded as a serious alternative to direct regression and
  useful for explaining retrieval/stitching boundaries.
- `Interpretation boundary`: It is gloss-dependent and must not be treated as direct support for
  a no-public-gloss How2Sign-compatible implementation without adaptation rationale.

### Source 2

- `Source corpus entry`: `SRC-MIXED-SIGNALS-2021`
- `Source link`: [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021)
- `Support type`: `transferable_method_support`
- `Claim supported`: Motion primitives or mixture-of-primitives approaches are serious
  alternatives to direct continuous pose regression.
- `Evidence note`: Mixed SIGNals is recorded as an important alternative representation family
  but not direct support for no-gloss implementation.
- `Interpretation boundary`: It is gloss-dependent and should remain counter-alternative or
  boundary evidence unless a gloss-free adaptation path is documented.

### Source 3

- `Source corpus entry`: `SRC-TEXT2SIGN-2020`
- `Source link`: [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020)
- `Support type`: `boundary_support`
- `Claim supported`: Earlier SLP pipelines used staged planning and intermediate assumptions
  before pose/video synthesis.
- `Evidence note`: Text2Sign is recorded as a foundational gloss-dependent text-to-pose/video
  pipeline.
- `Interpretation boundary`: It must not be treated as direct support for the project's
  no-public-gloss data regime.

### Source 4

- `Source corpus entry`: `SRC-EVERYBODY-SIGN-NOW-2020`
- `Source link`: [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020)
- `Support type`: `boundary_support`
- `Claim supported`: Staged systems can separate pose production from downstream signer
  visualization or photorealistic video synthesis.
- `Evidence note`: Everybody Sign Now is recorded as a pose-to-video/rendering boundary source.
- `Interpretation boundary`: This source clarifies staged production and rendering boundaries; it
  is not direct evidence for retrieval/stitching as a near-term implementation path.

### Source 5

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Support type`: `boundary_support`
- `Claim supported`: Sparse keyframes or anchors may offer a frontier alternative to dense
  frame-by-frame generation.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and may exceed immediate thesis
  implementation scope.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGN-STITCHING-2024`
- `Source link`: [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Gloss-dependent stitching assumptions cannot be treated as directly compatible
  with the project's no-public-gloss regime.
- `Evidence note`: Sign Stitching is recorded as a serious retrieval/stitching alternative, but
  it is gloss-dependent.
- `Interpretation boundary`: This limits direct adoption; it does not invalidate a no-gloss
  retrieval comparator built from available paired text/pose examples.

### Source 2

- `Source corpus entry`: `SRC-MIXED-SIGNALS-2021`
- `Source link`: [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Motion-primitives approaches may require gloss or manually defined
  intermediate units unavailable in the current project.
- `Evidence note`: Mixed SIGNals is recorded as technically relevant but not direct support for
  no-gloss implementation.
- `Interpretation boundary`: This limits primitive-based adoption; it does not prevent using
  retrieval as a comparator if retrieval units come from available paired data.

### Source 3

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Retrieval realism, low pose distance, or example reuse alone cannot prove sign
  intelligibility or semantic adequacy.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate retrieval comparison; it limits how
  retrieved outputs should be interpreted.

### Source 4

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Retrieval-based comparison should not be assumed competitive with modern
  latent generative systems without evaluation.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This is counter-evidence against treating retrieval as sufficient,
  not against using it as a serious comparator.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: A no-public-gloss retrieval-augmented pose comparator is useful as a
  serious counter-alternative and sanity check against learned generative claims.
- `Least-secure claim`: Whether retrieval can produce semantically adequate outputs under the
  project's available text/pose artifacts is not established by the candidate card alone.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: Retrieval, stitching, and primitive-based alternatives are
  technically well motivated, but many strong sources are gloss-dependent or frontier-heavy; a
  project-compatible comparator requires careful no-gloss adaptation and leakage-safe evaluation.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md), this candidate
uses retrieval or reuse rather than direct neural prediction. Compared with the
[Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md),
[Articulator-Disentangled Latent Modeling](articulator-disentangled-latent.md),
[Gloss-Free Latent Diffusion](gloss-free-latent-diffusion.md), and
[Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency.md), it is a comparator
and counter-alternative based on retrieval or reuse, not a primary neural generation mechanism.
Frontier sparse-keyframe/avatar approaches remain adjacent alternatives but should not be folded
into this minimal comparator.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Compare direct generation and future model candidates against
  retrieval-based nearest-neighbor or retrieved-reference outputs using the same split policy;
  verify that retrieval does not leak target references from validation/test sets.
- `Primary metric family`: Pose/keypoint distance and retrieval similarity diagnostics.
- `Secondary metric family`: Embedding-based motion checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Retrieved real motion may look plausible while being semantically
  mismatched, temporally discontinuous, or invalid for the source sentence.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Not applicable as a model mechanism; this is a comparator and
  counter-alternative.
- `Baseline changed`: No.
- `Primary isolated variable`: Retrieval-based reuse or nearest-neighbor pose comparison.
- `Requires another candidate`: No.
- `Independent ablation possible`: Yes; compare with and without retrieval-based outputs under
  the same data and evaluation protocol.
- `Likely interactions with other candidate families`: May interact with semantic alignment and
  evaluation protocol design; it should remain separate from primary model mechanisms unless a
  later retrieval-augmented model candidate is explicitly proposed.

## Complexity / Risk Estimate

- `Implementation complexity`: Low to medium.
- `Failure cost`: Low to medium; even weak retrieval can still serve as a useful sanity-check
  comparator.
- `Hidden dependency risk`: Medium; depends on leakage-safe indexing, feature choices, retrieval
  distance, and segment alignment.
- `Scope-creep risk`: Medium if the comparator expands into full gloss-dependent stitching or
  dictionary-based production.
- `Evaluation risk`: High if retrieval outputs are judged only by motion realism without semantic
  adequacy checks.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after retrieval feature choice, index construction,
  split-leakage policy, and evaluation protocol are fixed.
- `Open questions for scorecard`: retrieval feature type, train/validation/test leakage
  prevention, whether retrieval is text-based or pose-based, treatment of unmatched queries,
  transition smoothing, and semantic adequacy checks.
- `Evidence gaps to resolve before selection decision`: project-specific retrieval feasibility,
  no-gloss adaptation, comparison against direct baseline and primary model candidates, and
  metric limitations.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
