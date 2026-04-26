# Articulator-Disentangled Latent Modeling

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-ARTICULATOR-DISENTANGLED-LATENT`

## Candidate Name

Articulator-Disentangled Latent Modeling

## Candidate Record Type

- `Candidate record type`: `model_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING`
- `Family link`: [FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING](../candidate-universe/structure-aware-articulator-modeling.md)
- `Family type/status`: `primary_candidate_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the
  structure-aware and articulator-aware modeling family as a concrete test of whether body, hand,
  face, channel, or articulator-specific latent structure improves over flat text-to-pose
  generation.

## Candidate-Level Technical Definition

A gloss-free structure-aware modeling candidate that separates or regularizes sign pose generation
through articulator-specific latent groups, body/hand/face channels, skeleton partitions, or
channel-aware losses so that generated pose preserves sign-relevant structure better than flat
coordinate regression. The candidate focuses on the structure question: how should the model
preserve sign-relevant body, hand, face, channel, and articulator structure?

## Candidate Boundary

- `Included mechanism`: Articulator-specific latent grouping, body/hand/face channel separation,
  channel-aware regularization, skeleton or keypoint partitioning, and lightweight
  non-manual-aware modeling derived from pose/keypoint data.
- `Explicitly excluded mechanisms`: Learned pose-token bottleneck as the primary contribution,
  latent diffusion as the primary generation mechanism, semantic consistency objective as the
  primary contribution, retrieval/stitching, gloss-to-pose, HamNoSys/notation-to-pose, 3D avatar,
  SMPL-X, and rendering.
- `Not to be confused with`: Learned pose-token representation candidates, latent generative
  production candidates, semantic alignment candidates, retrieval comparators, gloss/notation
  boundary evidence, or frontier avatar candidates.

Variational articulator-wise modeling and iterative latent refinement are strong variants or
adjacent methods. This card focuses on the minimal structure-aware articulator-disentangled
candidate.

## Research Role In Current Audit

- `Research role label`: `primary_model_candidate`

This record defines a primary model candidate to compare against the M0 baseline and to score only
in downstream scorecard surfaces.

## Expected Improvement Mechanism

This candidate is expected to reduce flat-coordinate generation weaknesses by forcing the model to
preserve articulator-level structure. Expected mechanisms include reducing hand articulation loss,
reducing non-manual feature loss, reducing cross-channel inconsistency, preserving body/hand/face
structure, making errors more diagnosable by separating channel-specific behavior, and improving
pose plausibility without requiring manual gloss supervision. Gains are conditional on stable
keypoint partitions, reliable body/hand/face channels, and avoiding excessive model complexity.

## Targeted Failure Modes

- `Applicable failure modes`: `hand_articulation_loss`, `non_manual_loss`,
  `cross_channel_inconsistency`, `temporal_jitter`, `timing_misalignment`,
  `unnatural_transitions`, `sequence_drift`, `oversmoothing`; secondary risk:
  `semantic_mismatch`

This candidate mainly targets structure and articulator-quality failure modes. Semantic alignment
may require a separate candidate or auxiliary objective.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_compatible`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed; articulator groups are derived
  from pose/keypoint schema.
- `Segmentation required`: Sentence/video segment boundaries are assumed; no additional gloss
  segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for the minimal keypoint-level candidate.
- `Current project artifacts sufficient`: Conditional; sufficient if pose/keypoint artifacts
  expose stable body, hand, and face/non-manual channels or masks.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if the schema
  exposes stable body/hand/face keypoint groups, masks, or channel partitions.
- `Compatible with current experiment workflow`: Expected to be compatible if channel
  partitioning, losses, and diagnostics are implemented as reusable modules.
- `Heavy new preprocessing required`: Low to moderate; requires defining keypoint partitions and
  masks, but not a new dataset or manual linguistic annotation.
- `Thin-driver notebook principle affected`: Not expected if structure-aware losses and partition
  utilities live outside notebooks.
- `Reproducibility risk`: Medium; depends on stable schema definitions, deterministic channel
  partitioning, and documented loss weighting.

## Evidence Chain

- `Source-selection basis`: Structure-aware modeling relevance, articulator/channel/skeleton
  preservation, gloss-free or gloss-adaptable supervision, pose/keypoint production relevance,
  dataset compatibility, and downstream contribution-selection relevance.
- `Supporting source-corpus entries used`: `SRC-DARSLP-2025`, `SRC-MCST-2024`,
  `SRC-A2VSLP-2026`, `SRC-ILRSLP-2025`, `SRC-M3T-2026`, `SRC-MULTICHANNEL-MDN-2021`
- `Limiting source-corpus entries used`: `SRC-SIGNIDD-2025`, `SRC-TEXT2SIGNDIFF-2025`,
  `SRC-POSE-EVAL-2025`
- `Candidate-universe family used`: [FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING](../candidate-universe/structure-aware-articulator-modeling.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-DARSLP-2025`
- `Source link`: [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Articulator-specific latent structure and channel-aware regularization are
  credible structure-aware SLP directions.
- `Evidence note`: DARSLP is recorded as a core source for articulator-disentangled latent spaces
  and channel-aware regularization.
- `Interpretation boundary`: This supports the structure-aware candidate; it does not by itself
  select a concrete disentanglement architecture or loss design for this project.

### Source 2

- `Source corpus entry`: `SRC-MCST-2024`
- `Source link`: [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024)
- `Support type`: `transferable_method_support`
- `Claim supported`: Channel-aware spatial-temporal modeling is a credible architecture-level
  response to the structure of sign pose.
- `Evidence note`: MCST is recorded as supporting channel-aware architectural modeling across
  manual and non-manual articulators.
- `Interpretation boundary`: Dataset transfer must be argued because the source is recorded as
  `cross_lingual_transferable`, not How2Sign-direct.

### Source 3

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Articulator-wise latent modeling can be extended with distributional or
  variational structure.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: This is recent frontier evidence and should not automatically
  outweigh earlier reviewed structure-aware sources.

### Source 4

- `Source corpus entry`: `SRC-ILRSLP-2025`
- `Source link`: [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Structure-aware latent spaces can be refined iteratively rather than
  predicted once.
- `Evidence note`: ILRSLP is recorded as an important continuation of articulator-disentangled
  latent modeling.
- `Interpretation boundary`: Iterative refinement is adjacent to this candidate; it should not
  turn this card into a general latent-generation candidate.

### Source 5

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Support type`: `boundary_support`
- `Claim supported`: Recent work pressures structure-aware modeling toward body, hands, and
  face/non-manual separation.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a heavy 3D or parametric representation.

### Source 6

- `Source corpus entry`: `SRC-MULTICHANNEL-MDN-2021`
- `Source link`: [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Support type`: `boundary_support`
- `Claim supported`: Multi-channel 3D pose production is part of the baseline structure-aware
  context.
- `Evidence note`: The source is recorded as an extended multi-channel baseline.
- `Interpretation boundary`: It is foundational baseline evidence, not direct evidence for a
  modern structure-aware contribution candidate by itself.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGNIDD-2025`
- `Source link`: [SRC-SIGNIDD-2025](../../source-corpus.md#src-signidd-2025)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Bone direction, bone length, or relative-pose structure may be technically
  relevant while remaining gloss-to-pose constrained.
- `Evidence note`: Sign-IDD is recorded as useful for skeletal structure and bone-direction/length
  representation, but gloss-to-pose limits direct use.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 2

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Structure-aware modeling should not be assumed superior to latent generative
  approaches without evaluation.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This is counter-evidence against prematurely choosing the
  structure-aware route, not against evaluating it as a primary model candidate.

### Source 3

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Better channel separation or lower pose error alone cannot prove sign
  intelligibility, non-manual adequacy, or semantic fidelity.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate structure-aware modeling; it limits how
  structure-aware gains should be evaluated.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: Articulator-disentangled or channel-aware modeling is a credible primary
  model candidate for improving structure preservation over flat text-to-pose regression.
- `Least-secure claim`: The exact articulator partition, latent grouping, loss weighting, and
  expected improvement under the project's current keypoint schema are not established by the
  candidate card alone.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The structure-aware family is well supported across
  source-corpus entries, but project-specific feasibility depends on available body/hand/face
  channels, schema stability, loss design, and evaluation sensitivity.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md), this candidate
adds explicit articulator/channel structure. Compared with the
[Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md), it targets structure
preservation rather than representation tokenization. Gloss-free latent diffusion, text-pose
semantic consistency objectives, retrieval-augmented comparators, and frontier body/hand/face
tokenization or 3D/avatar approaches remain adjacent alternatives, but this candidate is distinct
because it does not center generative sampling, semantic conditioning, retrieval, or avatar
output.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Compare flat text-to-pose against structure-aware articulator/channel
  modeling with the same splits, input text, and output pose/keypoint schema; ablate
  channel-aware loss, articulator grouping, or structure-specific regularization.
- `Primary metric family`: Pose/keypoint distance metrics stratified by body, hands, and
  face/non-manual channels where possible.
- `Secondary metric family`: Embedding-based motion checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Improved channel-level error or structure preservation does not by
  itself prove sign intelligibility, timing, non-manual adequacy, or semantic fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Conditional; additive if articulator-aware structure is the primary
  mechanism changed relative to the direct baseline.
- `Baseline changed`: No, unless structure-aware preprocessing changes the output target
  representation enough to require a separate baseline variant.
- `Primary isolated variable`: Articulator/channel-aware latent grouping or regularization.
- `Requires another candidate`: No.
- `Independent ablation possible`: Yes; compare with and without articulator grouping,
  channel-aware regularization, or structure-specific losses under the same data and evaluation
  protocol.
- `Likely interactions with other candidate families`: May interact with learned pose-token
  bottlenecks, latent generative production, and semantic alignment; those should remain separate
  unless explicitly combined in a later candidate.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium.
- `Failure cost`: Medium; poor grouping or loss weighting may complicate training without
  improving production quality.
- `Hidden dependency risk`: Medium; depends on stable body/hand/face channel definitions, masks,
  and loss balancing.
- `Scope-creep risk`: Medium if variational modeling, iterative refinement, or heavy
  non-manual/3D modeling is added too early.
- `Evaluation risk`: Medium to high because channel-level improvements may not align with
  semantic adequacy or human-perceived quality.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after articulator partitions, structure-aware
  mechanism, loss weighting, and evaluation protocol are fixed.
- `Open questions for scorecard`: body/hand/face partition scheme, channel weights, non-manual
  handling, loss design, ablation plan, and whether variational or iterative refinement is
  included or excluded.
- `Evidence gaps to resolve before selection decision`: project-specific channel stability,
  effect of structure-aware losses, comparison against direct baseline and learned-token
  candidate, and metric limitations.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
