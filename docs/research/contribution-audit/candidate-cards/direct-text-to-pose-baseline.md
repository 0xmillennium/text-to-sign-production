# Direct Text-to-Pose Baseline

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-M0-DIRECT-TEXT-TO-POSE-BASELINE`

## Candidate Name

Direct Text-to-Pose Baseline

## Candidate Record Type

- `Candidate record type`: `baseline_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES`
- `Family link`: [FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES](../candidate-universe/foundational-text-to-pose-baselines.md)
- `Family type/status`: `boundary_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the
  baseline floor defined by the foundational text-to-pose family so later model candidates can be
  compared against a minimal direct text-to-pose reference.

It is acceptable that this candidate derives from a `boundary_family`, because it is a
baseline/ablation candidate, not a primary model contribution.

## Candidate-Level Technical Definition

A minimal text/transcript-to-pose/keypoint baseline that maps spoken-language text input directly
to sign pose/keypoint sequences without learned pose-token bottlenecks, latent diffusion,
articulator-disentangled latent groups, retrieval/stitching, gloss supervision, or 3D/avatar
output.

## Candidate Boundary

- `Included mechanism`: Direct text/transcript-to-pose or text/transcript-to-keypoint sequence
  prediction using a minimal baseline architecture.
- `Explicitly excluded mechanisms`: Learned pose tokens, VQ/dynamic VQ, latent diffusion,
  articulator-disentangled latent groups, semantic consistency objectives, retrieval/stitching,
  gloss-to-pose, HamNoSys/notation-to-pose, 3D avatar, SMPL-X, and rendering.
- `Not to be confused with`: Primary model candidates that introduce learned representations,
  latent generative architectures, structure-aware modeling, semantic alignment objectives, or
  frontier avatar outputs.

## Research Role In Current Audit

- `Research role label`: `baseline_or_ablation_candidate`

This record defines the M0 comparison point for later candidate evaluation. It is not itself a
final contribution selection.

## Expected Improvement Mechanism

This candidate is not expected to solve the main modeling weaknesses by itself. Its role is to
expose baseline failure modes and provide a minimum comparison point for later candidate
improvements. Later candidates should show improvement over this baseline on motion quality,
semantic alignment, structure preservation, or evaluability.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `temporal_jitter`, `timing_misalignment`,
  `hand_articulation_loss`, `non_manual_loss`, `cross_channel_inconsistency`,
  `semantic_mismatch`, `mode_averaging`, `sequence_drift`

This baseline primarily exposes these failure modes rather than resolving them.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_direct`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual linguistic annotation assumed beyond text/transcript and
  pose/keypoint supervision.
- `Segmentation required`: Sentence/video segment boundaries are assumed; no additional gloss
  segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required.
- `Current project artifacts sufficient`: Conditional; sufficient if text/transcript-to-pose/keypoint
  training pairs and compatible pose/keypoint artifacts are available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if the current
  artifact schema exposes text/transcript and pose/keypoint sequences.
- `Compatible with current experiment workflow`: Expected to be compatible as a minimal baseline
  experiment.
- `Heavy new preprocessing required`: Not expected, beyond standard text/pose pairing and batching.
- `Thin-driver notebook principle affected`: Not expected.
- `Reproducibility risk`: Lower than representation-heavy, diffusion, retrieval, or avatar
  candidates; still depends on deterministic preprocessing and documented splits.

## Evidence Chain

- `Source-selection basis`: Task relevance, direct text-to-pose baseline relevance,
  dataset/supervision compatibility, and downstream comparison relevance.
- `Source-corpus entries used`: `SRC-PROGTRANS-2020`, `SRC-MULTICHANNEL-MDN-2021`,
  `SRC-NSLPG-2021`, `SRC-MS2SL-2024`, `SRC-HOW2SIGN-2021`
- `Candidate-universe family used`: [FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES](../candidate-universe/foundational-text-to-pose-baselines.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-PROGTRANS-2020`
- `Source link`: [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Direct text-to-pose SLP is an established baseline task and comparison point.
- `Evidence note`: Progressive Transformers is recorded as a foundational text-to-pose baseline.
- `Interpretation boundary`: This source supports baseline framing, not a sufficient modern
  contribution direction by itself.

### Source 2

- `Source corpus entry`: `SRC-MULTICHANNEL-MDN-2021`
- `Source link`: [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Multi-channel 3D pose production and user-evaluation precedent are part of
  the baseline context.
- `Evidence note`: The source is recorded as an extended multi-channel baseline.
- `Interpretation boundary`: It strengthens baseline framing but does not define a current
  primary candidate by itself.

### Source 3

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Support type`: `transferable_method_support`
- `Claim supported`: Non-autoregressive latent-space SLP and evaluation caution are part of the
  foundational baseline landscape.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source that
  critiques back-translation reliability.
- `Interpretation boundary`: This source is older baseline/evaluation-caution evidence; it should
  not be treated as sufficient modern baseline methodology by itself.

### Source 4

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Support type`: `evaluation_support`
- `Claim supported`: Recent How2Sign-direct production work raises the expected baseline and
  evaluation standard for text-to-keypoint or text/audio-to-keypoint systems.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Its diffusion and multimodal architecture belong to later
  model-family records; this candidate uses it only for baseline pressure and evaluation
  precedent.

### Source 5

- `Source corpus entry`: `SRC-HOW2SIGN-2021`
- `Source link`: [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Support type`: `boundary_support`
- `Claim supported`: How2Sign defines the primary practical text/video/keypoint data regime that
  the baseline should respect.
- `Evidence note`: How2Sign is recorded as the project's primary ASL text/video/keypoint dataset
  anchor.
- `Interpretation boundary`: Dataset compatibility does not imply public manual gloss
  availability.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Automatic baseline improvement cannot be interpreted from one metric family
  alone.
- `Evidence note`: The evaluation family records pose-distance, embedding-based, and
  back-translation-based evaluation concerns.
- `Interpretation boundary`: This does not invalidate the baseline; it limits how baseline
  comparisons should be interpreted.

### Source 2

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Direct raw-pose baseline may be weaker than learned representation approaches
  for preserving motion structure.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This is counter-evidence against treating the baseline as a
  contribution, not against using it as M0.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: A minimal direct text-to-pose baseline is necessary as the M0 comparison
  floor for later candidate evaluation.
- `Least-secure claim`: The exact architecture and expected baseline quality are not fixed by the
  family record alone and depend on available artifacts and evaluation implementation.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The baseline role is strongly supported as an audit need, but
  implementation details and expected quality remain conditional.

## Comparison To Strong Alternatives

Learned pose-token bottlenecks, articulator-disentangled latent modeling, gloss-free latent
diffusion, text-pose semantic consistency objectives, and retrieval-augmented comparators may
outperform or extend this baseline. The baseline is still needed to measure additivity and
improvement for those alternatives without treating the minimal reference as a model contribution.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: This candidate itself is the M0 baseline.
- `Required ablation`: Later candidates should compare against this baseline without their added
  mechanism.
- `Primary metric family`: Pose/keypoint distance metrics.
- `Secondary metric family`: Embedding-based metrics and cautious back-translation checks where
  available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Automatic metrics are insufficient alone for sign intelligibility,
  timing, non-manuals, and semantic adequacy.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Not applicable; this candidate is the baseline.
- `Baseline changed`: No.
- `Primary isolated variable`: Minimal direct text/transcript-to-pose/keypoint mapping without
  additional candidate mechanisms.
- `Requires another candidate`: No.
- `Independent ablation possible`: Yes; later candidates can remove their added mechanism and
  compare against this baseline.
- `Likely interactions with other candidate families`: All primary model candidates should use
  this candidate as a comparison floor or ablation reference.

## Complexity / Risk Estimate

- `Implementation complexity`: Low to medium.
- `Failure cost`: Low; failure still informs baseline weakness.
- `Hidden dependency risk`: Low, provided the text/pose pairing and splits are stable.
- `Scope-creep risk`: Low.
- `Evaluation risk`: Medium, because weak baselines can exaggerate later candidate gains if
  metrics are incomplete.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after the exact baseline architecture, artifact
  schema, and evaluation protocol are fixed.
- `Open questions for scorecard`: exact architecture, split policy, metric set, qualitative
  inspection protocol.
- `Evidence gaps to resolve before selection decision`: baseline implementation details and
  evaluation reliability.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
