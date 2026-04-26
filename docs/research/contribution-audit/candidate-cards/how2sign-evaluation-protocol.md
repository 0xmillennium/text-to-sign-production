# How2Sign-Compatible Evaluation Protocol

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-HOW2SIGN-EVALUATION-PROTOCOL`

## Candidate Name

How2Sign-Compatible Evaluation Protocol

## Candidate Record Type

- `Candidate record type`: `evaluation_support_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-EVALUATION-BENCHMARK-METHODOLOGY`
- `Family link`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)
- `Family type/status`: `evaluation_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the
  evaluation and benchmark methodology family as a concrete protocol artifact for judging later
  candidate cards and scorecards under the project's How2Sign-style text/pose/keypoint regime.

It is acceptable that this candidate derives from an `evaluation_family`, because it is an
evaluation-support candidate, not a model contribution.

## Candidate-Level Technical Definition

A How2Sign-compatible evaluation protocol for text/transcript-to-pose/keypoint production that
combines pose/keypoint distance metrics, embedding-based checks where available, cautious
back-translation or recognition-based checks where appropriate, qualitative visual inspection,
and explicit metric-limitation documentation.

## Candidate Boundary

- `Included mechanism`: Evaluation protocol for generated sign pose/keypoint sequences under a
  How2Sign-style text/video/keypoint regime, including metric families, comparison rules,
  qualitative inspection, and failure-mode logging.
- `Explicitly excluded mechanisms`: Model architecture, training objective, learned
  representation, latent generator, retrieval system, 3D/avatar implementation, final scorecard
  score, selection decision, or whole-audit result.
- `Not to be confused with`: Scorecards that apply scores to a candidate, model candidate cards
  that propose mechanisms, or benchmark papers that define external tasks but do not directly
  instantiate the project's protocol.

## Research Role In Current Audit

- `Research role label`: `evaluation_support_candidate`

This record defines the evaluation support surface needed before primary model candidates are
scored or considered by downstream decision records.

## Expected Improvement Mechanism

This candidate does not improve generated motion directly. Its expected audit improvement is
methodological: it reduces metric overclaiming, exposes evaluation blind spots, standardizes
candidate comparisons, and forces later scorecards to document metric limitations. It supports
fair comparison across baseline, primary model, counter-alternative, and future-work candidates.

## Targeted Failure Modes

- `Applicable failure modes`: `evaluation_blind_spot`, `semantic_mismatch`,
  `timing_misalignment`, `hand_articulation_loss`, `non_manual_loss`, `temporal_jitter`,
  `unnatural_transitions`

These are failure modes the protocol should help detect or document, not necessarily solve.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_direct`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed for the minimum protocol;
  optional human inspection or annotation can be recorded separately if later introduced.
- `Segmentation required`: Sentence/video segment boundaries are assumed for aligning generated
  and reference sequences.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for the minimum keypoint-level protocol.
- `Current project artifacts sufficient`: Conditional; sufficient if generated pose/keypoint
  sequences, reference pose/keypoint sequences, text/transcript IDs, and stable split metadata are
  available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if generated
  and reference pose/keypoint arrays share a documented schema.
- `Compatible with current experiment workflow`: Expected to be compatible as a reusable
  evaluation module or notebook-backed report.
- `Heavy new preprocessing required`: Not expected for pose/keypoint distance metrics; embedding
  or back-translation metrics may require additional models or cached features.
- `Thin-driver notebook principle affected`: Not expected if metric computation is implemented as
  reusable library code and notebooks remain thin drivers.
- `Reproducibility risk`: Medium unless metric definitions, splits, preprocessing, and
  qualitative-inspection criteria are fixed.

## Evidence Chain

- `Source-selection basis`: Evaluation-method relevance, benchmark relevance, How2Sign
  compatibility, metric limitation awareness, and downstream audit relevance.
- `Source-corpus entries used`: `SRC-POSE-EVAL-2025`, `SRC-SLRTP2025-CHALLENGE`,
  `SRC-MS2SL-2024`, `SRC-NSLPG-2021`, `SRC-SURVEY-2024`, `SRC-HOW2SIGN-2021`
- `Candidate-universe family used`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Evaluation for sign-language production should consider multiple metric
  families rather than relying on one automatic score.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This source informs evaluation design; it does not select a model
  contribution.

### Source 2

- `Source corpus entry`: `SRC-SLRTP2025-CHALLENGE`
- `Source link`: [SRC-SLRTP2025-CHALLENGE](../../source-corpus.md#src-slrtp2025-challenge)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Standardized Text-to-Pose evaluation and benchmark design are important for
  comparing production systems.
- `Evidence note`: SLRTP2025 Challenge is recorded as a core source for standardized Text-to-Pose
  evaluation and benchmark design.
- `Interpretation boundary`: Its benchmark setting is `cross_lingual_transferable`; adaptation to
  the project's How2Sign-style regime must be documented.

### Source 3

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Support type`: `evaluation_support`
- `Claim supported`: Recent How2Sign-direct production work raises expectations for evaluating
  text-to-keypoint or text/audio-to-keypoint models.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Its architecture belongs to model-family records; this candidate
  uses it only for evaluation precedent and How2Sign-direct comparison pressure.

### Source 4

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Support type`: `boundary_support`
- `Claim supported`: Back-translation reliability should be treated cautiously when interpreting
  sign-language production results.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source that
  critiques back-translation reliability.
- `Interpretation boundary`: This source is older baseline/evaluation-caution evidence, not
  sufficient evaluation methodology by itself.

### Source 5

- `Source corpus entry`: `SRC-HOW2SIGN-2021`
- `Source link`: [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Support type`: `boundary_support`
- `Claim supported`: The evaluation protocol must respect the project's primary ASL
  text/video/keypoint data regime.
- `Evidence note`: How2Sign is recorded as the project's primary ASL text/video/keypoint dataset
  anchor.
- `Interpretation boundary`: Dataset compatibility does not imply public manual gloss
  availability.

### Source 6

- `Source corpus entry`: `SRC-SURVEY-2024`
- `Source link`: [SRC-SURVEY-2024](../../source-corpus.md#src-survey-2024)
- `Support type`: `evaluation_support`
- `Claim supported`: Survey-level coverage can help check that major dataset and evaluation
  categories are not ignored.
- `Evidence note`: The survey is recorded as useful for taxonomy and coverage sanity checking,
  not as a primary decision-bearing method source.
- `Interpretation boundary`: This is background context only; primary evaluation decisions must
  rely on specific evaluation or benchmark sources.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Back-translation or recognition-based checks cannot be treated as sufficient
  evidence of sign intelligibility or semantic adequacy.
- `Evidence note`: NSLP-G is recorded as a source that critiques back-translation reliability.
- `Interpretation boundary`: This limitation supports a multi-metric protocol; it does not
  invalidate using back-translation as one cautious secondary signal.

### Source 2

- `Source corpus entry`: `SRC-SLRTP2025-CHALLENGE`
- `Source link`: [SRC-SLRTP2025-CHALLENGE](../../source-corpus.md#src-slrtp2025-challenge)
- `Limitation type`: `dataset_limitation`
- `Claim limited`: External benchmark protocols may not transfer directly to the project's
  How2Sign-style regime.
- `Evidence note`: The SLRTP2025 Challenge is recorded as standardized Text-to-Pose benchmark
  evidence but with cross-lingual-transferable adaptation requirements.
- `Interpretation boundary`: This limits direct benchmark adoption; it does not prevent adapting
  benchmark principles.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: A How2Sign-compatible evaluation protocol is necessary before candidate
  scorecards can be interpreted reliably.
- `Least-secure claim`: The exact metric set and thresholds are not fixed by the candidate card
  alone and depend on available artifacts, generated outputs, and evaluation implementation.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The need for multi-metric and benchmark-aware evaluation is
  well supported, but the project-specific metric implementation remains conditional.

## Comparison To Strong Alternatives

Relying only on pose/keypoint distance, relying only on back-translation, relying only on visual
inspection, or adopting an external benchmark protocol without adaptation would each leave a
different evaluation blind spot. This protocol is stronger because it treats those approaches as
complementary signals with explicit limitations.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Each future model candidate should be evaluated with and without its added
  mechanism against the same baseline and split policy.
- `Primary metric family`: Pose/keypoint distance metrics.
- `Secondary metric family`: Embedding-based checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: No single automatic metric should be treated as sufficient evidence
  of sign intelligibility, timing, non-manual adequacy, or semantic fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Not applicable as a model mechanism; applicable as an
  evaluation-support layer over all candidate comparisons.
- `Baseline changed`: No.
- `Primary isolated variable`: Evaluation protocol design rather than model architecture.
- `Requires another candidate`: No, but it is most useful once baseline and model outputs exist.
- `Independent ablation possible`: Partially; metric-family choices can be compared, but the
  protocol itself is an audit support artifact.
- `Likely interactions with other candidate families`: All primary model, counter-alternative,
  baseline, and future-work candidates should use this protocol or explicitly justify deviations.

## Complexity / Risk Estimate

- `Implementation complexity`: Low to medium.
- `Failure cost`: Medium; weak evaluation can distort later scorecards and selection decisions.
- `Hidden dependency risk`: Medium if embedding or back-translation models are required.
- `Scope-creep risk`: Medium if human evaluation or external benchmark replication is expanded
  too early.
- `Evaluation risk`: High if the protocol relies on a single metric family; lower if multi-metric
  limitations are documented.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after metric definitions, split policy, output schema,
  and qualitative-inspection criteria are fixed.
- `Open questions for scorecard`: exact metric set, aggregation policy, qualitative review format,
  treatment of missing non-manual or face keypoints, and whether embedding/back-translation models
  are available.
- `Evidence gaps to resolve before selection decision`: implementation of metric scripts,
  documented metric limitations, and candidate-output comparability.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
