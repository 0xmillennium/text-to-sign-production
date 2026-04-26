# Evaluation and Benchmark Methodology

This record is for family-level audit-universe framing only. It is not a candidate-level record.
Representative sources cite stable entries in the [Source Corpus](../../source-corpus.md), using
`SRC-*` IDs and heading anchors rather than duplicated bibliography entries.

## Family ID

`FAM-EVALUATION-BENCHMARK-METHODOLOGY`

## Family Name

Evaluation and Benchmark Methodology

## Family Type / Status

`evaluation_family`

This is not a candidate-generating model family. It defines evaluation, benchmarking, and
metric-design evidence that later candidate families and candidate cards must respect.

## Scope Definition

This family covers evaluation methodology for text-to-sign-pose and sign language production. It
includes pose-based evaluation, distance-based metrics over keypoints or skeletons,
embedding-based pose or motion metrics, back-translation-based evaluation, benchmark and
challenge methodology, hidden test-set or standardized evaluation protocols, human-correlation or
perceptual evaluation concerns, metric limitations especially for sign-language production,
dataset/evaluation taxonomy as background evidence, and How2Sign-direct evaluation precedent
where relevant.

The family focuses on the evaluation question: how will candidate methods be compared, validated,
and interpreted? It does not collapse evaluation methodology into learned pose/motion
representation, latent generative production, structure-aware articulator modeling, text-pose
semantic alignment, retrieval/stitching, 3D avatar frontier, or dataset-supervision boundary;
those are separate family records.

## Included Directions / Subfamilies

- Pose-distance evaluation.
- Embedding-based pose or motion evaluation.
- Back-translation evaluation.
- Human-correlation or perceptual-evaluation concerns.
- Benchmark/challenge protocol.
- Hidden-test or standardized evaluation infrastructure.
- Metric limitations and negative evidence.
- Dataset and evaluation taxonomy as background context.
- How2Sign-direct model evaluation precedent.

## Explicit Exclusions / Out-of-Scope Directions

- Model architectures treated as candidates merely because they report evaluation results.
- Learned representation, latent generation, semantic alignment, or structure-aware methods as
  model families.
- Dataset-selection decisions, except where they affect benchmark interpretation.
- Rendering or avatar visualization as evaluation methodology unless explicitly tied to
  evaluation.
- Recognition-only benchmarks with no production or pose-output relevance.
- Treating back-translation metrics as sufficient without acknowledging their limitations.
- Treating evaluation-family evidence as a model selection result.

Model papers that provide useful evaluation precedent are not excluded categorically. In this
family, they are classified as evaluation-supporting evidence only.

## Source-Selection Basis

- `Relevant source-selection criteria`: Relation to text-to-sign-pose production,
  evaluation-method relevance, benchmark relevance, extractable metric/protocol detail, dataset
  compatibility, evidence quality, and downstream audit relevance.
- `Evidence classes represented`: `evaluation_evidence`, `dataset_evidence`,
  `reproducibility_evidence`, `direct_task_evidence`, `background_evidence`, and
  `negative_evidence`.
- `Dataset compatibility boundary`: `how2sign_direct` and `how2sign_compatible` sources provide
  strongest direct evaluation relevance when their metrics or protocols map to the project's
  pose/keypoint regime. `cross_lingual_transferable` sources may support evaluation methodology
  when the metric or benchmark concept transfers methodologically.
- `Gloss dependency boundary`: Evaluation sources can be useful under `gloss_free`,
  `gloss_adaptable`, or `gloss_dependent` regimes, but metrics relying on gloss, recognition, or
  back-translation must be interpreted with supervision and task assumptions stated explicitly.

## Representative Source-Corpus Entries

### Source 1

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Relevance type`: Core pose-based evaluation methodology evidence.
- `Claim supported`: Evaluation for sign-language production should consider distance-based,
  embedding-based, and back-translation-based pose evaluation families rather than relying on one
  metric family alone.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This source informs evaluation design; it does not select a model
  contribution.

### Source 2

- `Source corpus entry`: `SRC-SLRTP2025-CHALLENGE`
- `Source link`: [SRC-SLRTP2025-CHALLENGE](../../source-corpus.md#src-slrtp2025-challenge)
- `Relevance type`: Benchmark and challenge methodology evidence.
- `Claim supported`: Standardized Text-to-Pose evaluation and benchmark design are central to
  comparing production systems.
- `Evidence note`: SLRTP2025 Challenge is recorded as a core source for standardized Text-to-Pose
  evaluation and benchmark design.
- `Interpretation boundary`: Its benchmark setting is `cross_lingual_transferable`; adaptation
  to the project's How2Sign-style regime must be documented.

### Source 3

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Relevance type`: How2Sign-direct production evaluation precedent.
- `Claim supported`: Recent How2Sign-direct production work raises expectations for evaluating
  text-to-keypoint or text/audio-to-keypoint models.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Its architecture belongs to model-family records; this family uses
  it only for evaluation precedent and How2Sign-direct comparison pressure.

### Source 4

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Relevance type`: Evaluation caution and baseline-evaluation evidence.
- `Claim supported`: Back-translation reliability should be treated cautiously when interpreting
  SLP results.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source
  that critiques back-translation reliability.
- `Interpretation boundary`: This source is older baseline/evaluation caution evidence, not
  sufficient evaluation methodology by itself.

### Source 5

- `Source corpus entry`: `SRC-SURVEY-2024`
- `Source link`: [SRC-SURVEY-2024](../../source-corpus.md#src-survey-2024)
- `Relevance type`: Background evaluation taxonomy and coverage evidence.
- `Claim supported`: Survey-level coverage can help check whether the project acknowledges major
  dataset and evaluation categories.
- `Evidence note`: The survey is recorded as useful for taxonomy and coverage sanity checking,
  not as a primary decision-bearing method source.
- `Interpretation boundary`: This is background context only; primary evaluation decisions must
  rely on more specific evaluation or benchmark sources.

## Supervision And Representation Boundary

- `Gloss-free compatibility`: Evaluation methods are directly compatible when they assess
  generated pose, keypoints, skeletons, motion tokens, or learned representations without
  requiring manual gloss annotations.
- `Gloss-dependent risks`: Evaluation protocols involving glosses, sign recognition,
  back-translation, or gloss-conditioned references may be useful but must state their supervision
  assumptions and possible bias.
- `Intermediate representation type`: This family may evaluate continuous pose, skeletons,
  keypoints, learned tokens, latent motion representations, generated video, or back-translated
  text depending on the candidate being assessed.
- `Pose/keypoint/skeleton/3D compatibility`: Pose/keypoint/skeleton evaluation is directly
  aligned with the project. Avatar, video, or 3D parametric evaluation may be relevant for
  frontier or downstream visualization but should not be imposed on near-term keypoint-level
  candidates unless scoped explicitly.

## Method-Family Rationale

This family exists because contribution quality cannot be established only by model design. It
prevents three common errors: treating back-translation metrics as sufficient without pose-level
or semantic caution, comparing candidate families without a shared benchmark or metric plan, and
letting model-family claims outrun what the available evaluation can actually measure.

## Counter-Evidence And Limitations

- `Known limitations`: SLP evaluation may depend on imperfect back-translation systems, noisy
  keypoints, weak human-correlation evidence, dataset-specific benchmarks, or metrics that fail to
  capture sign intelligibility, timing, non-manuals, and semantic fidelity.
- `Conflicting evidence`: Some evaluation evidence is `how2sign_direct`, some is
  `how2sign_compatible`, some is `cross_lingual_transferable`, some is benchmark-specific, and
  some is background/taxonomy evidence rather than primary protocol evidence.
- `Reasons this family may not become a candidate`: This is an evaluation family, not a
  model-contribution family. It may produce evaluation-plan artifacts, but it should not become a
  model candidate unless the project explicitly defines evaluation methodology itself as a
  contribution.

## Potential Candidate Directions

This family may later produce evaluation-plan or candidate-support records such as a standard
pose-evaluation notebook, back-translation metric caution protocol, pose-distance and
embedding-metric comparison, How2Sign-compatible evaluation checklist, qualitative
visual-inspection protocol, benchmark adaptation note, or evaluation failure-mode ledger.
Candidate cards must not be instantiated here.

## Relation To The Overall Audit

This family is upstream of candidate scoring and selection. It should help later audit records
distinguish model-family evidence, evaluation evidence, benchmark compatibility, metric
limitations, baseline comparison, qualitative inspection, and final candidate-selection
confidence.

## Family-Level Evidence Boundary

This family does not select a model contribution. It does not rank candidate families, claim that
any single metric is sufficient, or convert evaluation precedent into model evidence. It only
defines the evaluation and benchmark methodology universe for later candidate records,
scorecards, and selection decisions.

## Downstream Record Policy

Downstream candidate-universe, candidate-card, scorecard, and selection-decision records must cite
`source-corpus.md` entries by `SRC-*` ID, distinguish model evidence from evaluation evidence,
state whether metrics are `how2sign_direct`, `how2sign_compatible`, or
`cross_lingual_transferable`, explicitly identify whether evaluation depends on back-translation,
pose distance, embedding similarity, human judgment, benchmark protocol, or qualitative
inspection, document metric limitations before using scores as selection evidence, and avoid
treating automatic metrics as complete evidence of sign intelligibility or semantic adequacy.

## Record Status

- No concrete candidate is instantiated here.
- No candidate-level evaluation is recorded here.
- No [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  or [Audit Result](../audit-result.md) content is recorded here.
