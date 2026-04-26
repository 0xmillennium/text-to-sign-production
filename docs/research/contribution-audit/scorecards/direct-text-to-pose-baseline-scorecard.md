# Direct Text-to-Pose Baseline Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-M0-DIRECT-TEXT-TO-POSE-BASELINE`
- `Candidate card link`: [Direct Text-to-Pose Baseline](../candidate-cards/direct-text-to-pose-baseline.md)
- `Candidate record type`: `baseline_candidate`
- `Originating family ID`: `FAM-FOUNDATIONAL-TEXT-TO-POSE-BASELINES`
- `Originating family link`: [Foundational Text-to-Pose Baselines](../candidate-universe/foundational-text-to-pose-baselines.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Baseline readiness scorecard; this is not a model contribution scorecard.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Foundational Text-to-Pose Baselines](../candidate-universe/foundational-text-to-pose-baselines.md)
- `Evaluation family consulted`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)
- `Model evidence separated from boundary/evaluation/counter/frontier evidence`: Established.

## Raw Score Scale

| Raw score | Meaning |
| ---: | --- |
| `0` | Unsupported, incompatible, or blocking weakness. |
| `1` | Weak support with major unresolved risk. |
| `2` | Partial or conditional support with material gaps. |
| `3` | Strong support with manageable limitations. |
| `4` | Very strong support with clear evidence, compatibility, and evaluability. |

For risk-oriented criteria, a higher score means lower practical risk and better audit readiness.

## Criteria

### Baseline Additivity and Isolation (`20`)

- `Raw score (0-4)`: `4.0`
- `Weighted contribution`: `20.0`
- `Evidence used`: Candidate-card role as M0 comparison floor; [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Criterion-level rationale`: The candidate is the baseline floor itself, so its isolated variable is clean: direct text/transcript-to-pose/keypoint prediction without learned tokens, diffusion, structure-aware grouping, semantic objectives, retrieval, gloss input, or avatar output.
- `Unresolved caveat`: The exact baseline architecture and preprocessing must still be fixed before implementation.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card data assumptions; [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020)
- `Criterion-level rationale`: The baseline is gloss-free at the linguistic-supervision level and aligns with text/transcript-to-pose/keypoint artifacts; How2Sign and recent How2Sign-direct work support the practical data regime.
- `Unresolved caveat`: Compatibility remains conditional on stable paired text/transcript, pose/keypoint arrays, split metadata, and batching.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `11.3`
- `Evidence used`: Candidate-card expected role; [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Criterion-level rationale`: Direct text-to-pose is mechanistically plausible as a minimal production baseline and failure-mode exposer, but the candidate is not intended to resolve oversmoothing, mode averaging, semantic drift, or articulator loss by itself.
- `Unresolved caveat`: A high readiness score here does not imply the baseline is expected to produce strong sign-language production quality.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `11.3`
- `Evidence used`: [Foundational Text-to-Pose Baselines](../candidate-universe/foundational-text-to-pose-baselines.md); [SRC-PROGTRANS-2020](../../source-corpus.md#src-progtrans-2020), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Criterion-level rationale`: The source corpus and baseline family strongly support a direct baseline as comparison context, but this is baseline and boundary evidence rather than evidence for a novel method contribution.
- `Unresolved caveat`: Several foundational sources are cross-dataset or older than current How2Sign-direct work.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card limiting evidence; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Learned representation evidence limits the baseline as a contribution candidate, and evaluation evidence limits metric interpretation, but neither undermines the need for a minimal M0 comparison floor.
- `Unresolved caveat`: A weak baseline could exaggerate later candidate gains if metric coverage is incomplete.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Criterion-level rationale`: The baseline gives later candidates a clear comparison floor, and the evaluation family supports using pose/keypoint, embedding, cautious back-translation, and qualitative checks rather than one metric alone.
- `Unresolved caveat`: Qualitative inspection and metric limitations must be recorded so baseline comparisons are not overread.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `4.0`
- `Weighted contribution`: `10.0`
- `Evidence used`: Candidate-card workflow and risk estimate; [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Criterion-level rationale`: Practical risk is low relative to representation-heavy, diffusion, retrieval, or avatar candidates because failure still provides useful baseline evidence and does not require specialized annotations or heavy new model machinery.
- `Unresolved caveat`: Reproducibility still depends on deterministic preprocessing, stable splits, and documented evaluation.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `86.3`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The baseline role is well supported by candidate-card, source-corpus, family, dataset, and evaluation evidence, while implementation details and expected baseline quality remain conditional.

## Score Interpretation Boundary

- `Score interpretation`: This score supports baseline readiness only. It indicates suitability as the M0 comparison floor, not expected task-solving quality and not model contribution strength.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: A missing paired artifact schema, unstable split policy, or evaluation protocol unable to compare later candidates would reduce readiness.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Exact baseline architecture and target pose/keypoint representation.
- `Evaluation gaps`: Metric set, qualitative inspection protocol, and handling of metric limitations.
- `Implementation unknowns`: Text/pose pairing, batching, sequence length handling, and deterministic preprocessing.
- `Decision risks`: Treating baseline readiness as model contribution strength would misread this record.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
