# Retrieval-Augmented Pose Comparator Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-RETRIEVAL-AUGMENTED-POSE-COMPARATOR`
- `Candidate card link`: [Retrieval-Augmented Pose Comparator](../candidate-cards/retrieval-augmented-pose-comparator.md)
- `Candidate record type`: `counter_alternative_candidate`
- `Originating family ID`: `FAM-RETRIEVAL-STITCHING-PRIMITIVES`
- `Originating family link`: [Retrieval, Stitching, and Motion-Primitives Alternatives](../candidate-universe/retrieval-stitching-primitives.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Comparator scorecard; this is not a primary model contribution scorecard.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020), [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020), [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Retrieval, Stitching, and Motion-Primitives Alternatives](../candidate-universe/retrieval-stitching-primitives.md)
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

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `15.0`
- `Evidence used`: Candidate-card comparator boundary; [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021)
- `Criterion-level rationale`: Retrieval can be isolated as a comparator or counter-alternative against direct generation, but it is not an additive model mechanism over the baseline.
- `Unresolved caveat`: Its score should not be compared one-to-one with primary model candidate scores as if it were the same contribution type.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `2.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card data assumptions; [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Criterion-level rationale`: A no-public-gloss retrieval comparator is possible from paired text/pose examples, but most strong retrieval, stitching, dictionary, or primitive evidence is gloss-dependent, dictionary-dependent, or adaptation-heavy.
- `Unresolved caveat`: Retrieval-safe train/validation/test separation and no-public-gloss adaptation must be specified.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `2.75`
- `Weighted contribution`: `10.3`
- `Evidence used`: Candidate-card expected mechanism; [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Criterion-level rationale`: Retrieval or reuse can preserve real motion and challenge learned generators, but it risks semantic mismatch, discontinuity, and poor coverage when no gloss dictionary is available.
- `Unresolved caveat`: Realistic retrieved motion may still be wrong for the source sentence.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `2.25`
- `Weighted contribution`: `8.4`
- `Evidence used`: [Retrieval, Stitching, and Motion-Primitives Alternatives](../candidate-universe/retrieval-stitching-primitives.md); [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-TEXT2SIGN-2020](../../source-corpus.md#src-text2sign-2020), [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020), [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Criterion-level rationale`: Evidence supports retrieval, stitching, staged production, motion-primitives, and sparse-keyframe alternatives as serious comparison pressure, but much of it is transferable, boundary, or frontier evidence rather than direct How2Sign no-gloss support.
- `Unresolved caveat`: Direct source support for a minimal no-public-gloss retrieval comparator remains limited.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `2.25`
- `Weighted contribution`: `5.6`
- `Evidence used`: Candidate-card limiting evidence; [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Criterion-level rationale`: Gloss-dependency, dictionary dependency, leakage risk, semantic-adequacy risk, and modern latent generative alternatives are serious counter-evidence against treating retrieval as a primary production mechanism.
- `Unresolved caveat`: Retrieval remains useful as a sanity-check comparator even if it is not a strong primary path.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `2.75`
- `Weighted contribution`: `10.3`
- `Evidence used`: Candidate-card evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Criterion-level rationale`: Retrieval can be evaluated with nearest-neighbor diagnostics and shared pose/embedding/qualitative checks, but leakage and semantic adequacy risks are high.
- `Unresolved caveat`: Evaluation must avoid rewarding copied or near-duplicate references from invalid splits.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `8.8`
- `Evidence used`: Candidate-card risk estimate; [SRC-SIGN-STITCHING-2024](../../source-corpus.md#src-sign-stitching-2024), [SRC-MIXED-SIGNALS-2021](../../source-corpus.md#src-mixed-signals-2021), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Implementation cost is relatively low and failure can still provide a useful sanity check, provided leakage-safe indexing and documented retrieval features are enforced.
- `Unresolved caveat`: Evaluation leakage or hidden dictionary assumptions would undermine the comparator.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `65.9`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: Retrieval and stitching evidence is technically meaningful, but direct no-public-gloss compatibility is conditional and much support remains boundary or counter-alternative evidence.

## Score Interpretation Boundary

- `Score interpretation`: This is a comparator/counter-alternative score.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: A leakage-safe, no-public-gloss retrieval design with clear semantic-adequacy evaluation would improve readiness; gloss or dictionary dependence would reduce it.
- This score should not be compared one-to-one with primary model candidate scores as if it were a primary neural generation mechanism.
- It may still be useful for downstream audit comparison.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Retrieval feature type, no-public-gloss adaptation, and example-bank construction.
- `Evaluation gaps`: Retrieval leakage policy, semantic adequacy checks, qualitative inspection, and metric limitations.
- `Implementation unknowns`: Index construction, tie-breaking, unmatched queries, transition handling, and split-safe feature caching.
- `Decision risks`: Treating retrieval realism as semantic correctness would overstate the comparator.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
