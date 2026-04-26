# Articulator-Disentangled Latent Modeling Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-ARTICULATOR-DISENTANGLED-LATENT`
- `Candidate card link`: [Articulator-Disentangled Latent Modeling](../candidate-cards/articulator-disentangled-latent.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING`
- `Originating family link`: [Structure-Aware and Articulator-Aware Modeling](../candidate-universe/structure-aware-articulator-modeling.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Full model scorecard.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021), [SRC-SIGNIDD-2025](../../source-corpus.md#src-signidd-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Structure-Aware and Articulator-Aware Modeling](../candidate-universe/structure-aware-articulator-modeling.md)
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

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `17.5`
- `Evidence used`: Candidate-card additivity plan; [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Criterion-level rationale`: Structure-aware losses, articulator grouping, and channel-aware regularization can be ablated against a flat text-to-pose baseline under shared data and output schema.
- `Unresolved caveat`: Isolation is conditional if partitioning or preprocessing changes the target representation enough to require a separate baseline variant.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `2.75`
- `Weighted contribution`: `10.3`
- `Evidence used`: Candidate-card data assumptions; [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Criterion-level rationale`: The approach is gloss-free when articulator groups are derived from pose/keypoint schema, but data and workflow compatibility are more conditional than token or latent-diffusion candidates because stable body/hand/face partitions and masks are required.
- `Unresolved caveat`: Missing or noisy channel partitions would materially weaken the candidate.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card expected mechanism; [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025)
- `Criterion-level rationale`: Articulator/channel modeling plausibly targets hand articulation loss, non-manual loss, and cross-channel inconsistency by making sign-relevant structure explicit instead of flattening all coordinates.
- `Unresolved caveat`: Structure preservation may not translate into sign-level semantic gains without sensitive evaluation.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: [Structure-Aware and Articulator-Aware Modeling](../candidate-universe/structure-aware-articulator-modeling.md); [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-MULTICHANNEL-MDN-2021](../../source-corpus.md#src-multichannel-mdn-2021)
- `Criterion-level rationale`: Evidence supports articulator-specific latent structure, channel-aware modeling, iterative or variational continuations, multi-modal token pressure, and foundational multi-channel baseline context.
- `Unresolved caveat`: Several sources are frontier, cross-lingual-transferable, or boundary context, so direct project fit is not fully settled.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card limiting evidence; [SRC-SIGNIDD-2025](../../source-corpus.md#src-signidd-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Gloss-dependent skeletal-structure evidence must remain boundary evidence, latent diffusion is a strong alternative, and evaluation evidence limits claims from channel-level gains alone.
- `Unresolved caveat`: If latent diffusion or token bottlenecks capture structure sufficiently, this candidate's incremental value may be harder to justify.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `2.75`
- `Weighted contribution`: `10.3`
- `Evidence used`: Candidate-card evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024)
- `Criterion-level rationale`: Channel-stratified metrics and ablations are possible, but proving sign-level gains from channel-level improvements is difficult and requires qualitative inspection and metric caution.
- `Unresolved caveat`: A channel-level error reduction may fail to reflect intelligibility, timing, non-manual adequacy, or semantic fidelity.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `2.75`
- `Weighted contribution`: `6.9`
- `Evidence used`: Candidate-card risk estimate; [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-MCST-2024](../../source-corpus.md#src-mcst-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Criterion-level rationale`: Risk is moderate-high because masks, partition definitions, channel weights, loss balancing, and evaluation sensitivity can create complexity without guaranteed production-quality gains.
- `Unresolved caveat`: Scope can expand quickly if variational, iterative, non-manual, or 3D features are added prematurely.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `78.8`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The scientific case is coherent and source-supported, but readiness depends on the project's artifact schema, stable partitions, loss weighting, and evaluation sensitivity.

## Score Interpretation Boundary

- `Score interpretation`: This is viable but conditional on project artifact schema and evaluation sensitivity.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: Unstable body/hand/face partitions, weak masks, poor loss balancing, or uninformative channel-aware metrics would reduce readiness.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Exact body/hand/face partition scheme and relationship to available keypoint masks.
- `Evaluation gaps`: Channel-aware metrics, sign-level interpretation of channel gains, qualitative inspection, and metric limitations.
- `Implementation unknowns`: Mask handling, channel weights, structure-aware loss design, and interaction with token or latent candidates.
- `Decision risks`: Treating better structure preservation as sufficient proof of semantic adequacy would overstate the candidate.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
