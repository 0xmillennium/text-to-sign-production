# Scorecard Template

This template scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: Not yet assigned.
- `Candidate card link`: Not yet assigned.
- `Candidate record type`: Not yet assigned.
- `Originating family ID`: Not yet assigned.
- `Originating family link`: Not yet assigned.

## Scorecard Preconditions

- `Candidate card populated`: Not yet established.
- `Evidence chain present`: Not yet established.
- `Supporting source-corpus evidence present`: Not yet established.
- `Limiting or contradicting evidence reviewed`: Not yet established.
- `Minimum evaluation and ablation plan present`: Not yet established.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Not yet established.
- `Primary source-corpus entries used`: Not yet documented.
- `Boundary families consulted`: Not yet documented.
- `Evaluation family consulted`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)
- `Model evidence separated from boundary/evaluation/counter/frontier evidence`: Not yet established.

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

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: Not yet recorded.
- `Weighted contribution`: Not yet recorded.
- `Evidence used`: Not yet documented.
- `Criterion-level rationale`: Not yet recorded.
- `Unresolved caveat`: Not yet documented.

## Weighted Total Score

- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: Not yet recorded.
- `Score calculation checked`: Not yet established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: Not yet assigned.
- `Confidence rationale`: Not yet documented.

## Score Interpretation Boundary

- `Score interpretation`: Not yet documented.
- `Blocking caveats`: Not yet documented.
- `Conditions that would change the score`: Not yet documented.

## Open Issues Before Selection Decision

- `Evidence gaps`: Not yet documented.
- `Evaluation gaps`: Not yet documented.
- `Implementation unknowns`: Not yet documented.
- `Decision risks`: Not yet documented.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
