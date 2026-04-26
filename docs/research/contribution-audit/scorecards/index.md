# Scorecards

## Purpose

Scorecards record contribution-audit readiness scores for populated
[Candidate Cards](../candidate-cards/index.md). They are downstream of candidate cards and upstream
of [Selection Decisions](../selection-decisions/index.md).

Scorecards do not select, reject, or rank candidates by themselves. They do not update the
[Audit Result](../audit-result.md).

## Scoring Basis

Scores use the current scorecard template, candidate-card evidence chains,
[Candidate Universe](../candidate-universe/index.md) family records, the
[Source Corpus](../../source-corpus.md), and evaluation constraints from
[Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md).

The scores are contribution-audit readiness scores. They are not experimental implementation
results and should not be read as downstream audit outcomes.

### Scoring Formula

Each criterion uses a raw score from `0` to `4`.

```text
weighted contribution = (raw score / 4) × criterion weight
weighted total = sum of weighted contributions
```

Weighted totals are calculated from raw scores before one-decimal display rounding. Displayed
criterion contributions may therefore not sum exactly to the shown total.

## Scored Candidate Set

- [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline-scorecard.md)
- [Learned Pose-Token Bottleneck](learned-pose-token-bottleneck-scorecard.md)
- [Gloss-Free Latent Diffusion](gloss-free-latent-diffusion-scorecard.md)
- [Articulator-Disentangled Latent Modeling](articulator-disentangled-latent-scorecard.md)
- [Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency-scorecard.md)
- [Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator-scorecard.md)

## Score Summary

This table is a scorecard summary only. It is not a ranking, and it does not determine
downstream selection decisions.

The table is ordered for readability by scorecard role and score summary; ordering is not a
selection ranking.

| Scorecard | Candidate type | Total | Evidence confidence | Downstream interpretation |
| --- | --- | ---: | --- | --- |
| [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline-scorecard.md) | Baseline readiness | `86.3` | `medium` | M0 comparison floor readiness, not a model contribution. |
| [Learned Pose-Token Bottleneck](learned-pose-token-bottleneck-scorecard.md) | Full model | `85.0` | `medium` | Strong near-term model candidate; tokenizer/reconstruction risk remains. |
| [Gloss-Free Latent Diffusion](gloss-free-latent-diffusion-scorecard.md) | Full model | `84.1` | `medium` | Strong evidence and high potential, but risk-heavy. |
| [Articulator-Disentangled Latent Modeling](articulator-disentangled-latent-scorecard.md) | Full model | `78.8` | `medium` | Viable structure-preservation candidate with conditional schema/evaluation fit. |
| [Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency-scorecard.md) | Additive/objective candidate | `69.1` | `medium` | Important but likely auxiliary unless standalone evaluation is justified. |
| [Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator-scorecard.md) | Comparator | `65.9` | `medium` | Useful no-public-gloss comparator; not a primary neural generation candidate. |

## Downstream Use

Selection decisions are downstream. They must review the relevant candidate card, scorecard,
source-corpus evidence, boundary evidence, evaluation constraints, and open scorecard issues before
recording any decision outcome.

## Record Status

- Six refreshed scorecards are instantiated.
- Scorecards remain downstream of populated candidate cards.
- Scorecards must distinguish model evidence from boundary, evaluation, counter-alternative, and
  frontier-watch evidence.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  remain downstream audit surfaces.
