# Candidate Cards

## Purpose

This surface stores candidate-level evidence records derived from the refreshed family-level
candidate universe. Candidate-card instantiation means a candidate is worth auditing under the
current source-corpus and candidate-universe architecture.

Instantiation does not mean scoring, ranking, selection, rejection, final audit outcome, or
implementation commitment.

## Design Basis

Candidate cards are downstream of the [Candidate Universe](../candidate-universe/index.md) and
follow the [Candidate Card Template](template.md). Their evidence chains cite the
[Source Corpus](../../source-corpus.md) under the standards defined in the
[Source Selection Criteria](../../source-selection-criteria.md).

Candidate cards are concrete audit records, not literature summaries and not implementation
plans. Boundary, evaluation, counter-alternative, and frontier families do not all produce primary
model candidates. Some produce baseline, comparator, evaluation-support, or future-work records.

## Why These Candidate Cards?

The candidate-card set is intentionally smaller than the 10-family candidate universe. Families
are analytic audit axes. Candidate cards are concrete downstream audit records. Not every family
should become a primary model candidate.

The 8-card set instantiates one baseline floor, one evaluation-support artifact, three primary
model candidates, one auxiliary/additive objective, one counter-alternative comparator, and one
frontier future-work record. This is enough to support scorecard generation without turning the
candidate-card layer into a repetition of the family records.

Detailed evidence belongs inside the candidate-card records.

## Candidate Record Type Labels

| Record type | Meaning |
| --- | --- |
| `model_candidate` | Concrete model-direction candidate that may be scored downstream. |
| `counter_alternative_candidate` | Serious non-primary alternative or comparator kept to avoid premature narrowing. |
| `baseline_candidate` | Baseline or ablation floor used for later comparison. |
| `evaluation_support_candidate` | Evaluation protocol or support artifact used to constrain scorecards and decisions. |
| `frontier_future_work_candidate` | Frontier direction tracked for positioning or future work, not near-term implementation by default. |

## Instantiated Candidate Cards

### Baseline / Ablation Candidate

- [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md)
  - `Candidate ID`: `CAND-M0-DIRECT-TEXT-TO-POSE-BASELINE`
  - `Record type`: `baseline_candidate`
  - `Role`: M0 comparison floor for later candidate evaluation.

### Evaluation-Support Candidate

- [How2Sign-Compatible Evaluation Protocol](how2sign-evaluation-protocol.md)
  - `Candidate ID`: `CAND-HOW2SIGN-EVALUATION-PROTOCOL`
  - `Record type`: `evaluation_support_candidate`
  - `Role`: Multi-metric evaluation protocol for later scorecards and decisions.

### Primary Model Candidates

- [Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md)
  - `Candidate ID`: `CAND-LEARNED-POSE-TOKEN-BOTTLENECK`
  - `Record type`: `model_candidate`
  - `Primary family`: `FAM-LEARNED-POSE-MOTION-REPRESENTATIONS`

- [Articulator-Disentangled Latent Modeling](articulator-disentangled-latent.md)
  - `Candidate ID`: `CAND-ARTICULATOR-DISENTANGLED-LATENT`
  - `Record type`: `model_candidate`
  - `Primary family`: `FAM-STRUCTURE-AWARE-ARTICULATOR-MODELING`

- [Gloss-Free Latent Diffusion](gloss-free-latent-diffusion.md)
  - `Candidate ID`: `CAND-GLOSS-FREE-LATENT-DIFFUSION`
  - `Record type`: `model_candidate`
  - `Primary family`: `FAM-LATENT-GENERATIVE-PRODUCTION`

### Auxiliary / Additive Objective

- [Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency.md)
  - `Candidate ID`: `CAND-TEXT-POSE-SEMANTIC-CONSISTENCY`
  - `Record type`: `model_candidate`
  - `Research role`: `auxiliary_additive_objective`
  - `Role`: Auxiliary/additive semantic-alignment mechanism, not a standalone primary model candidate.

### Counter-Alternative Candidate

- [Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator.md)
  - `Candidate ID`: `CAND-RETRIEVAL-AUGMENTED-POSE-COMPARATOR`
  - `Record type`: `counter_alternative_candidate`
  - `Role`: No-public-gloss retrieval/reuse comparator, not a primary neural generation mechanism.

### Frontier / Future-Work Candidate

- [Sparse-Keyframe / Avatar-Control Future Direction](sparse-keyframe-avatar-future.md)
  - `Candidate ID`: `CAND-SPARSE-KEYFRAME-AVATAR-FUTURE`
  - `Record type`: `frontier_future_work_candidate`
  - `Role`: 3D/avatar/sparse-keyframe frontier tracking without near-term implementation commitment.

## Downstream Use

Scorecards should be created from populated candidate cards, not directly from family records.
Primary model candidates may receive full scorecards. The baseline candidate may receive a
scorecard only if needed to document baseline readiness or comparison reliability. The
evaluation-support candidate should constrain scorecard interpretation rather than compete as a
model. Auxiliary/additive objectives may receive scorecards to bound their add-on role, but their
candidate-card presence does not make them standalone primary model candidates. The
counter-alternative candidate may receive a comparator scorecard. The frontier future-work
candidate should normally remain future-work unless a later audit explicitly authorizes scoring it
as an implementation candidate.

Candidate-card records do not by themselves determine scorecard outcomes, selection decisions, or
the final audit result.

## Record Status

- The refreshed candidate-card set is instantiated.
- Eight candidate cards are recorded.
- Candidate-card instantiation does not assign scores, rankings, selections, rejections, or final audit outcomes.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md), and [Audit Result](../audit-result.md) remain downstream audit surfaces.
