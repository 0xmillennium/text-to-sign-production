# Structure-Aware / Multi-Channel Improvement

## Family ID

`structure-aware-multi-channel-improvement`

## Family Name

Structure-Aware / Multi-Channel Improvement

## Family Type / Status

`primary_candidate`

## Scope Definition

This family covers sign language production directions that improve generation quality by modeling
sign articulation through explicit channel structure, articulator-sensitive organization, or
multi-channel inductive bias. In this family, body, hands, face, or other articulator groups are
not treated as a single undifferentiated target when a more structured organization is
scientifically justified. This record remains at the family level only.

## Included Directions / Subfamilies

- channel-aware modeling and regularization
- articulator-aware latent or decoder structure
- body/hand/face-sensitive architectural factorization
- explicit multi-channel spatial-temporal modeling
- closely related structure-aware sign production directions

## Explicit Exclusions / Out-of-Scope Directions

- no concrete candidate method is selected in this file
- no candidate-card leaf record is instantiated in this file
- no score, veto outcome, or selection outcome is recorded in this file
- no family member is assigned to `C1` or `C2` in this file
- no paper-by-paper candidate ranking is recorded in this file
- discrete/data-driven representation approaches are not part of this family and remain a separate
  primary family
- retrieval/stitching approaches are not part of this family and remain a separate
  counter-alternative family
- diffusion-based generation approaches are not part of this family and remain a separate
  counter-alternative family

## Representative Sources

### Source 1

- `Citation`: [6](../../bibliography.md#ref-06)
- `Relevance type`: `direct`
- `Claim supported`: Articulator-based disentanglement and channel-aware regularization define a serious gloss-free structure-aware sign language production direction.
- `Evidence note`: The paper explicitly proposes a gloss-free sign language production framework that models face, left hand, right hand, and body features separately and applies channel-aware regularization with articulator-weighted losses.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source strongly supports the articulator-aware and channel-aware branches of the family, but it does not define the entire family on its own.

### Source 2

- `Citation`: [1](../../bibliography.md#ref-01)
- `Relevance type`: `direct`
- `Claim supported`: Sign language production must explicitly account for multi-channel
  articulation, including both manual and non-manual features, to avoid robotic and under-expressive
  output.
- `Evidence note`: The paper frames sign languages as multi-channel visual languages and critiques
  prior sign language production methods for focusing mainly on manual features. It positions
  continuous 3D multi-channel production as necessary for natural and understandable sign output.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source is a strong family-level anchor for explicit multi-channel
  modeling, but it is broader than any single structure-aware sub-approach.

### Source 3

- `Citation`: [5](../../bibliography.md#ref-05)
- `Relevance type`: `direct`
- `Claim supported`: Unified feature representations can ignore structural correlations between
  articulatory channels, motivating explicit multi-channel structure in sign language production
  models.
- `Evidence note`: The paper argues that current transformer-based sign language production models
  often flatten multi-channel sign poses into unified representations and proposes channel-level
  spatial and temporal modeling to address the resulting structural loss.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source directly supports an explicit multi-channel branch of the
  family, but it does not imply that every possible structure-aware intervention is equally strong.

## Rationale For Inclusion In The Audit Universe

This family is included as a `primary_candidate` family because the literature strongly supports the
view that sign language production should not always model all articulators as a single
undifferentiated prediction target. Across gloss-free articulator-based disentanglement,
channel-aware regularization, and explicit multi-channel spatial-temporal modeling, the family is
scientifically defensible as a contribution direction with clear relevance to sign quality, hand
fidelity, channel coordination, and expressive realism.

## Relation To The Overall Audit

This record defines one family-level branch of the contribution audit. Its purpose is to fix the
scope and scientific justification of the structure-aware / multi-channel family before
candidate-level evaluation. Inclusion in the audit universe does not mean that any specific family
member has been selected, preferred, or rejected. Candidate-level evidence and additivity analysis
remain downstream, along with [Scorecards](../scorecards/index.md) and
[Selection Decisions](../selection-decisions/index.md).

## Family-Level Evidence Boundary

The family is well supported as a legitimate contribution direction, but the literature does not
imply that all structure-aware interventions are equally lightweight, equally additive, or equally
strong as thesis contributions. Some sub-directions may be modest optimization-level changes, while
others may introduce architectural complexity or entanglement risk. Those differences must be
resolved at the candidate level, not assumed at the family level.

## Downstream Record Policy

Downstream audit work may derive concrete [Candidate Cards](../candidate-cards/index.md) from this
family. Any family-derived
candidate must still be tested through candidate-level evidence, independent additivity over the
base model, compatibility with the four-model evidence structure (`M0 = Base`, `M1 = Base + C1`,
`M2 = Base + C2`, `M3 = Base + C1 + C2`), and explicit limiting-evidence review. This file does
not assign the family, or any family member, to `C1` or `C2`.

## Record Status

- This family remains instantiated as a `primary_candidate` family.
- The family-level scientific justification has been documented.
- No specific family member is selected in this file.
- [Candidate Cards](../candidate-cards/index.md), [Scorecards](../scorecards/index.md),
  [Selection Decisions](../selection-decisions/index.md), and
  [Audit Result](../audit-result.md) remain downstream.
