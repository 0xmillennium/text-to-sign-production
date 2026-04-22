# Retrieval / Stitching

## Family ID

`retrieval-stitching`

## Family Name

Retrieval / Stitching

## Family Type / Status

`counter_alternative`

## Scope Definition

This family covers sign language production directions that construct continuous signing from retrieved or example-backed sign units rather than relying primarily on a learned end-to-end generative mapping from text to motion. In this family, dictionary examples, isolated signs, retrieved units, or stitched sign segments are used as the core production mechanism, typically with additional transition or prosody handling to improve continuity. This record remains at the family level only.

## Included Directions / Subfamilies

- dictionary-example retrieval for sign production
- example-backed sign sequence construction
- sign stitching and transition-aware composition
- prosody-aware adjustment of retrieved sign units
- closely related retrieval/composition-based sign production directions

## Explicit Exclusions / Out-of-Scope Directions

- no concrete candidate method is selected in this file
- no candidate-card leaf record is instantiated in this file
- no score, veto outcome, or selection outcome is recorded in this file
- no family member is assigned to `C1` or `C2` in this file
- no paper-by-paper candidate ranking is recorded in this file
- discrete/data-driven learned representation approaches are not part of this family and remain a separate primary family
- structure-aware / multi-channel learned modeling approaches are not part of this family and remain a separate primary family
- diffusion-based generation approaches are not part of this family and remain a separate counter-alternative family

## Representative Sources

### Source 1

- `Citation`: [7](../../bibliography.md#ref-07)
- `Relevance type`: `direct`
- `Claim supported`: Retrieval/stitching is a serious sign language production direction in which expressive sign units are composed and explicitly adjusted to improve continuity and prosody.
- `Evidence note`: The paper proposes constructing sign sequences from dictionary examples and explicitly argues that simply concatenating signs would create robotic and unnatural sequences, motivating a dedicated stitching process with duration, cutoff, filtering, and prosody-oriented adjustments.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source strongly supports the retrieval/stitching family, but it does not imply that retrieval-based construction is necessarily preferable to all learned generative alternatives.

### Source 2

- `Citation`: [4](../../bibliography.md#ref-04)
- `Relevance type`: `indirect`
- `Claim supported`: Composition and continuity remain important issues even when sign production uses learned intermediate units.
- `Evidence note`: The paper includes sign stitching as an additional component alongside its discrete/data-driven representation, which reinforces the broader point that continuity and token-to-token composition are non-trivial in sign language production.
- `Source location`: Abstract; Introduction; contribution list.
- `Interpretation boundary`: This source does not define the retrieval/stitching family by itself, but it supports the general importance of explicit sequence-joining and continuity handling.

## Rationale For Inclusion In The Audit Universe

This family is included as a `counter_alternative` family because the literature shows that example-backed and stitching-based production offers a serious alternative route to continuous sign language generation, especially where direct regression suffers from under-articulation or poor continuity. Retrieval/stitching is therefore relevant not as a trivial baseline, but as a credible competing family that addresses expressiveness and transition quality through a different design philosophy than the primary learned-generation families.

## Relation To The Overall Audit

This record defines one family-level counter-alternative branch of the contribution audit. Its
purpose is to ensure that the project does not evaluate only learned generative contribution
families while ignoring a serious composition-based alternative. Inclusion in the audit universe
does not mean that any specific family member has been selected, preferred, or rejected.
Candidate-level evidence and additivity analysis remain downstream, along with
[Scorecards](../scorecards/index.md) and [Selection Decisions](../selection-decisions/index.md).

## Family-Level Evidence Boundary

The family is well supported as a legitimate counter-alternative direction, but the literature does not imply that retrieval/stitching cleanly satisfies the same thesis objectives as the primary learned contribution families. Its strengths lie in expressiveness and continuity control, while its weaknesses may include dependence on example inventories, explicit transition engineering, and potential problem drift away from learned representation design. Those tradeoffs must be handled at the candidate level, not assumed away at the family level.

## Downstream Record Policy

Downstream audit work may derive concrete [Candidate Cards](../candidate-cards/index.md) from this
family if retrieval/stitching must be compared as a serious alternative or comparator path. Any
family-derived candidate must still be tested through candidate-level evidence, compatibility with
the four-model evidence structure where applicable, and explicit limiting-evidence review. This
file does not assign the family, or any family member, to `C1` or `C2`.

## Record Status

- This family remains instantiated as a `counter_alternative` family.
- The family-level scientific justification has been documented.
- No specific family member is selected in this file.
- [Candidate Cards](../candidate-cards/index.md), [Scorecards](../scorecards/index.md),
  [Selection Decisions](../selection-decisions/index.md), and
  [Audit Result](../audit-result.md) remain downstream.
