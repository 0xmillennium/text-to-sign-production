# Discrete / Data-Driven Representation

## Family ID

`discrete-data-driven-representation`

## Family Name

Discrete / Data-Driven Representation

## Family Type / Status

`primary_candidate`

## Scope Definition

This family covers sign language production directions that replace or augment direct continuous
pose regression with reusable, learned, or data-driven intermediate representational units. In this
family, continuous sign motion is modeled through discrete tokens, motion units, motion primitives,
or closely related data-driven intermediate structures rather than only through full-sequence
framewise regression. This record remains at the family level only.

## Included Directions / Subfamilies

- dynamic vector quantization and variable-length discrete coding
- fixed or codebook-based pose-token representations
- data-driven motion lexicons or reusable motion units
- motion-primitives-based intermediate representations
- closely related discrete/data-driven sign representation directions

## Explicit Exclusions / Out-of-Scope Directions

- no concrete candidate method is selected in this file
- no candidate-card leaf record is instantiated in this file
- no score, veto outcome, or selection outcome is recorded in this file
- no family member is assigned to `C1` or `C2` in this file
- no paper-by-paper candidate ranking is recorded in this file
- retrieval/stitching approaches are not part of this family and remain a separate
  counter-alternative family
- diffusion-based generation approaches are not part of this family and remain a separate
  counter-alternative family

## Representative Sources

### Source 1

- `Citation`: [2](../../bibliography.md#ref-02)
- `Relevance type`: `direct`
- `Claim supported`: Dynamic discrete coding is a serious sign language production direction, especially where fixed-length vector quantization may mismatch the uneven information density of sign sequences.
- `Evidence note`: The paper explicitly proposes a two-stage sign language production paradigm that first encodes sign sequences into discrete codes and then generates those codes from text. It also argues that fixed-length encoding overlooks uneven information density in sign language and motivates dynamic vector quantization as a response.
- `Source location`: Abstract; Introduction; method discussion of DVQ-VAE and duration prediction.
- `Interpretation boundary`: This source strongly supports the dynamic-discrete branch of the family, but it does not by itself define the entire discrete/data-driven family.

### Source 2

- `Citation`: [4](../../bibliography.md#ref-04)
- `Relevance type`: `direct`
- `Claim supported`: Reframing continuous pose generation as discrete sequence generation is a legitimate and competitive sign language production direction.
- `Evidence note`: The paper explicitly transforms continuous pose generation into a discrete sequence generation problem by learning a codebook of short motions through vector quantization and translating spoken-language text into codebook tokens. It reports strong back-translation gains over previous methods.
- `Source location`: Abstract; Introduction; contribution list.
- `Interpretation boundary`: This source is a strong family-level anchor for discrete/data-driven representation, but it also includes additional components such as sign stitching and should not be reduced to a single-method endorsement.

### Source 3

- `Citation`: [3](../../bibliography.md#ref-03)
- `Relevance type`: `direct`
- `Claim supported`: Motion-primitives-based intermediate structure is a legitimate family member within discrete/data-driven sign language production.
- `Evidence note`: The paper formulates sign language production as two jointly trained sub-tasks and proposes a Mixture of Motion Primitives architecture in which distinct motion primitives are learned and temporally combined at inference.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source directly supports a compositional motion-primitives branch of the family, but its original formulation uses gloss supervision and therefore does not automatically justify gloss-free transfer without qualification.

## Rationale For Inclusion In The Audit Universe

This family is included as a `primary_candidate` family because the current literature supports a
serious line of sign language production research that replaces or augments direct continuous
regression with reusable learned intermediate units. Across dynamic quantization, codebook-based
tokenization, and motion-primitives-based composition, the family is scientifically defensible as a
contribution direction rather than a cosmetic implementation variant. The family is therefore broad
enough to warrant dedicated audit treatment, while still being specific enough to remain distinct
from retrieval/stitching and diffusion-based alternatives.

## Relation To The Overall Audit

This record defines one family-level branch of the contribution audit. Its purpose is to fix the
scope and scientific justification of the discrete/data-driven representation family before
candidate-level evaluation. Inclusion in the audit universe does not mean that any specific family
member has been selected, preferred, or rejected. Candidate-level evidence and additivity analysis
remain downstream, along with [Scorecards](../scorecards/index.md) and
[Selection Decisions](../selection-decisions/index.md).

## Family-Level Evidence Boundary

The family is well supported as a legitimate contribution direction, but the literature does not
imply that all discrete/data-driven members are equally strong, equally clean to integrate, or
equally compatible with the project’s gloss-free baseline. In particular, some members of the
family may introduce transition-quality, prosody, supervision, or additivity risks that must be
handled at the candidate level rather than assumed away at the family level.

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
