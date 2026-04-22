# Diffusion-Based Generation

## Family ID

`diffusion-based-generation`

## Family Name

Diffusion-Based Generation

## Family Type / Status

`counter_alternative`

## Scope Definition

This family covers sign language production directions that generate continuous sign sequences through diffusion-based generative modeling rather than through discrete intermediate units, explicit retrieval/composition pipelines, or lighter structure-aware modifications to a baseline predictor. In this family, sign motion is generated through iterative denoising or closely related diffusion-style sequence generation mechanisms. This record remains at the family level only.

## Included Directions / Subfamilies

- sequence diffusion for sign pose generation
- diffusion-based spoken-to-sign production
- multimodal diffusion conditioning for sign production
- latent or embedding-conditioned diffusion generation for sign sequences
- closely related diffusion-style sign production directions

## Explicit Exclusions / Out-of-Scope Directions

- no concrete candidate method is selected in this file
- no candidate-card leaf record is instantiated in this file
- no score, veto outcome, or selection outcome is recorded in this file
- no family member is assigned to `C1` or `C2` in this file
- no paper-by-paper candidate ranking is recorded in this file
- discrete/data-driven learned representation approaches are not part of this family and remain a separate primary family
- structure-aware / multi-channel learned modeling approaches are not part of this family and remain a separate primary family
- retrieval/stitching approaches are not part of this family and remain a separate counter-alternative family

## Representative Sources

### Source 1

- `Citation`: [8](../../bibliography.md#ref-08)
- `Relevance type`: `direct`
- `Claim supported`: Diffusion-based generation is a serious sign language production direction capable of producing continuous sign sequences directly from spoken text or speech.
- `Evidence note`: The paper proposes a unified multimodal spoken-to-sign framework in which a sequence diffusion model generates sign predictions conditioned on text or audio embeddings. It presents this as a viable continuous sign language production approach and reports competitive performance on How2Sign and PHOENIX14T.
- `Source location`: Abstract; Introduction; contribution summary.
- `Interpretation boundary`: This source strongly supports diffusion as a real family-level alternative, but it does not imply that diffusion is automatically preferable to the primary audited learned-contribution families.

## Rationale For Inclusion In The Audit Universe

This family is included as a `counter_alternative` family because the recent literature shows that diffusion-based generation is a real and technically credible route for continuous sign language production. It therefore deserves explicit placement in the audit universe so that the project’s chosen contribution path is not evaluated only against weaker alternatives. At the same time, diffusion represents a broader generative-family shift than the primary audited contribution families and may alter the implementation burden, modeling assumptions, and thesis emphasis of the project.

## Relation To The Overall Audit

This record defines one family-level counter-alternative branch of the contribution audit. Its
purpose is to ensure that the project explicitly acknowledges diffusion-based generation as a
serious competing family rather than ignoring it. Inclusion in the audit universe does not mean
that any specific diffusion-based approach has been selected, preferred, or rejected.
Candidate-level evidence and additivity analysis remain downstream, along with
[Scorecards](../scorecards/index.md) and [Selection Decisions](../selection-decisions/index.md).

## Family-Level Evidence Boundary

The family is well supported as a legitimate counter-alternative direction, but the literature does not imply that diffusion cleanly satisfies the same project constraints as the primary audited contribution families. Diffusion may offer strong generative flexibility, but it can also imply higher modeling complexity, different training dynamics, and a larger shift in project direction. Those tradeoffs must be handled at the candidate level, not assumed away at the family level.

## Downstream Record Policy

Downstream audit work may derive concrete [Candidate Cards](../candidate-cards/index.md) from this
family if diffusion must be compared as a serious alternative or comparator path. Any
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
