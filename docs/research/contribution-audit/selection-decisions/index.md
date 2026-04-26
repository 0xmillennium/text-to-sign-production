# Selection Decisions

## Purpose

Selection decisions record candidate-level decision reasoning. They are downstream of candidate
cards and scorecards, upstream of [Audit Result](../audit-result.md), and do not replace the
whole-audit result.

## Decision Basis

Selection decisions use [Candidate Cards](../candidate-cards/index.md),
[Scorecards](../scorecards/index.md), the [Selection Decision Template](template.md), the
[Source Corpus](../../source-corpus.md), and [Audit Result](../audit-result.md) as the downstream
whole-audit outcome surface.

Decisions use scorecard evidence, hard veto checks, role-specific candidate type, and open
scorecard issues.

## Decision Status Labels

| Decision status | Meaning |
| --- | --- |
| `advance_for_audit_result_consideration` | Candidate advances to the whole-audit result synthesis as a primary model candidate. |
| `keep_as_baseline_or_ablation` | Candidate is retained as a baseline or ablation reference, not a primary model contribution. |
| `keep_as_auxiliary_or_additive_objective` | Candidate is retained as an auxiliary/additive mechanism, not a standalone primary model direction. |
| `keep_as_counter_alternative` | Candidate is retained as a comparator or counter-alternative, not a primary model contribution. |
| `defer_pending_evidence` | Candidate is deferred because required evidence remains unresolved. |
| `defer_as_future_work` | Candidate is deferred to future work or frontier tracking. |
| `reject_for_current_audit` | Candidate is rejected for the current audit scope. |

## Recorded Selection Decisions

### Baseline / Ablation Retention

- [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline-selection-decision.md)

### Advanced Primary Model Candidates

- [Learned Pose-Token Bottleneck](learned-pose-token-bottleneck-selection-decision.md)
- [Gloss-Free Latent Diffusion](gloss-free-latent-diffusion-selection-decision.md)
- [Articulator-Disentangled Latent Modeling](articulator-disentangled-latent-selection-decision.md)

### Auxiliary / Additive Retention

- [Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency-selection-decision.md)

### Counter-Alternative Retention

- [Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator-selection-decision.md)

## Decision Summary

| Candidate | Score | Scorecard type | Decision status | Audit-result handoff |
| --- | ---: | --- | --- | --- |
| [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline-selection-decision.md) | `86.3` | Baseline readiness | `keep_as_baseline_or_ablation` | Retain as M0 comparison floor. |
| [Learned Pose-Token Bottleneck](learned-pose-token-bottleneck-selection-decision.md) | `85.0` | Full model | `advance_for_audit_result_consideration` | Advance as a strong near-term primary model candidate. |
| [Gloss-Free Latent Diffusion](gloss-free-latent-diffusion-selection-decision.md) | `84.1` | Full model | `advance_for_audit_result_consideration` | Advance as a strong but risk-heavy primary model candidate. |
| [Articulator-Disentangled Latent Modeling](articulator-disentangled-latent-selection-decision.md) | `78.8` | Full model | `advance_for_audit_result_consideration` | Advance as a viable structure-preservation model candidate. |
| [Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency-selection-decision.md) | `69.1` | Additive/objective candidate | `keep_as_auxiliary_or_additive_objective` | Retain as auxiliary/additive mechanism pending explicit ablation design. |
| [Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator-selection-decision.md) | `65.9` | Comparator | `keep_as_counter_alternative` | Retain as no-public-gloss comparator and sanity-check baseline. |

These decisions do not constitute the final whole-audit result. The final synthesis belongs in
[Audit Result](../audit-result.md).

## Downstream Use

[Audit Result](../audit-result.md) should use these decisions as candidate-level handoff records.
It should not repeat all scorecard details. It should preserve limitations and role boundaries. It
should distinguish primary model candidates from baseline, auxiliary objective, and comparator
records.

## Record Status

- Six candidate-level selection decisions are recorded.
- Three primary model candidates advance for audit-result consideration.
- One baseline is retained for comparison and ablation.
- One auxiliary/additive objective is retained.
- One counter-alternative comparator is retained.
- [Audit Result](../audit-result.md) remains the authoritative whole-audit outcome surface.
