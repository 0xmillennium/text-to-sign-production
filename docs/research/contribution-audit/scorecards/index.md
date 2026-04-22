# Scorecards

This surface stores weighted candidate scoring records derived from populated
[Candidate Cards](../candidate-cards/index.md). It records score-only judgment and
evidence-confidence labels, but it does not by itself assign final contribution-selection
outcomes.

## Instantiated Scorecards

- [Dynamic VQ Pose Tokens](dynamic-vq-pose-tokens.md): `87.50 / 100`, confidence `medium`
- [Channel-Aware Loss Reweighting](channel-aware-loss-reweighting.md): `85.00 / 100`, confidence `medium`
- [Articulator-Partitioned Latent Structure](articulator-partitioned-latent-structure.md): `76.25 / 100`, confidence `medium`
- [Motion Primitives Representation](motion-primitives-representation.md): `61.25 / 100`, confidence `medium`

## Record Status

- Scorecards are instantiated for all four audited candidates.
- The current highest-scoring discrete/data-driven candidate is `Dynamic VQ Pose Tokens`.
- The current highest-scoring structure-aware candidate is `Channel-Aware Loss Reweighting`.
- Final selection status is recorded downstream in
  [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md).
- This index remains a score surface, not a final outcome surface.
