# Selection Decisions

This surface stores decision-level records derived from completed [Scorecards](../scorecards/index.md),
confidence labels, and veto review. It records candidate-specific and pairwise selection logic, but
it does not replace [Audit Result](../audit-result.md) as the authoritative whole-audit outcome
surface.

## Instantiated Decision Records

- [Dynamic VQ Pose Tokens](dynamic-vq-pose-tokens.md): `selected_for_C1`
- [Channel-Aware Loss Reweighting](channel-aware-loss-reweighting.md): `selected_for_C2`
- [Articulator-Partitioned Latent Structure](articulator-partitioned-latent-structure.md): `kept_as_fallback`
- [Motion Primitives Representation](motion-primitives-representation.md): `deferred`

## Record Status

- Decision records are instantiated for all four audited candidates.
- The current selected contribution pair is `Dynamic VQ Pose Tokens` as `C1` and `Channel-Aware Loss Reweighting` as `C2`.
- `Articulator-Partitioned Latent Structure` is retained as the current fallback record.
- `Motion Primitives Representation` remains deferred.
- [Audit Result](../audit-result.md) remains the authoritative whole-audit outcome surface.
