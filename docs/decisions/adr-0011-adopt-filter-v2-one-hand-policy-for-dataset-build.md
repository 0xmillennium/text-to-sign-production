# ADR-0011: Adopt Filter V2 One-Hand Policy For Dataset Build

- Status: Accepted
- Date: 2026-04-17

## Context

The initial strict filter policy treated the core channels as one flat all-required tuple:
`body + left_hand + right_hand`. That made Dataset Build drop samples that still had usable signing
evidence when one hand was usable but the other hand had zero usable frames.

That behavior was too strict for one-hand samples and made the active retention policy harder to
explain. The processed dataset also needed a clear distinction between schema/layout stability and
dataset membership semantics.

## Decision

Dataset Build adopts `configs/data/filter-v2.yaml` as the active filter policy. Filter v2 requires
usable `body` data and at least one usable hand: `left_hand | right_hand`.

The legacy strict policy remains available at `configs/data/filter-v1.yaml` for reproducibility and
comparison. It keeps the all-hands-required behavior instead of being silently reinterpreted.

The one-hand rule better matches usable samples where body and one hand carry valid signing signal.
It also makes the policy explicit in configuration instead of hiding it inside a flat required
channel tuple.

Keeping the processed schema layout at `t2sp-processed-v1` separates data layout compatibility from
membership policy. The arrays, manifest fields, and processed sample paths can remain stable while
the set of retained samples changes under a named filter policy.

## Alternatives Considered

- Keep the strict all-hands-required rule. This preserved the previous membership behavior but
  continued dropping usable one-hand samples only because the other hand had zero usable frames.
- Silently change `filter-v1.yaml` semantics. This avoided another config file but would have made
  older runs and historical comparisons ambiguous.
- Add ad hoc one-sided exceptions. This could rescue selected cases but would be less symmetric and
  harder to validate than requiring at least one hand from the left/right group.

## Consequences

- Dataset membership semantics changed: filter v2 can retain samples that filter v1 would drop.
- The processed schema layout did not change and remains `t2sp-processed-v1`.
- Documentation and validation needed to describe and enforce body-required plus
  at-least-one-hand-required semantics.
- Historical comparisons with filter v1 builds must account for membership differences.
- Split sample archives remain manifest-driven and must match the processed manifests exactly.
