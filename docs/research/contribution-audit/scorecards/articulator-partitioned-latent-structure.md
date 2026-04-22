# Articulator-Partitioned Latent Structure

## Candidate Under Scoring

`STRUCTURE-B2 — Articulator-Partitioned Latent Structure`

## Record Context

First completed scorecard pass before veto review.

## Criteria

### Base-Model Additivity (`20`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The current candidate record supports an independent `Base + C2` path conceptually, but leaves that path conditional on defining a sufficiently clean independent implementation. This is a strong but not fully established additivity profile.

### Repository/Data Compatibility (`15`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The candidate is conceptually compatible with the current processed pose surface and does not currently require heavy new preprocessing, but it introduces added architecture, training, and artifact-packaging complexity relative to the baseline workflow. This is a conditional rather than a clean compatibility profile.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `4`
- `Criterion-level rationale`: The mechanism is technically strong: articulator-partitioned latent or prediction pathways can reduce cross-channel interference and improve structured hand/body coordination by avoiding a single undifferentiated regression channel. This is a clear and compelling structure-aware rationale.

### Literature Support (`15`)

- `Raw score (0-4)`: `4`
- `Criterion-level rationale`: The candidate has one strong direct source and two strong near-direct supporting sources. The direct gloss-free DARSLP evidence strongly validates articulator-based disentanglement, while the supporting sources reinforce the broader explicit-structure rationale.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The main limiting evidence does not refute the candidate. It narrows overclaiming by showing that structured factorization is promising rather than automatically sufficient, and that implementation quality still matters. This is a real but manageable limitation.

### Ablation Clarity / Evaluability (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The minimum ablation plan is meaningful and usable: `Base vs Base+ArticulatorStructure`, simpler vs stronger partitioning variants, and later `Base+ArticulatorStructure vs Base+C1+ArticulatorStructure` if needed. The deduction is that evaluability is good, but less clean than the lower-complexity channel-aware-loss alternative.

### Implementation Risk / Cost Of Being Wrong (`10`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The current audit record places this candidate in a medium implementation-complexity band with moderate failure cost, medium-high hidden-dependency risk, and medium scope-creep risk. This makes it materially riskier than the simpler structure-aware alternative.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Weighted total score`: `76.25 / 100`

## Evidence-Confidence Label

`medium`

## Score-Band Interpretation Note

Strong but complexity-constrained scorecard result. The candidate is scientifically well supported and mechanistically compelling, but its current audit profile is limited by implementation burden and by the unresolved question of whether its added structural complexity is justified relative to simpler structure-aware alternatives.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not by itself assign a final contribution-selection outcome.
