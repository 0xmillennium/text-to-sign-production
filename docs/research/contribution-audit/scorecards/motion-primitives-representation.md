# Motion Primitives Representation

## Candidate Under Scoring

`DISCRETE-A2 — Motion Primitives Representation`

## Record Context

First completed scorecard pass before veto review.

## Criteria

### Base-Model Additivity (`20`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The current candidate record supports an independent `Base + candidate` path only conditionally, because a clean gloss-free adaptation must first be defined. This keeps the candidate in the plausible range, but not in the established range.

### Repository/Data Compatibility (`15`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The candidate is conceptually compatible with the current processed pose surface and does not currently require heavy raw-data preprocessing, but it is more adaptation-heavy than Dynamic VQ and brings extra representation-learning and artifact-definition burden. This is a conditional rather than a clean compatibility profile.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The mechanism is technically plausible: reusable motion components can address some weaknesses of monolithic continuous regression and support more structured sequence modeling. The main deduction is that clean transfer into the current gloss-free repository setting is not yet established.

### Literature Support (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The candidate has one strong direct source and one strong family-level supporting source. The support is meaningful, but not maximally clean, because the strongest direct source depends on gloss supervision rather than directly matching the current repository framing.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The main limiting evidence does not refute the candidate, but it narrows overclaiming in two ways: the strongest direct source is gloss-supervised, and reusable units do not automatically guarantee natural continuous signing. This gives the candidate real constraints rather than a clean evidence surface.

### Ablation Clarity / Evaluability (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The minimum ablation plan is meaningful and usable: `Base vs Base+MotionPrimitives`, internal primitive variants, family-level comparison against Dynamic VQ, and later `+C2` if needed. The deduction is that evaluability is less clean than Dynamic VQ because additivity and gloss-free adaptation remain conditional.

### Implementation Risk / Cost Of Being Wrong (`10`)

- `Raw score (0-4)`: `2`
- `Criterion-level rationale`: The current audit record places this candidate in a medium-high implementation-complexity band with higher failure cost than Dynamic VQ, plus medium-high hidden-dependency and scope-creep risk. This makes it meaningfully riskier than the current leading discrete alternative.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Weighted total score`: `61.25 / 100`

## Evidence-Confidence Label

`medium`

## Score-Band Interpretation Note

Mixed scorecard result. The candidate remains scientifically plausible and literature-supported, but its current audit profile is constrained by gloss-dependent direct evidence, conditional additivity, and a meaningfully heavier implementation burden than Dynamic VQ.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not by itself assign a final contribution-selection outcome.
