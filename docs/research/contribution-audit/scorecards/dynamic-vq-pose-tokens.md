# Dynamic VQ Pose Tokens

## Candidate Under Scoring

`DISCRETE-A1 — Dynamic VQ Pose Tokens`

## Record Context

First completed scorecard pass before veto review.

## Criteria

### Base-Model Additivity (`20`)

- `Raw score (0-4)`: `4`
- `Criterion-level rationale`: The current candidate record supports an independent `Base + candidate` path, does not require `C2`, and supports the existence of an independent variant. This is the strongest possible additivity reading for the current audit record.

### Repository/Data Compatibility (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The candidate is compatible with the current processed `.npz` pose surface and the current Colab-centered workflow, and heavy new preprocessing is not currently indicated. The main deduction is that artifact formats and run packaging will likely require extension, while the thin-driver impact is not yet fully established.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `4`
- `Criterion-level rationale`: The candidate replaces unconstrained continuous frame regression with reusable discrete pose units while allowing variable information density. This mechanism is technically coherent and directly aligned with the targeted failure modes of oversmoothing, mode-averaging, and articulation loss.

### Literature Support (`15`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The candidate has one strong direct source for dynamic vector quantization and one strong family-level supporting source for discrete sequence reformulation in sign language production. The support is strong, but not maximally clean, because the second source is family-level rather than exact-method support.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The main limiting evidence does not refute Dynamic VQ. It narrows overclaiming by showing that a discrete intermediate representation does not automatically solve continuity and transition quality. The candidate remains defensible, but not limitation-free.

### Ablation Clarity / Evaluability (`15`)

- `Raw score (0-4)`: `4`
- `Criterion-level rationale`: The minimum ablation plan is clean and directly usable: `Base vs Base+DynamicVQ`, `DynamicVQ vs fixed-VQ internal variant`, and later `Base+DynamicVQ vs Base+DynamicVQ+C2` if needed. This gives the candidate strong evaluability.

### Implementation Risk / Cost Of Being Wrong (`10`)

- `Raw score (0-4)`: `3`
- `Criterion-level rationale`: The current audit record places this candidate in a medium implementation-complexity and medium hidden-dependency band, with moderate failure cost and medium scope-creep risk. This is a manageable but not low-risk profile.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Weighted total score`: `87.50 / 100`

## Evidence-Confidence Label

`medium`

## Score-Band Interpretation Note

Strong scorecard result. The candidate currently reads as a serious front-runner within the discrete/data-driven family, but its evidence base does not yet justify a `high` confidence label because continuity and transition quality remain an explicit boundary rather than a solved issue.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not by itself assign a final contribution-selection outcome.
