# Text-Pose Semantic Consistency Objective Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-TEXT-POSE-SEMANTIC-CONSISTENCY`
- `Candidate card link`: [Text-Pose Semantic Consistency Objective](../candidate-cards/text-pose-semantic-consistency.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-TEXT-POSE-SEMANTIC-ALIGNMENT`
- `Originating family link`: [Text-Pose Semantic Alignment and Conditioning](../candidate-universe/text-pose-semantic-alignment.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Model/additive-objective scorecard; standalone versus auxiliary use remains unresolved.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-LVMCN-2024](../../source-corpus.md#src-lvmcn-2024), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Text-Pose Semantic Alignment and Conditioning](../candidate-universe/text-pose-semantic-alignment.md)
- `Evaluation family consulted`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)
- `Model evidence separated from boundary/evaluation/counter/frontier evidence`: Established.

## Raw Score Scale

| Raw score | Meaning |
| ---: | --- |
| `0` | Unsupported, incompatible, or blocking weakness. |
| `1` | Weak support with major unresolved risk. |
| `2` | Partial or conditional support with material gaps. |
| `3` | Strong support with manageable limitations. |
| `4` | Very strong support with clear evidence, compatibility, and evaluability. |

For risk-oriented criteria, a higher score means lower practical risk and better audit readiness.

## Criteria

### Baseline Additivity and Isolation (`20`)

- `Raw score (0-4)`: `2.25`
- `Weighted contribution`: `11.3`
- `Evidence used`: Candidate-card additivity plan; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Criterion-level rationale`: The objective can be ablated with and without semantic consistency, but standalone versus additive use is unresolved and the mechanism likely interacts with another model candidate.
- `Unresolved caveat`: The candidate may be more appropriate as an auxiliary objective than a standalone model direction.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card data assumptions; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Criterion-level rationale`: Paired text/transcript and pose/keypoint artifacts can support gloss-free text-pose alignment or embedding-consistency signals without manual gloss annotation.
- `Unresolved caveat`: Workflow depends on reliable text encoders, pose encoders, cached features, and split-safe feature extraction.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `11.3`
- `Evidence used`: Candidate-card expected mechanism; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Criterion-level rationale`: Semantic alignment plausibly targets semantic mismatch and sequence drift by adding a text-pose consistency signal beyond raw pose reconstruction.
- `Unresolved caveat`: Alignment loss can improve embedding proximity without improving intelligibility or motion quality.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `11.3`
- `Evidence used`: [Text-Pose Semantic Alignment and Conditioning](../candidate-universe/text-pose-semantic-alignment.md); [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Criterion-level rationale`: Alignment evidence is conceptually important but often appears inside larger diffusion, multimodal, token, or variational systems rather than as a clean standalone contribution.
- `Unresolved caveat`: Direct support for this exact minimal objective remains less complete than for primary token or latent-generation candidates.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `2.5`
- `Weighted contribution`: `6.3`
- `Evidence used`: Candidate-card limiting evidence; [SRC-LVMCN-2024](../../source-corpus.md#src-lvmcn-2024), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Criterion-level rationale`: Gloss-pose alignment methods remain supervision-boundary evidence, evaluation literature warns against overreading embedding consistency, and token/diffusion alternatives may address more direct modeling bottlenecks.
- `Unresolved caveat`: Representation-bottleneck and gloss-dependent alignment counter-evidence limit confidence in standalone readiness.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `2.25`
- `Weighted contribution`: `8.4`
- `Evidence used`: Candidate-card evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Criterion-level rationale`: With/without-objective ablation is possible, but semantic improvements are hard to validate automatically and may not align with pose-distance or motion-quality metrics.
- `Unresolved caveat`: Negative sampling, encoder choice, and metric design can dominate the apparent result.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card risk estimate; [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Implementation risk is moderate: the objective can be relatively contained, but encoder selection, feature caching, negative sampling, and metric validity create hidden dependencies.
- `Unresolved caveat`: A weak alignment objective can add complexity without improving motion or semantic quality.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `69.1`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The semantic-alignment concern is well supported, but standalone readiness is conditional because evidence often appears as a component inside larger systems.

## Score Interpretation Boundary

- `Score interpretation`: The score should not be read as standalone model-selection readiness.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: A validated standalone semantic-evaluation protocol or a clearly bounded auxiliary role on another model candidate would change readiness.
- The candidate may be more appropriate as an additive objective on top of another model candidate.
- Selection-decision stage must decide whether it is evaluated standalone or as an auxiliary mechanism.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Text encoder choice, pose encoder choice, alignment loss, and negative sampling strategy.
- `Evaluation gaps`: Semantic metric validity, qualitative inspection protocol, and relationship to pose/motion metrics.
- `Implementation unknowns`: Feature caching, split safety, auxiliary loss weighting, and interaction with token, latent, or structure-aware candidates.
- `Decision risks`: Treating an auxiliary objective as a standalone model contribution without explicit justification would overstate readiness.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
