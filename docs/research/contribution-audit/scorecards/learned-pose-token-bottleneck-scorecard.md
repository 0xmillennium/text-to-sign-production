# Learned Pose-Token Bottleneck Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-LEARNED-POSE-TOKEN-BOTTLENECK`
- `Candidate card link`: [Learned Pose-Token Bottleneck](../candidate-cards/learned-pose-token-bottleneck.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-LEARNED-POSE-MOTION-REPRESENTATIONS`
- `Originating family link`: [Learned Pose/Motion Representations](../candidate-universe/learned-pose-motion-representations.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Full model scorecard.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024), [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026), [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Learned Pose/Motion Representations](../candidate-universe/learned-pose-motion-representations.md)
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

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `17.5`
- `Evidence used`: Candidate-card additivity plan; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Criterion-level rationale`: The text-to-token-to-pose path is comparatively well isolated against the direct text-to-pose baseline, and the token bottleneck can be ablated while holding splits, text input, and output schema stable.
- `Unresolved caveat`: Tokenizer training can change preprocessing or reconstruction targets enough to require a clearly documented baseline variant.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card data assumptions; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Criterion-level rationale`: Learned tokens can be derived from pose/keypoint data without manual gloss supervision, and direct How2Sign-compatible representation evidence supports the workflow fit.
- `Unresolved caveat`: The current artifact schema must support stable body, hand, face, mask, and reconstruction targets.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `3.5`
- `Weighted contribution`: `13.1`
- `Evidence used`: Candidate-card expected mechanism; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024), [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024)
- `Criterion-level rationale`: A learned pose-token bottleneck plausibly reduces raw-regression oversmoothing and mode averaging by forcing generation through a pose-derived intermediate representation.
- `Unresolved caveat`: Token collapse, discontinuity, or poor reconstruction could erase the intended mechanism.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `3.75`
- `Weighted contribution`: `14.1`
- `Evidence used`: [Learned Pose/Motion Representations](../candidate-universe/learned-pose-motion-representations.md); [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024), [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024), [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Criterion-level rationale`: Support spans direct gloss-free representation evidence, learned pose/motion representation evidence, dynamic tokenization evidence, non-gloss keypoint-derived representation evidence, and frontier pressure toward body/hand/face tokens.
- `Unresolved caveat`: Support is high but not perfect because some sources are cross-lingual-transferable or frontier evidence rather than immediate project-compatible implementation proof.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card limiting evidence; [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Gloss-to-pose discrete diffusion remains supervision-boundary evidence, latent diffusion is a serious modern alternative, and evaluation evidence prevents overclaiming token reconstruction as sign adequacy.
- `Unresolved caveat`: A latent generative route may prove more effective for sequence-level motion once evaluated.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `3.25`
- `Weighted contribution`: `12.2`
- `Evidence used`: Candidate-card minimum evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024)
- `Criterion-level rationale`: Direct baseline versus text-to-token-to-pose is a clean ablation frame, and reconstruction plus downstream pose/motion checks can isolate representation quality.
- `Unresolved caveat`: Evaluation must distinguish tokenizer reconstruction, generated motion naturalness, and semantic adequacy.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card complexity estimate; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024), [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Criterion-level rationale`: Risk is moderate because tokenizer quality, codebook stability, reconstruction quality, and stored code artifacts can determine whether the candidate clarifies or confounds comparison.
- `Unresolved caveat`: Expanding into dynamic duration, modality-specific tokenization, or larger token stacks too early would increase cost and scope.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `85.0`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: The evidence base is strong for a near-term model candidate, but project-specific feasibility depends on tokenizer design, reconstruction quality, codebook stability, and evaluation protocol.

## Score Interpretation Boundary

- `Score interpretation`: This is strong enough for downstream selection-decision consideration, but the scorecard itself does not select it.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: Poor tokenizer reconstruction, codebook instability, token collapse, or uninformative evaluation would reduce readiness.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Tokenizer type, codebook size, temporal granularity, and reconstruction target.
- `Evaluation gaps`: Reconstruction metrics, pose/keypoint metrics, embedding checks, and qualitative inspection protocol.
- `Implementation unknowns`: Codebook stability, mask handling, stored token artifacts, and decoder design.
- `Decision risks`: Treating token quality as equivalent to semantic adequacy would overstate the candidate.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
