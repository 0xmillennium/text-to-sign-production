# Gloss-Free Latent Diffusion Scorecard

This scorecard scores a populated candidate card. It is downstream of a candidate card and upstream of selection decisions. It does not select, reject, or rank a candidate by itself.

## Candidate Card Under Scoring

- `Candidate ID`: `CAND-GLOSS-FREE-LATENT-DIFFUSION`
- `Candidate card link`: [Gloss-Free Latent Diffusion](../candidate-cards/gloss-free-latent-diffusion.md)
- `Candidate record type`: `model_candidate`
- `Originating family ID`: `FAM-LATENT-GENERATIVE-PRODUCTION`
- `Originating family link`: [Latent Generative Production](../candidate-universe/latent-generative-production.md)

## Scorecard Preconditions

- `Candidate card populated`: Established.
- `Evidence chain present`: Established.
- `Supporting source-corpus evidence present`: Established.
- `Limiting or contradicting evidence reviewed`: Established.
- `Minimum evaluation and ablation plan present`: Established.
- `Scorecard type adjustment`: Full model scorecard.

## Evidence Chain Used For Scoring

- `Candidate card evidence chain reviewed`: Established.
- `Primary source-corpus entries used`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021), [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024), [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024), [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Boundary families consulted`: [Dataset and Supervision Boundary](../candidate-universe/dataset-supervision-boundary.md), [Gloss/Notation-Dependent Pose Generation Boundary](../candidate-universe/gloss-notation-dependent-boundary.md), [Latent Generative Production](../candidate-universe/latent-generative-production.md)
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

- `Raw score (0-4)`: `3.25`
- `Weighted contribution`: `16.3`
- `Evidence used`: Candidate-card additivity plan; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Criterion-level rationale`: The latent generative mechanism can be compared against direct text-to-pose, but latent autoencoding and preprocessing may alter the output target enough to make isolation slightly less clean than a minimal learned-token bottleneck.
- `Unresolved caveat`: A separate baseline variant may be needed if the latent setup changes targets or normalization materially.

### Data, Supervision, and Workflow Compatibility (`15`)

- `Raw score (0-4)`: `3.75`
- `Weighted contribution`: `14.1`
- `Evidence used`: Candidate-card data assumptions; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-HOW2SIGN-2021](../../source-corpus.md#src-how2sign-2021)
- `Criterion-level rationale`: The candidate is gloss-free and has direct or near-direct How2Sign-related support for text-conditioned or text/audio-conditioned keypoint generation.
- `Unresolved caveat`: Workflow fit remains conditional on latent reconstruction targets, compute assumptions, and reproducible scheduler configuration.

### Mechanistic Plausibility (`15`)

- `Raw score (0-4)`: `4.0`
- `Weighted contribution`: `15.0`
- `Evidence used`: Candidate-card expected mechanism; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Criterion-level rationale`: Latent generative modeling plausibly targets oversmoothing, mode averaging, sequence drift, and temporal naturalness by modeling motion through denoising or latent generation rather than direct frame regression.
- `Unresolved caveat`: Strong mechanism support does not remove sensitivity to latent quality, denoising schedule, and training stability.

### Source-Corpus and Family Evidence Support (`15`)

- `Raw score (0-4)`: `4.0`
- `Weighted contribution`: `15.0`
- `Evidence used`: [Latent Generative Production](../candidate-universe/latent-generative-production.md); [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025), [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025), [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026), [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Criterion-level rationale`: Source support is very strong because the corpus includes direct latent diffusion evidence, How2Sign-direct diffusion evidence, latent/refinement continuations, and foundational latent baseline context.
- `Unresolved caveat`: Some supporting sources are frontier or cross-lingual-transferable, so project-specific implementation evidence remains unproven.

### Resilience To Counter-Evidence (`10`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `7.5`
- `Evidence used`: Candidate-card limiting evidence; [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024), [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024), [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024), [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Learned-token alternatives, gloss-dependent discrete diffusion, avatar/frontier scope pressure, sparse-keyframe alternatives, and evaluation limitations all constrain overconfidence but do not contradict evaluating a gloss-free latent diffusion candidate.
- `Unresolved caveat`: Modern representation candidates may be lower-risk, and frontier methods may expose limitations in dense latent generation.

### Evaluation and Ablation Clarity (`15`)

- `Raw score (0-4)`: `3.0`
- `Weighted contribution`: `11.3`
- `Evidence used`: Candidate-card evaluation plan; [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025), [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Criterion-level rationale`: The direct baseline versus latent generation comparison is clear, but evaluating generative diversity, semantic adequacy, and motion realism remains difficult and metric-sensitive.
- `Unresolved caveat`: Automatic metrics cannot by themselves prove sign intelligibility, non-manual adequacy, or semantic fidelity.

### Implementation Risk and Cost of Being Wrong (`10`)

- `Raw score (0-4)`: `2.0`
- `Weighted contribution`: `5.0`
- `Evidence used`: Candidate-card risk estimate; [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025), [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024), [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Criterion-level rationale`: Implementation risk is high because stochastic training, latent reconstruction, denoising schedules, compute, inference cost, and evaluation sensitivity can consume substantial effort without reliable improvement.
- `Unresolved caveat`: The candidate remains strong scientifically, but it should not be treated as low-risk.

## Weighted Total Score

- `Calculation note`: Convert each raw `0-4` score into its weighted contribution under the criterion weight, then sum the seven contributions for the total score out of `100`.
- `Formula`: `weighted contribution = (raw score / 4) × criterion weight`; `weighted total = sum of weighted contributions`.
- `Rounding note`: Weighted totals are calculated from raw scores before one-decimal display rounding; displayed criterion contributions may not sum exactly to the shown total.
- `Weighted total score`: `84.1`
- `Score calculation checked`: Established.

## Evidence-Confidence Label

Allowed values:

- `low`
- `medium`
- `high`

- `Evidence-confidence label`: `medium`
- `Confidence rationale`: Method and source-corpus support are very strong, but project-specific readiness is moderated by latent reconstruction, stochastic training, compute cost, and evaluation risk.

## Score Interpretation Boundary

- `Score interpretation`: The score reflects high scientific support but high implementation and evaluation risk.
- `Blocking caveats`: The scorecard does not select or reject the candidate, does not update audit-result.md, and does not claim experimental implementation results.
- `Conditions that would change the score`: Unstable denoising, poor latent reconstruction, excessive compute cost, or weak semantic evaluation would reduce readiness.
- Scores are contribution-audit readiness scores based on source evidence and project compatibility.

## Open Issues Before Selection Decision

- `Evidence gaps`: Exact latent type, reconstruction target, denoising schedule, and compute assumptions.
- `Evaluation gaps`: Generative-quality metrics, semantic adequacy checks, qualitative inspection, and comparison against learned-token alternatives.
- `Implementation unknowns`: Scheduler design, seed control, cached latents, sequence length handling, and inference cost.
- `Decision risks`: High evidence support could be mistaken for low implementation risk unless risk is kept explicit.

## Record Boundary Note

- This document records scorecard content only.
- It is downstream of a populated candidate card.
- It does not create a candidate card.
- It does not select, reject, or rank the candidate by itself.
- [Selection Decisions](../selection-decisions/index.md) and [Audit Result](../audit-result.md)
  are maintained in their dedicated downstream audit surfaces.
