# Gloss-Free Latent Diffusion

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-GLOSS-FREE-LATENT-DIFFUSION`

## Candidate Name

Gloss-Free Latent Diffusion

## Candidate Record Type

- `Candidate record type`: `model_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-LATENT-GENERATIVE-PRODUCTION`
- `Family link`: [FAM-LATENT-GENERATIVE-PRODUCTION](../candidate-universe/latent-generative-production.md)
- `Family type/status`: `primary_candidate_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the latent
  generative production family as a concrete test of whether gloss-free text-conditioned latent
  diffusion or a lightweight latent generative mechanism improves over direct text-to-pose
  generation.

## Candidate-Level Technical Definition

A gloss-free latent generative production candidate that maps text/transcript input into a latent
pose or motion generation process, such as latent diffusion or another lightweight latent
generative mechanism, and decodes generated latent states into pose/keypoint sequences. The
candidate focuses on the generation question: how should the model generate sign pose or motion
from text-conditioned latent structure?

## Candidate Boundary

- `Included mechanism`: Text/transcript-conditioned latent pose or motion generation, latent
  diffusion or lightweight latent generative dynamics, denoising or iterative latent generation,
  and decoding generated latent states to pose/keypoint sequences.
- `Explicitly excluded mechanisms`: Learned pose-token bottleneck as the primary contribution,
  articulator-disentangled structure modeling as the primary contribution, semantic consistency
  objective as the primary contribution, retrieval/stitching, gloss-to-pose,
  HamNoSys/notation-to-pose, 3D avatar, SMPL-X, and rendering.
- `Not to be confused with`: Learned pose-token representation candidates, structure-aware
  articulator candidates, text-pose semantic alignment candidates, retrieval comparators,
  gloss/notation boundary evidence, or frontier avatar candidates.

Iterative latent refinement, variational articulator-wise latent modeling, and multimodal
audio-conditioned diffusion are strong variants or adjacent methods. This card focuses on a
gloss-free text-conditioned latent diffusion or latent generation candidate.

## Research Role In Current Audit

- `Research role label`: `primary_model_candidate`

This record defines a primary model candidate to compare against the M0 baseline and to score only
in downstream scorecard surfaces.

## Expected Improvement Mechanism

This candidate is expected to reduce direct-regression weaknesses by modeling pose/motion
generation as a latent generative process rather than predicting every pose frame directly.
Expected mechanisms include reducing oversmoothing and mode averaging, improving motion diversity
and temporal naturalness, better modeling uncertainty or multimodal motion trajectories, enabling
iterative refinement or denoising over pose/motion latents, and potentially improving
sequence-level coherence compared with direct regression. Gains are conditional on latent quality,
training stability, compute feasibility, and evaluation sensitivity.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `temporal_jitter`, `timing_misalignment`,
  `mode_averaging`, `sequence_drift`, `unnatural_transitions`, `reconstruction_loss`;
  secondary risks: `semantic_mismatch`, `hand_articulation_loss`, `non_manual_loss`

This candidate mainly targets generative motion-quality and sequence-level failure modes.
Structure preservation and semantic alignment may require separate candidates or auxiliary
mechanisms.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_direct`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed; latent supervision is derived
  from pose/keypoint/motion artifacts.
- `Segmentation required`: Sentence/video segment boundaries are assumed; no additional gloss
  segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for the minimal keypoint-level candidate.
- `Current project artifacts sufficient`: Conditional; sufficient if stable
  text/transcript-to-pose/keypoint training pairs, split metadata, and latent reconstruction
  targets are available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if latent
  generation is trained over the current pose/keypoint sequence representation.
- `Compatible with current experiment workflow`: Conditionally compatible, but likely heavier
  than baseline or learned-token candidates.
- `Heavy new preprocessing required`: Moderate; may require latent autoencoding, normalization,
  denoising schedules, or cached latent targets.
- `Thin-driver notebook principle affected`: Not expected if latent model, scheduler, and
  evaluation code live outside notebooks.
- `Reproducibility risk`: Medium to high; depends on stochastic training, denoising schedules,
  seed control, latent reconstruction quality, and evaluation protocol.

## Evidence Chain

- `Source-selection basis`: Latent/generative method relevance, gloss-free or gloss-adaptable
  supervision, pose/keypoint production relevance, dataset compatibility, evaluation relevance,
  and downstream contribution-selection relevance.
- `Supporting source-corpus entries used`: `SRC-TEXT2SIGNDIFF-2025`, `SRC-MS2SL-2024`,
  `SRC-DARSLP-2025`, `SRC-ILRSLP-2025`, `SRC-A2VSLP-2026`, `SRC-NSLPG-2021`
- `Limiting source-corpus entries used`: `SRC-SIGNVQNET-2024`, `SRC-G2PDDM-2024`,
  `SRC-NEURAL-SIGN-ACTORS-2024`, `SRC-SIGNSPARK-2026`, `SRC-POSE-EVAL-2025`
- `Candidate-universe family used`: [FAM-LATENT-GENERATIVE-PRODUCTION](../candidate-universe/latent-generative-production.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Latent diffusion is a credible gloss-free text-to-sign-pose production
  family.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This supports the latent generative candidate; it does not by itself
  settle implementation feasibility or evaluation adequacy for this project.

### Source 2

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Support type`: `near_direct_method_support`
- `Claim supported`: How2Sign can support recent text/audio-conditioned keypoint generation under
  a gloss-free setting.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Audio-conditioning is relevant evidence but does not change this
  candidate's primary text-to-pose framing.

### Source 3

- `Source corpus entry`: `SRC-DARSLP-2025`
- `Source link`: [SRC-DARSLP-2025](../../source-corpus.md#src-darslp-2025)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Latent-space sign production can be structured around
  articulator-disentangled representations.
- `Evidence note`: DARSLP is recorded as a core source for articulator-disentangled latent spaces
  and channel-aware regularization.
- `Interpretation boundary`: Structure-aware aspects belong to the structure-aware family; this
  candidate uses DARSLP only for latent-production relevance.

### Source 4

- `Source corpus entry`: `SRC-ILRSLP-2025`
- `Source link`: [SRC-ILRSLP-2025](../../source-corpus.md#src-ilrslp-2025)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Latent sign production can be refined iteratively instead of predicted once.
- `Evidence note`: ILRSLP is recorded as an important continuation of articulator-disentangled
  latent modeling.
- `Interpretation boundary`: Iterative refinement is adjacent evidence; it should not turn this
  card into a general structure-aware or iterative-refinement candidate.

### Source 5

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Support type`: `transferable_method_support`
- `Claim supported`: Latent sign production can model distributional uncertainty through
  variational articulator-wise prediction.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: This is recent frontier evidence and should not automatically
  outweigh more directly compatible latent-diffusion sources.

### Source 6

- `Source corpus entry`: `SRC-NSLPG-2021`
- `Source link`: [SRC-NSLPG-2021](../../source-corpus.md#src-nslpg-2021)
- `Support type`: `boundary_support`
- `Claim supported`: Latent-space non-autoregressive SLP is part of the foundational generative
  baseline landscape.
- `Evidence note`: NSLP-G is recorded as a non-autoregressive latent baseline and as a source that
  critiques back-translation reliability.
- `Interpretation boundary`: This source is older baseline/evaluation-caution evidence, not
  sufficient modern latent-generation support by itself.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Latent diffusion should not be assumed superior to learned pose-token
  bottlenecks without evaluation.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This is counter-evidence against prematurely choosing the
  latent-diffusion route, not against evaluating it as a primary model candidate.

### Source 2

- `Source corpus entry`: `SRC-G2PDDM-2024`
- `Source link`: [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Discrete diffusion or code-based generative methods may be technically
  relevant while remaining gloss-to-pose constrained.
- `Evidence note`: G2P-DDM is recorded as valuable for VQ/discrete diffusion design but not
  directly compatible with gloss-free How2Sign assumptions.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 3

- `Source corpus entry`: `SRC-NEURAL-SIGN-ACTORS-2024`
- `Source link`: [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024)
- `Limitation type`: `scope_limitation`
- `Claim limited`: Latent/diffusion production can extend toward 3D avatar-compatible signing,
  but such systems may exceed the current keypoint-level thesis scope.
- `Evidence note`: Neural Sign Actors is recorded as important for SMPL-X / 3D avatar-compatible
  production, with heavier-than-current-keypoint-pipeline caution.
- `Interpretation boundary`: This is frontier/avatar-compatible evidence and should not force
  immediate adoption of a 3D parametric pipeline.

### Source 4

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Better generative motion quality or lower pose error alone cannot prove sign
  intelligibility or semantic adequacy.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate latent diffusion; it limits how generative
  gains should be evaluated.

### Source 5

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Limitation type`: `scope_limitation`
- `Claim limited`: Sparse-keyframe or flow-based frontier generation may exceed the current minimal keypoint-level latent-diffusion candidate scope.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and should not force sparse-keyframe, CFM, or 3D/avatar control into the current candidate.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: Gloss-free latent diffusion or lightweight latent generative production
  is a credible primary model candidate for improving over direct raw-pose regression.
- `Least-secure claim`: The exact latent design, denoising setup, compute cost, training
  stability, and project-specific evaluation gains are not established by the candidate card
  alone.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The latent generative family has strong recent support and
  direct How2Sign-related evidence, but project-specific feasibility depends on latent
  reconstruction quality, training complexity, compute constraints, and evaluation sensitivity.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md), this candidate
adds text-conditioned latent generative production. Compared with the
[Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md), it targets generation dynamics
rather than representation tokenization. Compared with
[Articulator-Disentangled Latent Modeling](articulator-disentangled-latent.md), it targets latent
generation rather than structure preservation. Text-pose semantic consistency objectives,
retrieval-augmented comparators, and frontier sparse-keyframe/avatar approaches remain adjacent
alternatives, but this candidate is distinct because it does not center semantic conditioning,
retrieval, or avatar output.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Compare direct text-to-pose against text-conditioned latent diffusion or
  latent generation with the same splits, input text, and output pose/keypoint schema; ablate
  latent/generative mechanism against direct regression.
- `Primary metric family`: Pose/keypoint distance and sequence-level motion metrics where
  available.
- `Secondary metric family`: Embedding-based motion checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Improved generative diversity, denoising quality, or lower pose
  distance does not by itself prove sign intelligibility, non-manual adequacy, or semantic
  fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Conditional; additive if the latent generative mechanism is the
  primary mechanism changed relative to the direct baseline.
- `Baseline changed`: No, unless latent autoencoding or preprocessing changes the output target
  representation enough to require a separate baseline variant.
- `Primary isolated variable`: Text-conditioned latent generative production mechanism.
- `Requires another candidate`: No.
- `Independent ablation possible`: Yes; compare with and without the latent generative mechanism
  under the same data and evaluation protocol.
- `Likely interactions with other candidate families`: May interact with learned pose-token
  representations, structure-aware modeling, and semantic alignment; those should remain separate
  unless explicitly combined in a later candidate.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium to high.
- `Failure cost`: Medium to high; unstable latent generation can consume substantial effort
  without reliable improvement.
- `Hidden dependency risk`: Medium to high; depends on latent autoencoding, stochastic training,
  denoising schedules, compute budget, and evaluation sensitivity.
- `Scope-creep risk`: Medium to high if multimodal audio conditioning, iterative refinement, or
  3D/avatar extensions are added too early.
- `Evaluation risk`: High because generative diversity, motion quality, and semantic adequacy may
  diverge across metrics.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after latent representation, denoising/generation
  setup, compute assumptions, and evaluation protocol are fixed.
- `Open questions for scorecard`: latent type, denoising schedule, training cost, inference cost,
  sequence length handling, reconstruction target, comparison to learned-token bottleneck, and
  metric sensitivity.
- `Evidence gaps to resolve before selection decision`: project-specific training feasibility,
  generated motion quality, comparison against direct baseline and learned-token candidate,
  compute cost, and metric limitations.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
