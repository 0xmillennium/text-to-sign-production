# Text-Pose Semantic Consistency Objective

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-TEXT-POSE-SEMANTIC-CONSISTENCY`

## Candidate Name

Text-Pose Semantic Consistency Objective

## Candidate Record Type

- `Candidate record type`: `model_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-TEXT-POSE-SEMANTIC-ALIGNMENT`
- `Family link`: [FAM-TEXT-POSE-SEMANTIC-ALIGNMENT](../candidate-universe/text-pose-semantic-alignment.md)
- `Family type/status`: `primary_candidate_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the
  text-pose semantic alignment and conditioning family as an auxiliary/additive mechanism for
  testing whether explicit semantic consistency or cross-modal alignment improves generated
  pose/motion faithfulness to the source text.

## Candidate-Level Technical Definition

A gloss-free semantic-alignment candidate that adds an explicit text-pose consistency signal,
cross-modal embedding objective, auxiliary semantic grounding loss, or lightweight conditioning
adapter to encourage generated pose/keypoint sequences to remain semantically aligned with the
source text. The candidate focuses on the semantic-conditioning question: how should generated
sign motion remain semantically aligned with the source text?

## Candidate Boundary

- `Included mechanism`: Text-pose semantic consistency objective, cross-modal text/pose embedding
  alignment, embedding-consistency regularization, semantic grounding loss, or lightweight
  text-conditioning adapter applied to text/transcript-to-pose/keypoint production.
- `Explicitly excluded mechanisms`: Learned pose-token bottleneck as the primary contribution,
  latent diffusion as the primary generation mechanism, articulator-disentangled structure
  modeling as the primary contribution, retrieval/stitching, gloss-to-pose,
  HamNoSys/notation-to-pose, 3D avatar, SMPL-X, and rendering.
- `Not to be confused with`: Learned pose-token representation candidates, latent generative
  production candidates, structure-aware articulator candidates, retrieval comparators,
  gloss/notation boundary evidence, or frontier avatar candidates.

Auxiliary translation objectives, multimodal embedding consistency, and gloss-free alignment
biases are strong variants or adjacent mechanisms. This card focuses on a minimal text-pose
semantic consistency candidate.

## Research Role In Current Audit

- `Research role label`: `auxiliary_additive_objective`

This record defines an auxiliary/additive objective or mechanism, not a standalone primary model
direction and not one of the advanced primary model candidates. It may attach to another candidate
only if a later ablation defines the attachment point and evaluation method.

## Expected Improvement Mechanism

This candidate is expected to reduce semantic drift by aligning generated motion representations
with the source text. Expected mechanisms include reducing semantic mismatch, improving
text-conditioned motion faithfulness, reducing sequence drift away from the source sentence,
making semantic errors more diagnosable, providing an auxiliary training or evaluation signal
beyond raw pose reconstruction, and complementing but not replacing pose-quality metrics. Gains
are conditional on reliable text/pose embeddings, meaningful contrastive or consistency
objectives, and evaluation sensitivity.

## Targeted Failure Modes

- `Applicable failure modes`: `semantic_mismatch`, `sequence_drift`, `timing_misalignment`,
  `evaluation_blind_spot`, `hand_articulation_loss`, `non_manual_loss`,
  `cross_channel_inconsistency`; secondary risks: `oversmoothing`, `mode_averaging`

This candidate mainly targets semantic faithfulness and conditioning failure modes. Motion
naturalness, structure preservation, and generative diversity may require separate candidates or
auxiliary mechanisms.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_direct`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed; semantic consistency is
  derived from paired text/transcript and pose/keypoint or generated motion artifacts.
- `Segmentation required`: Sentence/video segment boundaries are assumed; no additional gloss
  segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for the minimal keypoint-level candidate.
- `Current project artifacts sufficient`: Conditional; sufficient if stable
  text/transcript-to-pose/keypoint pairs, generated pose/keypoint outputs, split metadata, and
  usable text/pose embedding or comparison features are available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if
  pose/keypoint outputs can be embedded, pooled, or otherwise compared with text-conditioned
  representations.
- `Compatible with current experiment workflow`: Conditionally compatible if alignment objectives
  and evaluation hooks are implemented as reusable modules.
- `Heavy new preprocessing required`: Low to moderate; may require text embeddings, pose
  embeddings, cached features, or auxiliary encoders.
- `Thin-driver notebook principle affected`: Not expected if embedding extraction, losses, and
  evaluation hooks live outside notebooks.
- `Reproducibility risk`: Medium; depends on pretrained encoder choices, embedding extraction,
  negative sampling or consistency design, and stable evaluation protocol.

## Evidence Chain

- `Source-selection basis`: Semantic-alignment relevance, gloss-free or gloss-adaptable
  supervision, text-to-pose conditioning relevance, dataset compatibility, evaluation relevance,
  and downstream contribution-selection relevance.
- `Supporting source-corpus entries used`: `SRC-TEXT2SIGNDIFF-2025`, `SRC-MS2SL-2024`,
  `SRC-T2S-GPT-2024`, `SRC-M3T-2026`, `SRC-A2VSLP-2026`
- `Limiting source-corpus entries used`: `SRC-LVMCN-2024`, `SRC-POSE-EVAL-2025`,
  `SRC-SIGNVQNET-2024`, `SRC-G2PDDM-2024`
- `Candidate-universe family used`: [FAM-TEXT-POSE-SEMANTIC-ALIGNMENT](../candidate-universe/text-pose-semantic-alignment.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Cross-modal text-pose alignment can support gloss-free text-to-sign-pose
  generation.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: Its diffusion mechanism belongs to the latent generative family;
  this candidate uses it only for semantic alignment and conditioning relevance.

### Source 2

- `Source corpus entry`: `SRC-MS2SL-2024`
- `Source link`: [SRC-MS2SL-2024](../../source-corpus.md#src-ms2sl-2024)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Text, audio, and sign representations can provide useful cross-modal
  conditioning signal for production.
- `Evidence note`: MS2SL is recorded as a core recent How2Sign source for text/audio-to-keypoint
  production via diffusion and embedding consistency.
- `Interpretation boundary`: Audio-conditioning is relevant only when it clarifies semantic
  conditioning for text-to-pose; it does not change this candidate's text-to-pose framing.

### Source 3

- `Source corpus entry`: `SRC-T2S-GPT-2024`
- `Source link`: [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Support type`: `transferable_method_support`
- `Claim supported`: Text-conditioned code or token generation requires semantic conditioning
  between input sentence and generated sign representation.
- `Evidence note`: T2S-GPT is recorded as core evidence for dynamic variable-length sign
  tokenization and autoregressive text-to-token generation.
- `Interpretation boundary`: Its primary role remains learned representation; this candidate uses
  it only for text-conditioned generation and semantic-conditioning relevance.

### Source 4

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Support type`: `boundary_support`
- `Claim supported`: Motion-token spaces may require auxiliary semantic grounding to preserve
  linguistic meaning.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: Its tokenization and 3D/parametric aspects belong to
  learned-representation and frontier families; this candidate uses it only for
  semantic-grounding pressure.

### Source 5

- `Source corpus entry`: `SRC-A2VSLP-2026`
- `Source link`: [SRC-A2VSLP-2026](../../source-corpus.md#src-a2vslp-2026)
- `Support type`: `transferable_method_support`
- `Claim supported`: Semantic alignment can be integrated with variational articulator-wise
  latent modeling.
- `Evidence note`: A²V-SLP is recorded as a strong recent continuation of articulator-wise latent
  modeling with preprint/frontier caution.
- `Interpretation boundary`: Its variational and articulator-wise aspects belong to
  latent-generative and structure-aware families; this candidate uses it only for alignment
  relevance.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-LVMCN-2024`
- `Source link`: [SRC-LVMCN-2024](../../source-corpus.md#src-lvmcn-2024)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Fine-grained linguistic-visual alignment is important, but gloss-pose
  alignment solutions may be supervision-incompatible with the project's no-public-gloss regime.
- `Evidence note`: LVMCN is recorded as showing that alignment matters, but its solution depends
  on gloss-pose semantic alignment.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 2

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Semantic consistency or embedding alignment alone cannot prove sign
  intelligibility, timing, non-manual adequacy, or semantic fidelity.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate semantic consistency objectives; it limits
  how alignment gains should be evaluated.

### Source 3

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Semantic consistency should not be assumed more useful than representation
  bottlenecks without evaluation.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This is counter-evidence against prematurely choosing the
  semantic-consistency route, not against evaluating it as an auxiliary/additive objective.

### Source 4

- `Source corpus entry`: `SRC-G2PDDM-2024`
- `Source link`: [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Alignment or discrete-code methods may be technically relevant while
  remaining gloss-to-pose constrained.
- `Evidence note`: G2P-DDM is recorded as valuable for VQ/discrete diffusion design but not
  directly compatible with gloss-free How2Sign assumptions.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: A gloss-free text-pose semantic consistency objective is a credible
  auxiliary/additive mechanism for reducing semantic drift in generated pose/motion.
- `Least-secure claim`: The exact embedding space, contrastive objective, auxiliary loss, negative
  sampling strategy, and project-specific semantic-evaluation signal are not established by the
  candidate card alone.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The semantic-alignment family is well supported as an audit
  concern, but project-specific feasibility depends on encoder choices, alignment-loss design,
  available text/pose features, and evaluation sensitivity.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md), this candidate
adds an explicit semantic consistency signal. Compared with the
[Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md),
[Articulator-Disentangled Latent Modeling](articulator-disentangled-latent.md), and
[Gloss-Free Latent Diffusion](gloss-free-latent-diffusion.md), it targets semantic faithfulness
and conditioning rather than representation tokenization, latent generation, or structure
preservation. Retrieval-augmented comparators and frontier 3D/avatar approaches remain adjacent
alternatives, but this candidate is distinct because it does not center retrieval or avatar
output.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Compare baseline or model output with and without the semantic-consistency
  objective under the same splits, input text, output pose/keypoint schema, and evaluation
  protocol.
- `Primary metric family`: Semantic alignment checks using text-pose embeddings or auxiliary
  consistency metrics where available.
- `Secondary metric family`: Pose/keypoint distance, embedding-based motion checks, and cautious
  back-translation or recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Improved embedding consistency does not by itself prove sign
  intelligibility, non-manual adequacy, timing, or semantic fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Conditional; additive if the semantic consistency objective is the
  primary mechanism changed relative to the direct baseline or another model candidate.
- `Baseline changed`: No, unless semantic feature extraction changes training targets or
  evaluation artifacts enough to require a separate baseline variant.
- `Primary isolated variable`: Text-pose semantic consistency or alignment objective.
- `Requires another candidate`: No for a minimal direct-baseline attachment, but it may be most
  useful as an auxiliary objective on top of learned-token, latent-generation, or structure-aware
  candidates.
- `Independent ablation possible`: Yes; compare with and without the semantic-consistency
  objective under the same data and evaluation protocol.
- `Likely interactions with other candidate families`: Strong interaction with learned pose-token
  bottlenecks, latent generative production, and structure-aware modeling; interactions must be
  controlled if combined later.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium.
- `Failure cost`: Medium; a weak alignment objective can add complexity without improving motion
  or semantic quality.
- `Hidden dependency risk`: Medium; depends on text encoders, pose encoders, feature extraction,
  negative sampling, and metric design.
- `Scope-creep risk`: Medium if auxiliary translation, recognition models, or multimodal audio
  conditioning are added too early.
- `Evaluation risk`: High because semantic faithfulness is difficult to verify with automatic
  metrics alone.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after embedding/encoder choices, objective design,
  ablation target, and evaluation protocol are fixed.
- `Open questions for scorecard`: text encoder, pose encoder, alignment loss, negative sampling,
  whether the objective is evaluated as a baseline-attached auxiliary objective or as an additive
  objective on another model candidate, and how semantic gains will be measured.
- `Evidence gaps to resolve before selection decision`: project-specific alignment feasibility,
  metric validity, interaction with other candidate mechanisms, and qualitative inspection
  reliability.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
