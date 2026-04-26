# Learned Pose-Token Bottleneck

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-LEARNED-POSE-TOKEN-BOTTLENECK`

## Candidate Name

Learned Pose-Token Bottleneck

## Candidate Record Type

- `Candidate record type`: `model_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-LEARNED-POSE-MOTION-REPRESENTATIONS`
- `Family link`: [FAM-LEARNED-POSE-MOTION-REPRESENTATIONS](../candidate-universe/learned-pose-motion-representations.md)
- `Family type/status`: `primary_candidate_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the learned
  pose/motion representation family as a concrete test of whether a learned pose-token or compact
  motion-code bottleneck improves over direct text-to-raw-pose generation.

## Candidate-Level Technical Definition

A gloss-free learned representation candidate that learns pose/keypoint-derived discrete or
compact motion codes and uses those codes as an intermediate bottleneck for
text/transcript-to-pose/keypoint production. The candidate focuses on the representation question:
what intermediate pose/motion representation should the model generate or condition on?

## Candidate Boundary

- `Included mechanism`: Learned pose-token or compact motion-code bottleneck derived from
  pose/keypoint/motion data; text/transcript-conditioned prediction of that representation;
  reconstruction or decoding back to pose/keypoint sequences.
- `Explicitly excluded mechanisms`: Latent diffusion, full generative diffusion,
  articulator-disentangled latent groups as the primary contribution, semantic consistency
  objectives as the primary contribution, retrieval/stitching, gloss-to-pose,
  HamNoSys/notation-to-pose, 3D avatar, SMPL-X, and rendering.
- `Not to be confused with`: Latent generative production candidates, structure-aware
  articulator candidates, text-pose semantic alignment candidates, retrieval comparators, or
  frontier avatar candidates.

Dynamic VQ, code-duration generation, and modality-specific tokenization are strong alternatives
or later variants. This card focuses on a minimal learned pose-token bottleneck.

## Research Role In Current Audit

- `Research role label`: `primary_model_candidate`

This record defines a primary model candidate to compare against the M0 baseline and to score only
in downstream scorecard surfaces.

## Expected Improvement Mechanism

This candidate is expected to reduce direct-regression weaknesses by forcing generated motion
through a learned pose/motion representation. Expected mechanisms include reducing oversmoothing
and mode averaging, improving motion structure by avoiding unconstrained raw-coordinate
prediction, making generated sequences more temporally coherent, and creating an intermediate
representation that can be inspected or ablated. These gains are conditional on token and
reconstruction quality, and on avoiding token collapse.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `temporal_jitter`, `timing_misalignment`,
  `hand_articulation_loss`, `non_manual_loss`, `cross_channel_inconsistency`, `mode_averaging`,
  `sequence_drift`, `token_collapse`, `reconstruction_loss`; secondary risk:
  `semantic_mismatch`

This candidate mainly targets motion and representation failure modes. Semantic alignment may
require a separate candidate or auxiliary objective.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_direct`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: No manual gloss annotation assumed; learned tokens are derived
  from pose/keypoint/motion data.
- `Segmentation required`: Sentence/video segment boundaries are assumed; no additional gloss
  segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for the minimal keypoint-level candidate.
- `Current project artifacts sufficient`: Conditional; sufficient if stable pose/keypoint arrays,
  text/transcript pairs, split metadata, and reconstruction targets are available.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Conditionally compatible if the schema
  supports consistent body/hand/face keypoint dimensions and masks.
- `Compatible with current experiment workflow`: Expected to be compatible if tokenizer training,
  code assignment, and decoding are implemented as reusable modules.
- `Heavy new preprocessing required`: Moderate; requires learning or fitting the pose-token
  bottleneck and storing token/code artifacts.
- `Thin-driver notebook principle affected`: Not expected if tokenization and decoding code live
  outside notebooks.
- `Reproducibility risk`: Medium; depends on deterministic tokenizer training, codebook stability,
  split handling, and reconstruction evaluation.

## Evidence Chain

- `Source-selection basis`: Representation relevance, gloss-free or gloss-adaptable supervision,
  pose/keypoint production relevance, dataset compatibility, and downstream contribution-selection
  relevance.
- `Supporting source-corpus entries used`: `SRC-SIGNVQNET-2024`, `SRC-DATA-DRIVEN-REP-2024`,
  `SRC-T2S-GPT-2024`, `SRC-UNIGLOR-2024`, `SRC-M3T-2026`
- `Limiting source-corpus entries used`: `SRC-G2PDDM-2024`, `SRC-TEXT2SIGNDIFF-2025`,
  `SRC-POSE-EVAL-2025`
- `Candidate-universe family used`: [FAM-LEARNED-POSE-MOTION-REPRESENTATIONS](../candidate-universe/learned-pose-motion-representations.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGNVQNET-2024`
- `Source link`: [SRC-SIGNVQNET-2024](../../source-corpus.md#src-signvqnet-2024)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Learned discrete pose-token production is a credible gloss-free
  representation family for sign language production.
- `Evidence note`: SignVQNet is recorded as core evidence for learned discrete pose-token SLP
  under a gloss-free setting.
- `Interpretation boundary`: This supports the representation-level candidate; it does not by
  itself select a specific tokenizer implementation for this project.

### Source 2

- `Source corpus entry`: `SRC-DATA-DRIVEN-REP-2024`
- `Source link`: [SRC-DATA-DRIVEN-REP-2024](../../source-corpus.md#src-data-driven-rep-2024)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Learned pose/motion representations can reduce dependence on scarce
  linguistic annotation.
- `Evidence note`: The source is recorded as core evidence that learned pose/motion
  representations can replace scarce linguistic annotation.
- `Interpretation boundary`: Transfer to this project still requires compatibility with the
  available pose/keypoint schema.

### Source 3

- `Source corpus entry`: `SRC-T2S-GPT-2024`
- `Source link`: [SRC-T2S-GPT-2024](../../source-corpus.md#src-t2s-gpt-2024)
- `Support type`: `transferable_method_support`
- `Claim supported`: Dynamic or variable-length sign tokenization is a serious representation
  direction.
- `Evidence note`: T2S-GPT is recorded as core evidence for dynamic variable-length sign
  tokenization and autoregressive text-to-token generation.
- `Interpretation boundary`: Dataset transfer must be argued because the source is recorded as
  `cross_lingual_transferable`, not How2Sign-direct.

### Source 4

- `Source corpus entry`: `SRC-UNIGLOR-2024`
- `Source link`: [SRC-UNIGLOR-2024](../../source-corpus.md#src-uniglor-2024)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Learned keypoint-derived representations can serve as non-gloss intermediate
  representations.
- `Evidence note`: UniGloR is recorded as central evidence for non-gloss learned intermediate
  representations from keypoints.
- `Interpretation boundary`: Learned intermediate representations must not be conflated with
  manually annotated gloss.

### Source 5

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Support type`: `boundary_support`
- `Claim supported`: Recent work pressures representation design toward body, hands, and
  face/non-manual motion tokenization.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: This is frontier evidence and should not force immediate adoption of
  a heavy 3D or parametric representation.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-G2PDDM-2024`
- `Source link`: [SRC-G2PDDM-2024](../../source-corpus.md#src-g2pddm-2024)
- `Limitation type`: `supervision_limitation`
- `Claim limited`: Discrete pose representations and diffusion over codes may be technically
  relevant while remaining gloss-to-pose constrained.
- `Evidence note`: G2P-DDM is recorded as valuable for VQ/discrete diffusion design but not
  directly compatible with gloss-free How2Sign assumptions.
- `Interpretation boundary`: This source must remain boundary evidence unless a gloss-free
  adaptation path is documented.

### Source 2

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Limitation type`: `counter_evidence`
- `Claim limited`: Learned pose-token bottlenecks should not be assumed superior to latent
  generative approaches without evaluation.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This is counter-evidence against prematurely choosing the
  token-bottleneck route, not against evaluating it as a primary model candidate.

### Source 3

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Token reconstruction quality or pose distance alone cannot prove sign
  intelligibility or semantic adequacy.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate learned tokens; it limits how representation
  gains should be evaluated.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: A learned pose-token or compact motion-code bottleneck is a credible
  primary model candidate for improving over direct raw-pose regression.
- `Least-secure claim`: The exact tokenizer type, codebook size, temporal granularity, and
  reconstruction quality needed for this project are not established by the candidate card alone.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The representation family is well supported across
  source-corpus entries, but project-specific feasibility depends on artifact schema, tokenizer
  stability, reconstruction quality, and evaluation results.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md), this candidate
adds a learned representation bottleneck before pose/keypoint decoding. Dynamic VQ and
code-duration tokenization remain strong variants or alternatives, while articulator-disentangled
latent modeling, gloss-free latent diffusion, text-pose semantic consistency objectives, and
retrieval-augmented comparators address adjacent modeling questions. This card intentionally
starts with a minimal learned pose-token bottleneck to avoid over-fragmenting the first audit
pass.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md).
- `Required ablation`: Compare direct text-to-pose against text-to-token-to-pose with the same
  splits, input text, and output pose/keypoint schema; ablate token bottleneck usage against
  direct regression.
- `Primary metric family`: Pose/keypoint distance and reconstruction metrics.
- `Secondary metric family`: Embedding-based motion checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Improved reconstruction or lower pose distance does not by itself
  prove sign intelligibility, non-manual adequacy, or semantic fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Conditional; additive if the learned token bottleneck is the primary
  mechanism changed relative to the direct baseline.
- `Baseline changed`: No, unless tokenizer training changes preprocessing or target
  representation enough to require a separate baseline variant.
- `Primary isolated variable`: Learned pose/motion token bottleneck.
- `Requires another candidate`: No.
- `Independent ablation possible`: Yes; compare with and without the learned token bottleneck
  under the same data and evaluation protocol.
- `Likely interactions with other candidate families`: May interact with structure-aware
  modeling, latent generative production, and semantic alignment; those should remain separate
  unless explicitly combined in a later candidate.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium.
- `Failure cost`: Medium; a weak tokenizer can reduce motion quality or obscure model comparison.
- `Hidden dependency risk`: Medium; depends on stable keypoint schema, masks, tokenization
  training, and reconstruction quality.
- `Scope-creep risk`: Medium if dynamic tokenization, duration modeling, or modality-specific
  tokens are added too early.
- `Evaluation risk`: Medium to high because token quality, reconstruction quality, motion
  naturalness, and semantic adequacy may diverge.

## Downstream Scoring Notes

- `Scorecard readiness`: Conditional; ready after tokenizer design, reconstruction target,
  codebook/training setup, and evaluation protocol are fixed.
- `Open questions for scorecard`: tokenizer type, codebook size, temporal granularity,
  reconstruction loss, handling of body/hand/face channels, and whether dynamic duration modeling
  is included or excluded.
- `Evidence gaps to resolve before selection decision`: project-specific reconstruction quality,
  token stability, comparison against direct baseline, and metric limitations.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
