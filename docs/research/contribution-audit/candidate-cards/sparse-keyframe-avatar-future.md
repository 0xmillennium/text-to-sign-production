# Sparse-Keyframe / Avatar-Control Future Direction

This document records candidate-level evidence only. It is not a scorecard, selection decision, or final audit result.

## Candidate ID

`CAND-SPARSE-KEYFRAME-AVATAR-FUTURE`

## Candidate Name

Sparse-Keyframe / Avatar-Control Future Direction

## Candidate Record Type

- `Candidate record type`: `frontier_future_work_candidate`

## Originating Candidate-Universe Family

- `Family ID`: `FAM-THREED-AVATAR-PARAMETRIC-FRONTIER`
- `Family link`: [FAM-THREED-AVATAR-PARAMETRIC-FRONTIER](../candidate-universe/threed-avatar-parametric-frontier.md)
- `Family type/status`: `frontier_watch_family`
- `Reason this candidate is derived from this family`: This candidate operationalizes the 3D
  avatar and parametric motion frontier family as a future-work record that preserves
  sparse-keyframe, avatar-compatible, and parametric output-surface directions without converting
  them into immediate thesis implementation scope.

It is acceptable that this candidate derives from a `frontier_watch_family`, because this card is
a `frontier_future_work_candidate`, not a near-term model contribution.

## Candidate-Level Technical Definition

A future-work candidate that tracks sparse-keyframe, avatar-compatible, 3D parametric, or
rendering-oriented sign production directions as possible downstream extensions beyond the current
keypoint-level text-to-pose artifact. The candidate focuses on the output-surface question: when
should the project eventually move beyond keypoint-level pose output toward avatar-compatible or
parametric signing?

## Candidate Boundary

- `Included mechanism`: Sparse-keyframe control, avatar-compatible motion targets, 3D or
  parametric body/hand/face representations, SMPL-X/SMPL-like future output surfaces, downstream
  visualization bridges, and future rendering-oriented sign production analysis.
- `Explicitly excluded mechanisms`: Near-term implementation requirement, required current
  thesis artifact, ordinary 2D keypoint baseline, learned pose-token bottleneck as the primary
  contribution, latent diffusion as the immediate implementation mechanism, structure-aware
  articulator modeling as the primary contribution, semantic consistency objective,
  retrieval/stitching comparator, gloss-to-pose, and HamNoSys/notation-to-pose.
- `Not to be confused with`: Primary model candidates, current keypoint-level implementation
  plan, scorecard selection, or a decision to implement 3D avatar generation now.

This card should prevent both under-acknowledging frontier directions and prematurely expanding
the thesis scope.

## Research Role In Current Audit

- `Research role label`: `future_work_candidate`

This record defines a future-work and frontier-awareness candidate, not a near-term primary model
candidate.

## Expected Improvement Mechanism

This candidate does not improve the current keypoint-level model directly. Its expected audit
value is strategic: it preserves awareness of sparse-keyframe and avatar-compatible frontier
directions, clarifies the boundary between current keypoint-level production and future
user-facing visualization, prevents accidental scope creep into heavy 3D/avatar systems, records
non-manual, face, hand, and body representation pressure, and identifies future interface and
visualization paths once keypoint-level production is stabilized. Any future gains are conditional
on 3D/parametric artifact availability, fitting quality, rendering infrastructure, evaluation
design, and thesis scope.

## Targeted Failure Modes

- `Applicable failure modes`: `non_manual_loss`, `hand_articulation_loss`,
  `cross_channel_inconsistency`, `timing_misalignment`, `unnatural_transitions`,
  `evaluation_blind_spot`; secondary risks: `semantic_mismatch`, `sequence_drift`

This candidate mainly tracks frontier output-surface and visualization limitations, not near-term
keypoint-level model failure correction.

## Data And Supervision Assumptions

- `Dataset compatibility`: `how2sign_compatible`
- `Gloss dependency`: `gloss_free`
- `Extra annotation required`: Not required for recording this future-work candidate, but likely
  required for future implementation if 3D fitting, avatar parameters, non-manual labels, or
  rendering assets are introduced.
- `Segmentation required`: Sentence/video segment boundaries are assumed for any future
  extension; no gloss segmentation is assumed.
- `Dictionary/example bank required`: Not required.
- `3D/parametric artifacts required`: Not required for this candidate card; likely required for
  future implementation.
- `Current project artifacts sufficient`: Sufficient only for documenting frontier pressure and
  future-work positioning; not sufficient by itself for implementing a full avatar/parametric
  pipeline.

## Repository / Workflow Compatibility

- `Compatible with current pose/keypoint artifact schema`: Partially compatible as a future
  output-surface direction; full implementation would require additional representation or
  fitting layers.
- `Compatible with current experiment workflow`: Compatible as a future-work documentation and
  analysis artifact; not yet compatible as a near-term experiment without additional
  infrastructure.
- `Heavy new preprocessing required`: Not for this record; likely high for future implementation
  because 3D fitting, avatar parameterization, or rendering pipelines may be needed.
- `Thin-driver notebook principle affected`: Not affected by this record; future implementation
  should keep fitting/rendering code outside notebooks.
- `Reproducibility risk`: Medium to high for future implementation because avatar fitting,
  rendering, model versions, and evaluation protocols can be difficult to standardize.

## Evidence Chain

- `Source-selection basis`: Frontier output-surface relevance, avatar/parametric representation
  relevance, sparse-keyframe relevance, non-manual/body/hand/face modeling pressure, dataset
  compatibility, and downstream positioning relevance.
- `Supporting source-corpus entries used`: `SRC-SIGNSPARK-2026`,
  `SRC-NEURAL-SIGN-ACTORS-2024`, `SRC-M3T-2026`, `SRC-EVERYBODY-SIGN-NOW-2020`
- `Limiting source-corpus entries used`: `SRC-TEXT2SIGNDIFF-2025`, `SRC-POSE-EVAL-2025`,
  `SRC-SIGNSPARK-2026`, `SRC-NEURAL-SIGN-ACTORS-2024`
- `Candidate-universe family used`: [FAM-THREED-AVATAR-PARAMETRIC-FRONTIER](../candidate-universe/threed-avatar-parametric-frontier.md)
- `Boundary families consulted`: [FAM-DATASET-SUPERVISION-BOUNDARY](../candidate-universe/dataset-supervision-boundary.md), [FAM-GLOSS-NOTATION-DEPENDENT-BOUNDARY](../candidate-universe/gloss-notation-dependent-boundary.md)
- `Evaluation family consulted`: [FAM-EVALUATION-BENCHMARK-METHODOLOGY](../candidate-universe/evaluation-benchmark-methodology.md)

## Supporting Source-Corpus Evidence

### Source 1

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Support type`: `direct_candidate_support`
- `Claim supported`: Sparse-keyframe and 3D/avatar-compatible production are relevant frontier
  directions for scalable sign language production.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This is frontier evidence and should not force sparse-keyframe, CFM,
  or 3D/avatar control into near-term thesis implementation.

### Source 2

- `Source corpus entry`: `SRC-NEURAL-SIGN-ACTORS-2024`
- `Source link`: [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024)
- `Support type`: `near_direct_method_support`
- `Claim supported`: Text-to-3D signing avatar systems show that SLP can move beyond
  keypoint-level pose output.
- `Evidence note`: Neural Sign Actors is recorded as important for SMPL-X / 3D avatar-compatible
  production, with heavier-than-current-keypoint-pipeline caution.
- `Interpretation boundary`: This supports long-term avatar-compatible direction, not immediate
  adoption of a 3D parametric pipeline.

### Source 3

- `Source corpus entry`: `SRC-M3T-2026`
- `Source link`: [SRC-M3T-2026](../../source-corpus.md#src-m3t-2026)
- `Support type`: `boundary_support`
- `Claim supported`: Future output-surface design must acknowledge body, hands, face, and
  non-manual richness.
- `Evidence note`: M3T is recorded as frontier/core evidence that body, hands, and face
  tokenization are needed for linguistically complete SLP.
- `Interpretation boundary`: Its tokenization and semantic-grounding aspects belong to other
  candidate families; this record uses it only for frontier output-surface pressure.

### Source 4

- `Source corpus entry`: `SRC-EVERYBODY-SIGN-NOW-2020`
- `Source link`: [SRC-EVERYBODY-SIGN-NOW-2020](../../source-corpus.md#src-everybody-sign-now-2020)
- `Support type`: `boundary_support`
- `Claim supported`: Pose production can be separated from downstream signer visualization or
  photorealistic video synthesis.
- `Evidence note`: Everybody Sign Now is recorded as a pose-to-video/rendering boundary source.
- `Interpretation boundary`: This source clarifies downstream visualization and rendering
  boundaries; it is not central to near-term pose modeling.

## Limiting Or Contradicting Evidence

### Source 1

- `Source corpus entry`: `SRC-TEXT2SIGNDIFF-2025`
- `Source link`: [SRC-TEXT2SIGNDIFF-2025](../../source-corpus.md#src-text2signdiff-2025)
- `Limitation type`: `scope_limitation`
- `Claim limited`: Strong keypoint-level latent generative methods may be closer to the current
  thesis scope than heavy avatar/parametric systems.
- `Evidence note`: Text2SignDiff is recorded as a core recent alternative to token/VQ approaches
  and directly evaluates on How2Sign.
- `Interpretation boundary`: This source supports keeping current implementation attention on
  keypoint-level production before expanding to avatar or parametric outputs.

### Source 2

- `Source corpus entry`: `SRC-POSE-EVAL-2025`
- `Source link`: [SRC-POSE-EVAL-2025](../../source-corpus.md#src-pose-eval-2025)
- `Limitation type`: `evaluation_limitation`
- `Claim limited`: Avatar realism or 3D output quality alone cannot prove sign intelligibility,
  timing, non-manual adequacy, or semantic fidelity.
- `Evidence note`: Meaningful Pose-Based Sign Language Evaluation is recorded as a core source
  for distance-based, embedding-based, and back-translation-based pose evaluation.
- `Interpretation boundary`: This does not invalidate avatar future work; it limits how future
  avatar outputs should be evaluated.

### Source 3

- `Source corpus entry`: `SRC-SIGNSPARK-2026`
- `Source link`: [SRC-SIGNSPARK-2026](../../source-corpus.md#src-signspark-2026)
- `Limitation type`: `scope_limitation`
- `Claim limited`: Sparse-keyframe / CFM / 3D SLP frontier systems may exceed the current
  keypoint-level implementation scope.
- `Evidence note`: SignSparK is recorded as a frontier source for keyframe-driven 3D SLP and CFM.
- `Interpretation boundary`: This source supports future-work tracking, not immediate thesis
  implementation.

### Source 4

- `Source corpus entry`: `SRC-NEURAL-SIGN-ACTORS-2024`
- `Source link`: [SRC-NEURAL-SIGN-ACTORS-2024](../../source-corpus.md#src-neural-sign-actors-2024)
- `Limitation type`: `scope_limitation`
- `Claim limited`: Text-to-3D signing avatar systems may require fitting, rendering, and
  parametric-body infrastructure beyond the current project scope.
- `Evidence note`: Neural Sign Actors is recorded as important for SMPL-X / 3D avatar-compatible
  production, with heavier-than-current-keypoint-pipeline caution.
- `Interpretation boundary`: This source should inform future-work positioning rather than
  near-term implementation commitment.

## Candidate-Level Evidence Synthesis

- `Best-supported claim`: Sparse-keyframe and avatar-compatible production should be preserved as
  a future-work direction and output-surface frontier, not ignored.
- `Least-secure claim`: The project can implement sparse-keyframe avatar control or parametric
  signing within the current thesis scope.
- `Evidence confidence`: `medium`
- `Evidence confidence rationale`: The frontier direction is well supported as positioning
  evidence, but immediate implementation feasibility is uncertain and likely exceeds the current
  keypoint-level artifact scope.

## Comparison To Strong Alternatives

Compared with the [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md),
[Learned Pose-Token Bottleneck](learned-pose-token-bottleneck.md),
[Articulator-Disentangled Latent Modeling](articulator-disentangled-latent.md),
[Gloss-Free Latent Diffusion](gloss-free-latent-diffusion.md),
[Text-Pose Semantic Consistency Objective](text-pose-semantic-consistency.md), and
[Retrieval-Augmented Pose Comparator](retrieval-augmented-pose-comparator.md), this candidate
tracks a frontier output surface and future visualization path rather than a near-term
keypoint-level model mechanism or comparator.

## Minimum Evaluation And Ablation Plan

- `Minimum comparison baseline`: Not applicable as a near-term model candidate; future
  implementation would still need comparison to [Direct Text-to-Pose Baseline](direct-text-to-pose-baseline.md)
  and relevant model candidates.
- `Required ablation`: Not required for this future-work record; future implementation should
  ablate sparse-keyframe control, avatar/parametric representation, or rendering bridge if
  instantiated later.
- `Primary metric family`: Future avatar or 3D output evaluation would need pose/keypoint,
  parametric consistency, and perceptual/qualitative checks.
- `Secondary metric family`: Embedding-based motion checks and cautious back-translation or
  recognition-based checks where available.
- `Qualitative inspection needed`: Yes.
- `Metric limitation note`: Avatar realism, rendering quality, or sparse-keyframe plausibility
  does not by itself prove sign intelligibility, non-manual adequacy, timing, or semantic
  fidelity.
- `Evaluation-family reference`: [Evaluation and Benchmark Methodology](../candidate-universe/evaluation-benchmark-methodology.md)

## Additivity And Isolation

- `Additive over baseline`: Not applicable for the current audit because this is a future-work
  candidate.
- `Baseline changed`: Not applicable unless future implementation changes the output
  representation from keypoints to avatar/parametric artifacts.
- `Primary isolated variable`: Sparse-keyframe, avatar-compatible, or parametric output-surface
  direction.
- `Requires another candidate`: Not required for documentation; future implementation would
  likely depend on a working keypoint-level model or representation pipeline.
- `Independent ablation possible`: Not in the current audit; possible only if a future
  implementation instantiates sparse-keyframe or avatar-control mechanisms.
- `Likely interactions with other candidate families`: Strong interaction with latent generative
  production, structure-aware modeling, learned motion representations, semantic alignment, and
  evaluation methodology if pursued later.

## Complexity / Risk Estimate

- `Implementation complexity`: High for future implementation.
- `Failure cost`: High if attempted prematurely, because infrastructure needs may distract from
  the core keypoint-level thesis artifact.
- `Hidden dependency risk`: High; depends on 3D/parametric fitting, avatar representation,
  rendering, non-manual modeling, and evaluation infrastructure.
- `Scope-creep risk`: High.
- `Evaluation risk`: High because avatar realism and sign intelligibility may diverge.

## Downstream Scoring Notes

- `Scorecard readiness`: Not ready as a near-term model candidate; suitable for future-work
  tracking.
- `Open questions for scorecard`: Whether the project will later support 3D/parametric artifacts,
  what avatar representation would be used, how non-manuals would be modeled, and how avatar
  outputs would be evaluated.
- `Evidence gaps to resolve before selection decision`: Implementation feasibility, artifact
  availability, evaluation protocol, and whether this remains future work rather than current
  thesis scope.

## Record Boundary Note

- This document records candidate-level evidence only.
- It does not assign a score.
- It does not select or reject the candidate.
- [Scorecards](../scorecards/index.md), [Selection Decisions](../selection-decisions/index.md),
  and [Audit Result](../audit-result.md) are maintained in their dedicated downstream audit surfaces.
