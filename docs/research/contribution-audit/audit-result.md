# Contribution Audit Result

## Purpose

This document records the whole-audit synthesis produced from the refreshed source corpus,
candidate universe, candidate cards, scorecards, and selection decisions.

It is the authoritative audit-result surface for the current research-planning state. It does not
claim experimental implementation results, does not select a final implementation model, and
controls downstream literature-positioning and roadmap refreshes.

## Audit Scope

The current audit is scoped to:

- gloss-free / no-public-gloss text-to-sign pose production,
- How2Sign-compatible supervision and artifact planning,
- pose/keypoint-level output as the near-term thesis artifact,
- baseline, model candidate, evaluation, auxiliary, comparator, and future-work boundary roles.

The current audit result is scoped to keypoint/pose-level thesis planning. It does not authorize
immediate 3D avatar or rendering implementation.

## Audit Inputs

This audit result is controlled by:

- [Source Selection Criteria](../source-selection-criteria.md)
- [Source Corpus](../source-corpus.md)
- [Candidate Universe](candidate-universe/index.md)
- [Candidate Cards](candidate-cards/index.md)
- [Scorecards](scorecards/index.md)
- [Selection Decisions](selection-decisions/index.md)

This surface is synthesized from those audit inputs rather than from an isolated manual choice.

## Whole-Audit Result Summary

| Role | Candidate / Artifact | Audit-result treatment |
| --- | --- | --- |
| Baseline / ablation floor | [Direct Text-to-Pose Baseline](candidate-cards/direct-text-to-pose-baseline.md) | Retained as M0 comparison floor. |
| Primary model candidate | [Learned Pose-Token Bottleneck](candidate-cards/learned-pose-token-bottleneck.md) | Advanced for audit-result consideration. |
| Primary model candidate | [Gloss-Free Latent Diffusion](candidate-cards/gloss-free-latent-diffusion.md) | Advanced for audit-result consideration. |
| Primary model candidate | [Articulator-Disentangled Latent Modeling](candidate-cards/articulator-disentangled-latent.md) | Advanced for audit-result consideration. |
| Auxiliary / additive mechanism | [Text-Pose Semantic Consistency Objective](candidate-cards/text-pose-semantic-consistency.md) | Retained as auxiliary/additive objective. |
| Counter-alternative comparator | [Retrieval-Augmented Pose Comparator](candidate-cards/retrieval-augmented-pose-comparator.md) | Retained as comparator and sanity-check alternative. |
| Evaluation support surface | [How2Sign-Compatible Evaluation Protocol](candidate-cards/how2sign-evaluation-protocol.md) | Retained as evaluation constraint surface. |
| Future-work boundary | [Sparse-Keyframe / Avatar-Control Future Direction](candidate-cards/sparse-keyframe-avatar-future.md) | Retained as future-work and output-surface boundary. |

"Advanced for audit-result consideration" does not mean final implementation selection. It means
the candidate remains in the primary model contribution space for downstream synthesis, roadmap
planning, and later experimental evaluation.

## Advanced Primary Model Candidates

### Learned Pose-Token Bottleneck

Learned Pose-Token Bottleneck is advanced as a leading near-term primary model direction. It should
be treated as the cleanest representation-focused candidate, provided tokenizer reconstruction
quality, codebook stability, and evaluation separation are preserved.

Its direct comparison against the M0 baseline is comparatively well isolated. Downstream planning
must keep representation quality distinct from sign adequacy rather than treating tokenizer
reconstruction as sufficient evidence by itself.

### Gloss-Free Latent Diffusion

Gloss-Free Latent Diffusion is advanced as a high-support, high-risk primary model direction. It
should remain in the thesis contribution space, but roadmap planning must not treat its evidence
strength as low implementation risk.

The candidate has strong scientific and source-corpus support for gloss-free latent generation, but
its latent setup, stochastic training, denoising behavior, compute demand, and evaluation
sensitivity make it materially riskier than the learned-token route.

### Articulator-Disentangled Latent Modeling

Articulator-Disentangled Latent Modeling is advanced as a viable structure-preservation candidate.
It should remain in the candidate set, but downstream roadmap priority must account for schema,
mask, loss-weighting, and channel-aware evaluation dependencies.

The candidate targets body, hand, face, channel, and articulator structure. It remains more
conditional than the learned-token and latent-diffusion candidates in workflow and evaluation fit
because artifact schema, masks, channel partitions, loss balancing, and channel-aware evaluation
must be fixed before its claims can be interpreted.

## Retained Baseline And Evaluation Surfaces

### Direct Text-to-Pose Baseline

Direct Text-to-Pose Baseline is retained as the M0 comparison floor. It is not a model contribution
claim; it is the baseline surface required to interpret later candidate gains.

Baseline readiness does not imply strong task-solving quality. It defines the comparison floor that
primary model candidates, auxiliary mechanisms, and comparator surfaces must be interpreted
against.

### How2Sign-Compatible Evaluation Protocol

How2Sign-Compatible Evaluation Protocol is retained as an evaluation-support surface. It is not a
model candidate and must not compete with model candidates.

This protocol constrains how primary candidates, auxiliary mechanisms, and comparators are
evaluated. It should include pose/keypoint metrics, embedding checks, cautious recognition or
back-translation checks where available, qualitative inspection, and explicit metric-limitation
documentation.

## Retained Auxiliary / Additive Mechanism

### Text-Pose Semantic Consistency Objective

Text-Pose Semantic Consistency Objective is retained as an auxiliary or additive mechanism. It
should not define the main thesis model direction by itself unless a later ablation explicitly
justifies a standalone role.

The semantic-alignment evidence is important, but it is not cleanly isolated as a standalone
primary model candidate in the current audit. It may strengthen another candidate only if a later
ablation defines where it attaches and how its effect is evaluated.

## Retained Counter-Alternative Comparator

### Retrieval-Augmented Pose Comparator

Retrieval-Augmented Pose Comparator is retained as a no-public-gloss comparator and sanity-check
comparator. It should test whether learned or generative candidates outperform retrieval/reuse, but
it should not be treated as a primary neural generation candidate.

This comparator must preserve leakage-safe evaluation and semantic-adequacy caveats. Retrieved
motion quality or example reuse must not be treated as semantic correctness without split-safe
comparison and qualitative review.

## Frontier / Future-Work Boundary

### Sparse-Keyframe / Avatar-Control Future Direction

Sparse-Keyframe / Avatar-Control Future Direction is retained as a future-work and output-surface
boundary. It should inform literature positioning and future visualization strategy, but it does
not authorize near-term 3D/avatar implementation.

This boundary preserves awareness of sparse-keyframe, avatar-compatible, 3D/parametric, and
rendering-oriented directions while keeping the current thesis planning surface focused on
keypoint/pose-level artifacts.

## Resulting Thesis Contribution Frame

The resulting thesis contribution frame is a reproducible, gloss-free, How2Sign-compatible
text-to-sign pose-production research pipeline that compares multiple primary model directions
under a shared baseline, evaluation, and reporting surface.

The contribution is not a single isolated architecture claim. It is the controlled construction and
comparison of representation-based, latent-generative, and structure-aware text-to-pose candidates,
supported by a retained baseline, auxiliary semantic-alignment mechanism, retrieval comparator,
and explicit future avatar boundary.

This frame should drive both the literature-positioning refresh and the roadmap refresh.
Literature positioning should explain where the contribution frame sits relative to existing work.
The roadmap should translate the frame into dataset, pipeline, baseline, evaluation, model,
comparator, visualization, and thesis-writing workstreams.

## Claims Not Made By This Audit Result

This audit result does not claim that the project has:

- selected a final implementation model,
- produced experimental results,
- solved full sign-language translation,
- produced linguistically validated human-grade signing,
- validated semantic adequacy through automatic metrics alone,
- obtained or relied on public manual gloss supervision,
- committed to a near-term 3D/avatar/rendering pipeline,
- ranked candidates as final thesis outcomes.

## Constraints To Preserve Downstream

- Preserve no-public-gloss assumptions unless a later documented scope decision changes them.
- Keep the M0 baseline separate from model contribution claims.
- Compare primary model candidates under a shared artifact schema and split policy.
- Keep semantic consistency auxiliary unless standalone ablation is later justified.
- Keep retrieval as a comparator unless a separate retrieval-augmented model candidate is introduced.
- Keep avatar/3D work as future boundary unless a later scope decision authorizes implementation.
- Do not treat automatic metrics as sufficient evidence of sign intelligibility.
- Preserve metric-limitation and qualitative-inspection requirements in later evaluation planning.

## Downstream Use

This audit result should be used as the controlling synthesis for the literature-positioning and
roadmap refresh. Literature positioning should explain where this contribution frame sits relative
to existing work. The roadmap should translate this contribution frame into dataset, pipeline,
baseline, evaluation, model, comparator, visualization, and thesis-writing workstreams.

`literature-positioning.md` should not re-run candidate scoring. `roadmap.md` should not invent
sprint priorities independent of this audit result. Downstream files should preserve role
boundaries.

## Maintenance Triggers

Update this file only when one of the following changes:

- the source corpus changes materially,
- candidate-universe families change,
- candidate cards change,
- scorecard totals or interpretations change,
- selection-decision statuses change,
- implementation results introduce new evidence,
- dataset access or supervision regime changes,
- thesis scope changes.
