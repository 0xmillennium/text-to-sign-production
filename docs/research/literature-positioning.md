# Literature Positioning

## Purpose

This document positions the thesis contribution frame against the refreshed
[Source Corpus](source-corpus.md) and
[Contribution Audit Result](contribution-audit/audit-result.md).

It is not the canonical source registry. It is not a scorecard. It is not a
selection-decision surface. It is not a roadmap. It does not select a final
model. It explains how the thesis contribution frame sits relative to the
literature.

## Controlling Audit Basis

This positioning is controlled by:

- [Source Selection Criteria](source-selection-criteria.md)
- [Source Corpus](source-corpus.md)
- [Candidate Universe](contribution-audit/candidate-universe/index.md)
- [Candidate Cards](contribution-audit/candidate-cards/index.md)
- [Scorecards](contribution-audit/scorecards/index.md)
- [Selection Decisions](contribution-audit/selection-decisions/index.md)
- [Contribution Audit Result](contribution-audit/audit-result.md)

The literature positioning is controlled by the audit result; it does not re-run candidate scoring or alter selection-decision statuses.

Source-level authority remains in [Source Corpus](source-corpus.md). This file
should not duplicate the 28-source registry, introduce independent
bibliography-style source claims, or replace the source-corpus classification
fields. Literature claims in this file should be interpreted through the
source-corpus and audit-result surfaces.

## Problem Frame

The project addresses gloss-free text-to-sign pose production under a How2Sign-compatible supervision regime, with pose/keypoint-level output as the near-term thesis artifact.

The research problem is not only how to generate motion from text, but how to do
so under a reproducible research pipeline that preserves no-public-gloss
supervision assumptions, shared artifacts, shared evaluation, and comparable
reporting. The project is positioned around English text or transcript input,
sign video, extracted pose/keypoint supervision, and How2Sign-compatible
artifact planning.

The project does not position itself as full sign-language translation,
human-validated signing generation, or immediate 3D/avatar production.

## Positioning Summary

The thesis is positioned as a controlled, reproducible comparison framework for
gloss-free text-to-pose production rather than as a single architecture claim.
It uses the literature to define baseline, representation, latent-generation,
structure-aware, semantic-alignment, retrieval-comparator, evaluation, and
future-avatar boundaries under one shared experimental surface.

The gap addressed by this thesis is not merely the absence of another
text-to-sign model. The gap is the absence of a controlled, reproducible,
no-public-gloss comparison surface that evaluates multiple text-to-pose
production directions under shared artifacts, a shared baseline, shared
evaluation constraints, comparator pressure, visualization boundaries, and
explicit metric limitations.

This positioning does not name a final selected model and does not rank
candidate families as final thesis outcomes.

## Position Relative To Dataset, Supervision, And Gloss / Notation Boundaries

How2Sign-compatible positioning is therefore a supervision and artifact-compatibility claim, not a blanket claim that every sign-language-production method is directly reusable.

The project treats How2Sign-style text/video/pose artifacts as the near-term
supervision anchor. Gloss-free and gloss-optional methods are most directly
compatible with that regime. Gloss-dependent, notation-dependent,
dictionary-based, isolated-sign, or gloss-pose-alignment methods may provide
boundary or transferable evidence, but they do not become direct support without
a documented adaptation path.

Output pose/keypoint compatibility alone is not sufficient if the method depends
on unavailable gloss, HamNoSys, notation, dictionary, or isolated-sign
supervision. This boundary is what allows the project to compare candidate model
directions without silently changing the supervision regime.

This positioning is represented by the
[Dataset and Supervision Boundary](contribution-audit/candidate-universe/dataset-supervision-boundary.md),
the
[Gloss/Notation-Dependent Pose Generation Boundary](contribution-audit/candidate-universe/gloss-notation-dependent-boundary.md),
and the source-level classifications in the [Source Corpus](source-corpus.md).

## Position Relative To Foundational Text-to-Pose Baselines

Foundational text-to-pose baselines define the comparison floor for the project.
They establish why a direct text/transcript-to-pose route is needed as a minimum
reference before stronger representation-based, latent-generative, or
structure-aware candidates can be interpreted.

The project treats direct text-to-pose as a necessary M0 baseline and ablation floor, not as the primary thesis contribution.

This positioning is represented by the
[Direct Text-to-Pose Baseline](contribution-audit/candidate-cards/direct-text-to-pose-baseline.md)
candidate and the
[Foundational Text-to-Pose Baselines](contribution-audit/candidate-universe/foundational-text-to-pose-baselines.md)
family. Baseline readiness is therefore used to support comparison discipline,
not to make a model-contribution claim.

## Position Relative To Learned Pose / Motion Representation Work

Learned pose-token and motion-representation literature defines the cleanest
near-term representation-focused primary candidate family. It motivates a
text-to-token-to-pose route in which pose/keypoint-derived tokens, compact motion
codes, or learned non-gloss intermediate representations can mediate between
text input and generated pose output.

Learned pose/motion representation work defines the cleanest near-term representation-focused route, but tokenization alone must not be treated as proof of semantic adequacy.

This positioning is represented by the
[Learned Pose-Token Bottleneck](contribution-audit/candidate-cards/learned-pose-token-bottleneck.md)
candidate and the
[Learned Pose/Motion Representations](contribution-audit/candidate-universe/learned-pose-motion-representations.md)
family. The route remains a primary model direction for comparison, not a final
selection.

## Position Relative To Latent Generative Production Work

Latent diffusion and related generative production work defines the strongest
high-potential generative route in the refreshed audit. It supports treating
text-conditioned latent pose or motion generation as a serious primary candidate
family, especially where generation quality, sequence-level coherence, and
motion diversity are central concerns.

Latent generative production provides a high-potential generative direction, but high method support must not be read as low implementation or evaluation risk.

This positioning is represented by the
[Gloss-Free Latent Diffusion](contribution-audit/candidate-cards/gloss-free-latent-diffusion.md)
candidate and the
[Latent Generative Production](contribution-audit/candidate-universe/latent-generative-production.md)
family. The project treats this direction as scientifically important and
risk-heavy, not as a low-risk default implementation choice.

## Position Relative To Structure-Aware Articulator Modeling

Structure-aware and articulator-aware work motivates explicit preservation of
sign-relevant body, hand, face, channel, and articulator structure. It keeps the
project from treating sign pose as an undifferentiated coordinate vector and
supports a primary candidate route focused on structure preservation.

Structure-aware and articulator-aware work motivates explicit body, hand, face, and channel preservation, but its project fit depends on artifact schema, masks, channel partitions, loss balancing, and channel-aware evaluation.

This positioning is represented by the
[Articulator-Disentangled Latent Modeling](contribution-audit/candidate-cards/articulator-disentangled-latent.md)
candidate and the
[Structure-Aware and Articulator-Aware Modeling](contribution-audit/candidate-universe/structure-aware-articulator-modeling.md)
family. The route is viable, but its claims require stable channel definitions
and evaluation that can observe channel-specific failures.

## Position Relative To Text-Pose Semantic Alignment Work

Semantic alignment is important because generated sign motion must remain
coupled to the source text rather than merely looking plausible as motion.
However, the current audit retains semantic alignment as an auxiliary or
additive mechanism rather than as a standalone primary model direction.

Semantic alignment is retained as an auxiliary/additive mechanism rather than as a standalone primary model direction.

This positioning is represented by the
[Text-Pose Semantic Consistency Objective](contribution-audit/candidate-cards/text-pose-semantic-consistency.md)
candidate and the
[Text-Pose Semantic Alignment and Conditioning](contribution-audit/candidate-universe/text-pose-semantic-alignment.md)
family. It may strengthen another candidate only if later ablation defines the
attachment point and evaluation method.

## Position Relative To Retrieval / Stitching Alternatives

Retrieval, stitching, and motion-primitives literature is retained as
counter-alternative pressure. It tests whether learned or generative production
directions actually improve over reuse, nearest-neighbor, or assembly-based
alternatives under leakage-safe evaluation.

Retrieval and stitching alternatives are retained as comparator pressure, not as the primary thesis model direction.

This positioning is represented by the
[Retrieval-Augmented Pose Comparator](contribution-audit/candidate-cards/retrieval-augmented-pose-comparator.md)
candidate and the
[Retrieval, Stitching, and Motion-Primitives Alternatives](contribution-audit/candidate-universe/retrieval-stitching-primitives.md)
family. Retrieval realism must not be treated as semantic correctness, and
no-public-gloss adaptation plus leakage-safe evaluation remain required.

## Position Relative To Evaluation Methodology

The project positions evaluation as a first-class contribution constraint, not as a post-hoc metric table. Evaluation defines whether baseline, primary model,
auxiliary, comparator, and future-boundary claims can be interpreted under a
shared surface.

The evaluation surface should include pose/keypoint metrics, embedding checks,
cautious recognition or back-translation checks where available, qualitative
inspection, and metric-limitation documentation. Automatic metrics are not sufficient proof of sign intelligibility.

This positioning is represented by the
[How2Sign-Compatible Evaluation Protocol](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
candidate and the
[Evaluation and Benchmark Methodology](contribution-audit/candidate-universe/evaluation-benchmark-methodology.md)
family. The evaluation protocol constrains candidate comparison; it is not a
competing model candidate.

## Position Relative To 3D Avatar / Parametric Frontier Work

3D avatar, sparse-keyframe, and parametric frontier work is retained as a future-work and output-surface boundary, not as near-term thesis implementation scope.

This literature clarifies where keypoint-level production may eventually lead:
avatar-compatible motion, sparse keyframes, richer body/hand/face
representations, parametric control, and rendering-oriented output surfaces.
Near-term visualization may use skeleton/keypoint playback and debugging
surfaces. Full avatar, SMPL-X, parametric rendering, or photorealistic signer
production remains future boundary unless later scope changes.

This positioning is represented by the
[Sparse-Keyframe / Avatar-Control Future Direction](contribution-audit/candidate-cards/sparse-keyframe-avatar-future.md)
candidate and the
[3D Avatar and Parametric Motion Frontier](contribution-audit/candidate-universe/threed-avatar-parametric-frontier.md)
family.

## Thesis Contribution Position

The thesis contribution is not "we propose model X."

The thesis contribution is a reproducible, gloss-free, How2Sign-compatible
comparison and production pipeline for text-to-sign pose generation, with
controlled evaluation of representation-based, latent-generative, and
structure-aware model directions.

The contribution includes shared dataset and artifact discipline, an M0
baseline, an evaluation harness, candidate model comparison, auxiliary semantic
alignment as optional/additive, a retrieval comparator, a visualization
boundary, and reproducibility documentation. The project contribution is the
controlled construction and comparison surface that makes these roles
interpretable together.

The literature positioning therefore justifies a roadmap that begins with data/artifact schema, baseline, evaluation harness, and visualization/reporting infrastructure before treating candidate models as comparable experimental units.

## Claims Not Made

This literature positioning does not claim that the project:

- solves full sign-language translation,
- produces linguistically validated human-grade signing,
- has selected a final implementation model,
- has produced experimental results,
- uses public manual gloss supervision,
- treats automatic pose metrics as sufficient proof of intelligibility,
- implements a near-term full 3D/avatar/rendering system,
- ranks the candidate families as final thesis outcomes.

## Downstream Use

The roadmap must translate this positioning into workstreams for dataset access, artifact schema, pose/keypoint processing, M0 baseline, evaluation harness, candidate models, auxiliary semantic objective, retrieval comparator, visualization, comparative analysis, and thesis writing.

The roadmap must not invent priorities independent of the audit result and this
literature positioning. It must preserve the no-public-gloss boundary, metric
limitation requirements, and future-avatar boundary.

For thesis writing, this document should control the literature-review
narrative: source-corpus entries should be discussed through the positioning
axes above, not as an undifferentiated chronological bibliography. The
methodology chapter should inherit the no-public-gloss, shared-artifact, M0
baseline, evaluation, comparator, and future-avatar boundaries defined here. The
limitations and future-work chapters should preserve the automatic-metric,
semantic-adequacy, and 3D/avatar-scope constraints recorded in this positioning.

## Maintenance Triggers

Update this file only when:

- the source corpus changes materially,
- the audit result changes,
- selection-decision statuses change,
- implementation results alter candidate interpretation,
- dataset access or supervision assumptions change,
- thesis scope changes,
- the roadmap introduces a new contribution claim not supported by this positioning.
