# Research

## Purpose

This section contains the research documentation foundation for the
text-to-sign-production project. It collects the source-selection and source-corpus
documentation, contribution-audit records, audit-result synthesis, literature positioning,
roadmap, and experiment documentation where applicable.

The research section is source-grounded and audit-derived. Higher-level planning claims
should remain traceable to the source corpus and contribution-audit records rather than to
unsupported implementation preference.

## Current Research Architecture

The current research documentation chain is:

```text
source-selection-criteria.md
  -> source-corpus.md
    -> contribution-audit/
      -> audit-result.md
        -> literature-positioning.md
          -> roadmap.md
```

Lower-level records define the evidence and role boundaries. Higher-level synthesis may
summarize and operationalize those records, but it must not change source roles, candidate
roles, confidence, supervision assumptions, or scope boundaries.

## Key Research Surfaces

| Surface | Role |
| --- | --- |
| [Source Selection Criteria](source-selection-criteria.md) | Defines which sources may support the research audit and how evidence boundaries are handled. |
| [Source Corpus](source-corpus.md) | Canonical public source registry using stable `SRC-*` records. |
| [Contribution Audit](contribution-audit/index.md) | Candidate-universe, candidate-card, scorecard, and selection-decision documentation. |
| [Contribution Audit Result](contribution-audit/audit-result.md) | Whole-audit synthesis for the current research-planning state. |
| [Literature Positioning](literature-positioning.md) | Explains where the contribution frame sits relative to the literature. |
| [Roadmap](roadmap.md) | Audit-derived execution map for dataset, artifact, baseline, evaluation, model, comparator, visualization, and thesis-writing phases. |
| [Experiment Records](../experiments/index.md) | Formal experiment documentation for empirical work and artifact-producing runs. |

## Current Thesis Contribution Frame

The current thesis contribution frame is a reproducible, gloss-free,
How2Sign-compatible text-to-sign pose-production research pipeline that compares
representation-based, latent-generative, and structure-aware text-to-pose candidates under
a shared baseline, evaluation, comparator, visualization, reporting, and reproducibility
surface.

This is not a final implementation-model selection, not an experimental result, and not a
final ranking.

Current roles are:

- M0 Direct Text-to-Pose Baseline as comparison floor.
- Learned Pose-Token Bottleneck as primary candidate.
- Gloss-Free Latent Diffusion as primary candidate.
- Articulator-Disentangled Latent Modeling as primary candidate.
- Text-Pose Semantic Consistency as auxiliary/additive mechanism.
- Retrieval-Augmented Pose Comparator as comparator.
- How2Sign-Compatible Evaluation Protocol as evaluation surface.
- Sparse-Keyframe / Avatar-Control Future Direction as future-work boundary.

## Navigation Guide

Use [Source Selection Criteria](source-selection-criteria.md) and
[Source Corpus](source-corpus.md) to verify evidence and source boundaries.

Use [Contribution Audit](contribution-audit/index.md) to inspect candidate-family and
candidate-level reasoning, including scorecards and selection decisions.

Use [Contribution Audit Result](contribution-audit/audit-result.md) for the authoritative
whole-audit synthesis.

Use [Literature Positioning](literature-positioning.md) for the literature narrative,
research gap, and thesis framing.

Use [Roadmap](roadmap.md) for implementation planning derived from the audit result and
literature positioning.

Use [Experiment Records](../experiments/index.md) for formal records of empirical work
when available.

## Non-Goals And Boundaries

This overview does not claim:

- final model selection,
- final candidate ranking,
- experimental results,
- full sign-language translation,
- human-grade signing,
- public manual gloss supervision,
- near-term full 3D/avatar/rendering implementation,
- legacy C1/C2 thesis outcomes.

This page is not a candidate-selection page, scorecard, selection-decision surface,
audit-result replacement, roadmap replacement, or thesis chapter.

## Maintenance Notes

Update this page only when the source-corpus architecture, contribution-audit architecture,
audit result, literature positioning, roadmap, or research navigation changes.

This overview must not reintroduce stale legacy outcome framing.
