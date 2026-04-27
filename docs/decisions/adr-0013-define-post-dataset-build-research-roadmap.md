# ADR-0013: Define the Post-Dataset-Build Research Roadmap

- Status: Superseded
- Date: 2026-04-17

Status note: this ADR is retained as historical context only. ADR-0023 is the current governing
decision for phase-based research governance and public documentation boundaries. This ADR must not
be used to reintroduce C1/C2 selected-pair framing, sprint-era roadmap sequencing as current
planning, M0/M1/M2/M3 final model-matrix framing, or released/loadable M0 claims without documented
release evidence.

## Context

Dataset Build is now the implemented project stage. Packaging hardening is complete, Filter V2 is
active, split sample archives are manifest-driven, and full `train / val / test` validation has been
recorded. The repository therefore needs a clear research roadmap for the thesis stages that follow
the processed dataset layer.

The next major stage is Baseline Modeling, but the baseline should not be mistaken for the main
thesis contribution. The literature also points to several possible directions: direct continuous
pose regression, symbolic/intermediate representations, discrete/data-driven motion units,
structure-aware multi-channel modeling, retrieval/stitching, diffusion, and evaluation work. Without
an explicit roadmap, future work could drift into a high-complexity model family before the project
has a reproducible baseline and stable evaluation stack.

## Decision

Freeze the post-Dataset-Build research roadmap as Sprint 3 through Sprint 8:

- Sprint 3 is Baseline Modeling. It is baseline-only and establishes the first reproducible English
  text-to-pose baseline on the processed dataset.
- Sprint 4 is Evaluation & Error Analysis. It establishes the evaluation stack before strong
  thesis-contribution claims are made.
- Sprint 5 is Thesis Contribution I: discrete/data-driven pose representation.
- Sprint 6 is Thesis Contribution II: structure-aware / multi-channel improvement.
- Sprint 7 is Inference / Playback / Minimal Demo for inspectable downstream capability.
- Sprint 8 is Thesis Packaging / Final Integration for final comparison, reproducibility, and
  thesis-facing packaging.

The project will not treat Sprint 3 as the main research-contribution sprint. The main planned
contribution path is Sprint 5 followed by Sprint 6. Diffusion and retrieval/stitching remain possible
later alternatives or future extensions, but they are not the chosen primary roadmap path now.

Sprint 3 is baseline-only because the repository needs a reproducible reference model before more
ambitious claims are meaningful. A weak baseline is still useful if its data, commands,
configuration, and outputs are traceable.

Sprint 4 is evaluation-first because pose-based sign-language production metrics are still
difficult to interpret. Stabilizing metrics, qualitative inspection, and error analysis before the
main contribution sprints reduces the risk of overclaiming.

Sprint 5 focuses on discrete/data-driven pose representation because the literature suggests that
continuous frame-level regression can under-articulate motion and collapse toward averaged poses.
Learning reusable pose units, motion primitives, or token-like representations is a plausible first
main contribution that fits the processed dataset.

Sprint 6 focuses on structure-aware / multi-channel improvement because sign production depends on
coordinated body, hand, and non-manual channels. After the baseline and representation work exist,
channel-aware or skeleton-aware ablations can test whether structural assumptions produce justified
improvements.

Demo and final packaging are deferred because inference/playback and thesis packaging should expose
and integrate research evidence, not replace the research contribution itself.

This ordering is better than immediately jumping to diffusion, retrieval, or another ambitious
generative direction because it keeps the project reproducible, evaluable, and contribution-focused
before taking on high-complexity alternatives.

## Consequences

- Future modeling work should start with a reproducible baseline, not a thesis-contribution claim.
- Evaluation work must precede the main contribution sprints.
- Sprint 5 and Sprint 6 are the primary planned contribution path.
- Diffusion and retrieval/stitching may be discussed as alternatives, later extensions, or future
  work, but should not be presented as the primary Sprint 3 through Sprint 6 path.
- Experiment records should support baseline runs, evaluation comparisons, discrete representation
  experiments, and structure-aware ablations.

## Alternatives Considered

- Pursue the main thesis contribution directly in Sprint 3. Rejected because this would blur
  baseline evidence with contribution evidence and leave no stable reference point for later
  comparison.
- Jump straight to diffusion or other high-complexity modeling. Rejected because that would add
  complexity before the project has baseline behavior, evaluation assumptions, and error analysis
  under control.
- Promote retrieval/stitching to the main roadmap path immediately. Rejected because it would
  redirect the thesis away from the frozen discrete representation and structure-aware contribution
  sequence.
- Keep the roadmap vague. Rejected because a vague future-work list would make it too easy to
  reorder evaluation, merge contribution sprints, or promote an alternative model family without
  recording the tradeoff.
- Freeze explicit sprint goals. Chosen because it preserves the baseline-first and evaluation-first
  order, identifies the two main contribution sprints, and leaves diffusion and retrieval/stitching
  available as later alternatives without making them the current primary path.
