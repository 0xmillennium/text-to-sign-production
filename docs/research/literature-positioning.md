# Literature Positioning

This page explains how the verified literature cited in [Bibliography](bibliography.md) shapes the
post-Dataset-Build roadmap and how the completed current contribution audit should be interpreted
at a research-framing level. It is a literature-synthesis and rationale surface, not the detailed
audit ledger itself.

## Problem Framing

The project targets English text-to-sign-pose generation on top of the processed Dataset Build
manifests. The output is not ordinary text: it is a time-varying pose sequence with body and hand
channels, optional face information, variable sequence length, signer motion, timing, and
linguistic content all coupled together.

This is difficult because the locally verified papers consistently describe sign languages as
visual, spatial, and multi-channel, and they repeatedly note that generated pose sequences can
remain under-articulated or unnatural even when they are numerically close to reference motion.
That is why the repository first established a reproducible baseline, then routed
contribution-path selection through the contribution-audit surface, and only after that fixed the
current thesis-facing contribution pair.

## Local-Source Boundary

This page uses only the verified paper corpus cited in [Bibliography](bibliography.md) as
literature support.
Broader symbolic or intermediate-pipeline literature may be relevant to the field, but it falls
outside the current verified paper set for this page and is therefore not used here as evidence.

## Relation To The Contribution Audit

The detailed contribution-path evaluation, scorecards, veto review, and final candidate-level
decisions are recorded through the [Contribution Audit](contribution-audit/index.md). This page
complements that audit rather than replacing it.

At the current framing level, the completed audit outcome is:

- `C1 = Dynamic VQ Pose Tokens`
- `C2 = Channel-Aware Loss Reweighting`
- fallback = `Articulator-Partitioned Latent Structure`
- deferred = `Motion Primitives Representation`

The locally verified literature supports the family-level rationale behind this framing. The
current audit then interprets the selected pair as complementary rather than redundant: the
selected `C1` primarily changes representation, while the selected `C2` is treated as a lighter
structure-aware optimization path. That complementarity remains an audit-level interpretation
rather than a direct paper claim.

## Method-Family Overview

Within the locally verified corpus, the literature supports six relevant framing areas:

1. direct text-to-continuous pose generation
2. discrete / data-driven pose representation
3. structure-aware / multi-channel generation
4. retrieval / stitching-based approaches
5. diffusion-based approaches
6. evaluation signals visible in the local corpus

These areas do not all play the same role. Some define baseline or historical framing, some define
the selected current contribution path, and some remain serious audited alternatives that were not
selected in the current outcome.

## Direct Text To Continuous Pose Generation

- Representative source: [1](bibliography.md#ref-01).
- What the local corpus supports: this line of work provides a clear baseline path from spoken
  language text to continuous multi-channel pose generation without requiring a manually supplied
  symbolic bottleneck.
- How it informs the current audit: the current audit treats direct continuous generation as the
  necessary implemented baseline rather than as the preferred thesis contribution route. That
  interpretation is consistent with local papers that keep visible the difficulties of
  under-articulation, regression-to-the-mean behavior, and transition quality in continuous sign
  production.

## Discrete / Data-Driven Pose Representation

- Representative sources: [2](bibliography.md#ref-02), [3](bibliography.md#ref-03),
  [4](bibliography.md#ref-04).
- What the local corpus supports: discrete or data-driven intermediate representations are a real
  sign language production direction. The local papers support dynamic vector quantization,
  reusable motion units, and motion-primitives-style structure as serious alternatives to direct
  continuous regression.
- How the current audit interprets this family: the current audit treats discrete/data-driven
  representation as the strongest current route for the `C1` role. Within that family, the current
  audit interprets `Dynamic VQ Pose Tokens` as the selected `C1` path because the local literature
  supports a gloss-free dynamic-discrete route directly, while `Motion Primitives Representation`
  remains more conditional in this repository due to its strongest direct source using gloss
  supervision. Relative additivity, adaptation burden, and repository fit remain audit-level
  conclusions rather than direct paper findings.

## Structure-Aware / Multi-Channel Generation

- Representative sources: [1](bibliography.md#ref-01), [5](bibliography.md#ref-05),
  [6](bibliography.md#ref-06).
- What the local corpus supports: sign production should not be treated as a single undifferentiated
  motion target. The local papers directly support multi-channel or articulator-aware modeling as a
  serious direction, and DARSLP directly supports a combined gloss-free method that includes both
  articulator-based disentanglement and channel-aware weighted regularization.
- How the current audit interprets this family: the current audit treats structure-aware /
  multi-channel improvement as the strongest current route for the `C2` role. Within that family,
  the current audit interprets `Channel-Aware Loss Reweighting` as the selected `C2` path because
  it is the lighter repository fit in the current audit record, while keeping visible that the
  exact standalone loss-only form is only partly supported by the local literature. The current
  audit retains `Articulator-Partitioned Latent Structure` as the principal fallback because local
  literature directly supports the richer combined mechanism, even though the current repository
  path treats it as operationally heavier.

## Why The Selected Pair Remains Defensible

The selected current pair is treated by the current audit as a coherent combined path.

- `Dynamic VQ Pose Tokens` primarily changes representation by introducing a discrete/data-driven
  pose-token mechanism.
- `Channel-Aware Loss Reweighting` is treated by the current audit as a lighter optimization-level
  structure-aware intervention that emphasizes channel-sensitive learning pressure.

The locally verified literature supports the relevance of both families. The further claim that the
selected pair is the best current combined route for this repository remains an audit-level
synthesis built on the literature plus the project's additivity, workflow, and implementation
constraints.

## Retrieval / Stitching-Based Approaches

- Representative sources: [7](bibliography.md#ref-07), [4](bibliography.md#ref-04).
- What the local corpus supports: retrieval, example-backed construction, and explicit stitching are
  serious alternatives that address continuity and prosody more explicitly than simple unit
  concatenation.
- How the current audit interprets this family: retrieval/stitching remained a serious audited
  counter-alternative family, but the current contribution audit did not select it as part of the
  final `C1/C2` pair.

## Diffusion-Based Approaches

- Representative source: [8](bibliography.md#ref-08).
- What the local corpus supports: diffusion-based sign production is a real and technically credible
  family-level alternative capable of generating sign sequences directly from spoken text or speech
  audio.
- How the current audit interprets this family: diffusion remained a serious audited
  counter-alternative family, but the current contribution audit did not select it as part of the
  final `C1/C2` pair. Relative repository fit and implementation burden remain audit-level
  judgments rather than direct paper findings.

## Evaluation Signals In The Local Corpus

- The verified corpus shows repeated use of back-translation evaluation, including in
  [1](bibliography.md#ref-01), [3](bibliography.md#ref-03), [4](bibliography.md#ref-04), and
  [7](bibliography.md#ref-07).
- The verified corpus also includes user evaluation as part of how sign quality is interpreted,
  most explicitly in [1](bibliography.md#ref-01), [3](bibliography.md#ref-03), and
  [7](bibliography.md#ref-07).
- Benchmark comparison and reported competitive or state-of-the-art results are also visible in the
  verified corpus, including MCST-Transformer [5](bibliography.md#ref-05), DARSLP
  [6](bibliography.md#ref-06), MS2SL [8](bibliography.md#ref-08), and the discrete/data-driven
  representation papers [2](bibliography.md#ref-02), [4](bibliography.md#ref-04).
- The current audit therefore treats later implementation evidence as needing more than a single
  metric, but that evaluation framing is derived from the local corpus rather than from absent 2025
  evaluation sources.

## Current Research-Direction Framing

The current literature-informed framing, restricted to the local verified corpus, is now:

- the baseline remains necessary but is not the main thesis contribution
- the selected current contribution pair is `Dynamic VQ Pose Tokens` as `C1` and
  `Channel-Aware Loss Reweighting` as `C2`
- locally verified literature supports both the discrete/data-driven `C1` family and the
  structure-aware / multi-channel `C2` family
- the current audit interprets the selected pair as complementary because it combines a
  representation-level contribution with a lighter structure-aware contribution
- `Articulator-Partitioned Latent Structure` remains the main fallback if the selected `C2` route
  later proves unsatisfactory
- `Motion Primitives Representation` remains deferred rather than rejected
- retrieval/stitching and diffusion remain serious alternative families in the broader local-source
  landscape even though they are not part of the current selected pair

## Limits / Notes

- This page is a literature-synthesis and rationale page, not the detailed audit ledger.
- Detailed scorecards, veto reasoning, candidate-level decisions, and the authoritative current
  outcome remain in the [Contribution Audit](contribution-audit/index.md).
- Only the verified paper corpus cited in [Bibliography](bibliography.md) is used as evidence on
  this page.
- Relative repository fit, additivity, operational simplicity, and pair coherence remain current
  audit interpretations rather than direct paper claims.
