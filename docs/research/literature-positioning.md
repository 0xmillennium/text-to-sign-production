# Literature Positioning

This page explains how selected literature shaped the post-Dataset-Build roadmap. It is a
decision-support page, not a general related-work chapter.

## Problem Framing

The project targets English text-to-sign-pose generation on top of the processed Dataset Build
manifests. The output is not ordinary text: it is a time-varying pose sequence with body and hand
channels, optional face information, variable sequence length, signer motion, timing, and linguistic
content all coupled together.

This is difficult because sign languages are visual, spatial, multi-channel languages. A model can
look numerically close to a reference pose while still being under-articulated, poorly timed, or hard
to interpret. That makes the roadmap order important: first create a reproducible baseline, then
stabilize evaluation, then make contribution claims around representation and structure.

## Method-Family Overview

The literature points to several viable families:

1. symbolic / intermediate pipelines: text to gloss, notation, or another symbolic form before
   animation
2. direct text to continuous pose generation
3. structure-aware / skeleton-aware / channel-aware generation
4. discrete / data-driven pose representation
5. retrieval / stitching-based approaches
6. diffusion-based approaches
7. evaluation literature for pose-based sign-language production

These families influence the roadmap differently. Some define near-term work; others remain useful
alternatives or future extensions.

## Symbolic / Intermediate Pipelines

- Representative works: [HamNoSys - Representing Sign Language Data in Language Resources and
  Language Processing Contexts](https://www.sign-lang.uni-hamburg.de/lrec/pub/04001.html),
  [Changing the Representation: Examining Language Representation for Neural Sign Language
  Production](https://aclanthology.org/2022.sltat-1.18/), and [Text2Sign: Towards Sign Language
  Production Using Neural Machine Translation and Generative Adversarial
  Networks](https://link.springer.com/article/10.1007/s11263-019-01281-2).
- What this family gets right: intermediate symbolic forms can encode linguistic structure, reduce
  some ambiguity, and make generation more inspectable than a single opaque text-to-pose mapping.
- Why it does or does not define our primary roadmap choice: it does not define the primary path
  because this repository's current auditable asset is a processed text-plus-pose dataset, not a
  complete gloss or notation corpus for the target English-to-pose setup.
- How it influenced our roadmap, if relevant: it motivates explicit representation work in Sprint 5
  and supports keeping generated pose sequences inspectable, but it does not replace the baseline or
  evaluation-first ordering.

## Direct Text To Continuous Pose Generation

- Representative works: [Progressive Transformers for End-to-End Sign Language
  Production](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1430_ECCV_2020_paper.php)
  and [Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and
  Mixture Density Networks](https://link.springer.com/article/10.1007/s11263-021-01457-9).
- What this family gets right: it provides a clear baseline path from text to pose without requiring
  a manually supplied symbolic bottleneck.
- Why it does or does not define our primary roadmap choice: direct continuous regression is a valid
  Sprint 3 baseline direction, but it is not the strongest final thesis-contribution target because
  continuous regression can smooth motion, struggle with variable timing, and under-articulate
  high-frequency hand detail.
- How it influenced our roadmap, if relevant: it directly motivates Sprint 3 as a baseline-only
  sprint and gives later representation and structure-aware work a measurable comparison point.

## Structure-Aware / Skeleton-Aware / Channel-Aware Generation

- Representative works: [Continuous 3D Multi-Channel Sign Language Production via Progressive
  Transformers and Mixture Density Networks](https://link.springer.com/article/10.1007/s11263-021-01457-9).
- What this family gets right: it treats sign production as multi-channel motion rather than a
  single undifferentiated vector, aligning with the importance of body, hands, and non-manual
  features.
- Why it does or does not define our primary roadmap choice: it defines the second main contribution
  direction, not the first one, because structure-aware improvements should be evaluated after a
  baseline and a representation-focused contribution exist.
- How it influenced our roadmap, if relevant: it motivates Sprint 6, where body/hand/channel-aware
  modeling and ablations can test whether structure improves over the baseline and Sprint 5
  representation work.

## Discrete / Data-Driven Pose Representation

- Representative works: [Mixed SIGNals: Sign Language Production via a Mixture of Motion
  Primitives](https://openaccess.thecvf.com/content/ICCV2021/html/Saunders_Mixed_SIGNals_Sign_Language_Production_via_a_Mixture_of_Motion_ICCV_2021_paper.html)
  and [A Data-Driven Representation for Sign Language
  Production](https://openresearch.surrey.ac.uk/esploro/outputs/conferenceProceeding/A-Data-Driven-Representation-for-Sign-Language/99874166102346).
- What this family gets right: it tries to avoid treating every frame as an unconstrained continuous
  regression target by learning reusable motion units, pose tokens, or data-driven primitives.
- Why it does or does not define our primary roadmap choice: it is the strongest candidate for the
  first main thesis contribution because it targets a core weakness of direct continuous pose
  regression while remaining grounded in the processed pose dataset.
- How it influenced our roadmap, if relevant: it defines Sprint 5 as the first main contribution
  sprint focused on discrete/data-driven pose representation.

## Retrieval / Stitching-Based Approaches

- Representative works: [Text2Sign](https://link.springer.com/article/10.1007/s11263-019-01281-2),
  [Sign Stitching: A Novel Approach to Sign Language
  Production](https://bmvc2024.org/proceedings/721/), and the retrieval-oriented result context in
  [SLRTP2025 Sign Language Production Challenge: Methodology, Results and Future
  Work](https://cvpr.thecvf.com/virtual/2025/35711).
- What this family gets right: retrieval and stitching can preserve realistic motion from observed
  examples and can reduce the regression-to-the-mean behavior of pure continuous generation.
- Why it does or does not define our primary roadmap choice: it is not the chosen primary path for
  Sprint 3 through Sprint 6 because it shifts the main contribution toward retrieval policy,
  segmentation, and transition engineering rather than the planned representation and
  structure-aware modeling path.
- How it influenced our roadmap, if relevant: it remains a meaningful later alternative or future
  extension, and it reinforces the need to inspect generated motion quality in Sprint 7.

## Diffusion-Based Approaches

- Representative works: [G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete
  Diffusion Model](https://slpdiffusier.github.io/g2p-ddm/) and [SignDiff: Diffusion Model for
  American Sign Language Production](https://signdiff.github.io/).
- What this family gets right: diffusion methods are promising for complex, multimodal sequence
  generation and can model distributional variation better than simple deterministic regression.
- Why it does or does not define our primary roadmap choice: diffusion is not the chosen primary
  roadmap path at this stage because it is a high-complexity generative direction that should not
  precede a reproducible baseline, stable evaluation, and the planned representation/structure
  contributions.
- How it influenced our roadmap, if relevant: it remains a later optional extension or alternative
  after Sprint 5 and Sprint 6 clarify the value of discrete/data-driven and structure-aware changes.

## Evaluation Literature For Pose-Based Sign-Language Production

- Representative works: back-translation evaluation in Progressive Transformers, [Meaningful
  Pose-Based Sign Language Evaluation](https://aclanthology.org/2025.wmt-1.4/), and [SLRTP2025 Sign
  Language Production Challenge](https://cvpr.thecvf.com/virtual/2025/35711).
- What this family gets right: it recognizes that pose distance, back translation, qualitative
  inspection, human judgment, and standardized challenge settings each expose different parts of
  production quality.
- Why it does or does not define our primary roadmap choice: evaluation is not a model-family
  contribution, but it must define the order of work. Without a stable evaluation stack, later model
  improvements would be easy to overclaim.
- How it influenced our roadmap, if relevant: it defines Sprint 4 as evaluation-first, immediately
  after the Sprint 3 baseline and before the Sprint 5 and Sprint 6 contribution sprints.

## Our Research-Direction Choice

The roadmap follows this literature-informed interpretation:

- Baseline first: direct continuous text-to-pose generation is a valid and necessary Sprint 3
  baseline, but Sprint 3 is not the main thesis-contribution sprint.
- Evaluation first: pose-based sign-language production evaluation remains a weak and developing
  area, so Sprint 4 stabilizes evaluation and error analysis before strong contribution claims.
- First main contribution: discrete/data-driven pose representation is the Sprint 5 contribution
  direction because it targets the limitations of direct continuous regression while using the
  processed pose data directly.
- Second main contribution: structure-aware / multi-channel improvement is the Sprint 6 contribution
  direction because body, hands, and optional face channels should be modeled and ablated explicitly.
- Later alternatives: diffusion and retrieval/stitching remain meaningful research directions, but
  they are not the primary chosen roadmap path for Sprint 3 through Sprint 6.

## Limits / Notes

- This page is a positioning and rationale page, not a full related-work chapter.
- It documents how representative literature influenced roadmap decisions; it does not make
  experimental claims about models that have not yet been implemented in this repository.
- The cited works are selective anchors for decision-making, not an exhaustive survey of
  sign-language production.
