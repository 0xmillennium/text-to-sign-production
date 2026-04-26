# Source Corpus

## Purpose

This document is the canonical source registry for the research corpus used by the project. It
records reviewed research sources, official source links, corpus status, evidence role, dataset
compatibility, gloss dependency, and downstream decision relevance.

It replaces the previous plain bibliography role with a structured, heading-addressable source
registry. The registry is intended to support later synthesis without selecting contributions or
preserving any previous contribution decision.

## Relationship to Source Selection Criteria

[Source Selection Criteria](source-selection-criteria.md) defines how sources are selected,
classified, and reassessed. This document records the resulting source corpus under that standard.

This file is upstream of later literature-positioning and contribution-audit updates. It does not
select contributions.

## Scope and Non-Goals

This document records source-level corpus decisions at the registry level.

The source corpus is the canonical public registry for source identity, role, compatibility, and
decision relevance. It is not a full extraction worksheet and does not duplicate every detailed
extraction field from [Source Selection Criteria](source-selection-criteria.md). Detailed
extraction metadata may be recorded in separate per-source detail records or future review
artifacts when the project needs that level of public traceability. Downstream research claims
should continue to cite stable `SRC-*` IDs and audit records.

It does not:

- Provide a full literature review.
- Select contribution candidates.
- Assign candidate scores.
- Define implementation scope.
- Preserve any previous contribution decision.

## Corpus Status Labels

| Status | Meaning |
| --- | --- |
| `include_core` | Decision-bearing source; directly informs research positioning, evaluation design, or contribution audit. |
| `include_core_caution` | Decision-bearing source, but with transfer, preprint, implementation, dataset, or scope caution. |
| `include_boundary` | Important for defining historical, technical, supervision, dataset, or compatibility boundaries. |
| `include_counter_alternative` | Serious alternative method family; included to avoid narrowing the search space prematurely. |
| `include_background` | Useful for taxonomy or coverage checking; not a direct contribution-decision source. |

## Classification Fields

| Field | Purpose |
| --- | --- |
| Source ID | Stable heading-level identifier used by downstream research docs. |
| Source | Paper title with official source link. |
| Year / venue | Publication or reviewed-source year and venue/archive. |
| Primary role | One-line role in the project corpus. |
| Corpus status | Inclusion/status decision. |
| Evidence class | Source role under [Source Selection Criteria](source-selection-criteria.md). |
| Dataset compatibility | Dataset-regime compatibility label. |
| Gloss dependency | Gloss-dependency label. |
| Decision relevance | Why the source matters for future research decisions. |

## Corpus Overview

Current corpus size: 28 sources.

| Corpus status | Count |
| --- | ---: |
| `include_core` | 13 |
| `include_core_caution` | 6 |
| `include_boundary` | 6 |
| `include_counter_alternative` | 2 |
| `include_background` | 1 |
| **Total** | **28** |

The corpus covers dataset anchors, foundational text-to-pose baselines, learned pose/motion
representations, diffusion-based production, structure-aware and articulator-aware modeling,
retrieval/stitching alternatives, pose-based evaluation, recent frontier systems, and
survey/background coverage.

## Source Index

| Source ID | Short name | Year | Corpus status | Primary role |
| --- | --- | ---: | --- | --- |
| [`SRC-HOW2SIGN-2021`](#src-how2sign-2021) | How2Sign | 2021 | `include_core` | Dataset anchor |
| [`SRC-TEXT2SIGN-2020`](#src-text2sign-2020) | Text2Sign | 2020 | `include_boundary` | Historical SLP pipeline |
| [`SRC-PROGTRANS-2020`](#src-progtrans-2020) | Progressive Transformers | 2020 | `include_core` | Foundational text-to-pose baseline |
| [`SRC-EVERYBODY-SIGN-NOW-2020`](#src-everybody-sign-now-2020) | Everybody Sign Now | 2020 | `include_boundary` | Pose-to-video / rendering boundary |
| [`SRC-MULTICHANNEL-MDN-2021`](#src-multichannel-mdn-2021) | Multi-Channel MDN | 2021 | `include_core` | Extended multi-channel baseline |
| [`SRC-MIXED-SIGNALS-2021`](#src-mixed-signals-2021) | Mixed SIGNals | 2021 | `include_counter_alternative` | Motion-primitives alternative |
| [`SRC-NSLPG-2021`](#src-nslpg-2021) | NSLPG | 2021 | `include_core` | Non-autoregressive latent baseline |
| [`SRC-G2PDDM-2024`](#src-g2pddm-2024) | G2P-DDM | 2024 | `include_boundary` | Gloss-to-pose discrete diffusion |
| [`SRC-HAM2POSE-2025`](#src-ham2pose-2025) | Ham2Pose | 2025 | `include_boundary` | Notation-to-pose boundary |
| [`SRC-SIGNVQNET-2024`](#src-signvqnet-2024) | SignVQNet | 2024 | `include_core` | Gloss-free discrete representation |
| [`SRC-NEURAL-SIGN-ACTORS-2024`](#src-neural-sign-actors-2024) | Neural Sign Actors | 2024 | `include_core_caution` | Text-to-3D avatar diffusion |
| [`SRC-DATA-DRIVEN-REP-2024`](#src-data-driven-rep-2024) | Data-Driven Representation | 2024 | `include_core` | Learned pose-token representation |
| [`SRC-SIGN-STITCHING-2024`](#src-sign-stitching-2024) | Sign Stitching | 2024 | `include_counter_alternative` | Retrieval/stitching alternative |
| [`SRC-T2S-GPT-2024`](#src-t2s-gpt-2024) | T2S-GPT | 2024 | `include_core` | Dynamic VQ / text-to-code generation |
| [`SRC-UNIGLOR-2024`](#src-uniglor-2024) | UniGloR | 2024 | `include_core` | Gloss-alternative learned representation |
| [`SRC-SIGNIDD-2025`](#src-signidd-2025) | Sign-IDD | 2025 | `include_boundary` | Bone/relative-pose diffusion |
| [`SRC-LVMCN-2024`](#src-lvmcn-2024) | LVMCN | 2024 | `include_boundary` | Gloss-pose semantic alignment |
| [`SRC-DARSLP-2025`](#src-darslp-2025) | DARSLP | 2025 | `include_core` | Gloss-free articulator disentanglement |
| [`SRC-SLRTP2025-CHALLENGE`](#src-slrtp2025-challenge) | SLRTP2025 Challenge | 2025 | `include_core` | Benchmark / challenge methodology |
| [`SRC-TEXT2SIGNDIFF-2025`](#src-text2signdiff-2025) | Text2Sign Diffusion | 2025 | `include_core` | Gloss-free latent diffusion |
| [`SRC-A2VSLP-2026`](#src-a2vslp-2026) | A²V-SLP | 2026 | `include_core_caution` | Variational articulator-wise latent modeling |
| [`SRC-SIGNSPARK-2026`](#src-signspark-2026) | SignSparK | 2026 | `include_core_caution` | Sparse keyframe / multilingual 3D SLP |
| [`SRC-M3T-2026`](#src-m3t-2026) | M3T | 2026 | `include_core_caution` | Multi-modal motion tokens / non-manual features |
| [`SRC-ILRSLP-2025`](#src-ilrslp-2025) | ILRSLP | 2025 | `include_core_caution` | Iterative latent refinement |
| [`SRC-POSE-EVAL-2025`](#src-pose-eval-2025) | Pose-Based Evaluation | 2025 | `include_core` | Pose-based evaluation methodology |
| [`SRC-MS2SL-2024`](#src-ms2sl-2024) | MS2SL | 2024 | `include_core` | How2Sign-direct multimodal production |
| [`SRC-MCST-2024`](#src-mcst-2024) | MCST | 2024 | `include_core_caution` | Multi-channel spatial-temporal modeling |
| [`SRC-SURVEY-2024`](#src-survey-2024) | SLP Survey | 2024 | `include_background` | Survey / coverage check |

## Source Records

### SRC-HOW2SIGN-2021

**Source:** [How2Sign: A Large-scale Multimodal Dataset for Continuous American Sign Language](https://arxiv.org/abs/2008.08143)

**Year / venue:** 2021 — CVPR 2021 / arXiv
**Primary role:** Dataset anchor
**Corpus status:** `include_core`
**Evidence class:** `dataset_evidence`, `evaluation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_optional`

**Decision relevance:** Defines the practical ASL text/video/keypoint data regime.

### SRC-TEXT2SIGN-2020

**Source:** [Text2Sign: Towards Sign Language Production Using Neural Machine Translation and Generative Adversarial Networks](https://doi.org/10.1007/s11263-019-01281-2)

**Year / venue:** 2020 — International Journal of Computer Vision
**Primary role:** Historical SLP pipeline
**Corpus status:** `include_boundary`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `background_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Foundational gloss-dependent text-to-pose/video pipeline; useful for historical and gloss-bottleneck framing.

### SRC-PROGTRANS-2020

**Source:** [Progressive Transformers for End-to-End Sign Language Production](https://arxiv.org/abs/2004.14874)

**Year / venue:** 2020 — ECCV 2020 / arXiv
**Primary role:** Foundational text-to-pose baseline
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `evaluation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_optional`

**Decision relevance:** Establishes direct text-to-pose SLP and back-translation evaluation baseline.

### SRC-EVERYBODY-SIGN-NOW-2020

**Source:** [Everybody Sign Now: Translating Spoken Language to Photo Realistic Sign Language Video](https://arxiv.org/abs/2011.09846)

**Year / venue:** 2020 — arXiv
**Primary role:** Pose-to-video / rendering boundary
**Corpus status:** `include_boundary`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `background_evidence`
**Dataset compatibility:** `dataset_private_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Supports long-term text-to-pose-to-video framing; not central to near-term pose modeling.

### SRC-MULTICHANNEL-MDN-2021

**Source:** [Continuous 3D Multi-Channel Sign Language Production via Progressive Transformers and Mixture Density Networks](https://arxiv.org/abs/2103.06982)

**Year / venue:** 2021 — International Journal of Computer Vision / arXiv
**Primary role:** Extended multi-channel baseline
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_optional`

**Decision relevance:** Strengthens multi-channel 3D pose baseline, user-evaluation precedent, and regression-to-mean framing.

### SRC-MIXED-SIGNALS-2021

**Source:** [Mixed SIGNals: Sign Language Production via a Mixture of Motion Primitives](https://arxiv.org/abs/2107.11317)

**Year / venue:** 2021 — ICCV 2021 / arXiv
**Primary role:** Motion-primitives alternative
**Corpus status:** `include_counter_alternative`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Important alternative representation family; not direct support for no-gloss implementation.

### SRC-NSLPG-2021

**Source:** [Non-Autoregressive Sign Language Production with Gaussian Space](https://www.bmva-archive.org.uk/bmvc/2021/assets/papers/1102.pdf)

**Year / venue:** 2021 — BMVC 2021
**Primary role:** Non-autoregressive latent baseline
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `representation_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_optional`

**Decision relevance:** Establishes latent-space non-autoregressive SLP and critiques back-translation reliability.

### SRC-G2PDDM-2024

**Source:** [G2P-DDM: Generating Sign Pose Sequence from Gloss Sequence with Discrete Diffusion Model](https://arxiv.org/abs/2208.09141)

**Year / venue:** 2024 — AAAI 2024 / arXiv
**Primary role:** Gloss-to-pose discrete diffusion
**Corpus status:** `include_boundary`
**Evidence class:** `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Valuable for VQ/discrete diffusion design, but not directly compatible with gloss-free How2Sign assumptions.

### SRC-HAM2POSE-2025

**Source:** [Ham2Pose: Animating Sign Language Notation into Pose Sequences](https://arxiv.org/abs/2211.13613)

**Year / venue:** 2025 — arXiv
**Primary role:** Notation-to-pose boundary
**Corpus status:** `include_boundary`
**Evidence class:** `representation_evidence`, `evaluation_evidence`, `background_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Useful for notation systems and pose-distance evaluation; not a text-to-pose path for this project.

### SRC-SIGNVQNET-2024

**Source:** [A Gloss-free Sign Language Production with Discrete Representation / SignVQNet](https://arxiv.org/abs/2309.12179)

**Year / venue:** 2024 — FG 2024 / arXiv
**Primary role:** Gloss-free discrete representation
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core evidence for learned discrete pose-token SLP under a gloss-free setting.

### SRC-NEURAL-SIGN-ACTORS-2024

**Source:** [Neural Sign Actors: A Diffusion Model for 3D Sign Language Production from Text](https://arxiv.org/abs/2312.02702)

**Year / venue:** 2024 — CVPR 2024 / arXiv
**Primary role:** Text-to-3D avatar diffusion
**Corpus status:** `include_core_caution`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `representation_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Important for SMPL-X / 3D avatar-compatible production; heavier than current keypoint pipeline.

### SRC-DATA-DRIVEN-REP-2024

**Source:** [A Data-Driven Representation for Sign Language Production](https://arxiv.org/abs/2404.11499)

**Year / venue:** 2024 — FG 2024 / arXiv
**Primary role:** Learned pose-token representation
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_adaptable`

**Decision relevance:** Core evidence that learned pose/motion representations can replace scarce linguistic annotation.

### SRC-SIGN-STITCHING-2024

**Source:** [Sign Stitching: A Novel Approach to Sign Language Production](https://arxiv.org/abs/2405.07663)

**Year / venue:** 2024 — BMVC 2024 / arXiv
**Primary role:** Retrieval/stitching alternative
**Corpus status:** `include_counter_alternative`
**Evidence class:** `architecture_evidence`, `negative_evidence`, `background_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Serious alternative to direct regression; useful for explaining retrieval/stitching boundaries.

### SRC-T2S-GPT-2024

**Source:** [T2S-GPT: Dynamic Vector Quantization for Autoregressive Sign Language Production from Text](https://arxiv.org/abs/2406.07119)

**Year / venue:** 2024 — arXiv
**Primary role:** Dynamic VQ / text-to-code generation
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `dataset_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core evidence for dynamic variable-length sign tokenization and autoregressive text-to-token generation.

### SRC-UNIGLOR-2024

**Source:** [A Spatio-Temporal Representation Learning as an Alternative to Traditional Glosses in Sign Language Translation and Production / UniGloR](https://arxiv.org/abs/2407.02854)

**Year / venue:** 2024 — arXiv
**Primary role:** Gloss-alternative learned representation
**Corpus status:** `include_core`
**Evidence class:** `representation_evidence`, `direct_task_evidence`, `architecture_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Central evidence for non-gloss learned intermediate representations from keypoints.

### SRC-SIGNIDD-2025

**Source:** [Sign-IDD: Iconicity Disentangled Diffusion for Sign Language Production](https://arxiv.org/abs/2412.13609)

**Year / venue:** 2025 — AAAI 2025 / arXiv
**Primary role:** Bone/relative-pose diffusion
**Corpus status:** `include_boundary`
**Evidence class:** `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Useful for skeletal structure and bone-direction/length representation; gloss-to-pose limits direct use.

### SRC-LVMCN-2024

**Source:** [Linguistics-Vision Monotonic Consistent Network for Sign Language Production](https://arxiv.org/abs/2412.16944)

**Year / venue:** 2024 — arXiv
**Primary role:** Gloss-pose semantic alignment
**Corpus status:** `include_boundary`
**Evidence class:** `architecture_evidence`, `evaluation_evidence`, `negative_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_dependent`

**Decision relevance:** Shows alignment matters, but solution depends on gloss-pose semantic alignment.

### SRC-DARSLP-2025

**Source:** [Disentangle and Regularize: Sign Language Production with Articulator-Based Disentanglement and Channel-Aware Regularization](https://arxiv.org/abs/2504.06610)

**Year / venue:** 2025 — arXiv
**Primary role:** Gloss-free articulator disentanglement
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core source for articulator-disentangled latent spaces and channel-aware regularization.

### SRC-SLRTP2025-CHALLENGE

**Source:** [SLRTP2025 Sign Language Production Challenge: Methodology, Results, and Future Work](https://arxiv.org/abs/2508.06951)

**Year / venue:** 2025 — arXiv
**Primary role:** Benchmark / challenge methodology
**Corpus status:** `include_core`
**Evidence class:** `evaluation_evidence`, `dataset_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core source for standardized Text-to-Pose evaluation and benchmark design.

### SRC-TEXT2SIGNDIFF-2025

**Source:** [Text2Sign Diffusion: A Generative Approach for Gloss-Free Sign Language Production](https://arxiv.org/abs/2509.10845)

**Year / venue:** 2025 — arXiv
**Primary role:** Gloss-free latent diffusion
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `representation_evidence`, `evaluation_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core recent alternative to token/VQ approaches; directly evaluates on How2Sign.

### SRC-A2VSLP-2026

**Source:** [A²V-SLP: Alignment-Aware Variational Modeling for Disentangled Sign Language Production](https://arxiv.org/abs/2602.11861)

**Year / venue:** 2026 — arXiv
**Primary role:** Variational articulator-wise latent modeling
**Corpus status:** `include_core_caution`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Strong recent continuation of articulator-wise latent modeling; preprint/frontier caution.

### SRC-SIGNSPARK-2026

**Source:** [SignSparK: Efficient Multilingual Sign Language Production via Sparse Keyframe Learning](https://arxiv.org/abs/2603.10446)

**Year / venue:** 2026 — arXiv
**Primary role:** Sparse keyframe / multilingual 3D SLP
**Corpus status:** `include_core_caution`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `representation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Frontier source for keyframe-driven 3D SLP and CFM; too heavy for immediate project scope.

### SRC-M3T-2026

**Source:** [M3T: Discrete Multi-Modal Motion Tokens for Sign Language Production](https://arxiv.org/abs/2603.23617)

**Year / venue:** 2026 — arXiv
**Primary role:** Multi-modal motion tokens / non-manual features
**Corpus status:** `include_core_caution`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Frontier/core evidence that body, hands, and face tokenization are needed for linguistically complete SLP.

### SRC-ILRSLP-2025

**Source:** [Iterative Latent Refinement for Robust Non-Autoregressive Sign Language Production](https://openaccess.thecvf.com/content/ICCV2025W/MSLR/html/Kiziltepe_Iterative_Latent_Refinement_for_Robust_Non-Autoregressive_Sign_Language_Production_ICCVW_2025_paper.html)

**Year / venue:** 2025 — ICCV Workshop 2025 / CVF OpenAccess
**Primary role:** Iterative latent refinement
**Corpus status:** `include_core_caution`
**Evidence class:** `direct_task_evidence`, `representation_evidence`, `architecture_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Important continuation of articulator-disentangled latent modeling; group with DARSLP/A²V rather than separate family.

### SRC-POSE-EVAL-2025

**Source:** [Meaningful Pose-Based Sign Language Evaluation](https://aclanthology.org/2025.wmt-1.4/)

**Year / venue:** 2025 — WMT 2025 / ACL Anthology
**Primary role:** Pose-based evaluation methodology
**Corpus status:** `include_core`
**Evidence class:** `evaluation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `how2sign_compatible`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core source for distance-based, embedding-based, and back-translation-based pose evaluation.

### SRC-MS2SL-2024

**Source:** [MS2SL: Multimodal Spoken Data-Driven Continuous Sign Language Production](https://aclanthology.org/2024.findings-acl.432/)

**Year / venue:** 2024 — Findings of ACL 2024 / ACL Anthology
**Primary role:** How2Sign-direct multimodal production
**Corpus status:** `include_core`
**Evidence class:** `direct_task_evidence`, `architecture_evidence`, `evaluation_evidence`, `reproducibility_evidence`
**Dataset compatibility:** `how2sign_direct`
**Gloss dependency:** `gloss_free`

**Decision relevance:** Core recent How2Sign source for text/audio-to-keypoint production via diffusion and embedding consistency.

### SRC-MCST-2024

**Source:** [Multi-Channel Spatio-Temporal Transformer for Sign Language Production](https://aclanthology.org/2024.lrec-main.1022/)

**Year / venue:** 2024 — LREC-COLING 2024 / ACL Anthology
**Primary role:** Multi-channel spatial-temporal modeling
**Corpus status:** `include_core_caution`
**Evidence class:** `architecture_evidence`, `representation_evidence`, `evaluation_evidence`
**Dataset compatibility:** `cross_lingual_transferable`
**Gloss dependency:** `gloss_optional`

**Decision relevance:** Supports channel-aware architectural modeling across manual and non-manual articulators.

### SRC-SURVEY-2024

**Source:** [A Survey on Recent Advances in Sign Language Production](https://doi.org/10.1016/j.eswa.2023.122846)

**Year / venue:** 2024 — Expert Systems With Applications
**Primary role:** Survey / coverage check
**Corpus status:** `include_background`
**Evidence class:** `background_evidence`, `dataset_evidence`, `evaluation_evidence`
**Dataset compatibility:** `dataset_unclear`
**Gloss dependency:** `gloss_unclear`

**Decision relevance:** Useful for taxonomy and coverage sanity checking; not a primary decision-bearing method source.

## Recommended Use by Downstream Documents

Downstream research documents should treat this file as the source of truth for source identity,
corpus status, and high-level evidence role.

`literature-positioning.md` should synthesize source families after the refreshed contribution
audit. `contribution-audit/*` should use this corpus as upstream evidence for candidate universe,
candidate cards, scorecards, and selection decisions.

Boundary and counter-alternative sources may be used to define limitations or competing method
families. Boundary and counter-alternative sources must not be treated as direct support for a
gloss-free How2Sign-compatible implementation unless an adaptation rationale is documented.

## Maintenance Policy

This file should be updated when:

- New decision-bearing sources are added.
- A source status changes after reassessment.
- Source-selection criteria change.
- The contribution audit is refreshed.
- Literature positioning is rewritten.
- Research audit closure is declared.
