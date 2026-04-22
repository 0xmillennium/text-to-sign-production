# Dynamic VQ Pose Tokens

## Candidate ID

`DISCRETE-A1`

## Candidate Name

Dynamic VQ Pose Tokens

## Family

- `Family label`: `discrete_data_driven`
- `Family-level universe record`: [Discrete / Data-Driven Representation](../candidate-universe/discrete-data-driven-representation.md)

## Initial Status

- `Initial status label`: `primary_candidate`

## One-Sentence Technical Definition

Learn a discrete pose-token representation over the processed pose sequences using dynamic vector
quantization, then predict token sequences from text and decode them back into pose trajectories.

## Research Role

- `Research role label`: `candidate_for_C1`

## Base-Model Additivity

- `additive_over_base`: Established: yes. The current audit record supports an independent
  `Base + candidate` path because the representation-learning and decoding stages can be attached to
  the base workflow without inherently requiring `C2`.
- `what_changes`: Learn a discrete pose-token representation over the processed pose sequences, then
  predict token sequences from text and decode them back into pose trajectories.
- `requires_other_contribution`: Established: this candidate does not inherently require `C2`.
- `independent_variant_exists`: Established: the current audit record supports the existence of an
  independent `Base + candidate` variant.

## Expected Improvement Mechanism

Replace unconstrained continuous frame regression with reusable discrete pose units while allowing
variable information density across the sequence, with the expectation of reducing mode-averaging
and preserving informative regions better than fixed-length tokenization.

## Targeted Failure Modes

- `Applicable failure modes`: `oversmoothing`, `mode_averaging`, `timing_misalignment`,
  `hand_articulation_loss`, `sequence_drift`

## Required Data / Supervision Assumptions

- `Current processed manifests sufficient`: Current audit assumption: current processed text-pose
  pairs are sufficient for a gloss-free text-to-discrete-to-pose setup.
- `Extra annotation required`: Current audit assumption: no extra annotation is required.
- `Gloss required`: Current audit assumption: no new gloss annotation is required.
- `Dictionary/example bank required`: Current audit assumption: no dictionary or example bank is
  required.
- `Segmentation required`: Current audit assumption: no segmentation annotation is required.

## Repository / Workflow Compatibility

- `Compatible with current .npz schema`: Current audit assessment: compatible with the current
  processed `.npz` pose surface.
- `Compatible with current Colab-centered workflow`: Current audit assessment: compatible with the
  current Colab-centered workflow.
- `Heavy new preprocessing required`: Current audit assessment: heavy new preprocessing is not
  currently indicated.
- `Thin-driver notebook principle affected`: Current audit assessment: artifact formats and run
  packaging will likely require extension. Impact on the thin-driver notebook principle is not yet
  established in the current audit record.

## Supporting Evidence

### Source 1

- `Citation`: [2](../../bibliography.md#ref-02)
- `Support type`: `direct`
- `Claim supported`: Dynamic vector quantization is a plausible candidate for discrete sign representation because fixed-length vector quantization can overlook uneven information density in sign language, and a variable-length coding strategy is intended to address that mismatch.
- `Evidence note`: The paper explicitly proposes a two-stage sign language production paradigm in which sign sequences are first encoded into discrete codes and then generated from text. It further argues that existing fixed-length encodings overlook uneven information density in sign language and motivates a dynamic vector quantization design with adaptive downsampling and duration prediction.
- `Source location`: Abstract; Introduction; method discussion of DVQ-VAE and duration prediction.
- `Interpretation boundary`: This source directly supports the dynamic-VQ idea and its motivation. It does not by itself prove that the same gain will transfer unchanged to this repository’s current baseline or processed data surface.

### Source 2

- `Citation`: [4](../../bibliography.md#ref-04)
- `Support type`: `indirect`
- `Claim supported`: Reframing continuous pose generation as discrete sequence generation is a legitimate sign language production direction and can outperform prior continuous-generation baselines.
- `Evidence note`: The paper explicitly reformulates continuous pose generation as a discrete sequence generation problem by learning a codebook of short motions through vector quantization and translating spoken-language text into codebook tokens. It reports substantial back-translation gains over previous methods.
- `Source location`: Abstract; Introduction; contribution list.
- `Interpretation boundary`: This source strongly supports the discrete/data-driven family that Dynamic VQ belongs to, but it is not itself a dynamic-VQ paper. Its contribution also includes sign stitching, so it should not be cited as evidence for dynamic quantization alone.

## Contradicting Or Limiting Evidence

### Limitation / Boundary Source

- `Citation`: [7](../../bibliography.md#ref-07)
- `Claim limited`: A discrete or reusable-unit representation alone is sufficient to produce natural continuous signing.
- `Evidence note`: The paper argues that simply concatenating expressive sign units creates robotic and unnatural sequences, and it introduces explicit stitching and prosody-oriented processing to address that problem. This is an important boundary for Dynamic VQ: discrete units may help representation, but continuity and transition quality still require explicit evaluation.
- `Source location`: Abstract; Introduction.
- `Interpretation boundary`: This source does not refute Dynamic VQ. It limits overclaiming by showing that a discrete intermediate representation does not automatically solve continuity and transition quality.

### Candidate-Level Evidence Synthesis

- `Best-supported claim`: Dynamic VQ is a directly supported discrete-representation strategy for sign language production, especially where uneven information density makes fixed-length vector quantization a poor fit.
- `Least-secure claim`: Dynamic VQ will necessarily produce natural continuous transitions in this repository without additional transition-aware mechanisms.
- `Evidence confidence rationale`: The candidate has one strong direct source and one strong family-level supporting source. The main remaining uncertainty is not whether dynamic discrete representation is scientifically plausible, but how completely it addresses continuous transition quality in this specific repository setting.

## Comparison To Strong Alternatives

Relative to retrieval/stitching, this candidate keeps the project centered on learned generative
representation rather than example retrieval policy; relative to diffusion, it is lower-risk and
easier to align with the current baseline; relative to motion-primitives, it is more naturally
gloss-free but may lose some explicit compositional structure.

## Minimum Ablation Plan

Base vs Base+DynamicVQ; DynamicVQ vs a fixed-VQ internal variant; Base+DynamicVQ vs
Base+DynamicVQ+C2 if later selected; qualitative inspection of token-decoded motion stability and
articulation.

## Position In The Four-Model Matrix

- `fits_as_C1`: Established: yes. The current audit record supports this candidate as a plausible
  `C1`-family fit.
- `fits_as_C2`: Established: no in the current research-role framing. This candidate is documented
  as a `C1`-family discrete/data-driven path rather than as a structure-aware `C2` path.
- `fits_joint_model`: Conditional: joint-model compatibility with a later structure-aware
  candidate appears plausible but remains to be tested.
- `breaks_factorial_design`: Not currently indicated. The current audit record supports an
  independent `Base + candidate` variant and does not identify an inherent requirement for `C2`.

## Complexity / Risk Estimate

- `Implementation complexity`: Medium implementation complexity.
- `Failure cost (days or bucket)`: Moderate failure cost if wrong.
- `Hidden dependency risk`: Medium because the representation learner and decoder both need to
  work.
- `Scope-creep risk`: Medium but still lower than switching to diffusion or retrieval-centered
  redesign.

## Record Boundary Note

- This document records candidate-level evidence only.
- [Scorecards](../scorecards/index.md) and
  [Selection Decisions](../selection-decisions/index.md) are maintained in their dedicated audit
  surfaces.
- No final selection outcome is recorded in this document.
