# Roadmap

This page is the authoritative roadmap after Dataset Build. Sprint 1 through Sprint 4 are
completed historical foundation: Sprint 1 established repository infrastructure, Sprint 2
established the Dataset Build foundation, Sprint 3 produced the implemented baseline milestone, and
Sprint 4 completed the Contribution Audit. Sprint 5 through Sprint 8 now define the remaining
implementation, evaluation, demo, comparison, and thesis-integration path under the accepted
current audit outcome.

## Strategy

Contribution selection is already fixed in this roadmap. Sprint 4 completed the current
contribution-audit milestone and froze the downstream comparison architecture as:

- `M0 = Base`
- `M1 = Base + C1`
- `M2 = Base + C2`
- `M3 = Base + C1 + C2`

The accepted current audit outcome under that structure is:

- `C1 = Dynamic VQ Pose Tokens`
- `C2 = Channel-Aware Loss Reweighting`
- fallback = `Articulator-Partitioned Latent Structure`
- deferred = `Motion Primitives Representation`

The remaining work is therefore not a naive "implement C1, then implement C2" sequence. The next
step is to harden and validate the implemented baseline as a trustworthy `M0` reference, then
build a minimal reusable visual inspection surface, then execute the controlled selected-pair
implementation and `M0/M1/M2/M3` comparison program, and finally package the integrated
thesis-facing result. Detailed candidate-selection reasoning remains in the
[Contribution Audit](contribution-audit/index.md) rather than in this roadmap.

## Historical Context

The following completed sprints explain how the current repository foundation was established.

### Sprint 1

Sprint 1 delivers repository infrastructure only:

- packaging
- tests
- documentation
- CI/CD
- historical DVC bootstrap, later removed from the active workflow by ADR-0012
- ADR and experiment logging support

### Sprint 2

Sprint 2 established the current Dataset Build foundation:

- raw How2Sign layout inspection and documentation
- raw manifest generation and validation
- OpenPose 2D normalization into versioned `.npz` samples
- active `filter-v2.yaml` policy with usable body plus at least one usable hand
- manifest-driven processed dataset export and packaging foundation
- data-quality, split-integrity, and assumption reporting
- full `train / val / test` validation recorded after Filter V2 and packaging hardening

## Sprint 3 - Baseline Modeling

### Purpose

Sprint 3 establishes the current implemented baseline milestone: a reproducible English
text-to-pose baseline on the processed Dataset Build manifests. This sprint is baseline-only and is
not the main thesis-contribution sprint. It creates the current reference point that later sprints
must validate, reuse, and compare against as `M0`.

### In Scope

- manifest-driven training and validation data loading from processed Dataset Build outputs
- a simple, reproducible text-to-continuous-pose baseline
- baseline execution evidence, qualitative export, and record/package outputs
- formal experiment recording that makes the baseline reproducible and comparable

### Out of Scope

- strong thesis-contribution claims
- discrete/data-driven pose representation work
- structure-aware / multi-channel contribution work
- diffusion as the primary model family
- retrieval/stitching as the primary model family
- demo polish, artifact publication, or final thesis packaging

### Acceptance Criteria

- A baseline run is reproducible from recorded configuration, command, and processed manifest
  references.
- Baseline execution evidence, qualitative exports, and record/package outputs are available as
  current repo surfaces and are tied to a formal experiment record.
- The baseline uses processed manifests rather than direct raw-file access.
- The baseline is clearly labeled as baseline evidence, not the main thesis contribution.
- The baseline exists as the starting reference point for later hardening and controlled comparison
  work.

### Main Risk

Direct continuous regression may produce weak, smoothed, or under-articulated output. That risk is
acceptable here because Sprint 3 exists to create a measurable reference point.

## Sprint 4 - Completed Contribution Audit

### Purpose

Record the completed Contribution Audit milestone that fixed the current thesis-facing selected
pair and the frozen comparison structure for later work.

### In Scope

- completion of the contribution-audit process for the current thesis path
- recording the minimum required four-model evidence structure:
  - `M0 = Base`
  - `M1 = Base + C1`
  - `M2 = Base + C2`
  - `M3 = Base + C1 + C2`
- recording the fixed current outcome:
  - `C1 = Dynamic VQ Pose Tokens`
  - `C2 = Channel-Aware Loss Reweighting`
  - fallback = `Articulator-Partitioned Latent Structure`
  - deferred = `Motion Primitives Representation`
- recording that later sprints now proceed under this fixed outcome

### Out of Scope

- re-running contribution-selection reasoning inside the roadmap
- re-scoring candidates
- treating fallback or deferred candidates as newly selected
- implementing the downstream model-comparison program

### Acceptance Criteria

- Sprint 4 is clearly recorded as a completed milestone.
- The selected pair and frozen `M0/M1/M2/M3` architecture are clearly stated.
- Later sprints are explicitly framed as downstream execution under the fixed Sprint 4 outcome.

### Main Risk

The main risk is decision drift after audit completion, including accidental reopening of
contribution selection during downstream implementation.

## Sprint 5 - Baseline Evaluation / Error Analysis / Hardening

### Purpose

Validate the implemented base model as a trustworthy reference point for later comparison work.

### In Scope

- verify that base-model training was run correctly
- verify that base-model testing was run correctly
- complete checks on epoch outputs and baseline behavior
- error analysis of baseline outputs
- notebook-based baseline testing surfaces
- reusable evaluation and test surfaces that later models can reuse
- package the validated baseline artifact
- publish the validated baseline through Hugging Face
- document downstream usage against the published artifact

### Out of Scope

- implementing `C1`
- implementing `C2`
- treating this sprint as final contribution-comparison work
- demo polish beyond what is needed for validation surfaces

### Acceptance Criteria

- Training provenance, commands, configs, summaries, checkpoints, and baseline test outputs are
  reviewed and recorded well enough to verify that the base-model training and testing path ran
  correctly.
- Epoch outputs and baseline behavior have documented review outcomes, and any required follow-up
  checks for the baseline reference point are completed.
- Notebook surfaces can run baseline testing or inspection workflows without relying on ad hoc
  local-only procedures.
- Reusable evaluation and test surfaces exist for later `M1/M2/M3` use.
- A validated baseline artifact is packaged, published on Hugging Face, and documented as the
  preferred downstream baseline reference whenever practical.

### Main Risk

The main risk is comparing later contribution models against a weak or poorly validated baseline
reference.

## Sprint 6 - Minimal Visual Demo / Inspectable Output Surface

### Purpose

Provide a minimal inspectable visual surface for model outputs that supports evaluation and
communication.

### In Scope

- consume the published baseline artifact from Hugging Face
- provide reproducible minimal visual playback, skeleton inspection, or output inspection
- connect inspected outputs back to published artifacts and experiment identity
- keep the surface reusable for later models, not only the base model
- support later side-by-side inspection and comparison across `M0/M1/M2/M3` outputs through the
  same minimal surface

### Out of Scope

- main research contribution claims
- broad avatar or polished product/demo ambitions
- ad hoc checkpoint-only usage when a published artifact is available
- final thesis packaging

### Acceptance Criteria

- The visual inspection surface works against the published baseline artifact.
- Inspected examples can be reproduced from documented artifact references.
- The surface is intentionally minimal and clearly framed as evaluation-support infrastructure.
- The same surface can later be reused for `M1/M2/M3` outputs and later comparison-oriented
  inspection.

### Main Risk

The main risk is letting demo work expand beyond a minimal reusable inspection surface.

## Sprint 7 - Selected-Pair Implementation / Controlled 2x2 Comparison

### Purpose

Implement the selected contribution pair and execute the controlled comparison program under the
frozen audit architecture.

### In Scope

- implement `C1 = Dynamic VQ Pose Tokens`
- implement `C2 = Channel-Aware Loss Reweighting`
- train and evaluate the frozen comparison set:
  - `M0 = Base`
  - `M1 = Base + C1`
  - `M2 = Base + C2`
  - `M3 = Base + C1 + C2`
- reuse Sprint 5 evaluation and testing surfaces
- reuse Sprint 6 visual inspection surfaces
- continue published-artifact discipline as far as practical for downstream consumption

### Out of Scope

- reopening candidate selection
- substituting fallback or deferred candidates unless later explicitly triggered outside this
  roadmap step
- turning this sprint into a new literature review or a new audit
- treating the visual surface as the main thesis contribution

### Acceptance Criteria

- All four planned comparison conditions are clearly defined and evaluated under the frozen
  architecture.
- The selected `C1/C2` pair is implemented as the current audited pair.
- Downstream evaluation and testing reuse the Sprint 5 surfaces rather than creating a separate ad
  hoc comparison path.
- Visual inspection and comparison reuse the Sprint 6 surface rather than introducing a separate
  model-specific inspection workflow.
- Artifact handling remains reproducible and publication-oriented where practical, including later
  selected-pair outputs.

### Main Risk

The main risk is uncontrolled comparison design drift, or contribution implementation that bypasses
the frozen `M0/M1/M2/M3` structure.

## Sprint 8 - Final Integration / Thesis Packaging

### Purpose

Integrate the final thesis-facing narrative and reproducibility package.

### In Scope

- integrate the baseline
- integrate the completed audit outcome
- integrate Sprint 5 hardened testing and evaluation surfaces
- integrate Sprint 6 visual inspection surface
- integrate Sprint 7 `M0/M1/M2/M3` comparisons
- final thesis-facing documentation and comparison packaging

### Out of Scope

- new primary model-family selection
- replacing the fixed selected pair
- expanding the demo beyond minimal inspection needs
- broad redesign of earlier pipeline stages

### Acceptance Criteria

- The final narrative clearly links historical foundation, baseline, completed audit, reusable
  evaluation surfaces, visual inspection, and controlled 2x2 comparison evidence.
- Final documentation and artifacts support reproducible thesis-facing comparison claims.
- No new main selection logic is introduced in Sprint 8.

### Main Risk

The main risk is late-stage packaging gaps that weaken provenance or the final comparison story.

## Guardrails

Future implementation should preserve the core Sprint 1 principles and the post-audit sequencing
discipline:

- code in `src/`
- notebooks as thin drivers
- documented decisions
- reproducible commands and logs
- raw-data access behind documented manifests and schemas
- large generated artifacts outside GitHub unless there is a strong reason to change policy
- downstream usage should prefer published artifacts over informal local checkpoints whenever
  practical
- reusable testing and reusable visual inspection surfaces should be favored over one-off
  sprint-specific tooling
- later `M0/M1/M2/M3` comparison work should extend shared surfaces rather than reintroduce
  ad hoc evaluation or demo paths
