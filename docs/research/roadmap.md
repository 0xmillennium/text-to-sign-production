# Roadmap

This page is the authoritative roadmap after Dataset Build. Sprint 1 and Sprint 2 are completed
historical context, Sprint 3 is the current implemented baseline milestone, and Sprint 4 through
Sprint 8 define the remaining planned thesis path.

## Strategy

Sprint 3 now anchors the implemented baseline evidence. Sprint 4 establishes evaluation and error
analysis before strong thesis-contribution claims are made. Sprint 5 and Sprint 6 remain the main
planned thesis-contribution path: Sprint 5 focuses on discrete/data-driven pose representation, and
Sprint 6 focuses on structure-aware / multi-channel improvement. Diffusion and retrieval/stitching
remain meaningful later alternatives or future extensions, but they are not the primary chosen
roadmap path at this stage.

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
not the main thesis-contribution sprint.

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
- demo polish, avatar integration, or final thesis packaging

### Acceptance Criteria

- A baseline run is reproducible from recorded configuration, command, and processed manifest
  references.
- Baseline execution evidence, qualitative exports, and record/package outputs are available as
  current repo surfaces and are tied to a formal experiment record.
- The baseline uses processed manifests rather than direct raw-file access.
- The baseline is clearly labeled as baseline evidence, not the main thesis contribution.

### Main Risk

Direct continuous regression may produce weak, smoothed, or under-articulated output. That risk is
acceptable here because Sprint 3 exists to create a measurable reference point.

## Sprint 4 - Evaluation & Error Analysis

### Purpose

Stabilize the evaluation stack before strong thesis-contribution claims are made.

### In Scope

- baseline evaluation protocol
- pose-level and sequence-level metric selection
- qualitative inspection workflow for generated pose sequences
- error taxonomy for baseline failures
- comparison templates for later Sprint 5 and Sprint 6 experiments

### Out of Scope

- merging evaluation work into Sprint 3
- moving evaluation after the main thesis-contribution sprints
- introducing a new main model family
- claiming final contribution strength before the evaluation stack is stable

### Acceptance Criteria

- Baseline outputs can be evaluated consistently with documented commands and inputs.
- Evaluation results distinguish metric outputs from qualitative interpretation.
- A baseline error-analysis record identifies major failure modes and comparison needs.
- Sprint 5 experiments have a stable evaluation reference instead of inventing metrics ad hoc.

### Main Risk

Pose-based sign-language production metrics are imperfect. The sprint reduces that risk by making
the evaluation assumptions explicit before contribution claims depend on them.

## Sprint 5 - Thesis Contribution I: Discrete/Data-Driven Pose Representation

### Purpose

Develop the first main thesis contribution: a discrete/data-driven pose representation for
text-to-sign-pose production.

### In Scope

- learned or data-driven pose units over the processed pose dataset
- codebook-like, motion-primitive, tokenized, or otherwise discrete pose representation experiments
- reconstruction and generation comparisons against the Sprint 3 baseline
- evaluation through the Sprint 4 stack

### Out of Scope

- replacing this sprint with diffusion
- replacing this sprint with retrieval/stitching
- collapsing this sprint into a generic modeling sprint
- final demo or thesis packaging work

### Acceptance Criteria

- The discrete/data-driven representation can be trained or derived from processed manifests.
- Reconstruction or generation behavior is evaluated against the baseline and Sprint 4 metrics.
- Experiment records make representation choices, data provenance, and limitations explicit.
- Results are framed as the first main thesis-contribution path, not as final system packaging.

### Main Risk

The learned representation may collapse, reconstruct poorly, or fail to improve generation. The
evaluation stack should expose that failure mode rather than hiding it.

## Sprint 6 - Thesis Contribution II: Structure-Aware / Multi-Channel Improvement

### Purpose

Develop the second main thesis contribution: structure-aware / multi-channel improvement over the
baseline and discrete/data-driven representation work.

### In Scope

- skeleton-aware, body/hand-aware, or channel-aware modeling changes
- ablations across body, left hand, right hand, and optional face channels where supported by the
  processed schema
- comparisons against Sprint 3 baseline and Sprint 5 representation experiments
- documentation of which structure or channel assumptions improve results

### Out of Scope

- replacing this sprint with retrieval/stitching
- replacing this sprint with diffusion
- introducing a new unrelated main model family
- treating demo/playback work as the main contribution

### Acceptance Criteria

- Structure-aware or multi-channel variants are evaluated using the Sprint 4 stack.
- Comparisons against Sprint 3 and Sprint 5 are recorded with reproducible provenance.
- Ablations make clear which pose channels or structural assumptions contribute to observed changes.
- Claims remain bounded by the available evidence.

### Main Risk

Additional structure can add complexity without measurable gain. The sprint must preserve ablations
that show whether the complexity is justified.

## Sprint 7 - Inference / Playback / Minimal Demo

### Purpose

Provide inspectable downstream demo capability for generated pose sequences.

### In Scope

- reproducible inference path for selected trained artifacts
- minimal pose playback or skeleton inspection
- example generation records that connect text inputs to output pose sequences
- lightweight demo documentation for thesis inspection

### Out of Scope

- redefining the main research contribution
- replacing Sprint 5 or Sprint 6 with demo work
- broad avatar or photorealistic rendering commitments
- final thesis packaging

### Acceptance Criteria

- Selected generated examples can be replayed or inspected reproducibly.
- Demo artifacts are connected to the experiment records that produced them.
- The demo is clearly downstream inspection support, not the central research claim.

### Main Risk

Demo work can distract from thesis evidence. Keep it minimal and tied to already-evaluated outputs.

## Sprint 8 - Thesis Packaging / Final Integration

### Purpose

Package final comparisons, reproducibility evidence, and thesis-facing integration.

### In Scope

- final comparison tables and thesis-facing summaries
- reproducibility review for configs, manifests, commands, and experiment records
- integration of Sprint 3 through Sprint 7 evidence into a coherent final narrative
- final documentation updates needed to explain the implemented research path

### Out of Scope

- introducing a new main model family
- redefining the Sprint 5 or Sprint 6 contribution path
- broad Dataset Build redesign
- new demo ambitions beyond minimal inspection needs

### Acceptance Criteria

- Final evidence clearly separates baseline, evaluation, thesis contributions, demo, and packaging.
- Experiment records and docs support reproduction of the final comparison claims.
- The thesis-facing story aligns with the frozen Sprint 3 through Sprint 8 roadmap.

### Main Risk

Late integration may reveal missing provenance or weak comparisons. Sprint 8 should fix evidence
gaps, not introduce new primary research directions.

## Guardrails

Future implementation should preserve the core Sprint 1 principles:

- code in `src/`
- notebooks as thin drivers
- documented decisions
- reproducible commands and logs
- raw-data access behind documented manifests and schemas
- large generated artifacts outside GitHub unless there is a strong reason to change policy
