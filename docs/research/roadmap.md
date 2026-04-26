# Roadmap

## Purpose

This roadmap translates the refreshed [Contribution Audit Result](contribution-audit/audit-result.md)
and [Literature Positioning](literature-positioning.md) into an execution map for a working thesis
artifact.

It is not a source registry. It is not a literature review. It is not a scorecard or selection
decision. It does not select a final implementation model.

The roadmap sequences shared infrastructure, baseline, evaluation, candidate models, comparator,
visualization, comparative analysis, and thesis writing for a reproducible, gloss-free,
How2Sign-compatible text-to-sign pose-production research pipeline.

## Controlling Research Basis

This roadmap is controlled by:

- [Source Selection Criteria](source-selection-criteria.md)
- [Source Corpus](source-corpus.md)
- [Contribution Audit Result](contribution-audit/audit-result.md)
- [Literature Positioning](literature-positioning.md)
- [Candidate Universe](contribution-audit/candidate-universe/index.md)
- [Candidate Cards](contribution-audit/candidate-cards/index.md)
- [Scorecards](contribution-audit/scorecards/index.md)
- [Selection Decisions](contribution-audit/selection-decisions/index.md)

The roadmap must not invent priorities independent of the audit result and literature positioning.

The roadmap is audit-derived: phases must trace back to the audit result, literature positioning,
candidate cards, scorecards, and selection decisions rather than to an unsupported implementation
preference.

## Roadmap Principles

- Audit-derived sequencing: execution order follows the audit result, literature positioning,
  candidate cards, scorecards, and selection decisions.
- Shared artifact discipline before model comparison: dataset, schema, split, metadata, and
  pose/keypoint conventions must exist before candidate claims are interpretable.
- Evaluation before selection: the evaluation harness must be usable before candidate outputs are
  compared or synthesized.
- Baseline before candidate claims: the M0 direct text-to-pose baseline is the comparison floor,
  not a model contribution claim.
- Candidate isolation before combination: learned-token, latent-generative, structure-aware,
  semantic-objective, and retrieval-comparator roles should be tested separately before any
  combined interpretation.
- Visualization as inspection and communication layer: skeleton/keypoint playback supports
  debugging, qualitative review, thesis figures, and demo communication.
- Thesis writing as continuous synthesis: audit, method, experiment, limitation, and
  reproducibility material should remain thesis-ready as work progresses.
- Future-avatar boundary preserved: sparse-keyframe, 3D, parametric, and rendering directions are
  tracked as future work unless a later audit or scope decision changes the boundary.
- Model phase order is execution sequence, not final ranking: learned-token, latent-diffusion, and
  articulator-aware phases are ordered for controlled execution and dependency management, not as
  final thesis outcome ranking.
- Reproducibility release respects dataset boundaries: public release should include code,
  configuration, documentation, reproducible commands, and derived reports where allowed, not
  restricted dataset redistribution.
- Human-facing claims require separate validation: user-facing accessibility, human-grade signing,
  or intelligibility claims require appropriate human evaluation and must not be inferred from
  automatic metrics or visual plausibility alone.

Candidate models must not be compared unless they share artifact schema, split policy, baseline,
evaluation protocol, and reporting conventions.

Candidate phases are comparable experimental units, not final implementation commitments.

## Workstream Map

| Workstream | Purpose | Audit basis |
| --- | --- | --- |
| Research audit and documentation | Maintain source-corpus, audit-result, and positioning foundation. | Source corpus and audit result. |
| Dataset and supervision | Enforce How2Sign-compatible, no-public-gloss artifact regime. | Dataset/supervision and gloss/notation boundaries. |
| Pose/keypoint artifacts | Extract, normalize, validate body/hand/face/keypoint data. | Candidate-universe and artifact discipline. |
| Baseline | Maintain M0 direct text-to-pose comparison floor. | Direct Text-to-Pose Baseline. |
| Evaluation | Maintain shared metric, qualitative inspection, and reporting harness. | How2Sign-Compatible Evaluation Protocol. |
| Primary model candidates | Evaluate learned-token, latent diffusion, and articulator-aware routes. | Selection decisions. |
| Auxiliary/comparator | Evaluate semantic consistency and retrieval comparator as controls. | Selection decisions. |
| Visualization | Inspect generated/reference skeleton or keypoint outputs. | Future avatar boundary. |
| Thesis writing | Convert audit, methods, experiments, limitations, and reproducibility into thesis chapters. | Literature positioning and audit result. |

## Phase Plan

### Phase 0 — Research Audit Consolidation

- `Audit basis`:
  - [source-corpus](source-corpus.md)
  - [candidate universe](contribution-audit/candidate-universe/index.md)
  - [candidate cards](contribution-audit/candidate-cards/index.md)
  - [scorecards](contribution-audit/scorecards/index.md)
  - [selection decisions](contribution-audit/selection-decisions/index.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
- `Goal`: Establish the source-to-roadmap research foundation; this phase is already established
  in the documentation layer.
- `Entry criteria`: Existing research documents and source records are available for
  consolidation; no implementation result is required.
- `Inputs`: Refreshed source corpus and audit chain.
- `Core tasks`: Maintain traceability for:
  - source-corpus
  - candidate-universe
  - candidate-card
  - scorecard
  - selection-decision
  - audit-result
  - literature-positioning
  - roadmap
- `Outputs`:
  - source-selection criteria
  - source-corpus
  - candidate-universe records
  - candidate cards
  - scorecards
  - selection decisions
  - audit-result
  - literature-positioning
- `Validation`:
  - `python -m mkdocs build --strict`
  - all public docs link cleanly
- `Exit criteria`:
  - audit chain builds in strict mode
  - all public links resolve
  - downstream roadmap phases can cite the audit basis without legacy selection framing
- `Risk controls`:
  - do not reintroduce old bibliography anchors
  - do not reintroduce local source paths
  - do not reintroduce legacy selection language
  - do not reintroduce experimental claims
- `Out of scope`:
  - model implementation
  - dataset preprocessing
  - experimental claims
- `Downstream dependency`: Controls all later phases.

### Phase 1 — Dataset Access And Supervision Boundary

- `Audit basis`:
  - [source selection criteria](source-selection-criteria.md)
  - [source corpus](source-corpus.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [dataset/supervision boundary](contribution-audit/candidate-universe/dataset-supervision-boundary.md)
  - [gloss/notation boundary](contribution-audit/candidate-universe/gloss-notation-dependent-boundary.md)
- `Goal`: Convert How2Sign-compatible, no-public-gloss supervision regime into actionable
  artifact decisions.
- `Entry criteria`: Source-corpus dataset records and supervision-boundary records are available;
  dataset access route is known or explicitly unresolved.
- `Inputs`:
  - official dataset/source references
  - source-corpus dataset classifications
  - supervision constraints
  - existing dataset-access assumptions
- `Core tasks`:
  - confirm access status
  - confirm usage constraints
  - confirm split assumptions
  - confirm segment metadata
  - confirm allowed supervision signals
  - confirm no-public-gloss boundary
- `Outputs`:
  - dataset access status
  - allowed usage notes
  - train/validation/test split policy
  - metadata schema
  - sentence/video segment mapping
  - supervision boundary note
  - no-public-gloss compliance note
- `Validation`:
  - official dataset/source references are traceable through source-corpus
  - split policy is documented
  - no gloss-dependent assumption is silently introduced
- `Exit criteria`: Dataset access and supervision assumptions are documented enough to support
  artifact schema work without silent gloss-dependency.
- `Risk controls`:
  - prevent manual gloss assumptions from entering as hidden requirements
  - prevent notation assumptions from entering as hidden requirements
  - prevent dictionary assumptions from entering as hidden requirements
  - prevent isolated-sign assumptions from entering as hidden requirements
  - prevent gloss-pose assumptions from entering as hidden requirements
- `Out of scope`:
  - model training
  - avatar output
  - manual gloss annotation
- `Downstream dependency`: Defines the supervision boundary for:
  - artifact schema
  - preprocessing
  - baseline
  - evaluation
  - candidate models
  - comparator work

### Phase 2 — Artifact Schema And Data Pipeline

- `Audit basis`:
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [candidate universe](contribution-audit/candidate-universe/index.md)
  - [dataset/supervision boundary](contribution-audit/candidate-universe/dataset-supervision-boundary.md)
  - [evaluation protocol](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
  - [direct baseline candidate](contribution-audit/candidate-cards/direct-text-to-pose-baseline.md)
- `Goal`: Define shared text/transcript, video segment, pose/keypoint sequence, and metadata
  artifact schema.
- `Entry criteria`: Dataset access/supervision assumptions and split policy draft exist.
- `Inputs`:
  - dataset access decisions
  - split policy
  - sentence/video segment mapping
  - pose/keypoint availability
  - existing reproducibility conventions
- `Core tasks`:
  - define schema fields
  - define artifact IDs
  - define metadata relationships
  - define split handling
  - define validation checks
  - define data registry conventions
- `Outputs`:
  - artifact schema document
  - data registry conventions
  - data versioning or artifact tracking rules if applicable
  - sample artifact manifest
  - schema validation script or check
  - leakage-aware split handling
- `Validation`:
  - sample entries load end-to-end
  - split leakage checks are possible
  - baseline and evaluation code can consume the same schema
- `Exit criteria`: A sample artifact manifest can be validated and consumed by baseline/evaluation
  planning.
- `Risk controls`:
  - prevent train/validation/test leakage
  - prevent schema drift
  - prevent undocumented artifact versions
  - prevent model-specific schema forks
- `Out of scope`:
  - candidate model architecture
  - score comparisons
- `Downstream dependency`: Provides the shared artifact contract for:
  - pose extraction
  - M0 baseline
  - evaluation harness
  - primary candidates
  - auxiliary objective
  - retrieval comparator

### Phase 3 — Pose / Keypoint Extraction And Normalization

- `Audit basis`:
  - [source corpus](source-corpus.md)
  - [candidate universe](contribution-audit/candidate-universe/index.md)
  - [literature positioning](literature-positioning.md)
  - [direct baseline](contribution-audit/candidate-cards/direct-text-to-pose-baseline.md)
  - [learned-token records](contribution-audit/candidate-cards/learned-pose-token-bottleneck.md)
  - [latent diffusion records](contribution-audit/candidate-cards/gloss-free-latent-diffusion.md)
  - [articulator-aware records](contribution-audit/candidate-cards/articulator-disentangled-latent.md)
  - [evaluation records](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
  - [future-avatar boundary records](contribution-audit/candidate-cards/sparse-keyframe-avatar-future.md)
- `Goal`: Produce model-trainable pose/keypoint artifacts from sign video or available pose
  sources.
- `Entry criteria`: Artifact schema and dataset/video or pose-source availability are
  sufficiently defined.
- `Inputs`:
  - artifact schema
  - dataset manifest
  - video or pose source availability
  - split policy
  - body/hand/face requirements
  - missing-data expectations
- `Core tasks`:
  - select extraction tool
  - define body/hand/face schema
  - define normalization
  - define masks
  - define missing-keypoint handling
  - define sequence handling
  - define visual debug samples
- `Outputs`:
  - extraction tool decision
  - body/hand/face keypoint schema
  - normalization policy
  - missing-keypoint and mask policy
  - sequence length policy
  - visual debug samples
- `Validation`:
  - extracted keypoints load
  - masks and missing data are documented
  - skeleton playback or debug visualization can inspect samples
- `Exit criteria`: Representative extracted sequences load under the shared schema and can be
  visually inspected.
- `Risk controls`:
  - prevent undocumented coordinate normalization
  - prevent silent missing-data handling
  - prevent inconsistent keypoint ordering
  - prevent premature avatar/SMPL-X scope
- `Out of scope`:
  - full avatar
  - SMPL-X
  - photorealistic rendering
- `Downstream dependency`: Supplies trainable pose/keypoint targets and masks for:
  - baseline
  - evaluation
  - learned-token
  - latent diffusion
  - articulator-aware
  - visualization
  - thesis reporting phases

### Phase 4 — M0 Direct Text-to-Pose Baseline

- `Audit basis`:
  - [Direct Text-to-Pose Baseline candidate card](contribution-audit/candidate-cards/direct-text-to-pose-baseline.md)
  - [baseline selection decision](contribution-audit/selection-decisions/direct-text-to-pose-baseline-selection-decision.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [evaluation protocol](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
- `Goal`: Build the M0 direct text/transcript-to-pose baseline as the comparison floor.
- `Entry criteria`: Shared artifact schema, split policy, normalized pose/keypoint artifacts, and
  baseline input/output format exist.
- `Inputs`:
  - shared artifact schema
  - split policy
  - normalized pose/keypoint artifacts
  - metadata
  - text/transcript inputs
  - baseline architecture constraints
- `Core tasks`:
  - specify baseline architecture
  - implement training/inference path
  - produce baseline outputs
  - document baseline failure modes
- `Outputs`:
  - baseline architecture specification
  - training script
  - inference script
  - baseline report
  - baseline failure-mode inventory
- `Validation`:
  - same splits
  - same artifact schema
  - same evaluation harness draft
  - reproducible run command
- `Exit criteria`: Baseline can run reproducibly on the agreed split and produce outputs
  consumable by the evaluation harness.
- `Risk controls`: Prevent baseline readiness from being interpreted as contribution strength or
  task-solving quality.
- `Out of scope`:
  - learned token mechanisms
  - diffusion mechanisms
  - articulator-specific mechanisms
- `Downstream dependency`: Establishes the comparison floor for:
  - evaluation
  - primary model candidate phases
  - auxiliary objective ablations
  - retrieval comparator pressure
  - comparative synthesis

### Phase 5 — Evaluation Harness And Reporting Protocol

- `Audit basis`:
  - [How2Sign-Compatible Evaluation Protocol candidate card](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
  - [evaluation and benchmark methodology family](contribution-audit/candidate-universe/evaluation-benchmark-methodology.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [direct baseline candidate](contribution-audit/candidate-cards/direct-text-to-pose-baseline.md)
- `Goal`: Build the shared evaluation and reporting surface for all candidate comparisons.
- `Entry criteria`: M0 baseline outputs and generated/reference artifact format exist.
- `Inputs`:
  - M0 outputs
  - shared artifact schema
  - generated/reference pose format
  - split policy
  - metric limitations from audit records
  - qualitative-inspection needs
- `Core tasks`:
  - implement pose/keypoint metrics
  - implement embedding checks where available
  - implement qualitative inspection checklist
  - implement metric-limitation reporting
  - implement comparison report format
- `Outputs`:
  - pose/keypoint metrics
  - embedding checks
  - cautious recognition or back-translation checks where available
  - qualitative inspection checklist
  - metric-limitation section template
  - comparison report template
- `Validation`:
  - works on the M0 baseline
  - can compare generated and reference outputs
  - preserves the boundary that automatic metrics are not proof of sign intelligibility
- `Exit criteria`: The harness can evaluate M0 baseline outputs and produce a report with
  automatic metrics plus limitations.
- `Risk controls`: Prevent automatic metrics, embedding similarity, or visual plausibility from
  being treated as sufficient proof of sign intelligibility.
- `Out of scope`:
  - final thesis conclusion
  - human-grade signing claim
- `Downstream dependency`: Defines the reporting surface for:
  - learned-token
  - latent diffusion
  - articulator-aware
  - semantic auxiliary
  - retrieval comparator
  - visualization
  - comparative analysis phases

### Phase 6 — Learned Pose-Token Bottleneck Candidate

- `Audit basis`:
  - [Learned Pose-Token Bottleneck candidate card](contribution-audit/candidate-cards/learned-pose-token-bottleneck.md)
  - [scorecard](contribution-audit/scorecards/learned-pose-token-bottleneck-scorecard.md)
  - [selection decision](contribution-audit/selection-decisions/learned-pose-token-bottleneck-selection-decision.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
- `Goal`: Evaluate text-to-token-to-pose as the learned representation candidate.
- `Entry criteria`:
  - shared artifacts exist
  - M0 baseline exists
  - evaluation harness exists
  - masks exist
  - reconstruction targets exist
- `Inputs`:
  - shared artifact schema
  - M0 baseline
  - evaluation harness
  - normalized keypoints
  - masks
  - split policy
  - reconstruction targets
- `Core tasks`:
  - design tokenizer/motion-code representation
  - train or fit representation
  - evaluate reconstruction
  - train text-to-token or equivalent predictor
  - decode to pose
  - compare against M0
- `Outputs`:
  - tokenizer or motion-code design
  - reconstruction report
  - text-to-token model or equivalent predictor
  - token-to-pose decoder
  - ablation against M0 baseline
- `Validation`:
  - reconstruction and generation are evaluated separately
  - codebook stability is tracked
  - semantic adequacy is not inferred from token reconstruction alone
- `Exit criteria`:
  - reconstruction and generated-pose reports exist
  - codebook stability is documented
  - comparison against M0 is possible
- `Risk controls`:
  - prevent token reconstruction quality from being treated as semantic adequacy
  - keep diffusion, semantic auxiliary, and avatar mechanisms out of this phase
- `Out of scope`:
  - diffusion
  - semantic auxiliary objective
  - avatar output
- `Downstream dependency`: Provides representation-based candidate evidence for comparative
  analysis and may become an attachment point for later auxiliary semantic ablation only if kept
  separately evaluated.

### Phase 7 — Gloss-Free Latent Diffusion Candidate

- `Audit basis`:
  - [Gloss-Free Latent Diffusion candidate card](contribution-audit/candidate-cards/gloss-free-latent-diffusion.md)
  - [scorecard](contribution-audit/scorecards/gloss-free-latent-diffusion-scorecard.md)
  - [selection decision](contribution-audit/selection-decisions/gloss-free-latent-diffusion-selection-decision.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
- `Goal`: Evaluate gloss-free latent generative production under the shared artifact and
  evaluation surface.
- `Entry criteria`:
  - shared artifacts exist
  - M0 baseline exists
  - evaluation harness exists
  - latent target decision exists
  - seed policy exists
  - compute assumptions exist
- `Inputs`:
  - shared artifact schema
  - M0 baseline
  - evaluation harness
  - normalized pose/keypoint artifacts
  - latent targets or encoder decisions
  - seed policy
  - compute assumptions
- `Core tasks`:
  - define latent representation
  - configure denoising/generation setup
  - document seed/reproducibility controls
  - produce inference outputs
  - compare under shared evaluation
- `Outputs`:
  - latent representation decision
  - denoising or generation setup
  - seed and reproducibility controls
  - inference outputs
  - comparison report
- `Validation`:
  - same splits and artifact schema
  - baseline comparison
  - generative-quality and metric caveats
  - compute and failure-cost documentation
- `Exit criteria`:
  - latent generation report exists with baseline comparison
  - reproducibility notes
  - compute/failure-cost notes
  - metric caveats
- `Risk controls`:
  - prevent high literature support from being treated as low implementation risk
  - prevent audio-conditioned scope creep
  - prevent sparse-keyframe scope creep
  - prevent CFM scope creep
  - prevent 3D/avatar scope creep
- `Out of scope`:
  - audio-conditioned expansion
  - sparse-keyframe or CFM scope
  - 3D avatar
- `Downstream dependency`: Provides latent-generative candidate evidence for comparative
  analysis and identifies compute, stability, and evaluation risks for thesis limitations.

### Phase 8 — Articulator-Aware Structure Candidate

- `Audit basis`:
  - [Articulator-Disentangled Latent Modeling candidate card](contribution-audit/candidate-cards/articulator-disentangled-latent.md)
  - [scorecard](contribution-audit/scorecards/articulator-disentangled-latent-scorecard.md)
  - [selection decision](contribution-audit/selection-decisions/articulator-disentangled-latent-selection-decision.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
- `Goal`: Evaluate whether body/hand/face/channel-aware modeling improves relevant failure modes.
- `Entry criteria`:
  - body/hand/face keypoint schema exists
  - masks exist
  - baseline outputs exist
  - evaluation harness exists
  - channel failure-mode needs exist
- `Inputs`:
  - shared artifact schema
  - M0 baseline
  - evaluation harness
  - normalized keypoints
  - masks
  - body/hand/face partitions
  - channel failure-mode inventory
- `Core tasks`:
  - define channel partitions
  - define mask strategy
  - define loss weighting
  - define structure-aware model variant
  - define channel-stratified evaluation
- `Outputs`:
  - channel partition scheme
  - mask strategy
  - loss weighting policy
  - structure-aware model variant
  - channel-stratified evaluation report
- `Validation`:
  - partitions are stable
  - masks are documented
  - hand/face/body errors are separately visible
  - sign-level claims are not overextended from channel-level metrics
- `Exit criteria`: Structure-aware report exists with per-channel diagnostics and comparison to
  M0 or relevant candidate outputs.
- `Risk controls`:
  - prevent channel-level improvements from being overclaimed as sign-level adequacy
  - avoid full non-manual linguistic annotation
  - avoid 3D parametric expansion
- `Out of scope`:
  - full non-manual linguistic annotation
  - 3D parametric avatar
- `Downstream dependency`: Provides structure-aware candidate evidence for comparative analysis
  and exposes channel-level diagnostics for visualization and thesis limitations.

### Phase 9 — Auxiliary Semantic Consistency And Retrieval Comparator

- `Audit basis`:
  - [Text-Pose Semantic Consistency Objective candidate card](contribution-audit/candidate-cards/text-pose-semantic-consistency.md)
  - [Retrieval-Augmented Pose Comparator candidate card](contribution-audit/candidate-cards/retrieval-augmented-pose-comparator.md)
  - [scorecards](contribution-audit/scorecards/index.md)
  - [selection decisions](contribution-audit/selection-decisions/index.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
- `Goal`: Evaluate semantic consistency and retrieval as non-primary control surfaces.
- `Entry criteria`: At least one primary candidate output or planned attachment point exists;
  shared artifact schema and evaluation harness exist; split-safe retrieval features can be
  defined.
- `Inputs`:
  - shared artifact schema
  - M0 and candidate outputs where available
  - evaluation harness
  - text/pose embedding choices
  - split-safe retrieval features
  - generated/reference metadata
- `Core tasks`:
  - decide semantic objective attachment point
  - run with/without-objective ablation where applicable
  - build retrieval index
  - enforce split-safe retrieval policy
  - produce comparator report
- `Outputs`:
  - semantic objective attachment decision
  - with/without semantic objective ablation
  - retrieval index
  - leakage-safe retrieval policy
  - retrieval comparator report
- `Validation`:
  - semantic objective is not presented as a standalone primary model
  - retrieval leakage is prevented
  - retrieval realism is not treated as semantic correctness
- `Exit criteria`: Semantic auxiliary effect and retrieval comparator results can be interpreted
  without treating either as a primary model candidate.
- `Risk controls`:
  - prevent semantic objective from becoming a standalone model without new evidence
  - prevent retrieval leakage
  - prevent gloss-dependent stitching
  - prevent dictionary-based production
  - prevent retrieval realism-as-correctness claims
- `Out of scope`:
  - full gloss-dependent stitching
  - dictionary-based production
  - independent semantic-only final model
- `Downstream dependency`: Adds auxiliary and comparator evidence that pressures primary
  candidate interpretation during comparative analysis.

### Phase 10 — Comparative Analysis And Candidate Synthesis

- `Audit basis`:
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [selection decisions](contribution-audit/selection-decisions/index.md)
  - [scorecards](contribution-audit/scorecards/index.md)
  - [evaluation protocol](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
  - [candidate outputs](../experiments/index.md)
  - [auxiliary and comparator reports](contribution-audit/selection-decisions/index.md)
- `Goal`: Interpret primary candidates, auxiliary objective, and comparator under one evaluation
  surface.
- `Entry criteria`:
  - M0 report exists
  - evaluation harness outputs exist
  - primary candidate reports exist
  - semantic auxiliary evidence exists
  - retrieval comparator report exists
  - qualitative samples exist
- `Inputs`:
  - M0 baseline report
  - evaluation harness outputs
  - learned-token report
  - latent diffusion report
  - articulator-aware report
  - semantic objective ablation
  - retrieval comparator report
  - qualitative samples
- `Core tasks`:
  - build consolidated results matrix
  - compare failure modes
  - synthesize qualitative examples
  - document limitations
  - write candidate synthesis memo
- `Outputs`:
  - consolidated results matrix
  - error analysis
  - qualitative samples
  - model failure taxonomy
  - candidate synthesis memo
  - thesis results chapter source material
- `Validation`:
  - all comparisons use the same artifact schema
  - all comparisons use the same splits
  - all comparisons use the same evaluation protocol
  - metric limitations are visible
  - synthesis does not overclaim final performance
- `Exit criteria`: Candidate synthesis can support thesis results and limitations without
  claiming unsupported final performance.
- `Risk controls`:
  - prevent late-stage new model family invention
  - prevent metric cherry-picking
  - prevent incomparable artifacts
  - prevent claims not backed by shared evaluation
- `Out of scope`:
  - new model family invention
  - late-stage scope expansion
- `Downstream dependency`: Supplies the evidence base for:
  - thesis results
  - limitations
  - future-work framing
  - reproducibility release

### Phase 11 — Visual Representation Layer

- `Audit basis`:
  - [literature positioning](literature-positioning.md)
  - [sparse-keyframe/avatar future-work boundary](contribution-audit/candidate-cards/sparse-keyframe-avatar-future.md)
  - [evaluation protocol](contribution-audit/candidate-cards/how2sign-evaluation-protocol.md)
  - [pose/keypoint artifact discipline](contribution-audit/candidate-universe/index.md)
  - [comparative-analysis needs](#phase-10-comparative-analysis-and-candidate-synthesis)
- `Goal`: Provide skeleton/keypoint playback and comparison surfaces for inspection, debugging,
  thesis figures, and demo communication.
- `Entry criteria`:
  - normalized generated/reference pose outputs exist
  - masks exist
  - channel metadata exists where available
  - qualitative sample selections exist
- `Inputs`:
  - normalized pose/keypoint artifacts
  - generated/reference outputs
  - masks
  - channel-partition metadata where available
  - qualitative sample selections
- `Core tasks`:
  - build skeleton/keypoint playback
  - build generated-versus-reference visualization
  - build per-channel visualization where available
  - build thesis/demo export surfaces
- `Outputs`:
  - skeleton playback viewer
  - generated-versus-reference visualization
  - per-channel visualization where available
  - thesis figures or demo clips
- `Validation`:
  - pose output is interpretable
  - failure modes are visually inspectable
  - no near-term full avatar claim is introduced
- `Exit criteria`: Visual outputs support debugging, qualitative inspection, thesis figures, and
  demo communication without implying full avatar generation.
- `Risk controls`:
  - prevent visualization quality from being interpreted as sign correctness
  - prevent SMPL-X scope creep
  - prevent photorealistic rendering scope creep
  - prevent full avatar-control scope creep
- `Out of scope`:
  - SMPL-X
  - photorealistic signer rendering
  - full avatar control
- `Downstream dependency`: Supports:
  - qualitative inspection
  - thesis communication
  - reproducibility review
  - future-avatar boundary documentation

### Phase 12 — Thesis Writing And Reproducibility Release

- `Audit basis`:
  - [source corpus](source-corpus.md)
  - [audit result](contribution-audit/audit-result.md)
  - [literature positioning](literature-positioning.md)
  - [candidate cards](contribution-audit/candidate-cards/index.md)
  - [scorecards](contribution-audit/scorecards/index.md)
  - [selection decisions](contribution-audit/selection-decisions/index.md)
  - [experiment outputs](../experiments/index.md)
  - [comparative analysis](#phase-10-comparative-analysis-and-candidate-synthesis)
  - [future-work boundary](contribution-audit/candidate-cards/sparse-keyframe-avatar-future.md)
- `Goal`: Convert the research artifacts into thesis chapters, public documentation, and
  reproducibility package.
- `Entry criteria`:
  - literature positioning exists
  - roadmap exists
  - methodology records exist
  - artifact schema exists
  - experiment reports exist
  - comparison outputs exist
  - visualizations exist
  - limitations exist
- `Inputs`:
  - literature positioning
  - roadmap
  - methodology records
  - artifact schema
  - experiment reports
  - comparative analysis
  - visualization outputs
  - limitations
  - reproducibility commands
- `Core tasks`:
  - draft literature review
  - draft methodology
  - draft experiments/results
  - draft limitations/future work
  - draft reproducibility appendix
  - draft public docs
  - draft release checklist
- `Outputs`:
  - literature review draft
  - methodology chapter
  - experiments/results chapter
  - limitations/future-work chapter
  - reproducibility appendix
  - final repo docs
  - release checklist
- `Validation`:
  - claims trace back to source-corpus
  - claims trace back to audit-result
  - claims trace back to literature-positioning
  - claims trace back to experiment outputs
  - no unsupported performance claim is introduced
  - limitations and future-work boundaries are preserved
- `Exit criteria`:
  - thesis claims trace to source-corpus
  - thesis claims trace to audit-result
  - thesis claims trace to literature-positioning
  - thesis claims trace to experiment outputs
  - thesis claims trace to documented limitations
- `Risk controls`:
  - prevent unsupported performance claims
  - prevent restricted dataset redistribution
  - prevent hidden local-path dependencies
  - prevent late-stage scope expansion
  - prevent human-grade/accessibility claims without human evaluation
  - preserve dataset-license and redistribution boundaries for reproducibility release
  - include code, configuration, documentation, and reproducible commands in public artifacts
  - do not include restricted dataset redistribution in public artifacts
- `Out of scope`:
  - new research claims not supported by audit or experiments
  - late-stage scope expansion
- `Downstream dependency`: Produces the thesis-facing and public reproducibility surface for the
  completed research artifact.

## Cross-Cutting Requirements

- No-public-gloss boundary must be preserved unless a later documented scope decision changes it.
- Official/source-corpus-backed claims must control dataset, method, evaluation, and literature
  statements.
- Shared split policy must apply across baseline, candidates, auxiliary objective, comparator, and
  evaluation.
- Artifact schema versioning must keep text, video segment, pose/keypoint, mask, metadata, and
  generated-output formats traceable.
- Reproducible commands must exist for data checks, training, inference, evaluation, visualization,
  and report generation where applicable.
- Experiment logging must record configuration, data version, split, seed, artifact version,
  command, output location, and known limitations.
- Metric limitation notes must accompany automatic metrics and comparison reports.
- Qualitative inspection must remain part of evaluation and thesis evidence.
- Dataset/license awareness must be preserved in artifact planning and public documentation.
- Model phase order is not a final ranking.
- Reproducibility release must not redistribute restricted dataset artifacts.
- User-facing accessibility or human-grade signing claims require separate human evaluation.
- No local paths in public docs.
- No automatic metric as intelligibility proof.
- No hidden change to supervision regime.

## Deferred / Future-Work Scope

Deferred does not mean irrelevant. It means outside the current near-term thesis implementation
scope unless a later audit or scope decision changes it.

- Full 3D avatar generation.
- SMPL-X or parametric body fitting.
- Photorealistic rendering.
- Public manual gloss supervision.
- Full sign-language translation claim.
- Human-grade signing validation.
- New model family outside the audited candidate set.
- Production deployment claims.

## Maintenance Triggers

Update this file only when:

- the audit result changes,
- literature positioning changes,
- source corpus changes materially,
- selection-decision statuses change,
- implementation results change candidate feasibility,
- dataset access or supervision assumptions change,
- thesis scope changes,
- a new workstream is introduced that is not traceable to the audit result.
