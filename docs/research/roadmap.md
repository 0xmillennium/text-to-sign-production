# Roadmap

This roadmap defines the ideal project sequence from the beginning, using the current research and
engineering understanding as the planning baseline. The project progresses from research planning
to repository infrastructure, dataset construction, base model release, reusable testing,
contribution models, comparative analysis, visual interface work, and thesis packaging.

`C1` and `C2` are contribution roles defined by the Sprint 1 research-planning and
contribution-selection work. Downstream model work uses the frozen comparison architecture:

- `M0 = Base`
- `M1 = Base + C1`
- `M2 = Base + C2`
- `M3 = Base + C1 + C2`

Published artifacts and reusable evaluation surfaces are part of the reproducibility contract.
Validated `M0`, `M1`, `M2`, and `M3` artifacts should be published through Hugging Face or an
equivalent documented artifact surface. Downstream notebooks, tests, demos, and comparison
surfaces should consume published artifacts whenever practical. Informal local checkpoints must not
become the default downstream interface, and every released model should include sufficient
metadata for reproducible loading and use.

Notebooks are thin drivers for defined workflows, not primary implementation locations. Shared
dataset, model, training, evaluation, artifact-loading, and visualization logic should live in
reusable project code. The notebook set should cover a dedicated dataset build / export notebook,
base model training and release notebook, Sprint 5 standard model testing notebook, Sprint 5 visual
testing / visual inspection notebook, `C1` training and release notebook, `C2` training and release
notebook, final combined model training and release notebook, and the later Sprint 10 final visual
interface or demonstration surface. The Sprint 5 visual notebook is an evaluation-support surface;
Sprint 10 owns the final use-facing or thesis-demonstration interface.

## Roadmap Overview

| Sprint | Focus | Primary Output |
| --- | --- | --- |
| [1](#sprint-1) | Research planning and contribution selection | Selected `C1`/`C2` roles and frozen comparison architecture |
| [2](#sprint-2) | Repository infrastructure | Reproducible project foundation |
| [3](#sprint-3) | Dataset build | Reusable processed dataset artifacts |
| [4](#sprint-4) | Base model | Released `M0` artifact |
| [5](#sprint-5) | Testing pipelines | Reusable model evaluation pipeline |
| [6](#sprint-6) | `C1` model | Released `M1` artifact |
| [7](#sprint-7) | `C2` model | Released `M2` artifact |
| [8](#sprint-8) | Combined model | Released `M3` artifact |
| [9](#sprint-9) | 2x2 analysis | Comparative evidence |
| [10](#sprint-10) | Visual interface | Thesis demonstration surface |
| [11](#sprint-11) | Thesis packaging | Final thesis package |

<a id="sprint-1"></a>

## Sprint 1 - Literature Review, Research Planning, and Contribution Selection

### Purpose

Define the research problem, establish the source-grounded project scope, and select the two main
thesis contribution roles. This sprint determines `C1` and `C2`, freezes the `M0/M1/M2/M3`
comparison architecture, and produces the research-planning documents required before downstream
implementation begins.

### Inputs

- Project objective and thesis goals
- Available data assumptions and constraints
- Candidate source papers and method families
- Thesis timeline, evaluation, and writing constraints
- Existing project documentation that affects scope or reproducibility expectations

### Outputs

- Verified bibliography and source corpus
- Source-selection criteria and verified source-corpus boundaries
- Literature positioning for the research problem
- Contribution-selection record
- Selected `C1` and `C2` contribution roles
- Fallback and deferred alternatives, documented without changing the selected roles
- Frozen comparison architecture:
  - `M0 = Base`
  - `M1 = Base + C1`
  - `M2 = Base + C2`
  - `M3 = Base + C1 + C2`
- Research assumptions, source-grounding policy, and rationale for key research framing choices
- Rationale for whether the project is gloss-free or uses another intermediate representation
  strategy
- Conditions under which `C1` or `C2` selection should be revisited
- Roadmap baseline for later implementation

### In Scope

- Defining the research problem and project scope
- Defining source-selection criteria and establishing the verified literature and source corpus
  boundary
- Identifying candidate method families
- Selecting `C1` and `C2` as the two main thesis contribution roles
- Defining fallback and deferred alternatives if needed
- Freezing the four-model comparison structure
- Defining evidence, source-grounding, and reproducibility expectations for later work
- Documenting the rationale for key framing assumptions, including supervision and intermediate
  representation assumptions
- Defining what new evidence, feasibility result, or scope change would require revisiting the
  `C1` or `C2` selection

### Out of Scope

- Training models
- Building the dataset pipeline
- Implementing `C1` or `C2`
- Building demos or visual interfaces
- Replacing research selection with implementation convenience

### Acceptance Criteria

- Research scope is clear enough to guide implementation.
- Source corpus is concrete and verifiable.
- Source-selection criteria and verified source-corpus boundaries are documented.
- `C1` and `C2` contribution roles are selected and recorded.
- The `M0/M1/M2/M3` comparison architecture is frozen.
- Downstream implementation can proceed without reopening research selection.
- Research assumptions and source-grounding rules are documented.
- Key framing assumptions are justified, including whether the project is gloss-free or uses
  another intermediate representation strategy.
- Conditions for revisiting `C1` or `C2` selection are explicit.

### Main Risk

Premature implementation before the research direction is sufficiently grounded.

<a id="sprint-2"></a>

## Sprint 2 - Repository Infrastructure

### Purpose

Establish the technical foundation for reproducible research software, documentation, experiments,
and notebooks.

### Inputs

- Sprint 1 roadmap baseline and research assumptions
- Target repository conventions
- Expected Python package, notebook, documentation, and test workflows
- Reproducibility and artifact-management requirements

### Outputs

- Package structure and `src/` layout
- Test structure and initial test commands
- Linting, formatting, and type-checking configuration
- CI workflow for tests and documentation checks
- Documentation site foundation
- ADR and experiment-log conventions
- Reproducibility conventions for commands, configs, seeds, data versions, and artifacts
- Notebook-as-thin-driver policy

### In Scope

- Establishing reusable project code organization
- Creating test, lint, formatting, type-checking, and CI foundations
- Setting documentation and decision-recording surfaces
- Defining experiment logging conventions
- Defining how notebooks call shared project code rather than carrying core logic
- Documenting reproducibility expectations for later dataset, training, and evaluation work

### Out of Scope

- Dataset processing
- Model training
- Research selection
- Visual interface implementation
- Adding dependencies without a clear infrastructure need

### Acceptance Criteria

- Repository can be tested and documented reproducibly.
- Development conventions are explicit.
- Experiments and decisions have documented recording surfaces.
- Shared code location and notebook responsibilities are clear.
- Later sprints can add dataset, model, evaluation, and interface work without changing the basic
  repository architecture.

### Main Risk

Weak infrastructure causing later research work to rely on ad hoc scripts, undocumented decisions,
or notebook-only implementation.

<a id="sprint-3"></a>

## Sprint 3 - Dataset Build

### Purpose

Turn the raw dataset into reusable model-training artifacts with documented assumptions, quality
checks, and stable split handling.

### Inputs

- Raw dataset files and metadata
- Dataset assumptions from Sprint 1
- Repository infrastructure from Sprint 2
- Required model input and output expectations
- Reproducibility conventions for generated artifacts

### Outputs

- Raw dataset inspection notes
- Dataset manifests
- Filtering policy
- Normalization procedure
- Train, validation, and test split handling
- Processed `.npz` or equivalent samples
- Dataset quality reports
- Reusable dataset artifacts for downstream training and testing
- Dedicated dataset build / export notebook that orchestrates shared project code

### In Scope

- Inspecting raw data layout and metadata
- Generating manifests
- Defining and applying filtering policy
- Normalizing pose and related model inputs
- Preserving split identity and split assumptions
- Creating processed samples in a reusable format
- Running quality checks and documenting dataset assumptions
- Creating a dedicated thin dataset build / export notebook that orchestrates raw-data intake,
  filtering, normalization, and export through shared project code
- Keeping core dataset-processing logic in reusable project code rather than in notebook cells

### Out of Scope

- Model training
- `C1` or `C2` implementation
- Visual UI work
- Changing the research contribution selection

### Acceptance Criteria

- Downstream training uses processed artifacts rather than direct raw-file access.
- Dataset artifacts are reproducible from documented inputs and commands.
- Splits and assumptions are documented.
- Later `M0`, `M1`, `M2`, and `M3` models can reuse the same dataset format.
- A dedicated dataset build / export notebook exists, is separate from model training notebooks,
  and orchestrates reusable project code rather than embedding core processing logic.

### Main Risk

Dataset ambiguity or inconsistent preprocessing undermining every downstream model comparison.

<a id="sprint-4"></a>

## Sprint 4 - Base Model Training and Release

### Purpose

Produce `M0 = Base` as the trusted reference model for all later contribution comparisons.

### Inputs

- Processed dataset artifacts from Sprint 3
- Repository training infrastructure
- Base model configuration
- Reproducibility conventions for experiments and released artifacts
- Base model training and release notebook requirements

### Outputs

- Base model implementation
- Base model training run
- Validation and test outputs
- Epoch-output review notes
- Baseline error inspection
- Dedicated base model training and release notebook
- Packaged `M0` artifact
- Hugging Face release or equivalent documented model publication
- Model card or usage metadata
- Downstream loading instructions

### In Scope

- Implementing the base model
- Training `M0` on the processed dataset artifacts
- Running validation and test procedures
- Reviewing epoch outputs and correctness signals
- Inspecting baseline errors
- Packaging the validated base model artifact
- Publishing `M0` through Hugging Face or an equivalent documented artifact surface
- Documenting reproducible loading and use from the published artifact
- Creating a dedicated base model training and release notebook that loads processed dataset
  artifacts, drives `M0` training, supports validation/test review, and supports artifact packaging
  and release through reusable project code

### Out of Scope

- `C1` implementation
- `C2` implementation
- Final 2x2 comparison
- Final visual interface
- Treating informal local checkpoints as the downstream model interface

### Acceptance Criteria

- Base model training and testing are reproducible from documented configuration, data references,
  and commands.
- Validation and test outputs are documented.
- Epoch behavior has been reviewed.
- Known baseline failure modes are documented.
- The reviewed baseline is accepted as the `M0` reference for downstream comparison.
- `M0` can be loaded from its published artifact.
- Released `M0` artifact includes model metadata, configuration reference, and usage instructions
  for reproducible loading.
- Downstream workflows do not depend on informal local checkpoints.
- Base model training and release notebook loads processed dataset artifacts, supports
  validation/test review and artifact packaging/release, and is a thin driver over reusable project
  code.

### Main Risk

Building later comparisons on a weak, incorrectly trained, or unreleased baseline.

<a id="sprint-5"></a>

## Sprint 5 - Reusable Testing Pipelines

### Purpose

Create model-agnostic testing infrastructure that can evaluate every released model through the
same comparable procedure.

### Inputs

- Processed dataset artifacts and split definitions
- Published `M0` artifact and loading metadata
- Repository evaluation conventions
- Expected comparison metadata requirements
- Standard model testing notebook requirements
- Visual testing / visual inspection notebook requirements

### Outputs

- Terminal-based model testing interface
- Standard model testing notebook
- Visual testing / visual inspection notebook for inspectable pose outputs
- Published-artifact model loading path
- Dataset split selection support
- Standard inference and test execution
- Comparable test reports
- Error-analysis outputs
- Captured metadata for model version, dataset version, config, split, and run identity

### In Scope

- Loading models from published artifact references whenever practical
- Testing `M0`, `M1`, `M2`, and `M3` through the same interface
- Selecting dataset splits for evaluation
- Running standard inference and test procedures
- Producing comparable reports and error-analysis outputs
- Capturing model, dataset, config, split, and run identity metadata
- Keeping the terminal-based testing interface and standard model testing notebook on the same
  shared evaluation logic
- Creating a visual testing / visual inspection notebook that loads released artifacts, generates
  inspectable pose outputs, and supports visual comparison or inspection of model outputs
- Keeping visual testing as a reusable evaluation-support surface, not the final use-facing
  interface

### Out of Scope

- Training new contribution models
- Final user-facing visual interface work; Sprint 5 visual work is limited to reusable
  evaluation-support inspection
- Thesis writing
- Model-specific testing shortcuts that bypass the shared interface

### Acceptance Criteria

- Any released model can be put through the same test interface.
- The pipeline can run at least one released model end-to-end and produce a persisted,
  reproducible test report.
- Outputs are comparable across models.
- Test metadata is sufficient for reproducibility.
- Pipeline supports later 2x2 comparison work.
- Terminal and notebook testing paths use the same shared evaluation logic.
- The standard model testing notebook can run a released model through the standard evaluation path
  and produce persisted, comparable test reports.
- The visual testing / visual inspection notebook can load a released model and produce
  inspectable outputs for model-output review.
- Both testing notebooks use shared project code rather than duplicating core evaluation or
  visualization logic.

### Main Risk

Each model being tested through a different, non-comparable procedure.

<a id="sprint-6"></a>

## Sprint 6 - Main Thesis Contribution 1 Implementation and Release

### Purpose

Produce `M1 = Base + C1` by adding Main Thesis Contribution 1 to the base model while preserving a
clear implementation boundary between the base functionality and `C1`.

### Inputs

- Sprint 1 definition of `C1`
- Published `M0` artifact and base model code
- Processed dataset artifacts
- Reusable testing pipeline from Sprint 5
- Artifact publication and metadata conventions
- `C1` training and release notebook requirements

### Outputs

- `C1` implementation delta from the base model
- Trained `M1` model
- Validation outputs
- Standard test reports from the reusable testing pipeline
- Documented `C1` implementation boundary
- Dedicated `C1` training and release notebook for `M1`
- Packaged `M1` artifact
- Hugging Face release or equivalent documented model publication
- Usage documentation for loading and evaluating `M1`

### In Scope

- Implementing Main Thesis Contribution 1 as `C1`
- Separating base functionality from the `C1` implementation delta
- Training and validating `M1`
- Testing `M1` through the reusable testing pipeline
- Packaging and publishing the released `M1` artifact
- Documenting usage and reproducible loading
- Creating a dedicated `C1` training and release notebook that drives `M1 = Base + C1`
  training/release and remains a thin driver over reusable project code

### Out of Scope

- `C2`
- `M3`
- Final 2x2 analysis
- Final visual interface
- Unrelated baseline changes that obscure the `C1` effect

### Acceptance Criteria

- `M1` is trained.
- `M1` is tested using the standard testing pipeline.
- `C1` implementation delta is documented.
- `M1` is released as a reusable artifact.
- Released artifact includes model metadata, configuration reference, and usage instructions.
- Downstream comparison can consume the released `M1`.
- `C1` training and release notebook delegates shared logic to reusable project code.

### Main Risk

`C1` implementation becoming entangled with unrelated baseline or `C2` changes.

<a id="sprint-7"></a>

## Sprint 7 - Main Thesis Contribution 2 Implementation and Release

### Purpose

Produce `M2 = Base + C2` by adding Main Thesis Contribution 2 to the base model while preserving a
clear implementation boundary between the base functionality and `C2`.

### Inputs

- Sprint 1 definition of `C2`
- Published `M0` artifact and base model code
- Processed dataset artifacts
- Reusable testing pipeline from Sprint 5
- Artifact publication and metadata conventions
- `C2` training and release notebook requirements

### Outputs

- `C2` implementation delta from the base model
- Trained `M2` model
- Validation outputs
- Standard test reports from the reusable testing pipeline
- Documented `C2` implementation boundary
- Dedicated `C2` training and release notebook for `M2`
- Packaged `M2` artifact
- Hugging Face release or equivalent documented model publication
- Usage documentation for loading and evaluating `M2`

### In Scope

- Implementing Main Thesis Contribution 2 as `C2`
- Separating base functionality from the `C2` implementation delta
- Training and validating `M2`
- Testing `M2` through the reusable testing pipeline
- Packaging and publishing the released `M2` artifact
- Documenting usage and reproducible loading
- Creating a dedicated `C2` training and release notebook that drives `M2 = Base + C2`
  training/release and remains a thin driver over reusable project code

### Out of Scope

- `C1` implementation work
- `M3`
- Final 2x2 analysis
- Final visual interface
- Unrelated baseline changes that obscure the `C2` effect

### Acceptance Criteria

- `M2` is trained.
- `M2` is tested using the standard testing pipeline.
- `C2` implementation delta is documented.
- `M2` is released as a reusable artifact.
- Released artifact includes model metadata, configuration reference, and usage instructions.
- Downstream comparison can consume the released `M2`.
- `C2` training and release notebook delegates shared logic to reusable project code.

### Main Risk

`C2` effects being hard to isolate because the implementation is not cleanly separated from the
base model.

<a id="sprint-8"></a>

## Sprint 8 - Final Combined Model Implementation and Release

### Purpose

Produce `M3 = Base + C1 + C2` by integrating the two selected contribution roles into one model
without changing the meaning of the frozen comparison structure.

### Inputs

- Published `M0`, `M1`, and `M2` artifacts and metadata
- Implemented `C1` and `C2` deltas
- Processed dataset artifacts
- Reusable testing pipeline from Sprint 5
- Artifact publication and metadata conventions
- Final combined model training and release notebook requirements

### Outputs

- Combined `C1` and `C2` model implementation
- Compatibility and interaction checks
- Trained `M3` model
- Validation outputs
- Standard test reports from the reusable testing pipeline
- Documented `C1`/`C2` interaction issues or compatibility assumptions
- Dedicated final combined model training and release notebook for `M3`
- Packaged `M3` artifact
- Hugging Face release or equivalent documented model publication
- Usage documentation for loading and evaluating `M3`

### In Scope

- Integrating `C1` and `C2` into the same model
- Checking compatibility and interaction behavior
- Training and validating `M3`
- Testing `M3` through the same pipeline used for `M0`, `M1`, and `M2`
- Packaging and publishing the released `M3` artifact
- Documenting usage and reproducible loading
- Creating a dedicated final combined model training and release notebook that drives
  `M3 = Base + C1 + C2` training/release and remains a thin driver over reusable project code

### Out of Scope

- Reselecting `C1` or `C2`
- Introducing a new main contribution
- Final comparative analysis as the main work
- Changing the base model in ways unrelated to combining `C1` and `C2`

### Acceptance Criteria

- `M3` is trained.
- `M3` is tested through the same pipeline as `M0`, `M1`, and `M2`.
- `C1`/`C2` interaction issues are documented.
- `M3` is released as a reusable artifact.
- Released artifact includes model metadata, configuration reference, and usage instructions.
- Model is ready for 2x2 comparative analysis.
- Final combined model training and release notebook delegates shared logic to reusable project code.

### Main Risk

Combined-model effects being impossible to interpret because integration changes more than `C1`
plus `C2`.

<a id="sprint-9"></a>

## Sprint 9 - 2x2 Comparative Analysis

### Purpose

Compare the frozen four-model set under one standardized protocol and produce thesis-ready
evidence about base performance, individual contribution effects, and combined contribution
behavior.

### Inputs

- Published `M0 = Base` artifact
- Published `M1 = Base + C1` artifact
- Published `M2 = Base + C2` artifact
- Published `M3 = Base + C1 + C2` artifact
- Processed dataset artifacts and split definitions
- Reusable testing pipeline and comparison metadata
- Error-analysis and qualitative inspection surfaces

### Outputs

- Standardized test runs for all four models
- Numerical comparison tables
- Qualitative comparison outputs
- Error-analysis reports
- Interpretation of base performance
- Interpretation of `C1` effect
- Interpretation of `C2` effect
- Interpretation of `C1 + C2` combined effect
- Thesis-ready comparison tables and reports

### In Scope

- Comparing:
  - `M0 = Base`
  - `M1 = Base + C1`
  - `M2 = Base + C2`
  - `M3 = Base + C1 + C2`
- Running standardized testing across all four released artifacts
- Producing numerical and qualitative comparison evidence
- Performing error analysis
- Interpreting individual and combined contribution effects
- Preparing comparison outputs for thesis use

### Out of Scope

- New model implementation
- Contribution reselection
- Final thesis writing as the primary task
- Creating new test procedures that break comparability with earlier runs

### Acceptance Criteria

- All four models are compared under the same test protocol.
- Results are documented in comparable form.
- Contribution effects are interpretable.
- Analysis can support thesis claims.
- Comparison inputs are traceable to released artifacts and documented dataset versions.

### Main Risk

Drawing thesis conclusions from non-comparable runs or incomplete artifacts.

<a id="sprint-10"></a>

## Sprint 10 - Final Visual Interface

### Purpose

Produce the final use-facing visual interface for project use and thesis demonstration while
keeping it subordinate to the research evidence. This sprint builds on released artifacts and
reusable inference/visualization code after Sprint 5 has already established reusable visual
testing and inspection as evaluation support.

### Inputs

- Released model artifacts for `M0`, `M1`, `M2`, and `M3` as needed
- Reusable artifact-loading code
- Text-to-pose inference interface
- Visualization or skeleton-rendering code
- Thesis demonstration requirements

### Outputs

- Final use-facing visual interface for text input and pose output inspection
- Released-artifact loading support
- Skeleton playback or equivalent visual rendering
- Optional model selection among released models when useful
- Final demonstration notebook or interface surface when a notebook best supports reproducible
  thesis demonstration
- Documentation for running the interface from released artifacts

### In Scope

- Loading released model artifacts
- Accepting text input
- Generating pose output
- Rendering skeleton playback or an equivalent visual output
- Optionally selecting between released models when useful
- Showing outputs in a form suitable for thesis demonstration
- Keeping final demonstration notebooks or interface surfaces as thin drivers over shared
  visualization and inference code
- Separating the final use-facing interface from Sprint 5 visual testing / visual inspection
  notebooks

### Out of Scope

- Photorealistic avatar requirements
- Production-grade product deployment
- New research claims
- New contribution selection
- Replacing the Sprint 5 reusable visual testing / visual inspection notebook
- Interface polish that distracts from thesis evidence

### Acceptance Criteria

- User can provide text input and inspect visual output.
- Interface works from released artifacts.
- Interface is stable enough for thesis presentation.
- Interface does not become the main research claim.
- Final demonstration notebook or interface surface uses reusable inference and visualization code.

### Main Risk

Visual interface work expanding into product development and distracting from thesis evidence.

<a id="sprint-11"></a>

## Sprint 11 - Thesis Writing and Final Packaging

### Purpose

Package the project into its final thesis-facing form with a coherent narrative, traceable
evidence, released artifacts, and reproducibility references.

### Inputs

- Research-planning outputs from Sprint 1
- Infrastructure, dataset, model, testing, analysis, and interface documentation
- Released `M0`, `M1`, `M2`, and `M3` artifacts and metadata
- 2x2 comparative analysis results
- Final figures, tables, limitations, and future-work notes

### Outputs

- Thesis writing
- Methodology explanation
- Dataset description
- Model descriptions
- Evaluation results
- 2x2 comparative analysis
- Limitations and future work
- Final figures and tables
- Reproducibility package
- Final documentation polish

### In Scope

- Writing the thesis narrative
- Explaining methodology, dataset, models, and evaluation
- Integrating numerical and qualitative comparison evidence
- Documenting limitations and future work
- Preparing final figures and tables
- Packaging reproducibility references for data, configs, model artifacts, notebooks, and commands
- Aligning documentation with the thesis story

### Out of Scope

- New model family
- New contribution selection
- Major dataset redesign
- New experimental direction
- Reworking results without a documented reproducibility path

### Acceptance Criteria

- Thesis narrative is complete.
- Final evidence is organized and traceable.
- Figures and tables are ready.
- Model artifacts and reproducibility references are documented.
- Documentation and thesis story are aligned.
- Final package points to published artifacts rather than informal local checkpoints whenever
  practical.

### Main Risk

Late packaging revealing missing provenance, incomplete comparisons, or unclear claims.
