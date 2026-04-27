# Decisions

Architecture Decision Records (ADRs) capture the long-lived architecture, workflow, and
reproducibility decisions that shaped this repository over time.

ADRs are historical decision records, not the active operator guide. For current behavior, use
[Execution](../execution.md), [Data](../data/index.md), [Experiments](../experiments/index.md),
and [Testing](../testing/index.md).

Use the [ADR Template](template.md) when creating or updating a decision record.

## Records

- [ADR-0001: Repository-Centered Workflow](adr-0001-repo-centered-workflow.md): keep the repository as the primary source of truth for project state.
- [ADR-0002: Notebooks As Thin Drivers](adr-0002-notebooks-as-thin-drivers.md): keep notebooks orchestration-only instead of letting them own production logic.
- [ADR-0003: DVC Inclusion](adr-0003-dvc-inclusion.md): record the original decision to include DVC before later workflow changes removed it.
- [ADR-0004: English As Project Language](adr-0004-english-as-project-language.md): standardize English for code, docs, and collaboration surfaces.
- [ADR-0005: Sprint 1 Excludes ML Implementation](adr-0005-sprint-1-excludes-ml-implementation.md): mark Sprint 1 as infrastructure-only rather than ML-implementation work.
- [ADR-0006: Sprint 2 Is Data-First](adr-0006-sprint-2-data-first.md): prioritize building the dataset layer before downstream modeling.
- [ADR-0007: Preserve Official How2Sign Splits](adr-0007-preserve-official-how2sign-splits.md): preserve the observed `train`, `val`, and `test` split boundaries.
- [ADR-0008: Manifest-Driven Processed Dataset](adr-0008-manifest-driven-processed-dataset.md): make processed manifests the canonical model-facing dataset interface.
- [ADR-0009: Use `.npz` For Processed Samples](adr-0009-npz-processed-sample-format.md): store each processed sample as one compressed `.npz` payload.
- [ADR-0010: Face Is Optional In v1](adr-0010-face-optional-in-v1.md): keep face support in the schema without making it mandatory for sample retention.
- [ADR-0011: Adopt Filter V2 One-Hand Policy For Dataset Build](adr-0011-adopt-filter-v2-one-hand-policy-for-dataset-build.md): make usable body plus at least one usable hand the active retention policy.
- [ADR-0012: Remove DVC from the Active Workflow](adr-0012-remove-dvc-active-workflow.md): remove DVC while preserving the current notebook-first reproducibility surface.
- [ADR-0013: Define the Post-Dataset-Build Research Roadmap](adr-0013-define-post-dataset-build-research-roadmap.md): superseded historical roadmap context for the earlier post-Dataset-Build sprint plan.
- [ADR-0014: Adopt Sprint 3 Stage-Oriented Baseline Workflow And Notebook Extension](adr-0014-adopt-sprint-3-stage-oriented-baseline-workflow-and-notebook-extension.md): preserve the earlier workflow framing that was later superseded.
- [ADR-0015: Define Runtime Artifact, Archive/Resume, And Evidence Strategy](adr-0015-define-runtime-artifact-archive-resume-and-evidence-strategy.md): keep the active Baseline Modeling artifact layout, archive names, and resume order.
- [ADR-0016: Adopt Notebook-Only Supported Workflow And Single Execution Guide](adr-0016-adopt-notebook-only-supported-workflow-and-single-execution-guide.md): make the single Colab notebook and `docs/execution.md` the only supported public workflow surface.
- [ADR-0017: Standardize Structured Documentation Surfaces With Index And Template](adr-0017-standardize-structured-documentation-surfaces-with-index-and-template.md): adopt the `index.md` plus `template.md` plus leaf-files documentation pattern as repository documentation architecture.
- [ADR-0018: Roadmap as Normative Sprint Contract](adr-0018-roadmap-as-normative-sprint-contract.md): superseded historical context for sprint-era roadmap governance.
- [ADR-0019: Split Notebook Architecture as Target Workflow](adr-0019-split-notebook-architecture-as-target-workflow.md): superseded historical context for target notebook-surface planning.
- [ADR-0020: Published Model Artifacts as Downstream Interface](adr-0020-published-model-artifacts-as-downstream-interface.md): superseded historical context for model-artifact release planning.
- [ADR-0021: Four-Model 2x2 Comparison Architecture](adr-0021-four-model-2x2-comparison-architecture.md): superseded historical context for the M0/M1/M2/M3 comparison matrix.
- [ADR-0022: Standard Testing Surfaces Before Contribution Implementation](adr-0022-standard-testing-surfaces-before-contribution-implementation.md): superseded historical context for pre-contribution testing-surface sequencing.
- [ADR-0023: Phase-Based Research Governance And Public Documentation Boundaries](adr-0023-phase-based-research-governance-and-public-documentation-boundaries.md): govern the current Phase 0-12 research frame and public documentation boundaries.
