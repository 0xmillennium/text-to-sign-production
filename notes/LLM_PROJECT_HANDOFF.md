# LLM Project Handoff

## Project Identity

- Repository: `text-to-sign-production`
- Project type: graduation thesis / research engineering repository
- Domain: English text-to-sign-pose production
- Long-term direction: `English text -> pose tokens -> keypoints -> skeleton/avatar`

## Current Status

- Sprint 1 is complete.
- Sprint 2 is complete.
- Sprint 3 is complete as a bounded baseline milestone.
- Dataset Build remains the public implemented stage.
- Baseline Modeling exists as internal downstream baseline evidence, not as the main thesis contribution.
- DVC is not part of the active workflow.
- Broader evaluation, contribution modeling, and playback/demo remain unimplemented.

## Canonical Docs Map

- `docs/execution.md`: operator-facing guide for the supported Colab notebook workflow.
- `docs/repository-map.md`: repository structure map and docs-surface locator.
- `docs/data/index.md`: canonical artifact and data dictionary.
- `docs/testing/index.md`: canonical testing surface and layer reference.
- `docs/decisions/index.md`: ADR index for governing and historical repository decisions.
- `docs/experiments/index.md`: canonical experiment-record index.
- `docs/research/roadmap.md`: authoritative roadmap for planned post-baseline work.
- `docs/research/literature-positioning.md`: literature-informed rationale for the current research direction.

## Completed Dataset Build Surface

- Fixed How2Sign/BFH raw input layout.
- Raw, normalized, filtered, and processed manifests.
- Processed schema `t2sp-processed-v1`.
- Compressed processed `.npz` samples.
- Active filter policy `configs/data/filter-v2.yaml`.
- Reports for assumptions, filtering, data quality, and split integrity.
- Published Dataset Build archives:
  `dataset_build_manifests_reports.tar.zst`,
  `dataset_build_samples_train.tar.zst`,
  `dataset_build_samples_val.tar.zst`,
  `dataset_build_samples_test.tar.zst`.
- Available through the project-wide Colab notebook; lower-level scripts and workflow helpers remain secondary implementation surfaces.

## Completed Baseline Modeling Surface

Sprint 3 is complete as a bounded baseline milestone. This surface is implemented baseline evidence,
not the main thesis contribution.

- Modeling data contracts built on processed Dataset Build manifests and processed `.npz` samples.
- Baseline model surface.
- Masked loss and metric utilities.
- Train/validation runtime.
- Checkpointing plus `run_summary.json`.
- Qualitative panel export.
- Runtime evidence bundle via `baseline_evidence_bundle.json`.
- Runtime record/package outputs, including `baseline_modeling_package.json`.
- Deterministic published archives:
  `baseline_training_outputs.tar.zst`,
  `baseline_qualitative_outputs.tar.zst`,
  `baseline_record_package.tar.zst`.
- Archive/resume behavior follows `reuse -> extract -> run/publish` for training, qualitative, and record/package outputs.

## Supported Workflow

- The single supported operator workflow is the project-wide Colab notebook:
  `notebooks/colab/text_to_sign_production_colab.ipynb`.
- `docs/execution.md` is the operator-facing execution guide.
- Notebook sections cover setup, Dataset Build, Baseline Modeling, qualitative export, record/package assembly, and final artifact inspection.

## Artifact Storage And Run Roots

- Large generated artifacts stay outside Git.
- Fixed Colab raw input root:
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`.
- Fixed Dataset Build artifact root:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`.
- Fixed Colab Baseline Modeling run root:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`.
- Local developer-facing Baseline Modeling default run root:
  `outputs/modeling/baseline-modeling/runs/<run_name>/`.

## Experiment Records

- `docs/experiments/index.md` is the canonical index for durable Markdown experiment records.
- `docs/experiments/template.md` is the canonical template for new experiment records.
- The current Baseline Modeling record is:
  `docs/experiments/2026-04-baseline-modeling-experiment-record-colab-run-190420261845.md`.
- Runtime evidence and package artifacts remain the execution evidence surface.
- The Markdown experiment record is the durable comparison and interpretation surface that cites those runtime artifacts rather than replacing them.

## Key Governing ADRs

- ADR-0002: notebooks remain thin drivers and do not own core project logic.
- ADR-0012: DVC is removed from the active workflow.
- ADR-0015: Baseline Modeling uses stable run roots, deterministic archives, and defined archive/resume evidence behavior.
- ADR-0016: the single supported operator workflow is the Colab notebook, with `docs/execution.md` as the execution guide.
- ADR-0017: structured documentation surfaces use `index.md`, `template.md`, and real leaf records.

## Next Planned Work

- Sprint 3 is complete as the bounded baseline milestone.
- Sprint 4: broader evaluation and error analysis.
- Sprint 5: first thesis contribution, focused on discrete or data-driven pose representation.
- Sprint 6: second thesis contribution, focused on structure-aware or multi-channel improvement.
- Sprint 7: inference, playback, and minimal demo.
- Sprint 8: thesis packaging and final integration.

## Testing Snapshot

- Active test layers are `unit`, `integration`, `e2e`, and `operational`.
- Automated CI-safe coverage centers on unit, integration, and e2e layers.
- Operational checks remain the manual or external-runtime validation surface for real Colab, Drive, large-archive, and resume/publish behavior.
- High-level contract coverage includes domain and workflow logic, notebook/docs wording, workflow inventory, and thin wrapper behavior.
- `docs/testing/index.md` is the canonical reference for testing layers, infrastructure, and contracts.

## Documentation Maintenance Policy

- Execution behavior changes belong in `docs/execution.md`.
- Artifact shape, data shape, or path changes belong in `docs/data/`.
- Testing behavior or test-surface changes belong in `docs/testing/`.
- Decision or governance changes belong in `docs/decisions/`.
- Experiment-record surface changes belong in `docs/experiments/`.
- Research-plan changes belong in `docs/research/roadmap.md`.
- Research-positioning changes belong in `docs/research/literature-positioning.md`.
- Repository-structure or docs-map changes belong in `docs/repository-map.md`.
- Update this handoff only after the owning canonical docs surfaces are updated.
