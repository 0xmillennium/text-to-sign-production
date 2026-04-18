# LLM Project Handoff

This file is the project-wide briefing artifact for new chats and future implementation sessions.
It should be used with `notes/SPRINT3_BRIEFING.md` for Sprint 3 context.

## Project Identity

- Repository: `text-to-sign-production`
- Project type: graduation thesis / research engineering repository
- Domain: English text-to-sign-pose production
- Long-term direction: `English text -> pose tokens -> keypoints -> skeleton/avatar`
- Project language: English

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Current Status

- Sprint 1 is complete: repository foundation, packaging, tests, docs, CI, ADR support, and
  experiment-record support.
- Sprint 2 is complete: Dataset Build from fixed How2Sign/BFH inputs to processed v1 artifacts.
- Sprint 3 is complete through Phase 7C as a baseline-only surface.
- Dataset Build remains the public implemented stage.
- Baseline Modeling is implemented as internal downstream baseline evidence, not as the main thesis
  contribution.
- DVC is not part of the active workflow.

## Completed Dataset Build Surface

Dataset Build provides:

- fixed How2Sign/BFH raw input layout
- raw, normalized, filtered, and processed manifests
- processed schema `t2sp-processed-v1`
- compressed `.npz` processed samples
- active filter policy `configs/data/filter-v2.yaml`
- reports for assumptions, quality, filtering, and split integrity
- manifest-driven split archives
- CLI: `python scripts/dataset_build.py`
- workflow entrypoint:
  `text_to_sign_production.workflows.dataset_build.run_dataset_build`
- Colab operation through `notebooks/colab/text_to_sign_production_colab.ipynb`

Dataset Build archive names:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

## Completed Sprint 3 Baseline Surface

Sprint 3 Baseline Modeling provides:

- processed manifest and processed `.npz` modeling data contracts
- baseline model surface
- mask-aware loss and metric utilities
- train/validation runtime
- checkpointing and `run_summary.json`
- qualitative panel export
- `baseline_evidence_bundle.json`
- stage-oriented CLI
- Colab archive/resume workflow support
- runtime-side record/package outputs
- formal experiment-record guide/schema and template docs
- ADR-0014 and ADR-0015 documenting workflow and artifact strategy

Baseline Modeling CLI:

```bash
python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]
```

Baseline Modeling workflow entrypoint:

`text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`

Default local run root:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

Default Colab run root:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

Run root structure:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

Baseline archive names:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

Resume priority:

1. Reuse extracted outputs.
2. Extract archived outputs.
3. Run and publish outputs.

## Main Notebook

The single project-wide Colab notebook is:

`notebooks/colab/text_to_sign_production_colab.ipynb`

Notebook rules:

- one code cell equals one operational job
- every code cell has a preceding markdown cell explaining purpose, inputs, outputs, and skip
  conditions
- environment setup and repo acquisition are separate
- runtime settings and Drive integration are separate
- archive extraction and actual execution are separate
- from Dataset Build outputs onward, major steps have separate reuse, extract, and run/publish cells

Notebook section layout:

- Section 0: runtime and session setup
- Section 1: Drive mount
- Section 2: repo acquisition and install
- Section 3: Dataset Build outputs
- Section 4: Baseline Modeling training outputs
- Section 5: qualitative panel outputs
- Section 6: record/package outputs
- Section 7: final artifact inspection

## Storage Policy

Large artifacts stay outside GitHub:

- raw How2Sign data
- processed `.npz` samples
- generated manifests and reports
- Dataset Build archives
- baseline checkpoints
- qualitative outputs
- runtime record/package outputs
- baseline archives

Fixed Colab roots:

- Raw inputs: `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Dataset Build artifacts:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`
- Baseline run roots:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

## Experiment Records

Sprint 3 formal baseline records are Markdown docs under `docs/experiments/`. They cite runtime
artifacts rather than replacing them.

Use:

- `docs/experiments/sprint3-baseline-record-guide.md`
- `docs/experiments/sprint3-baseline-record-template.md`

A formal baseline record should cite:

- Dataset Build manifests or archives
- baseline config files
- `run_summary.json`
- `last.pt` and `best.pt`
- qualitative panel outputs
- `baseline_evidence_bundle.json`
- `baseline_modeling_package.json`
- deterministic archives under `archives/`

The runtime record package is baseline evidence. The formal Markdown record is the comparison
surface for later sprints.

## Key ADRs

- ADR-0002: notebooks as thin drivers
- ADR-0012: remove DVC from the active workflow
- ADR-0013: define the post-Dataset-Build research roadmap
- ADR-0014: adopt Sprint 3 stage-oriented baseline workflow and notebook extension
- ADR-0015: define Sprint 3 runtime artifact, archive/resume, and evidence strategy

## Roadmap

- Sprint 3: Baseline Modeling, complete as baseline-only evidence
- Sprint 4: broader evaluation and error analysis
- Sprint 5: thesis contribution I, discrete/data-driven pose representation
- Sprint 6: thesis contribution II, structure-aware / multi-channel improvement
- Sprint 7: inference/playback/minimal demo
- Sprint 8: thesis packaging/final integration

Do not implement Sprint 4 or later work while closing Sprint 3 documentation.

## Testing Strategy

Test layers:

- `tests/unit/`
- `tests/integration/`
- `tests/e2e/`
- `tests/operational/`

Operational checks are manual and cover real Colab, Drive, large archives, publish/resume, and real
How2Sign behavior.

Phase 7C validation used the `t2sp-py311` conda environment because the active shell may be `base`.

## Documentation Update Practice

When changing execution behavior, update execution docs, storage docs, notebook surfaces, public
workflow references, ADRs when needed, experiment-record docs when evidence surfaces change, and
handoff/briefing notes last.

When wording current maturity, use:

- Current public stage: Dataset Build
- Implemented internal downstream surface: Baseline Modeling
- Not yet implemented: broader evaluation, contribution modeling, playback/demo
