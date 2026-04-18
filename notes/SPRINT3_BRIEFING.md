# Sprint 3 Briefing: Baseline Modeling

This file is the Sprint-3-specific companion to `notes/LLM_PROJECT_HANDOFF.md`.

## Sprint 3 Identity

- Sprint name: Baseline Modeling
- Sprint role: first reproducible baseline after Dataset Build
- Sprint maturity: baseline-only, not the main thesis contribution

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Sprint 3 Completion State

Sprint 3 has completed:

- modeling/data contracts
- baseline model surface
- masked loss and metric utilities
- train/validation runtime
- checkpointing and `run_summary.json`
- qualitative panel export
- evidence bundle writing
- stage-oriented baseline workflow CLI
- Colab archive/resume workflow support
- testing hardening to repository standard
- documentation, notebook, ADR, experiment-record, and handoff closure

Sprint 3 did not implement:

- broader evaluation
- contribution modeling
- new metrics beyond the baseline utilities already implemented
- playback/demo/rendering
- Dataset Build behavior changes
- Sprint 4 work

## Frozen Sprint 3 Decisions

Default text backbone:

- `google/flan-t5-base`

Modeling input contract:

- processed Dataset Build manifests only
- processed `.npz` samples only
- no raw OpenPose JSON, raw videos, raw translation CSVs, or raw keypoint archives

Text input:

- raw English sentence from the processed manifest

Default pose targets:

- `body`
- `left_hand`
- `right_hand`

Face:

- retained in the processed schema
- not part of the default Sprint 3 training target

Test split:

- not used for training, tuning, or model selection in Sprint 3

## Public Workflow Surface

Baseline Modeling CLI:

```bash
python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]
```

Workflow entrypoint:

`text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`

Main notebook:

`notebooks/colab/text_to_sign_production_colab.ipynb`

The notebook is project-wide, not one notebook per stage.

## Notebook Requirements

The Phase 7C notebook closure enforces:

- one code cell equals one operational job
- explanatory markdown before every code cell
- no combined environment setup and repo acquisition
- no combined runtime settings and Drive integration
- no combined archive extraction and actual execution
- separate reuse, extract, and run/publish cells from Dataset Build outputs onward

Major sections:

- Section 0: runtime and session setup
- Section 1: Drive mount
- Section 2: repo acquisition and install
- Section 3: Dataset Build outputs
- Section 4: Baseline Modeling training outputs
- Section 5: qualitative panel outputs
- Section 6: record/package outputs
- Section 7: final artifact inspection

## Runtime Artifact Strategy

Baseline run roots contain:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

Archive names:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

Resume priority:

1. Reuse extracted outputs.
2. Extract archived outputs.
3. Run and publish outputs.

This strategy is recorded in ADR-0015.

## Formal Experiment Records

Sprint 3 formal baseline records are documented by:

- `docs/experiments/sprint3-baseline-record-guide.md`
- `docs/experiments/sprint3-baseline-record-template.md`

A formal record should cite:

- Dataset Build provenance
- baseline config files
- `run_summary.json`
- `last.pt` and `best.pt`
- qualitative outputs
- `baseline_evidence_bundle.json`
- `baseline_modeling_package.json`
- deterministic archives under `archives/`

The runtime record package is evidence. The Markdown experiment record is the durable comparison
surface for Sprint 4 and later sprints.

## ADRs Added During Phase 7C

- ADR-0014: adopt Sprint 3 stage-oriented baseline workflow and notebook extension
- ADR-0015: define Sprint 3 runtime artifact, archive/resume, and evidence strategy

## Handoff For Later Sprints

Sprint 4 should consume Sprint 3 records and runtime artifacts to implement broader evaluation and
error analysis.

Sprint 5 and Sprint 6 should compare contribution models against formal Sprint 3 baseline records.

Sprint 7 may use selected baseline or later artifacts for playback/demo once those capabilities are
implemented.

Do not treat Sprint 3 outputs as final thesis-contribution evidence. Treat them as baseline
evidence with documented provenance.
