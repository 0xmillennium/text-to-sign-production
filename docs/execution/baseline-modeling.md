# Baseline Modeling Execution

Implemented internal downstream surface: Baseline Modeling.

Baseline Modeling is a Sprint 3 baseline-only workflow on top of processed Dataset Build outputs.
It does not replace Dataset Build as the current public stage, and it is not the main thesis
contribution.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Public Surface

The public stage-oriented CLI is:

```bash
python scripts/baseline_modeling.py [prepare|train|export-panel|package|all]
```

The reusable workflow entrypoint is:

`text_to_sign_production.workflows.baseline_modeling.run_baseline_modeling`

Lower-level utilities remain available for development and debugging:

- `python scripts/train_baseline.py`
- `python scripts/export_qualitative_panel.py`

`python scripts/evaluate_baseline.py` remains an intentional placeholder until broader evaluation
is implemented in a later phase.

## Inputs

The default Sprint 3 baseline config is:

`src/text_to_sign_production/modeling/config/baseline.yaml`

The default text backbone is `google/flan-t5-base`. The workflow remains modular and
backbone-agnostic through config and model interfaces.

The config must point at processed Dataset Build manifests and `.npz` samples. The workflow
validates that train and validation manifests are present, non-empty, and resolve to existing
processed sample files. It does not read raw How2Sign files.

## Local Run Root

For local CLI use, Baseline Modeling run roots default to:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

Use `--run-name` for separate runs and `--artifact-runs-root` when the run root should live
elsewhere.

Each run root contains:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

## Colab Run Root

The notebook publishes run roots to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

The single project-wide notebook is:

`notebooks/colab/text_to_sign_production_colab.ipynb`

From Baseline Modeling onward, the notebook separates reuse, extract, and run/publish cells for
training outputs, qualitative panel outputs, and record/package outputs.

## Step Semantics

- `prepare` validates processed Dataset Build inputs and writes run config files.
- `train` ensures `checkpoints/` and `metrics/` exist.
- `export-panel` ensures qualitative panel outputs exist.
- `package` ensures record/package outputs exist.
- `all` runs the full chain.

The `config/` directory contains:

- `config/baseline.yaml`: effective config used by the workflow, with checkpoint output pointing at
  the run root.
- `config/source_baseline.yaml`: original operator-provided config.

## Archive-Aware Resume

For training, qualitative panel export, and record/package outputs, the workflow priority is:

1. Reuse already-extracted outputs in the expected run root.
2. Otherwise extract the deterministic archive from `archives/`.
3. Otherwise run the step and publish both extracted outputs and archived outputs.

The deterministic archive names are:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

Archive extraction merges step outputs into the run root without deleting sibling directories such
as `archives/`.

## Runtime Artifacts

Training writes:

- `checkpoints/last.pt`
- `checkpoints/best.pt` when a best checkpoint exists
- `checkpoints/run_summary.json`
- `metrics/run_summary.json`

Qualitative panel export writes:

- `qualitative/panel_definition.json`
- `qualitative/records.jsonl`
- `qualitative/panel_summary.json`
- `qualitative/baseline_evidence_bundle.json`
- `qualitative/references/*.npz`
- `qualitative/predictions/*.npz`

Record/package assembly writes:

- `record/baseline_modeling_package.json`
- `record/baseline_evidence_bundle.json`
- `record/run_summary.json`

The runtime record package is baseline evidence and an input to the formal Sprint 3 experiment
record described in the experiment-record docs.
