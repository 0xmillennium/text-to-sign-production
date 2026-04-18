# Baseline Modeling Execution

Sprint 3 Baseline Modeling is a baseline-only workflow on top of processed Dataset Build outputs.
It is not the main thesis-contribution stage and it does not replace Dataset Build as the current
implemented public project stage.

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

`python scripts/evaluate_baseline.py` remains an intentional placeholder. Broader evaluation is
outside Sprint 3 Phase 7A.

## Inputs

The workflow reads a Sprint 3 baseline YAML config. By default this is:

`src/text_to_sign_production/modeling/config/baseline.yaml`

The default text backbone is `google/flan-t5-base`. The workflow remains modular and
backbone-agnostic through the config and model interfaces.

The config must point at processed Dataset Build manifests and samples. The workflow validates
that the train and validation manifests are present, non-empty, and resolve to existing processed
`.npz` sample files. It does not read raw How2Sign files.

## Steps

- `prepare` validates processed Dataset Build inputs and copies the config into the run layout.
- `train` prepares the run layout, then resolves or runs baseline training.
- `export-panel` prepares the run layout, resolves or runs training, then resolves or runs the
  qualitative panel export.
- `package` prepares the run layout, resolves or runs training and qualitative export, then
  resolves or builds the runtime-side record package.
- `all` is equivalent to the full `prepare -> train -> export-panel -> package` chain.

For local CLI use, runs default to:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

Use `--run-name` for separate runs and `--artifact-runs-root` when the run root should live
elsewhere.

## Colab Drive Layout

The supported Colab notebook uses:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

Each run has this stable structure:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

The `config/` directory contains two config files:

- `config/baseline.yaml` is the effective config used by the workflow. Its
  `checkpoint.output_dir` points at the run's `checkpoints/` directory.
- `config/source_baseline.yaml` preserves the original operator-provided config.

The deterministic archive names are:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

## Archive-Aware Resume

For training, qualitative export, and record packaging, the workflow uses this priority order:

1. Reuse already-extracted outputs in the expected run directory.
2. Otherwise extract the corresponding archive from `archives/`.
3. Otherwise run the step and write both extracted outputs and the archive under the run root.

The extraction path merges archived step outputs into the existing run root without deleting sibling
directories such as `archives/`.

## Colab Notebook

The existing main notebook is:

`notebooks/colab/dataset_build_colab.ipynb`

It covers Dataset Build first, verifies processed outputs, then calls the Sprint 3 baseline
workflow step by step. Notebook cells stay orchestration-only; implementation logic stays in
`src/`.
