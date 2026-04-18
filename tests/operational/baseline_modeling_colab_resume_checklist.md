# Sprint 3 Baseline Modeling Colab Resume Checklist

Use this checklist before release-time claims about real Sprint 3 Colab and Google Drive resume
behavior.

## Preconditions

- The notebook is `notebooks/colab/dataset_build_colab.ipynb`.
- Google Drive is mounted at `/content/drive`.
- Dataset Build processed outputs are present in the Colab worktree after the Dataset Build step:
  - `data/processed/v1/manifests/train.jsonl`
  - `data/processed/v1/manifests/val.jsonl`
  - referenced `data/processed/v1/samples/<split>/<sample_id>.npz` files
- The Sprint 3 run root is:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`.
- The default run name is `baseline-default` unless intentionally changed in the notebook.

## Fresh Publish Scenario

1. Remove or rename the existing Drive run root for the selected `<run_name>`.
2. Run the notebook through the Sprint 3 prepare, training, qualitative export, and package cells.
3. Confirm the run root contains:
   - `config/`
   - `checkpoints/`
   - `metrics/`
   - `qualitative/`
   - `record/`
   - `archives/`
4. Confirm `config/baseline.yaml` points `checkpoint.output_dir` at the run's `checkpoints/`
   directory.
5. Confirm `config/source_baseline.yaml` preserves the original operator-provided config.
6. Confirm `archives/` contains:
   - `baseline_training_outputs.tar.zst`
   - `baseline_qualitative_outputs.tar.zst`
   - `baseline_record_package.tar.zst`
7. Confirm `record/baseline_modeling_package.json` states that the package is runtime-side
   baseline evidence, not a formal experiment record.

## Archive-Present, Extracted-Absent Scenario

1. Keep the three archive files under `archives/`.
2. Remove the extracted `checkpoints/`, `metrics/`, `qualitative/`, and `record/` directories.
3. Re-run the Sprint 3 training cell and confirm it reports `train: extract_archive`.
4. Re-run the Sprint 3 qualitative cell and confirm it reports `export-panel: extract_archive`.
5. Re-run the Sprint 3 package cell and confirm it reports `package: extract_archive`.
6. Confirm `archives/` still exists and still contains all three archive files after extraction.

## Extracted-Present Reuse Scenario

1. Keep extracted `checkpoints/`, `metrics/`, `qualitative/`, and `record/` directories in place.
2. Re-run the Sprint 3 training cell and confirm it reports `train: reuse_extracted`.
3. Re-run the Sprint 3 qualitative cell and confirm it reports `export-panel: reuse_extracted`.
4. Re-run the Sprint 3 package cell and confirm it reports `package: reuse_extracted`.
5. Confirm no new model training or qualitative export work starts.

## Notes

- Drive mount failures are operational environment failures, not unit-test failures.
- Hugging Face model availability and Colab accelerator behavior are operational conditions.
- If a config changes, use a new `BASELINE_RUN_NAME` instead of reusing an existing run root.
