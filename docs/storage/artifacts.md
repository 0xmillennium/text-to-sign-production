# Artifact Storage Strategy

Large generated artifacts are required for reproducibility but do not belong in GitHub. This page
defines the current run roots, extracted outputs, archived outputs, and publish locations.

## Policy

Git stores source, docs, tests, configs, ADRs, and experiment-record templates. Git does not store:

- raw How2Sign downloads
- processed `.npz` samples
- generated manifests and reports
- Dataset Build archives
- Baseline Modeling checkpoints
- qualitative panel outputs
- runtime record/package outputs
- baseline run archives

## Fixed Colab Drive Layout

The supported notebook uses one fixed Drive layout:

- Raw inputs:
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Dataset Build archived outputs:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`
- Baseline Modeling run roots:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

There is no storage-provider switch, no `gdown` path, and no user-edited artifact destination in
the notebook.

## Dataset Build Outputs

Dataset Build extracted outputs live in the worktree:

- `data/interim/raw_manifests/`
- `data/interim/normalized_manifests/`
- `data/interim/filtered_manifests/`
- `data/interim/reports/`
- `data/processed/v1/manifests/`
- `data/processed/v1/reports/`
- `data/processed/v1/samples/<split>/`

Local archived outputs live under:

`data/archives/`

Published archived outputs live under:

`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`

Archive names:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Split sample archives are manifest-driven and use `dataset_build_samples_<split>.tar.zst`.

## Baseline Modeling Run Roots

Local CLI run roots default to:

`outputs/modeling/baseline-modeling/runs/<run_name>/`

Colab run roots publish to:

`/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`

Each run root has this stable structure:

- `config/`
- `checkpoints/`
- `metrics/`
- `qualitative/`
- `record/`
- `archives/`

## Baseline Modeling Extracted Outputs

Training extracted outputs:

- `config/baseline.yaml`
- `config/source_baseline.yaml`
- `checkpoints/last.pt`
- `checkpoints/best.pt` when present
- `checkpoints/run_summary.json`
- `metrics/run_summary.json`

Qualitative extracted outputs:

- `qualitative/panel_definition.json`
- `qualitative/records.jsonl`
- `qualitative/panel_summary.json`
- `qualitative/baseline_evidence_bundle.json`
- `qualitative/references/*.npz`
- `qualitative/predictions/*.npz`

Record/package extracted outputs:

- `record/baseline_modeling_package.json`
- `record/baseline_evidence_bundle.json`
- `record/run_summary.json`

## Baseline Modeling Archived Outputs

Archives are stored under each run root's `archives/` directory:

- `archives/baseline_training_outputs.tar.zst`
- `archives/baseline_qualitative_outputs.tar.zst`
- `archives/baseline_record_package.tar.zst`

Resume behavior is stable:

1. Reuse extracted outputs when required files already exist.
2. Extract archived outputs when the archive exists and extracted outputs are absent.
3. Run and publish when neither extracted outputs nor archived outputs exist.

## Runtime Evidence And Formal Records

The `record/` directory is a runtime-side evidence package. It is not a tracking database and it is
not a replacement for the formal Baseline Modeling experiment record.

The formal record is a Markdown experiment record that cites:

- Dataset Build processed manifests or archives
- effective baseline config
- `run_summary.json`
- `last.pt` and `best.pt`
- qualitative panel outputs
- `baseline_evidence_bundle.json`
- `baseline_modeling_package.json`
- deterministic training, qualitative, and record archives

Later phases should consume the formal record for comparison context and the runtime artifacts for
reproduction or inspection.
