# Data Versioning

Sprint 2 uses DVC plus explicit schema versioning to keep the dataset reproducible.

## Canonical Roots

- raw: `data/raw/how2sign/`
- interim: `data/interim/`
- processed: `data/processed/v1/`

## DVC Stages

The implemented `dvc.yaml` stages are:

1. `prepare_raw`
2. `normalize_keypoints`
3. `filter_samples`
4. `export_training_manifest`

Run the full pipeline with:

```bash
dvc repro
```

## Colab Heavy-Execution Model

DVC remains the reproducibility standard for the repository, but the primary heavy-execution path
in Google Colab uses the existing script entry points directly:

```bash
python scripts/prepare_raw.py
python scripts/normalize_keypoints.py
python scripts/filter_samples.py --config configs/data/filter-v1.yaml
python scripts/export_training_manifest.py
```

This avoids the extra hashing cost of `dvc repro` over very large raw trees while keeping the
pipeline logic aligned with the same implemented stages.

## Schema Versioning

The processed dataset contract is versioned separately from the folder name through:

`processed_schema_version = "t2sp-processed-v1"`

This value travels with every processed sample and every processed manifest record so downstream
code can reject incompatible layouts explicitly.

## Access Policy

- Raw files are not the training interface.
- Future models must read the processed manifests.
- Processed manifests identify the canonical `.npz` samples to consume.

## Git And Large Files

- Raw downloads stay out of Git.
- Interim and processed outputs are ignored in Git and are intended to be regenerated.
- Packaged Colab output archives stay out of Git.
- DVC provides the reproducible pipeline definition for rebuilding the dataset state.
