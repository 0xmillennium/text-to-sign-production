# Data Versioning

Dataset Build uses DVC plus explicit schema versioning to keep the dataset reproducible.

## Canonical Roots

- raw: `data/raw/how2sign/`
- interim: `data/interim/`
- processed: `data/processed/v1/`

## DVC Stage

The implemented `dvc.yaml` stage is:

`dataset_build`

Run it with:

```bash
dvc repro
```

DVC runs the stage without local archive packaging. The primary terminal workflow is:

```bash
python scripts/dataset_build.py
```

Both routes call the same stage-level workflow in `src/`.

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
- Packaged Dataset Build output archives stay out of Git.
- DVC provides the reproducible pipeline definition for rebuilding the dataset state.
