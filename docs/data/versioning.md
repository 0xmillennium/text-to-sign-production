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

## Filter Policy Versioning

The active filter config is `configs/data/filter-v2.yaml`. It changes dataset membership semantics
from the legacy `configs/data/filter-v1.yaml` all-hands-required policy to body-required plus
at-least-one-hand-required.

Regenerating outputs with filter v2 may change processed row counts compared with earlier filter v1
builds. The processed schema version remains `t2sp-processed-v1` because required manifest fields,
sample arrays, and file layout did not change.

## Access Policy

- Raw files are not the training interface.
- Future models must read the processed manifests.
- Processed manifests identify the canonical `.npz` samples to consume.

## Git And Large Files

- Raw downloads stay out of Git.
- Interim and processed outputs are ignored in Git and are intended to be regenerated.
- Packaged Dataset Build output archives stay out of Git.
- DVC provides the reproducible pipeline definition for rebuilding the dataset state.
