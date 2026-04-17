# Getting Started

## Prerequisites

- Python 3.11 or newer
- `git`

## Install

Full contributor setup:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Or install one extra set at a time:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
```

## Smoke Check

```bash
python -c "from text_to_sign_production import smoke_check; print(smoke_check())"
```

Expected output:

```text
t2sp-smoke-ok
```

## Dataset Build Start Path

For local Dataset Build runs, use the primary CLI against the canonical raw layout under
`data/raw/how2sign/`:

```bash
python scripts/dataset_build.py
```

The CLI uses the active `configs/data/filter-v2.yaml` policy by default, writes manifests and
reports, and packages local archives under `data/archives/`.

To build without local archive packaging, run:

```bash
python scripts/dataset_build.py --no-package
```

For heavy Colab execution, use the fixed runner notebook:

```text
notebooks/colab/dataset_build_colab.ipynb
```

The supported Colab workflow publishes archives only to
`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`.

## Next Steps

- Read the development setup guide before contributing.
- Read the [Dataset Build execution guide](execution/dataset-build.md) before running the full data
  pipeline.
- Use ADRs for non-trivial architectural decisions.
- Record future empirical work with the experiment log template.
