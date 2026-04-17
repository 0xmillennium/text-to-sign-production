# Operational Tests

Operational testing is a first-class layer for validations that need the real external runtime.
These checks are intentionally separate from normal CI-safe pytest runs.

Use this layer for:

- Real Colab notebook execution.
- Real Google Drive mount validation.
- Real `.tar.zst` archive copy, extract, local packaging, and fixed Drive publish checks on Colab.
- Real How2Sign/BFH smoke runs against the private raw data layout.
- Release-time manual validation that depends on external storage, large files, or runtime speed.

Do not use this layer for local deterministic unit, integration, or e2e tests. Those belong under
`tests/unit/`, `tests/integration/`, and `tests/e2e/`.

Operational Dataset Build checks should use the repo default `configs/data/filter-v2.yaml` unless
the legacy strict `configs/data/filter-v1.yaml` policy is intentionally selected for comparison.
Successful Colab runs publish this archive set under
`/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`:

- `dataset_build_manifests_reports.tar.zst`
- `dataset_build_samples_train.tar.zst`
- `dataset_build_samples_val.tar.zst`
- `dataset_build_samples_test.tar.zst`

Each split sample archive is manifest-driven: `dataset_build_samples_<split>.tar.zst` must contain
exactly the `.npz` files referenced by `data/processed/v1/manifests/<split>.jsonl`, rather than
copying the whole split sample directory.

Normal CI runs:

```bash
python -m pytest
```

That default excludes `operational` tests through pytest configuration. If an operational pytest
test is ever added, it must be marked with `@pytest.mark.operational` and run only by explicit
operator choice:

```bash
python -m pytest -m operational
```

For the current repository state, operational validation is represented as documentation and
manual checklists rather than fake always-on pytest tests.
