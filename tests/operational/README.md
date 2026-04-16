# Operational Tests

Operational testing is a first-class layer for validations that need the real external runtime.
These checks are intentionally separate from normal CI-safe pytest runs.

Use this layer for:

- Real Colab notebook execution.
- Real Google Drive mount validation.
- Real `.tar.zst` archive copy, extract, and publish checks on Colab.
- Real How2Sign/BFH smoke runs against the private raw data layout.
- Release-time manual validation that depends on external storage, large files, or runtime speed.

Do not use this layer for local deterministic unit, integration, or e2e tests. Those belong under
`tests/unit/`, `tests/integration/`, and `tests/e2e/`.

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
