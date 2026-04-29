# Operational Tests

Operational testing is a first-class layer for validations that need the real external runtime.
These checks are intentionally separate from normal CI-safe pytest runs.

Use this layer for:

- Real Colab notebook execution.
- Real Google Drive mount validation.
- Real How2Sign/BFH smoke runs against the private raw data layout.
- Release-time manual validation that depends on external storage, large files, or runtime speed.

Do not use this layer for local deterministic unit, integration, or e2e tests. Those belong under
`tests/unit/`, `tests/integration/`, and `tests/e2e/`.

Path placeholders:

- `<COLAB_DRIVE_PROJECT_ROOT>` is the mounted Drive project directory that contains this project's
  raw inputs and artifacts.
- `<PROJECT_ARTIFACT_ROOT>` is the artifact root under that Drive project directory.

## Phase 1.1 Transition Status

Phase 1 removed the repository Python archive helper infrastructure. The old operational notes that
validated Python-driven archive copy, extraction, packaging, publish, and archive-aware resume
behavior are obsolete pending the notebook rewrite.

The checklist files in this directory are retained only as transition placeholders. They should not
be used as release signoff for archive movement or publish/resume behavior until the post-Phase-1
notebook architecture is defined. This README intentionally does not define the future notebook
shell extraction standard.

Expected stale archive references remain in notebooks during Phase 1.1 and are deferred to the
notebook rewrite.

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
