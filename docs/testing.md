# Testing Architecture

The test suite is layered around the repository's real risk surfaces: domain correctness, stage
correctness, workflow correctness, and operational runtime validation.

## Test Layers

- `tests/unit/` contains isolated, fast, deterministic logic tests. These cover OpenPose parsing,
  filtering policy validation, manifest validation, report guards, progress helpers, archive helper
  internals, package smoke behavior, small workflow validation, and Sprint 3 baseline contracts such
  as processed manifest/sample loading, backbone boundaries, training numerics, qualitative
  artifacts, run layout, resume decisions, and archive member validation.
- `tests/integration/` contains CI-safe tests where multiple modules collaborate. These cover data
  stages on tiny workspaces, local archive operations, Colab workflow helpers with fake local
  paths, public notebook/script contracts, thin CLI wrappers, and Sprint 3 baseline workflow
  chaining with mocked runners.
- `tests/e2e/` contains local happy-path tests that cross Dataset Build stage boundaries. These use
  tiny synthetic How2Sign-like inputs and run raw manifest creation, normalization, filtering, final
  export, and CLI/workflow entrypoints without real external services. Sprint 3 has a CPU-safe
  e2e-like workflow test on tiny processed data with fake runners so automated tests do not
  download Hugging Face assets.
- `tests/operational/` contains manual or external-runtime validation guidance. Real Colab,
  Google Drive mount, large archive transfer, publish, Sprint 3 archive/resume checks, and real
  How2Sign smoke checks belong there, not in normal CI-safe pytest runs.

The active Dataset Build policy under test is `configs/data/filter-v2.yaml`: body must be usable
and at least one hand must be usable. `configs/data/filter-v1.yaml` is covered as the legacy strict
policy only when a test intentionally selects it.

Packaging regression coverage is independent from filter-policy coverage. Execution-facing tests
verify local archive creation, fixed Colab publish behavior, and manifest-driven
`dataset_build_samples_<split>.tar.zst` membership so sample archives match processed manifests
instead of unfiltered whole-directory contents.

Sprint 3 workflow tests verify the public `scripts/baseline_modeling.py` surface, stable run layout,
archive/extract/reuse decision order, config and processed-output validation, and chaining across
training, qualitative export, and runtime package assembly. Automated tests use tiny processed
Dataset Build workspaces, fake runners, dummy torch models, or dummy artifacts and must not perform
network-dependent model downloads.

## Fixtures And Support

`tests/fixtures/` holds small static examples when a checked-in file is clearer than generated
data: tiny OpenPose JSON, translation TSV, JSONL manifest examples, and stable expected values.

`tests/support/` holds reusable programmatic test infrastructure: fake OpenPose payload builders,
translation and manifest builders, `.npz` sample builders, tiny workspace scenarios, archive
helpers, Sprint 3 modeling workflow builders, torch-only dummy baseline helpers, transformer
loading fakes, path patching helpers, and shared assertions.

Sprint 3 tests should use support builders for synthetic processed manifests, samples, configs,
checkpoints, qualitative artifacts, and run directories. Keep setup literals local only when the
test is directly asserting that literal contract. Monkeypatching should stay at explicit boundaries:
repo-root path resolution, CLI `sys.argv`, external transformer imports, torch checkpoint/device
loading, archive create/extract process calls, and fake model construction used to prevent Hugging
Face downloads. Do not monkeypatch core internal behavior when a tiny workspace, fake collaborator,
or support builder can exercise the contract directly.

`tests/conftest.py` is intentionally narrow. It provides pytest wiring only: import path setup and
fixture composition such as `fixtures_dir`, `tmp_project_root`, `patched_dataset_roots`, and
`tiny_dataset_workspace`.

## Commands

Normal CI-safe test run:

```bash
python -m pytest
```

Focused layer runs:

```bash
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m e2e
python -m pytest -m "unit or integration or e2e"
```

Operational validation is excluded by default. If operational pytest tests are ever added, run them
only by explicit operator choice:

```bash
python -m pytest -m operational
```

For the current repository, operational validation is documented under `tests/operational/`.
