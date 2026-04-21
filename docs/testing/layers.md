# Test Layers

The repository uses pytest markers as the canonical layer boundary. The current marker set is `unit`, `integration`, `e2e`, and `operational`, and the default pytest configuration excludes `operational` with `-m 'not operational'`.

## Unit Tests

Unit tests cover isolated, fast, deterministic logic and contract checks. In this repository that includes parsing and validation helpers, small workflow validation, archive and progress helpers, processed manifest and sample contracts, modeling numerics, checkpointing, and package smoke behavior.

Unit tests do not prove multi-module stage wiring, full CLI behavior, notebook wording, or stage-to-stage happy paths. If a test needs a real tiny workspace, multiple collaborating modules, or a public wrapper boundary, it no longer belongs here.

A test belongs in `unit` when it can exercise one narrow behavior with direct inputs, local fakes, or small in-memory fixtures and still stay deterministic.

Unit tests are CI-safe.

```bash
python -m pytest -m unit
```

## Integration Tests

Integration tests cover CI-safe collaboration between modules and public-facing glue. In the current repo that includes data stage chaining on tiny workspaces, local archive and Colab helper behavior, public workflow contract checks, and thin CLI wrappers such as `baseline_modeling.py`, `validate_manifest.py`, `view_sample.py`, and `export_qualitative_panel.py`.

Integration tests do not run real Colab, real Google Drive, large private datasets, or external-runtime validation. They also do not replace full local happy-path runs across stage boundaries when those paths are better represented as `e2e`.

A test belongs in `integration` when multiple modules or a public wrapper need to cooperate, but the scenario can still run locally with tiny inputs, fake collaborators, and no external services.

Integration tests are CI-safe.

```bash
python -m pytest -m integration
```

## End-to-End Tests

End-to-end tests cover tiny local happy paths that cross stage boundaries. In the current repo that means Dataset Build workflow and CLI runs on synthetic How2Sign-like inputs, manifest-driven archive checks, and a CPU-safe baseline-modeling workflow run on tiny processed data with fake runners.

End-to-end tests do not validate the real Colab runtime, Google Drive publish behavior, large archive movement, or heavy model downloads. Those external or operator-facing checks remain outside the normal CI-safe pytest path.

A test belongs in `e2e` when it should exercise a real happy-path entrypoint or workflow chain end to end, but can still do so with tiny local workspaces and CI-safe fakes.

End-to-end tests are CI-safe.

```bash
python -m pytest -m e2e
```

## Operational Tests

Operational tests cover manual or external-runtime validation. In this repository that surface lives under `tests/operational/` and documents real Colab notebook runs, Drive mount checks, large archive transfer and publish behavior, archive-aware resume checks, and real How2Sign smoke validation.

Operational tests do not replace automated `unit`, `integration`, or `e2e` coverage. They are intentionally separate because they depend on a real runtime, private data, or operator action.

A test belongs in `operational` when the behavior only makes sense against the real external environment or when the repo intentionally keeps the validation as a manual checklist instead of a normal pytest case.

Operational tests are not part of the default CI-safe pytest run. Pytest excludes them by default with `-m 'not operational'`.

```bash
python -m pytest -m operational
```

## Layer Selection Rules

- Use `unit` for isolated logic or contract checks that do not need a full workspace or public wrapper.
- Use `integration` for CI-safe collaboration between modules, ops helpers, notebook or docs contract checks, and thin wrapper behavior.
- Use `e2e` for tiny local happy paths that cross stage boundaries through real entrypoints or workflows.
- Use `operational` for manual or external-runtime validation under `tests/operational/`.
- If a test requires real Colab, real Drive, large archives, or private data, it does not belong in the default CI-safe layers.
- If a tiny workspace, builder, or fake collaborator can exercise the behavior directly, prefer that over broad monkeypatching or promoting the test to `operational`.
