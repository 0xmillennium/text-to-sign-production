# Test Infrastructure

The shared test support is intentionally small and explicit. The repository uses a narrow `conftest`, a small checked-in fixtures set, and reusable support helpers so tests can stay close to the real contracts without needing large local state.

## Conftest Scope

`tests/conftest.py` provides pytest wiring only. It adds the project and `src/` roots to `sys.path`, exposes `fixtures_dir`, creates a temporary project root, patches repo-relative dataset paths, and provides `tiny_dataset_workspace`.

It is not the place for broad hidden setup, global monkeypatches, or test-specific business logic. Scenario-specific behavior stays in the test itself or in `tests/support/`.

## Fixtures

`tests/fixtures/` is the checked-in static fixture surface. It currently holds small files that are easier to understand and reuse as real artifacts than as generated literals:

- tiny OpenPose JSON
- a translation CSV
- a raw manifest example
- a stable expected keys file

Keep this directory small. It is for durable sample files and golden snippets, not for full synthetic workspaces or large generated trees.

## Support Builders And Helpers

`tests/support/` holds reusable programmatic test support. The current modules cover:

- builders for manifests, translations, media, OpenPose payloads, samples, and archive inspection
- shared assertions for JSONL, reports, and processed sample payloads
- path helpers that patch repo-relative dataset roots
- scenario builders for tiny Dataset Build and publish-ready workspaces
- modeling helpers that create tiny processed workspaces, dummy configs, dummy torch-backed models, and fake transformer-loading boundaries

This layer exists to keep test setup explicit and reusable without turning each test into a long setup script.

## Tiny Workspaces And Dummy Artifacts

Most workflow-oriented tests use tiny local workspaces instead of broad mocks. `create_tiny_dataset_workspace` writes a synthetic raw layout with translations, minimal OpenPose frames, videos, and both filter configs. The modeling helpers write tiny processed manifests, `.npz` samples, baseline configs, run layouts, checkpoints, qualitative outputs, and package artifacts.

Dummy artifacts and fake builders are used where the contract depends on a file or output shape but does not need the real heavy runtime. That is how the suite keeps Dataset Build, baseline-modeling, archive, and wrapper tests CI-safe without hiding the public surface they exercise.

## Monkeypatch Boundaries

Monkeypatching stays at explicit boundaries:

- repo-root and dataset path resolution
- CLI `sys.argv`
- archive create, copy, extract, and process-call boundaries
- external transformer import or model-loading boundaries
- torch checkpoint or device boundaries when the real runtime would be heavy or network-dependent

Tests should not monkeypatch core internal behavior when a tiny workspace, support builder, or fake collaborator can exercise the contract directly. Prefer patching the boundary around the dependency, not the logic under test.
