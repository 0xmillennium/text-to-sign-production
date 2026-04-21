# Testing

This section explains the current repository test surface, how it is organized, and where to look when you need layer rules, test support guidance, or public contract coverage.

## Current Test Surface

The test suite is organized around four active pytest marker layers: `unit`, `integration`, `e2e`, and `operational`. Most automated runs stay inside the CI-safe `unit`, `integration`, and `e2e` layers, while `operational` remains the manual or external-runtime validation surface documented under `tests/operational/`.

The current automated surface still protects more than notebook execution alone. It covers isolated domain and workflow logic, tiny local stage-to-stage paths, public notebook and docs wording, workflow entrypoint inventory, and thin script or wrapper behavior that the repository still exposes today.

## Testing Documents

- [Test Layers](layers.md): canonical definitions for the `unit`, `integration`, `e2e`, and `operational` markers.
- [Test Infrastructure](infrastructure.md): how `conftest`, fixtures, builders, tiny workspaces, and monkeypatch boundaries are organized.
- [Test Contracts](contracts.md): what public notebook, docs, workflow, and wrapper expectations the CI-safe tests currently protect.

## Common Commands

```bash
python -m pytest
python -m pytest -m unit
python -m pytest -m integration
python -m pytest -m e2e
```
