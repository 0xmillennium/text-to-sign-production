# Test Contracts

The contract layer protects the public expectations that contributors and operators are most likely to notice when they drift: notebook shape, docs wording, workflow inventory, and the thin scripts and wrappers that still define the current repo surface.

## Public Surface Contracts

The current suite protects more than internal implementation details. It checks that the repository still exposes one main Colab notebook, the expected workflow scripts, and the current stage-oriented public wording around Dataset Build and Baseline Modeling.

The explicit public workflow contract lives in `tests/integration/ops/test_public_workflow_contract.py`. That file verifies the main notebook name and structure, the expected public workflow surface inventory, and the current status wording that appears across durable public text surfaces.

## Notebook And Docs Contracts

CI-safe integration tests catch drift between the main notebook and the docs that reference it. The contract checks verify that public docs still point to `notebooks/colab/text_to_sign_production_colab.ipynb`, preserve the current public status lines, and do not regress to removed notebook names or removed workflow wording.

The notebook contract also checks operator-facing structure inside the notebook itself: required section headings, stable code-cell identifiers, and the split between reuse, extract, and run cells for Dataset Build and Baseline Modeling outputs.

## Script And Wrapper Contracts

The current test surface still protects thin script and wrapper behavior where those scripts remain part of the repo surface. That includes:

- `scripts/baseline_modeling.py` stage dispatch and argument forwarding
- `scripts/validate_manifest.py` wrapper behavior on static manifest input
- `scripts/view_sample.py` manifest and processed-sample boundary checks
- `scripts/export_qualitative_panel.py` argument forwarding, shared defaults, and direct error reporting
- `scripts/dataset_build.py` execution on a tiny end-to-end workspace

These tests do not treat the scripts as the home of business logic. They protect the wrapper contract that callers still rely on while the underlying implementation stays in `src/`.

```bash
python -m pytest tests/integration/ops/test_public_workflow_contract.py
python -m pytest tests/integration/cli/test_baseline_modeling_cli.py
```

## CI-Safe Contract Boundaries

The contract checks are CI-safe because they use local repository files, tiny workspaces, fake builders, and explicit boundary fakes instead of real external services. They are allowed to verify notebook text, docs references, public entrypoint inventory, wrapper argument forwarding, and tiny local workflow runs.

This layer is not a full CI pipeline description. It only explains which contract checks can run safely inside normal automated test runs.

## What Is Not Guaranteed Here

These contract tests do not guarantee real Colab execution, real Google Drive mount or publish behavior, large archive movement, private How2Sign runtime success, or full release and pipeline correctness. Those checks remain outside the normal CI-safe contract layer and are documented under `tests/operational/`.

They also do not guarantee every internal implementation detail. A passing contract layer means the public surface still looks stable from the repository boundary, not that every deeper runtime path has been exercised.
