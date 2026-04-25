# LLM Project Handoff

## 1. Project Identity

- Repository name: `text-to-sign-production`
- Project type: graduation thesis / research engineering repository
- Domain: English text-to-sign-pose production
- High-level target: build a reproducible research pipeline for English text input to sign-pose
  model artifacts, evaluation surfaces, comparison evidence, and thesis packaging.
- Long-term modeling direction:
  `English text -> sign-pose representation -> keypoints -> skeleton/avatar`.
- GitHub repository: `https://github.com/0xmillennium/text-to-sign-production`
- GitHub Pages docs: `https://0xmillennium.github.io/text-to-sign-production/`
- Default branch: `master`

Comparison architecture:

- `M0 = Base`
- `M1 = Base + C1`
- `M2 = Base + C2`
- `M3 = Base + C1 + C2`

## 2. Quick Access

- GitHub repository: `https://github.com/0xmillennium/text-to-sign-production`
- GitHub Pages docs: `https://0xmillennium.github.io/text-to-sign-production/`
- Default branch: `master`
- GitHub file URL template:
  `https://github.com/0xmillennium/text-to-sign-production/blob/master/<path>`
- Raw file URL template:
  `https://raw.githubusercontent.com/0xmillennium/text-to-sign-production/master/<path>`
- Published docs URL pattern:
  `https://0xmillennium.github.io/text-to-sign-production/<docs-relative-page>/`

Examples:

- GitHub roadmap file:
  `https://github.com/0xmillennium/text-to-sign-production/blob/master/docs/research/roadmap.md`
- Raw roadmap:
  `https://raw.githubusercontent.com/0xmillennium/text-to-sign-production/master/docs/research/roadmap.md`
- Roadmap page:
  `https://0xmillennium.github.io/text-to-sign-production/research/roadmap/`
- Contribution audit result page:
  `https://0xmillennium.github.io/text-to-sign-production/research/contribution-audit/audit-result/`

## 3. How To Use This Handoff

This file is a context bridge for a new ChatGPT/Codex session. It is meant to help a new assistant
start from current project reality without relying on previous conversation history.

Canonical docs remain the source of truth. Verify claims against the repository before editing,
especially `docs/research/roadmap.md`, `docs/research/contribution-audit/`, `docs/execution.md`,
`docs/data/`, `docs/experiments/`, and `docs/testing/`.

Use `docs/research/roadmap.md` as the normative ideal roadmap. If this handoff and a canonical
surface disagree, inspect the canonical surface first and update this handoff only after the owning
documentation surface is corrected.

## 4. Current Roadmap Baseline

`docs/research/roadmap.md` is the normative ideal roadmap.

The current ideal roadmap has 11 sprints:

1. Literature Review, Research Planning, and Contribution Selection
2. Repository Infrastructure
3. Dataset Build
4. Base Model Training and Release
5. Reusable Testing Pipelines
6. Main Thesis Contribution 1 Implementation and Release
7. Main Thesis Contribution 2 Implementation and Release
8. Final Combined Model Implementation and Release
9. 2x2 Comparative Analysis
10. Final Visual Interface
11. Thesis Writing and Final Packaging

The comparison architecture is:

- `M0 = Base`
- `M1 = Base + C1`
- `M2 = Base + C2`
- `M3 = Base + C1 + C2`

Published artifacts and reusable evaluation surfaces are part of the target reproducibility
contract. Future `M0`, `M1`, `M2`, and `M3` artifacts should be published through Hugging Face or an
equivalent documented artifact surface, with metadata and loading instructions.

## 5. Current Sprint Status Baseline

- Sprint 1: `mostly completed`
- Sprint 2: `completed`
- Sprint 3: `mostly completed`
- Sprint 4: `partially completed`
- Sprint 5-11: `planned`

Current active sprint:

- Sprint 4 - Base Model Training and Release

Highest sprint honestly completed:

- Sprint 2 - Repository Infrastructure

Work sequencing:

- Sprint 4 closure is the active path.
- Sprint 1 closure can proceed in parallel with Sprint 4.
- Sprint 3 notebook closure can proceed in parallel with Sprint 4.
- C1/C2 implementation should not start before Sprint 4 is closed and Sprint 5 testing surfaces
  exist.
- Do not preserve older sprint numbering from stale plans.

## 6. Where To Look For What

- Project overview: `README.md`
- Published docs home: `docs/index.md`
- Execution / Colab workflow: `docs/execution.md`
- Repository structure: `docs/repository-map.md`
- Roadmap / sprint plan: `docs/research/roadmap.md`
- Literature positioning: `docs/research/literature-positioning.md`
- Bibliography: `docs/research/bibliography.md`
- Contribution audit result: `docs/research/contribution-audit/audit-result.md`
- Candidate decisions: `docs/research/contribution-audit/selection-decisions/`
- Dataset docs: `docs/data/`
- Experiments: `docs/experiments/`
- Testing docs: `docs/testing/`
- ADRs: `docs/decisions/`
- Source code: `src/text_to_sign_production/`
- Scripts / CLI wrappers: `scripts/`
- Configs: `configs/`
- Notebooks: `notebooks/`

## 7. Docs URL Mapping

GitHub Pages publishes `docs/` paths under:

`https://0xmillennium.github.io/text-to-sign-production/`

Mapping rules:

- `docs/index.md` maps to the docs root.
- Other `docs/<path>.md` files usually map to
  `https://0xmillennium.github.io/text-to-sign-production/<path-without-md>/`.
- Directory `index.md` files usually map to the directory URL.

Examples:

- `docs/index.md`
  -> `https://0xmillennium.github.io/text-to-sign-production/`
- `docs/research/roadmap.md`
  -> `https://0xmillennium.github.io/text-to-sign-production/research/roadmap/`
- `docs/research/literature-positioning.md`
  -> `https://0xmillennium.github.io/text-to-sign-production/research/literature-positioning/`
- `docs/research/contribution-audit/audit-result.md`
  -> `https://0xmillennium.github.io/text-to-sign-production/research/contribution-audit/audit-result/`

## 8. Research Decisions Snapshot

Current contribution-audit result, from `docs/research/contribution-audit/audit-result.md`:

- Current selected `C1`: `Dynamic VQ Pose Tokens`
- Current selected `C2`: `Channel-Aware Loss Reweighting`
- Current fallback: `Articulator-Partitioned Latent Structure`
- Current deferred candidate: `Motion Primitives Representation`

Caution:

- These are current Sprint 1 research-planning outcomes.
- Sprint 1 is only `mostly completed`.
- New source evidence, stronger source-selection criteria, or a better gloss-free/intermediate
  representation rationale may require revisiting C1/C2 before implementation.
- Do not frame C1/C2 as immutable.
- Do not reopen contribution selection casually; revisit it only if Sprint 1 closure evidence
  requires it.

## 9. Current Repo Reality vs Target Roadmap

Current repo reality:

- One project-wide Colab notebook exists:
  `notebooks/colab/text_to_sign_production_colab.ipynb`.
- Dataset Build pipeline is implemented.
- Baseline implementation and one recorded Colab run exist.
- Checkpointing, qualitative export, runtime records, and baseline package evidence exist.
- No released/loadable `M0` artifact exists yet.
- No standard released-artifact testing pipeline exists yet.
- No dedicated notebook split exists yet.
- The existing baseline run is useful evidence, but it is not a released `M0`.
- Local checkpoints are not published artifacts.

Target roadmap:

- Dedicated dataset build/export notebook.
- Dedicated base model training/release notebook.
- Standard model testing notebook.
- Visual testing/inspection notebook.
- Dedicated C1 training/release notebook.
- Dedicated C2 training/release notebook.
- Dedicated final combined model training/release notebook.
- Published artifacts for `M0`, `M1`, `M2`, and `M3`.

## 10. First Four Sprint Closure Work

Sprint 1 closure:

- Document source-selection criteria.
- Verify source-corpus boundaries.
- Document the gloss-free / intermediate representation rationale.
- Define C1/C2 revisit conditions.
- Clean up contribution-audit index wording if stale.
- Consider expanding source evidence before C1/C2 implementation if current corpus boundaries
  remain too narrow.

Sprint 2 closure:

- No material closure task.
- Sprint accepted as completed.

Sprint 3 closure:

- Create a dedicated dataset build/export notebook.
- Keep the dataset notebook separate from model training notebooks.
- Ensure the notebook calls shared project code rather than owning core logic.
- Optionally verify external artifact/archive availability.

Sprint 4 closure:

- Train and accept `M0`.
- Run validation/test evaluation.
- Review epoch behavior.
- Document known failure modes / baseline error analysis.
- Create a dedicated base model training/release notebook.
- Package `M0`.
- Publish `M0` to Hugging Face or an equivalent documented artifact surface.
- Add model card / metadata / config / usage instructions.
- Prove downstream workflows can load `M0` from the published artifact.

## 11. Notebook Architecture

| Notebook surface | Sprint | Purpose |
|---|---:|---|
| Dataset build/export notebook | Sprint 3 | Build reusable dataset artifacts |
| Base model training/release notebook | Sprint 4 | Train, evaluate, package, and publish `M0` |
| Standard model testing notebook | Sprint 5 | Run non-visual comparable test reports |
| Visual testing / visual inspection notebook | Sprint 5 | Inspect model outputs visually for evaluation support |
| C1 training/release notebook | Sprint 6 | Train and publish `M1` |
| C2 training/release notebook | Sprint 7 | Train and publish `M2` |
| Final combined model training/release notebook | Sprint 8 | Train and publish `M3` |
| Final visual interface/demo surface | Sprint 10 | Thesis-facing visual interface |

Notebook rules:

- Notebooks are thin drivers.
- Shared logic belongs in `src/`.
- Notebooks should not become the primary implementation location.
- Notebook cells should orchestrate reusable project code, not duplicate dataset, modeling,
  evaluation, packaging, or visualization logic.

## 12. Artifact / Release Discipline

- Large generated artifacts stay outside Git.
- Drive/run-root paths remain current execution surfaces.
- Future model artifacts should be published through Hugging Face or an equivalent documented
  artifact surface.
- Downstream notebooks, tests, and demos should consume published artifacts where practical.
- Local checkpoints must not become the default downstream interface.
- Do not treat local checkpoints, Drive run roots, or runtime archives as released `M0/M1/M2/M3`
  artifacts unless a documented release surface exists.

Current useful run-root paths:

- Raw How2Sign root:
  `/content/drive/MyDrive/text-to-sign-production/raw/how2sign/`
- Dataset Build artifact root:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/dataset-build/processed-v1/`
- Baseline Modeling run root:
  `/content/drive/MyDrive/text-to-sign-production/artifacts/baseline-modeling/runs/<run_name>/`
- Local modeling run root:
  `outputs/modeling/baseline-modeling/runs/<run_name>/`

## 13. Current Implemented Surfaces

### Dataset Build Surface

The Dataset Build implementation is strong. Manifests, filtering, normalization, reports, artifact
docs, archive docs, and processed `.npz` sample contracts exist. Processed artifacts are external
to Git and documented under `docs/data/`.

Sprint 3 remains `mostly completed` because the dedicated dataset build/export notebook is missing.
The current project-wide Colab notebook can run Dataset Build, but it does not satisfy the target
dedicated-notebook architecture by itself.

### Existing Baseline Modeling Evidence

Baseline implementation exists. One recorded Colab run exists:

`docs/experiments/2026-04-baseline-modeling-experiment-record-colab-run-190420261845.md`

Checkpointing, qualitative export, runtime records, package artifacts, and useful evidence exist.
This evidence supports Sprint 4 work, but it is not a released/loadable `M0`. Sprint 4 remains
`partially completed`.

## 14. Experiment Records

- `docs/experiments/index.md` is the canonical index for durable Markdown experiment records.
- `docs/experiments/template.md` is the canonical template for new experiment records.
- Current Dataset Build record:
  `docs/experiments/2026-04-dataset-build-experiment-record-filter-v2-full-validation.md`
- Current baseline experiment record:
  `docs/experiments/2026-04-baseline-modeling-experiment-record-colab-run-190420261845.md`
- Runtime evidence and package artifacts remain supporting evidence.
- Experiment records are durable evidence surfaces, not proof that Sprint 4 is completed.
- The current baseline record documents a one-epoch Colab run and its supporting runtime
  artifacts; it does not establish a released/loadable `M0`.

## 15. Key Governing ADRs

- ADR-0002: notebooks remain thin drivers and do not own core project logic.
- ADR-0012: DVC is removed from the active workflow.
- ADR-0015: Baseline Modeling uses stable run roots, deterministic archives, and defined
  archive/resume evidence behavior.
- ADR-0016: the single supported operator workflow is the Colab notebook, with `docs/execution.md`
  as the execution guide.
- ADR-0017: structured documentation surfaces use `index.md`, `template.md`, and real leaf records.
- ADR-0018: `docs/research/roadmap.md` is the normative ideal sprint contract; sprint status is
  assessed separately.
- ADR-0019: the current single Colab workflow is transitional; the target is split thin-driver
  notebooks.
- ADR-0020: downstream model use should consume published artifacts rather than informal local
  checkpoints.
- ADR-0021: thesis-facing model comparison uses `M0/M1/M2/M3`.
- ADR-0022: released `M0` and reusable testing surfaces should precede full C1/C2 implementation.

The current single-notebook workflow is the present supported execution surface, while the roadmap
target, formalized by ADR-0019, is a split notebook architecture. Treat this as a transition state,
not a contradiction: current execution should follow the supported Colab workflow, but future
sprint closure should move toward the dedicated notebook surfaces defined in the roadmap.

ADRs are historical and governing context. Current operator behavior should still be checked
against `docs/execution.md`, `docs/data/`, `docs/testing/`, and the current source code before
making changes.

## 16. Testing Snapshot

Existing repo test suite:

- Active pytest layers are `unit`, `integration`, `e2e`, and `operational`.
- CI-safe coverage focuses on `unit`, `integration`, and `e2e`.
- Operational checks cover real Colab, Drive, large archives, private data, and external runtime
  behavior.
- Existing tests cover dataset workflow logic, baseline helpers, archive/resume behavior, notebook
  and docs wording, public workflow contracts, and thin wrappers.

Existing baseline helpers:

- Baseline training helpers, checkpoint loading, qualitative export, evidence bundles, and package
  helpers exist.
- Existing qualitative helpers are useful supporting surfaces, but they are not the final reusable
  testing pipeline.

Missing Sprint 5 surfaces:

- No standard released-artifact model testing pipeline exists yet.
- No standard model testing notebook exists yet.
- No visual testing / visual inspection notebook exists yet.
- Existing baseline qualitative export does not replace the target Sprint 5 reusable testing
  pipeline.

## 17. Operating Rules For A New Chat

Do:

- Treat `docs/research/roadmap.md` as the normative sprint contract per ADR-0018.
- Treat the split notebook architecture as the target workflow per ADR-0019.
- Treat published artifacts as the downstream model interface per ADR-0020.
- Treat `M0/M1/M2/M3` as the comparison architecture per ADR-0021.
- Verify repo files before claiming completion.
- Treat Sprint 4 as active work.
- Close Sprint 1 and Sprint 3 gaps in parallel if needed.
- Preserve notebook thin-driver discipline.
- Preserve artifact publication discipline.
- Keep large generated artifacts outside Git.
- Use canonical docs as the first place to inspect before changing implementation.

Do not:

- Treat old sprint numbering as authoritative.
- Treat the existing baseline run as released `M0`.
- Treat local checkpoints as published artifacts.
- Treat the single Colab notebook as satisfying all future notebook requirements.
- Start C1/C2 work before Sprint 4 and Sprint 5 dependencies are closed, per ADR-0022.
- Reopen contribution selection casually.
- Invent Hugging Face releases, notebooks, test reports, model cards, release metadata, or artifact
  publications.

## 18. Immediate Next Actions

1. Active path - execute Sprint 4 closure:
   - train/validate/test accepted `M0`
   - write baseline error analysis
   - create dedicated base model training/release notebook
   - publish/load `M0`
   - document metadata/usage

2. Parallel closure - close Sprint 1 research-planning gaps:
   - source-selection criteria
   - source-corpus boundaries
   - gloss-free/intermediate-representation rationale
   - C1/C2 revisit conditions

3. Parallel closure - close Sprint 3 notebook gap:
   - create dedicated dataset build/export notebook

4. After Sprint 4:
   - build Sprint 5 standard and visual testing pipelines

## 19. Documentation Maintenance Policy

- Execution changes -> `docs/execution.md`
- Data/artifact changes -> `docs/data/`
- Testing changes -> `docs/testing/`
- Research-plan changes -> `docs/research/roadmap.md`
- Source/citation changes -> `docs/research/bibliography.md`
- Contribution decision changes -> `docs/research/contribution-audit/`
- Experiment records -> `docs/experiments/`
- ADR/governance changes -> `docs/decisions/`
- Repository structure -> `docs/repository-map.md`
- Research-positioning changes -> `docs/research/literature-positioning.md`
- Roadmap governance decisions should be recorded in ADRs before being treated as stable project
  policy.
- Update this handoff only after canonical surfaces are updated.
