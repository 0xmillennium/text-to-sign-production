# Asama 0 SRC Contract Freeze and Authority Map

Date: 2026-05-25

Scope: architecture-boundary and public-contract freeze only. This document does not authorize feature implementation, notebook refactors, generated-pose schema changes, gate/tier/debug behavior changes, or model behavior changes.

## 0A. Notebook/Public API Freeze

Notebooks inspected:

- `notebooks/gate.ipynb`
- `notebooks/tier.ipynb`
- `notebooks/debug_sample.ipynb`
- `notebooks/model.ipynb`
- `notebooks/test_model.ipynb`

No notebook files were modified.

### Notebook imports

- All notebooks import `Path` from `pathlib`.
- All notebooks import `display_review_sections` from `text_to_sign_production.workflows.foundation.review`.
- `gate.ipynb` imports `GateWorkflow`, `GateWorkflowConfig` from `text_to_sign_production.workflows.gate`.
- `tier.ipynb` imports `TierWorkflow`, `TierWorkflowConfig` from `text_to_sign_production.workflows.tier`.
- `debug_sample.ipynb` imports `DebugSampleRequest`, `DebugWorkflowConfig`, `SampleDebugWorkflow` from `text_to_sign_production.workflows.debug`.
- `model.ipynb` imports `ModelWorkflow`, `ModelWorkflowConfig` from `text_to_sign_production.workflows.model` and `ensure_model_provider_registered` from `text_to_sign_production.modeling.candidates.bootstrap`.
- `test_model.ipynb` imports `CheckpointPolicy`, `TestModelRequest`, `TestModelWorkflow`, `TestModelWorkflowConfig` from `text_to_sign_production.workflows.test_model`.

### Constructor signatures that must not break

- `GateWorkflowConfig(project_root, drive_project_root, splits, gates_config_relpath=..., person_selection_policy=...)`
- `GateWorkflow(config)`
- `TierWorkflowConfig(project_root, drive_project_root, splits, filters_config_relpath=..., tier_config_relpath=...)`
- `TierWorkflow(config)`
- `DebugWorkflowConfig(project_root, drive_project_root, runtime_root, gates_config_relpath=..., filters_config_relpath=..., tiers_config_relpath=..., person_selection_policy=..., debug_reports_relroot=...)`
- `DebugWorkflowConfig.from_roots(project_root=..., drive_project_root=..., runtime_root=None)`
- `DebugSampleRequest(debug_splits, target_sentence_name)`
- `SampleDebugWorkflow(config)`
- `ModelWorkflowConfig(project_root, drive_project_root, model_key, manifest_family, model_config_relpath=None, run_mode=..., run_name=None, auxiliary_objectives=...)`
- `ModelWorkflow(config)`
- `TestModelWorkflowConfig(project_root, drive_project_root, runtime_root=None)`
- `TestModelRequest(model_run_name, checkpoint_policy, target_sentence_name)`
- `TestModelWorkflow(config)`

### Config/request fields used by notebooks

- Gate: `project_root`, `drive_project_root`, `splits`, `gates_config_relpath`, `person_selection_policy`.
- Tier: `project_root`, `drive_project_root`, `splits`, `filters_config_relpath`, `tier_config_relpath`.
- Debug: `project_root`, `drive_project_root`, `runtime_root`; request uses `debug_splits`, `target_sentence_name`.
- Model: `project_root`, `drive_project_root`, `model_key`, `manifest_family`, `model_config_relpath`, `run_mode`, `auxiliary_objectives`; notebook reads `run_name`, `model_key.value`, `manifest_family.family_id`, `run_mode.value`.
- Test-model: `project_root`, `drive_project_root`; request uses `model_run_name`, `checkpoint_policy`, `target_sentence_name`; notebook reads `test_request.checkpoint_policy` and `test_request.target_sentence_name`.

### Public method sequences assumed by notebooks

- Gate: `plan_runtime` -> `review_runtime_plan` -> `validate_runtime_plan` -> `restore_runtime` -> `review_runtime_restore` -> `verify_runtime` -> `review_runtime_verification` -> `execute_processing` -> `review_processing` -> `review_outputs` -> `write_reports` -> `review_written_artifacts` -> `review_final_operator_summary` -> `build_publish_plan` -> publish review/execute/verify/result review.
- Tier: same restore/verify/process/report/publish lifecycle, plus `review_calibration` and `review_written_reports`.
- Debug: `review_request` -> `build_restore_plan` -> restore plan review/validation/execute/review/verify -> `resolve_target_sample` -> `collect_sample_evidence` -> `debug_gate` -> `debug_tier` -> `debug_visualization` -> `write_debug_reports` -> publish plan/publish/verify -> `build_final_result` -> `print_final_result`.
- Model: `plan_runtime` -> runtime review/validation -> `run_preflight` -> smoke protocol -> restore/verify -> `resolve_research` -> `resolve_provider` -> `load_provider_config` -> `plan_model_stages` -> `execute_model_stages` -> artifact/materialization/report/validation/objective methods -> publish lifecycle -> `build_final_result` -> `review_final_operator_summary`.
- Test-model: `review_request` -> `run_preflight` -> smoke protocol -> restore plan/validation/restore/verify -> model run resolution -> checkpoint selection -> target resolution -> evidence -> single-sample inference -> visualization -> reports -> publish lifecycle -> final result/review.

## 0B. Authority Ledger

### core

- Owns: canonical identities, base immutable DTOs, generic filesystem, root discovery, digest helpers, generic text/JSON IO helpers.
- May call: Python standard library only and same-level core modules.
- Must not own: gate/tier policy, artifact topology, model candidate behavior, workflow lifecycle, notebook behavior.
- Must not call: data, artifacts, modeling, workflows.
- Extension points: stable identity enums, digest helpers, generic IO.
- Risk: none found for upward imports.

### core.progress

- Owns: progress specs, sessions, sinks.
- May call: core/stdlib.
- Must not own: domain state/results/decisions.
- Must not call: data/artifacts/modeling/workflows.
- Extension points: optional progress sinks for long-running logic.
- Risk: raw progress loops may still exist in candidate/training code; track for later where long-running.

### artifacts.store

- Owns: physical topology, path refs, relative path validation, store validation, resolution helpers.
- May call: core.
- Must not own: manifest content parsing, domain decisions, model training.
- Must not call: workflows/model internals.
- Extension points: generated-pose/model output path topology remains here.
- Risk: workflow-specific layouts hand-build some report/output roots; acceptable boundary when workflow-owned runtime/publish layout, but candidate code should use topology/context.

### artifacts.catalog

- Owns: logical sample catalog loading/lookup and sample handles.
- May call: artifacts.store, core, data.dataset manifest readers.
- Must not own: gate/tier policy, model tensor construction.
- Must not call: workflows.
- Extension points: catalog lookup should be reused by candidate-agnostic validation where sample handles are needed.
- Risk: artifacts -> data.dataset is an intentional schema-reader dependency for catalog loading; acceptable.

### data.dataset

- Owns: prepared sample payload schema, passed/dropped/tiered manifest schema and IO, dataset build/validation primitives.
- May call: core.
- Must not own: gate/tier policy, model tensor shape contracts, artifact topology.
- Must not call: workflows.
- Extension points: prepared-sample schema remains source of truth.
- Risk: none found for workflow imports.

### data.gate

- Owns: source discovery/matching, pose parsing, anatomy/tensor/tracking/person selection, gate policy evaluation, gate reports.
- May call: core and data.dataset.
- Must not own: runtime restore/publish, notebook UI, artifact topology, tier policy, model pipeline.
- Must not call: workflows.
- Extension points: gate workflow must keep orchestrating these functions instead of reimplementing.
- Risk: none found for workflow imports.

### data.tier

- Owns: tier context/facts, quality metric families, leakage detection, admission/filter policy, tier reports/manifests.
- May call: core, data.dataset, data.gate pose/context facts as needed.
- Must not own: gate pass/drop policy, model candidate evaluation, workflow publish behavior.
- Must not call: workflows.
- Extension points: tier metrics/leakage remain source for debug diagnostics.
- Risk: none found for workflow imports.

### workflows.foundation

- Owns: operation/executor/review contracts, review rendering, provenance/receipts.
- May call: core.
- Must not own: domain logic, artifact topology, model behavior, gate/tier decisions.
- Must not call: candidate internals.
- Extension points: public review display API used by notebooks.
- Risk: old generic write helpers were too high-level for modeling; fixed by moving generic IO to `core.io` and re-exporting for compatibility.

### workflows.gate

- Owns: gate operator config/result/runtime/publish DTOs, restore/verify/validate, layout binding, orchestration, reviews, publish lifecycle, `GateWorkflow`.
- May call: core, artifacts.store/catalog as needed, data.gate, data.dataset, workflows.foundation.
- Must not own: gate policy, pose parsing, source matching, model pipeline.
- Must not call: modeling internals.
- Extension points: restore/verify/process/review/publish separation is frozen.
- Risk: workflow layout builds workflow-owned runtime/publish roots; acceptable.

### workflows.tier

- Owns: tier operator config/result/runtime/publish DTOs, layout binding, catalog/metric/leakage/decision/report orchestration, public `TierWorkflow`.
- May call: core, artifacts.catalog/store, data.tier, data.dataset, workflows.foundation.
- Must not own: gate pass/drop, prepared-sample schema, model selection/evaluation, debug behavior.
- Extension points: catalog + data.tier authorities are frozen.
- Risk: none requiring immediate fix.

### workflows.debug

- Owns: debug config/request/result/verdicts, layout/runtime, evidence aggregation, debug reviews, public `SampleDebugWorkflow`.
- May call: core, artifacts, data.gate, data.tier, visualization.
- Must not own: independent gate policy, independent tier metrics, model inference, dataset build logic.
- Extension points: cross-domain diagnostic composition.
- Risk: debug-specific verdict/status enums are acceptable diagnostic states, not core identity replacements.

### workflows.model

- Owns: model operator config/preflight/result/runtime/output/report/generated-pose layout, provider lookup/orchestration, auxiliary objective orchestration, publish lifecycle, public `ModelWorkflow`.
- May call: core, artifacts, modeling providers/validation/artifacts/objectives, workflows.foundation.
- Must not own: candidate training loops, text encoder implementation, temporal tokenization, candidate architecture internals, gate/tier decisions.
- Extension points: provider stages, generated-pose publishing, validation orchestration.
- Risk: validation orchestration reads manifests and generated manifests, but metric/pairing authority is already in `modeling.validation`; acceptable now, future engine should centralize more.

### workflows.test_model

- Owns: test-model config/request/result, checkpoint selection orchestration, target resolution, provider single-sample inference orchestration, visualization bridge, public `TestModelWorkflow`.
- May call: core, artifacts, modeling provider/data/artifacts, visualization, workflows.foundation.
- Must not own: training, candidate provider logic, debug gate/tier diagnostics, dataset catalog internals.
- Extension points: single-sample inference/debug surface only.
- Risk: target/preflight code reads model metadata JSON directly; acceptable workflow runtime metadata handling, but should not grow into general manifest/catalog logic.

### modeling.candidates

- Owns: provider protocols, stage/result contracts, candidate-specific config/training/inference/export/report internals.
- May call: core, artifacts.store topology, modeling.data/artifacts/backbones/validation/training/research.
- Must not own: workflow lifecycle, notebook behavior, gate/tier decisions, common text encoder backends, shared temporal representation.
- Must not call: workflows.
- Extension points: `ModelProvider`, `ModelStageExecutionContext`, `ModelSingleSampleInferenceContext`, stage specs, result artifacts.
- Risk: local ad-hoc JSON writers remain in some candidate modules; acceptable local helpers for candidate-owned artifacts, but later consolidation may be useful.

### modeling.data

- Owns: modeling adapters over canonical dataset manifests/payloads, BFH schema, masks, pairing keys, generated-pose surface loaders.
- May call: core, data.dataset, modeling.artifacts.
- Must not own: prepared sample schema, gate/tier policy, candidate-specific temporal windows.
- Must not call: workflows.
- Extension points: future temporal window/chunk contracts should live here.
- Risk: `modeling.data.dataset` is explicitly legacy M0 adapter; later temporal work should avoid adding candidate-specific windows elsewhere.

### modeling.artifacts

- Owns: generated-pose schema/IO/manifest/run metadata/validation helpers.
- May call: core, modeling.data where needed.
- Must not own: provider training loops, workflow publish, gate/tier policy.
- Must not call: workflows.
- Extension points: generated-pose contract remains common output for future retrieval comparator.
- Risk: generated-pose schema must not be changed in later work without explicit migration; no schema change was made here.

### modeling.backbones

- Owns: shared model backbones and text-conditioning contracts/backends.
- May call: core/stdlib/numpy and optional backend libraries only inside backend implementations.
- Must not own: candidate training loops, workflow orchestration.
- Must not call: workflows.
- Extension points: `TextEncoder`, `TextEncoderConfig`, `TextEncoderOutput`, deterministic hash backend.
- Risk: HuggingFace backend remains unimplemented by design.

### modeling.validation

- Owns: validation pairing, limitations, metrics, aggregation, validation artifact IO and records.
- May call: core, modeling.data/artifacts, data.gate pose channel identities.
- Must not own: workflow lifecycle, candidate-specific report formats.
- Must not call: workflows.
- Extension points: future candidate-agnostic `CandidateEvaluationResult` should live here.
- Risk: current workflow still orchestrates input resolution and summary markdown; acceptable now, needs later design for full validation engine.

## 0B. Import Direction Audit

- `core -> non-core`: none found. Classification: acceptable.
- `data -> workflows`: none found. Classification: acceptable.
- `modeling -> workflows`: found before this stage in candidate report/provider modules via `write_json`/`write_markdown`; fixed now by `core.io`.
- `artifacts.catalog -> data.dataset.manifests`: catalog uses canonical dataset manifest readers. Classification: acceptable boundary.
- `workflows -> data/artifacts/modeling`: expected orchestration direction. Classification: acceptable.
- `workflows.model -> modeling.validation`: expected orchestration of validation authority. Classification: acceptable now; track for later validation engine centralization.
- `workflows.test_model -> modeling.data/artifacts`: expected target/inference orchestration. Classification: acceptable.

## 0C. Duplication and Unusage Findings

### Duplication findings

- Location: old modeling candidate imports from `workflows.foundation.review`.
  - Duplicated authority: generic file writing was stored under review workflow layer.
  - Risk: future candidate code would normalize modeling -> workflows imports.
  - Decision: fix now. Generic helpers moved to `core.io`; old workflow imports re-export for compatibility.
- Location: `workflows.gate.publish.verify` and `workflows.tier.publish.verify` `_file_sha256` wrappers.
  - Duplicated authority: none; wrappers call `core.integrity.sha256_file` and only adapt `OSError` to `None`.
  - Risk: low.
  - Decision: acceptable local helper.
- Location: debug verdict/status enums.
  - Duplicated authority: not core split/status/tier identity; they are diagnostic workflow states.
  - Risk: low if they remain diagnostic only.
  - Decision: acceptable boundary.
- Location: candidate-local JSON/text writers in some modules.
  - Duplicated authority: generic IO style is partly duplicated.
  - Risk: low-to-medium for consistency; changing all would be broad.
  - Decision: leave as transitional seam; later refactor only where helpful.

### Unusage findings

- Location: workflow model/test-model metadata JSON reads.
  - Bypassed authority: no lower-level runtime metadata reader authority currently exists.
  - Risk: low now; could grow into duplicated artifact parsing.
  - Decision: track for later if metadata access expands.
- Location: `modeling.data.dataset` raw document read before dispatching to `data.dataset` manifest readers.
  - Bypassed authority: minimal manifest-kind sniffing before canonical readers.
  - Risk: low; module is documented as legacy adapter.
  - Decision: acceptable local helper.
- Location: workflow layouts build workflow-specific reports/runtime roots.
  - Bypassed authority: not artifact store topology when the paths are workflow-owned runtime/publish surfaces.
  - Risk: low; candidate/model artifact topology should still use `artifacts.store`.
  - Decision: acceptable boundary.
- Location: long-running candidate/training code may have local file writes/progress style.
  - Bypassed authority: `core.progress` for new long-running orchestration.
  - Risk: medium for future work.
  - Decision: track for later; no broad cleanup in Asama 0.

## 0D. Extension Point Freeze

### Text conditioning

- Authority: `modeling.backbones`.
- Frozen contract: candidates consume `TextEncoder`, `TextEncoderConfig`, `TextEncoderOutput`; they must not import HuggingFace/T5 directly.
- Deterministic hash backend remains supported.
- `TextEncoderOutput.embedding` remains the pooled vector for backward compatibility.
- Optional `token_embeddings`, `attention_mask`, and `pooled_embedding` property now reserve token-level support without implementing a HuggingFace backend.

### Temporal representation

- Authority: future contracts under `modeling.data`.
- Candidates must not introduce private temporal window extraction as the primary representation authority.
- `window_size=1` must remain frame-level behavior in later work.
- No temporal extraction was implemented in this stage.

### Candidate providers

- Authority: `modeling.candidates.provider`, `registry`, `stages`, `results`.
- Provider protocol remains unchanged.
- New capabilities should be optional/context-driven and carried through provider context/result objects, not workflow internals.
- No provider behavior was changed.

### Validation

- Authority: `modeling.validation`.
- Workflow may orchestrate input/output lifecycle only.
- Future `CandidateEvaluationResult` should live in `modeling.validation` or a submodule there, with candidate-agnostic records/IO.
- No validation engine was implemented in this stage.

### Retrieval comparator

- Authority: future modeling comparator/provider/index/query modules, tied to `modeling.registry.comparators` and `modeling.research.ComparatorKey`.
- Retrieval is a comparator, not a primary model workflow feature.
- Retrieval outputs must eventually use the same generated-pose artifact contract.
- No retrieval comparator was implemented in this stage.

## Code Changes Made

- Added `src/text_to_sign_production/core/io.py`.
  - Why: generic text/JSON writing is lower-level than workflow review and was being imported by modeling candidates.
  - Compatibility: old `workflows.foundation.review.write_*` functions still re-export the same names.
  - Authority protected: modeling no longer depends on workflows for generic IO.
  - Behavior change: intended none; JSON formatting remains sorted, indented, UTF-8, trailing newline.
- Updated `src/text_to_sign_production/workflows/foundation/review/write.py`.
  - Why: re-export core IO helpers for notebook/workflow compatibility.
  - Compatibility: public import path is preserved.
- Updated `src/text_to_sign_production/workflows/foundation/review/markdown.py`.
  - Why: reuse the lower-level `JsonValue`/`jsonable` authority without changing rendering behavior.
  - Compatibility: rendering fallback still uses `json.dumps(..., ensure_ascii=False, sort_keys=True)`.
- Updated low-risk modeling candidate imports in:
  - `modeling/candidates/base_direct/provider.py`
  - `modeling/candidates/base_direct/reports.py`
  - `modeling/candidates/latent_diffusion/provider.py`
  - `modeling/candidates/latent_diffusion/reports.py`
  - `modeling/candidates/articulator_aware/trainer.py`
  - `modeling/candidates/articulator_aware/reports.py`
  - `modeling/candidates/learned_pose_token/reports.py`
- Updated `modeling/backbones/text_encoder.py`.
  - Why: reserve backward-compatible token-level text output fields.
  - Compatibility: all existing positional fields remain unchanged; new fields default to `None`; deterministic hash behavior remains unchanged.
  - Authority protected: shared text conditioning stays in `modeling.backbones`.

## Explicit Non-Changes

- Notebooks were not modified.
- Generated-pose payload schema and `t2sp-generated-pose-v1` were not changed.
- Gate/tier/debug behavior was not changed.
- No HuggingFace backend was implemented.
- No temporal windows, learned-token changes, true latent diffusion refactor, articulator-factorized model, semantic consistency loss, retrieval comparator, or validation engine was implemented.
- Workflow constructor signatures were not broken.
- No dependencies were added.

## Risks Left For Later Stages

- Candidate-local ad-hoc JSON/text writers remain in some modules; avoid expanding them when adding shared infrastructure.
- Text conditioning still has only deterministic hash backend; HuggingFace support requires a later optional backend implementation.
- Temporal representation needs a `modeling.data` contract before candidate-specific windows appear.
- Validation orchestration should eventually converge on a candidate-agnostic result object under `modeling.validation`.
- Retrieval comparator should be designed under modeling/registry/comparator authority and not added to workflow internals.

## Recommended Next Stage

Next stage: Aşama 1 -- Shared Text Conditioning Backbone.

Do not implement Aşama 1 inside this stage.
