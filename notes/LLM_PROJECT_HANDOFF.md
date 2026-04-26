# LLM Project Handoff

## 1. Project Identity

- Repository name: `text-to-sign-production`
- Project type: graduation thesis / research engineering repository
- Domain: English text-to-sign pose production
- High-level goal: a reproducible research pipeline for English text input to sign-pose /
  pose-keypoint artifacts, evaluation, comparison, and thesis packaging.
- GitHub repository: `https://github.com/0xmillennium/text-to-sign-production`
- GitHub Pages docs: `https://0xmillennium.github.io/text-to-sign-production/`
- Default branch: `master`

The current handoff must not preserve legacy selected-pair or final model-matrix framing. M0, when
used, means only the Direct Text-to-Pose Baseline comparison floor.

## 2. Quick Access

- GitHub repository: `https://github.com/0xmillennium/text-to-sign-production`
- GitHub Pages docs: `https://0xmillennium.github.io/text-to-sign-production/`
- Default branch: `master`
- File URL template:
  `https://github.com/0xmillennium/text-to-sign-production/blob/master/<path>`
- Raw file URL template:
  `https://raw.githubusercontent.com/0xmillennium/text-to-sign-production/master/<path>`
- Published docs URL pattern:
  `https://0xmillennium.github.io/text-to-sign-production/<docs-relative-page>/`

Examples:

- Roadmap file:
  `https://github.com/0xmillennium/text-to-sign-production/blob/master/docs/research/roadmap.md`
- Roadmap page:
  `https://0xmillennium.github.io/text-to-sign-production/research/roadmap/`
- Audit result page:
  `https://0xmillennium.github.io/text-to-sign-production/research/contribution-audit/audit-result/`
- Literature positioning page:
  `https://0xmillennium.github.io/text-to-sign-production/research/literature-positioning/`
- Source corpus page:
  `https://0xmillennium.github.io/text-to-sign-production/research/source-corpus/`

## 3. How To Use This Handoff

This file is a context bridge for a new ChatGPT/Codex session. It exists so a new assistant can
start from the current project state without relying on previous conversation history.

Canonical docs remain the source of truth. Verify claims against repository files before editing,
especially the research roadmap, audit result, literature positioning, data docs, experiment
records, execution guide, testing docs, source code, scripts, and tests.

Do not treat this handoff as authoritative if it conflicts with canonical docs. If canonical docs
conflict with each other, do not silently resolve the conflict here; record the conflict as current
state and update canonical surfaces in their own task.

## 4. Current Source-Of-Truth Hierarchy

Current research source-of-truth chain:

```text
docs/research/source-selection-criteria.md
  -> docs/research/source-corpus.md
    -> docs/research/contribution-audit/
      -> docs/research/contribution-audit/audit-result.md
        -> docs/research/literature-positioning.md
          -> docs/research/roadmap.md
```

Implementation and evidence surfaces to inspect alongside the research chain:

- `docs/data/`
- `docs/experiments/`
- `docs/testing/`
- `docs/execution.md`
- `src/`
- `scripts/`
- `configs/`
- `notebooks/`
- `tests/`

Known conflict: `README.md`, `docs/index.md`, public workflow contract tests, and several ADRs
still contain stale C1/C2 or sprint-era wording. Treat the refreshed `docs/research/` chain as the
current research-planning authority until those surfaces are cleaned up.

## 5. Current Research Architecture

The merged research chain is source-grounded and audit-derived:

- Source-selection criteria define which sources may support the audit and how evidence boundaries
  are handled.
- Source corpus is the canonical public source registry, currently recording 28 reviewed sources
  with stable `SRC-*` IDs.
- Candidate universe records define family-level audit axes.
- Candidate cards instantiate eight concrete audit records.
- Scorecards record six readiness scorecards without selecting or ranking final outcomes.
- Selection decisions record candidate-level role decisions.
- Audit result synthesizes the whole audit.
- Literature positioning explains the thesis contribution frame relative to the source corpus and
  audit result.
- Roadmap translates the audit result and positioning into a Phase 0-12 execution map.

No repository evidence was found for the exact claim that a final broad research review passed
with no blocker/major/minor/nit findings. Do not repeat that claim unless a future canonical record
documents it.

The old plain bibliography-centered architecture is no longer the current research authority.

## 6. Current Thesis Contribution Frame

The current thesis contribution frame is a reproducible, gloss-free, How2Sign-compatible
text-to-sign pose-production research pipeline that compares representation-based,
latent-generative, and structure-aware text-to-pose candidates under a shared baseline,
evaluation, comparator, visualization, reporting, and reproducibility surface.

Current boundaries:

- No final implementation model is selected.
- No final candidate ranking is assigned.
- No experimental result is claimed by the research audit.
- No near-term full 3D/avatar implementation is authorized.
- Automatic metrics and visual plausibility are not proof of sign intelligibility.

## 7. Current Candidate Role Boundaries

| Surface | Current role |
| --- | --- |
| Direct Text-to-Pose Baseline | M0 comparison floor, not a model contribution claim. |
| Learned Pose-Token Bottleneck | Primary model candidate. |
| Gloss-Free Latent Diffusion | Primary high-support/high-risk model candidate. |
| Articulator-Disentangled Latent Modeling | Primary structure-preservation model candidate. |
| Text-Pose Semantic Consistency | Auxiliary/additive objective, not standalone primary model direction. |
| Retrieval-Augmented Pose Comparator | Comparator/counter-alternative, not primary neural generation candidate. |
| How2Sign-Compatible Evaluation Protocol | Evaluation surface, not model candidate. |
| Sparse-Keyframe / Avatar-Control Future Direction | Future-work/avatar boundary, not near-term implementation. |

## 8. Current Roadmap Baseline

The current roadmap uses Phase, not legacy sprint numbering. Phase order is execution sequence,
not final ranking. Candidate phases are comparable experimental units, not final implementation
commitments.

- Phase 0 - Research Audit Consolidation
- Phase 1 - Dataset Access And Supervision Boundary
- Phase 2 - Artifact Schema And Data Pipeline
- Phase 3 - Pose / Keypoint Extraction And Normalization
- Phase 4 - M0 Direct Text-to-Pose Baseline
- Phase 5 - Evaluation Harness And Reporting Protocol
- Phase 6 - Learned Pose-Token Bottleneck Candidate
- Phase 7 - Gloss-Free Latent Diffusion Candidate
- Phase 8 - Articulator-Aware Structure Candidate
- Phase 9 - Auxiliary Semantic Consistency And Retrieval Comparator
- Phase 10 - Comparative Analysis And Candidate Synthesis
- Phase 11 - Visual Representation Layer
- Phase 12 - Thesis Writing And Reproducibility Release

## 9. Current Roadmap Implementation Status

| Phase | Current status | Summary |
| --- | --- | --- |
| Phase 0 | mostly_complete | Research chain merged; public root docs, ADRs, and public contract tests still contain stale selected-pair/sprint wording. |
| Phase 1 | mostly_complete | Dataset/supervision evidence exists; consolidated access/license/redistribution note still needed. |
| Phase 2 | completed | Artifact schema and data pipeline are implemented and validated through docs, code, tests, and experiment records. |
| Phase 3 | mostly_complete | OpenPose/BFH parsing and normalization exist; visual debug/playback surface is missing. |
| Phase 4 | mostly_complete | Baseline implementation and one baseline experiment record exist; failure-mode inventory and evaluation linkage remain missing. |
| Phase 5 | partial | Metric primitives and qualitative helpers exist; reusable evaluation harness/reporting protocol remains missing. |
| Phase 6-12 | planned | Later model, comparator, synthesis, visualization, and thesis-release phases are not yet implemented. |

Do not mark a phase complete unless repository evidence supports the claim.

## 10. Immediate Next Work

1. Phase 0 cleanup: update stale public root docs, ADR wording, and public contract expectations
   that still expose old selected-pair or outcome framing.
2. Phase 1 consolidation: document dataset access, supervision, license, and redistribution
   boundaries in a canonical, reviewable form.
3. Phase 3 visual debug: add a skeleton/keypoint inspection or playback surface for processed and
   generated pose artifacts.
4. Phase 4 linkage: add baseline failure-mode inventory and connect baseline outputs to the
   emerging evaluation/reporting surface.
5. Phase 5 harness: build a reusable evaluation harness and reporting protocol that can evaluate
   M0 outputs reproducibly.

Keep Phase 6+ model candidate work planned until Phase 5 can evaluate M0 outputs reproducibly.

## 11. Repository Surfaces To Inspect

- `README.md`
- `docs/index.md`
- `docs/research/roadmap.md`
- `docs/research/contribution-audit/audit-result.md`
- `docs/research/literature-positioning.md`
- `docs/research/source-corpus.md`
- `docs/data/`
- `docs/experiments/`
- `docs/testing/`
- `docs/execution.md`
- `docs/decisions/`
- `src/text_to_sign_production/`
- `scripts/`
- `configs/`
- `notebooks/`
- `tests/`

Do not use the old plain bibliography page as the current canonical source registry.

## 12. Current Implementation Evidence Snapshot

Supported by current repo evidence:

- Dataset build pipeline exists.
- Processed-v1 artifact schema, docs, validation, and archive documentation exist.
- OpenPose/BFH keypoint parsing and normalization exist.
- Baseline model, training, inference, checkpointing, packaging, and qualitative export surfaces
  exist.
- One dataset build experiment record exists.
- One baseline modeling experiment record exists.
- Metric primitive(s), including masked keypoint L2 evaluation, exist.
- Qualitative export helpers exist.

Important limits:

- The reusable evaluation harness/reporting protocol is incomplete.
- `scripts/evaluate_baseline.py` explicitly reports that baseline evaluation is not implemented
  yet.
- No learned-token implementation exists yet.
- No latent-diffusion implementation exists yet.
- No articulator-aware candidate implementation exists yet.
- No retrieval comparator implementation exists yet.
- No near-term full avatar implementation exists.
- No released/loadable M0 artifact is documented.
- No Hugging Face or other external model release is documented.

## 13. Experiments And Evidence Records

`docs/experiments/` is the durable experiment-record surface. Experiment records are evidence, not
automatic proof of phase completion.

Current records:

- `docs/experiments/2026-04-dataset-build-experiment-record-filter-v2-full-validation.md`
  records the validated Dataset Build run and processed-v1 artifact evidence.
- `docs/experiments/2026-04-baseline-modeling-experiment-record-colab-run-190420261845.md`
  records one baseline modeling Colab run with checkpoints, qualitative outputs, runtime evidence,
  and archives.

Do not invent additional experiment records or claim unrecorded results.

## 14. ADR And Governance Notes

ADRs are historical/governance context. Current research planning is controlled by:

- `docs/research/roadmap.md`
- `docs/research/contribution-audit/audit-result.md`
- `docs/research/literature-positioning.md`

ADRs that mention old C1/C2 or M0/M1/M2/M3 framing should not be treated as current final
model-selection policy unless they are updated to align with the refreshed roadmap. ADR cleanup may
belong to Phase 0 foundation cleanup if stale public wording remains.

Stable ADR ideas that still appear valid:

- Notebooks remain thin drivers; shared logic belongs in `src/`.
- Large generated artifacts stay outside Git.
- Published or reusable artifacts are preferred as downstream interfaces.
- The current operator surface is the project-wide Colab workflow documented in
  `docs/execution.md`.
- DVC is removed from the active workflow.

Do not use ADR-0021 or ADR-0022 to reintroduce final C1/C2 selection.

## 15. Testing And Validation Snapshot

Active pytest layers are `unit`, `integration`, `e2e`, and `operational`. The default CI-safe
pytest run excludes `operational`; operational checks cover real Colab, Drive, private data, large
archives, and external runtime behavior.

Current CI-safe layers cover dataset workflow logic, baseline helpers, archive/resume behavior,
notebook and docs wording, public workflow contracts, and thin wrappers.

Before major changes, run the most targeted useful checks:

```bash
python -m mkdocs build --strict
python -m pytest <relevant-subset>
python -m pytest
```

Build/test commands may create local ignored artifacts. Check `git status --short` after
validation.

## 16. Operating Rules For A New Chat

Do:

- Verify canonical docs before claiming status.
- Treat roadmap phases as the current planning frame.
- Keep Phase 6+ planned until the Phase 5 evaluation harness exists.
- Preserve the no-public-gloss boundary.
- Preserve metric-limitation and human-evaluation boundaries.
- Keep large generated artifacts outside Git.
- Keep notebooks as thin drivers.
- Treat experiment records as evidence surfaces, not automatic completion proofs.

Do not:

- Reintroduce C1/C2 selected-pair framing.
- Treat model phase order as final ranking.
- Treat the existing baseline run as released/loadable M0 unless documented.
- Treat local checkpoints as published artifacts.
- Claim experimental results not recorded.
- Start later model candidate work before evaluation harness readiness.
- Infer sign intelligibility from automatic metrics or visual plausibility.
- Authorize near-term full 3D/avatar implementation.

## 17. Things Not To Reintroduce

Do not reintroduce:

- selected-C1 wording
- selected-C2 wording
- final two-contribution pair framing
- current chosen-pair wording
- C1/C2 active outcome framing
- the old 11-sprint roadmap
- the old active path from the sprint-era roadmap
- a fixed M0/M1/M2/M3 final model matrix
- the old plain bibliography page as canonical source registry
- near-term full avatar implementation
- automatic metrics as proof of sign intelligibility
- local source paths in public docs
- dedicated legacy contribution-pair notebooks as current targets
- published contribution-matrix artifacts as current roadmap commitments
- old closure work from the superseded sprint plan

## 18. Documentation Maintenance Policy

- Research source policy / source registry changes ->
  `docs/research/source-selection-criteria.md` and `docs/research/source-corpus.md`
- Contribution audit changes -> `docs/research/contribution-audit/`
- Audit synthesis changes -> `docs/research/contribution-audit/audit-result.md`
- Literature positioning changes -> `docs/research/literature-positioning.md`
- Roadmap changes -> `docs/research/roadmap.md`
- Data/artifact changes -> `docs/data/`
- Experiment records -> `docs/experiments/`
- Execution changes -> `docs/execution.md`
- Testing changes -> `docs/testing/`
- ADR/governance changes -> `docs/decisions/`
- Repository structure changes -> `docs/repository-map.md`
- Handoff updates -> update only after canonical surfaces are updated
