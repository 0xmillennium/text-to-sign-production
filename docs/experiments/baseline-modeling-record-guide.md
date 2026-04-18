# Baseline Modeling Record Guide

This guide defines the formal Baseline Modeling experiment-record surface.

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## What The Formal Record Is

The formal Baseline Modeling record is a Markdown document stored under `docs/experiments/`. It is
the durable comparison surface for later phases. It explains what was run, which Dataset Build
artifacts were consumed, which baseline config was used, which runtime outputs were produced, and
what the run means.

It is not a database, tracking server, notebook output dump, or replacement for runtime artifacts.

## Runtime Artifacts It Depends On

A complete Baseline Modeling record should cite these runtime artifacts:

- run root: `outputs/modeling/baseline-modeling/runs/<run_name>/` locally or the matching Colab
  Drive run root
- effective config: `config/baseline.yaml`
- source config: `config/source_baseline.yaml`
- training summary: `checkpoints/run_summary.json` or `metrics/run_summary.json`
- checkpoints: `checkpoints/last.pt` and `checkpoints/best.pt` when present
- qualitative panel definition: `qualitative/panel_definition.json`
- qualitative records: `qualitative/records.jsonl`
- qualitative summary: `qualitative/panel_summary.json`
- evidence bundle: `qualitative/baseline_evidence_bundle.json`
- record package manifest: `record/baseline_modeling_package.json`
- copied record evidence: `record/baseline_evidence_bundle.json`
- copied record summary: `record/run_summary.json`
- archived outputs under `archives/`

## Artifact Roles

`run_summary.json` is the training provenance and scalar outcome summary. It records config
summary, backbone name, seed, checkpoint references, final losses, validation metric, best epoch,
target channels, and history.

`last.pt` and `best.pt` are checkpoint payloads. `last.pt` is required for a completed training
step. `best.pt` is present when a best validation checkpoint was selected.

Qualitative outputs provide deterministic validation-panel inspection evidence. They include the
panel definition, per-sample records, exported reference and prediction `.npz` files, and a panel
summary.

`baseline_evidence_bundle.json` ties the qualitative panel to the training summary and selected
checkpoint.

`baseline_modeling_package.json` is the runtime-side package manifest. It records the run root,
config references, training references, qualitative references, copied evidence, and deterministic
archive paths.

The archives are the resume and transfer surface. They preserve training outputs, qualitative
outputs, and record/package outputs with stable names.

## Minimum Formal Record Fields

A formal record should include:

- objective
- run identifier and run root
- environment and command or notebook cells used
- Dataset Build provenance
- baseline config provenance
- training artifact references
- qualitative artifact references
- evidence/package artifact references
- archive references
- result summary
- interpretation and limits
- later-phase consumption notes

## Later-Phase Consumption

Later broader-evaluation work should use the formal record to attach error analysis to a known
baseline run. Later contribution-modeling work should cite the formal record when comparing new
models against Baseline Modeling. Playback/demo work may use the referenced checkpoints and
qualitative artifacts for downstream inspection once those capabilities exist.
