# Experiment Records

Experiment records capture empirical work that needs provenance beyond routine generated reports.

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Not yet implemented: broader evaluation, contribution modeling, playback/demo.

## Current Practice

Routine Dataset Build runs are documented by generated manifests, reports, and archived outputs.
Create an experiment record when a run supports a research or engineering claim that needs
interpretation across data, configs, metrics, or artifacts.

Baseline Modeling has a formal experiment-record documentation surface. A formal baseline record is
a Markdown document that cites runtime artifacts rather than replacing them with a database or
tracking server.

Use:

- [Baseline Modeling Record Guide](experiments/baseline-modeling-record-guide.md)
- [Baseline Modeling Record Template](experiments/baseline-modeling-record-template.md)

## Dataset Build Provenance

When Dataset Build output is used, record:

- processed schema version, currently `t2sp-processed-v1`
- split scope
- filter config, usually `configs/data/filter-v2.yaml`
- processed manifest paths or archived output names
- relevant reports or validation records

Use `configs/data/filter-v1.yaml` only when it is intentionally selected for legacy
reproducibility or comparison.

## Baseline Modeling Provenance

A Baseline Modeling experiment record should cite:

- run root
- effective config and source config
- Dataset Build manifests or Dataset Build archives consumed
- `checkpoints/run_summary.json` or `metrics/run_summary.json`
- `last.pt` and `best.pt`
- qualitative panel outputs
- `baseline_evidence_bundle.json`
- `baseline_modeling_package.json`
- deterministic archives under `archives/`

The runtime record package is baseline evidence. The formal Markdown record is the durable
experiment comparison surface for later sprints.

## Existing Records

- [Dataset Build Filter V2 Full-Run Validation](experiments/2026-04-dataset-build-filter-v2-full-validation.md):
  full `train / val / test` validation after manifest-driven packaging hardening and Filter V2
  adoption.

## Later Use

Later broader-evaluation work should consume formal Baseline Modeling records when adding error
analysis. Later contribution-modeling work should cite the relevant baseline record when comparing
new models against the baseline.
