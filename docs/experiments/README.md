# Experiment Records

Create one Markdown record per meaningful Dataset Build validation, Sprint 3 baseline run,
modeling ablation, or intentional comparison.

## Recording Guidance

- Use the closest template as a starting point.
- Prefer one record per command or tightly related run group.
- Record exact processed manifest or archive references, config identifiers, split scope, and
  commands.
- Summarize outcomes clearly enough that another researcher can understand what happened.
- Treat routine Dataset Build runs as generated manifests, reports, and archives unless they are
  being compared or interpreted as experiment evidence.

## Current Dataset Context

Current public stage: Dataset Build.

Implemented internal downstream surface: Baseline Modeling.

Experiment records that consume Dataset Build output should name the processed schema version,
filter config, and artifact set used. The active filter config is `configs/data/filter-v2.yaml`;
`configs/data/filter-v1.yaml` is legacy and should appear only when a run intentionally uses it.

Sprint 3 baseline records should use `sprint3-baseline-record-guide.md` and
`sprint3-baseline-record-template.md`.
