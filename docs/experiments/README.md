# Experiment Records

Create one Markdown record per meaningful modeling run, modeling ablation, or intentional Dataset
Build comparison.

## Recording Guidance

- Use the experiment template as a starting point.
- Prefer one log per command or tightly related run group.
- Record exact processed manifest or archive references, config identifiers, split scope, and
  commands.
- Summarize outcomes clearly enough that another researcher can understand what happened.
- Treat routine Dataset Build runs as generated manifests, reports, and archives unless they are
  being compared or interpreted as experiment evidence.

## Current Dataset Context

Dataset Build is the current implemented stage. Baseline Modeling is the next major stage.
Experiment records that consume Dataset Build output should name the processed schema version,
filter config, and artifact set used. The active filter config is `configs/data/filter-v2.yaml`;
`configs/data/filter-v1.yaml` is legacy and should appear only when a run intentionally uses it.
