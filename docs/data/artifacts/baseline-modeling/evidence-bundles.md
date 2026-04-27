# Baseline Evidence Bundles

## Purpose

Evidence bundles tie a baseline run config, summary, checkpoint, and qualitative panel outputs into
a runtime evidence surface.

## Artifact Role

`evidence`

## Artifact Unit

`file family`

## Canonical Path Pattern

- Qualitative evidence: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/qualitative/baseline_evidence_bundle.json`
- Record-side copied evidence: `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/<run_name>/record/baseline_evidence_bundle.json`

## Verified Example Path

- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/qualitative/baseline_evidence_bundle.json`
- `<PROJECT_ARTIFACT_ROOT>/baseline-modeling/runs/190420261845/record/baseline_evidence_bundle.json`

## Produced By

Section 5.3, Qualitative export/publish, writes `qualitative/baseline_evidence_bundle.json`.
Section 6.3 copies it to `record/baseline_evidence_bundle.json`.

## Consumed By

Section 6.3 uses the qualitative evidence bundle when assembling the runtime package. Section 7.1
checks the final artifact surface.

## Lifecycle

The qualitative evidence bundle is a runtime output under `qualitative/`. The record-side evidence
bundle is a copied record-side surface under `record/`.

## Structure / Contents

Schema version is `t2sp-baseline-evidence-v1`. Keys include `artifact_role`, `config_path`,
`export_checkpoint`, `phase5_checkpoint_references`, `qualitative_panel`, `run_summary_path`, and
`schema_version`.

## Validation

Evidence writing requires a readable run summary, checkpoint payload, config path, panel definition,
panel summary, records path, sample IDs, and target channels.

## Related Artifacts

- [Configs](configs.md)
- [Checkpoints](checkpoints.md)
- [Run summaries](run-summaries.md)
- [Qualitative metadata](qualitative-metadata.md)
- [Package manifest](package-manifest.md)
- [Qualitative archive](qualitative-archive.md)
- [Record archive](record-archive.md)

## Related Docs

- [Data leaf template](../../template.md)
- [Baseline Modeling execution](../../../execution.md)
