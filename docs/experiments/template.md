# Experiment Record Template

Use this file as the only canonical standard for formal records under `docs/experiments/`.

## Document Type

`docs/experiments/index.md` and `docs/experiments/template.md` are the only canonical non-record
documents in this directory. Every other durable Markdown file under `docs/experiments/` is an
`Experiment Record`.

Do not create parallel durable document systems such as logs, guides, validation-doc variants, or
baseline-specific record systems outside this standard.

## File Naming

Real record filenames must use this exact pattern:

`YYYY-MM-<stage>-experiment-record-<slug>.md`

Supported stage slugs include at least:

- `dataset-build`
- `baseline-modeling`

## Record Type Vocabulary

`Record Type` must be one of:

- `validation`
- `baseline-run`
- `comparison`
- `ablation`
- `rerun`

## Required Metadata

Every `Experiment Record` must explicitly record:

- date or time context
- git revision
- execution surface
- environment
- stage
- record type
- artifact references
- relevant schema versions where applicable

When a required field is unavailable, write `not recorded` instead of omitting it.

## Section Order

Every real `Experiment Record` must use exactly this section order:

1. `# <Title>`
2. `## Objective`
3. `## Record Type`
4. `## Run Identity`
5. `## Input Provenance`
6. `## Config Provenance`
7. `## Artifact References`
8. `## Results`
9. `## Interpretation`
10. `## Limits`
11. `## Follow-up`

Do not rename, reorder, or replace these section headings.

## Artifact References

Use `docs/data` as the canonical artifact dictionary. `Experiment Record` files should cite the
relevant `docs/data` pages for raw inputs, manifests, reports, configs, checkpoints, run
summaries, qualitative metadata, qualitative sample artifacts, evidence bundles, package
manifests, and archives instead of duplicating artifact schema details in the record itself.

## Formal Record Vs Runtime Artifacts

Runtime package manifests, evidence bundles, run summaries, checkpoints, qualitative exports, and
archives are not the formal experiment record. The Markdown `Experiment Record` is the durable
formal record, and it must cite those runtime artifacts rather than replacing them.

## Template

```markdown
# <Title>

## Objective
State the concrete research or engineering outcome being verified.

## Record Type
- Document type: `Experiment Record`
- Record type: `<validation|baseline-run|comparison|ablation|rerun>`
- Stage: `<Dataset Build|Baseline Modeling|...>`

## Run Identity
| Field | Value |
| --- | --- |
| Date / time context | `<YYYY-MM or ISO 8601>` |
| Git revision | `<revision>` |
| Execution surface | `<CLI or notebook surface>` |
| Environment | `<Python, accelerator, env, or not recorded>` |

## Input Provenance
Document the input roots, consumed manifests or archives, split scope, and relevant input schema
versions. Link to the canonical `docs/data` pages instead of restating schema details.

## Config Provenance
Document the source config, effective config, repo-default config where applicable, and the
configuration values needed to understand the run.

## Artifact References
List the runtime-relative artifact paths cited by the record and the matching canonical
`docs/data` references.

## Results
Summarize the validated counts, metrics, schema versions, artifact roles, or archive composition
facts established by the run.

## Interpretation
State what the record supports and what it does not support.

## Limits
List the known bounds, missing surfaces, and any `not recorded` provenance that matters.

## Follow-up
List the next concrete record, rerun, comparison, ablation, or evaluation step.
```
