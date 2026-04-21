# ADR Template

This file defines the only canonical ADR standard for `docs/decisions/`.

## Document Type

`docs/decisions/index.md` and `docs/decisions/template.md` are the only canonical non-ADR files in
this directory. Every other durable Markdown file under `docs/decisions/` must be a real ADR.

Do not create parallel decision sources such as extra overview files, governance essays, or
alternate ADR-standard documents outside this pattern.

## Naming Rule

ADR filenames must use this exact pattern:

`adr-XXXX-<slug>.md`

## Status Vocabulary

`Status` must be one of:

- `Accepted`
- `Proposed`
- `Superseded`
- `Deprecated`
- `Narrowed`

## Required Structure

Every ADR must use this exact core structure:

1. `# ADR-XXXX: Title`
2. `- Status: ...`
3. `- Date: ...`
4. optional status note
5. `## Context`
6. `## Decision`
7. `## Consequences`

The optional status note belongs immediately after the metadata lines. Use it to explain when an
ADR is historical-only, narrowed by later implementation, or superseded by a later governing
decision.

## Optional Sections

Use these sections only when they are needed:

- `## Alternatives Considered`
- `## Superseded By`
- `## Narrowed By`

Do not introduce additional standing ADR sections as parallel structure.

## Single-Decision Rule

One ADR must record one decision. If a document starts carrying multiple independent decisions, it
should be split into separate ADRs unless a later ADR explicitly keeps the earlier decision intact
and only updates its current status.

## Current-Guidance Rule

ADRs are historical decision records. They are not the active operator guide.

Current behavior belongs in surfaces such as:

- `docs/execution.md`
- `docs/data`
- `docs/experiments`
- `docs/testing/index.md`

## Update Rule

Use `Narrowed` when the original decision still matters but later implementation reduced its scope.

Use `Superseded` when a later decision replaces it as the governing decision.

## Template

```markdown
# ADR-XXXX: Title

- Status: <Accepted|Proposed|Superseded|Deprecated|Narrowed>
- Date: YYYY-MM-DD

Optional status note: explain whether this ADR is historical-only, narrowed, or superseded, and
link the current guidance surface when the ADR is not the active operator reference.

## Context

Describe the problem, pressure, or need that required the decision.

## Decision

Describe the chosen approach clearly and directly.

## Consequences

Describe the expected benefits, tradeoffs, constraints, and follow-up implications.

## Alternatives Considered

Use only when meaningful alternatives were evaluated.

## Superseded By

Use only when a later ADR replaces this ADR as the governing decision.

## Narrowed By

Use only when later implementation narrows the original decision without replacing it.
```
