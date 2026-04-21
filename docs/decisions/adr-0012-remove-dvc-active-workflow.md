# ADR-0012: Remove DVC from the Active Workflow

- Status: Accepted
- Date: 2026-04-17

Historical status note: later notebook-only workflow changes narrowed the supported public operator
surface while leaving DVC removal intact. ADR-0016 defines the current notebook-only workflow
surface, and `docs/execution.md` is the active execution guide.

## Context

ADR-0003 included DVC during Sprint 1 as a reproducibility bootstrap before real data workflow
stages existed. Dataset Build is now implemented and has a validated operational path based on the
single Colab notebook, `docs/execution.md`, manifest and report contracts, packaged `.tar.zst`
artifacts, and the mounted Google Drive publish location.

DVC is not the operational backbone for current day-to-day Dataset Build work. It also adds active
maintenance and security-audit cost: the installed DVC dependency chain pulls in `dvc-data`, which
pulls in `diskcache 5.6.3`, and local auditing reported `CVE-2025-69872` for that transitive
dependency.

## Decision

Remove DVC from the active repository workflow. This includes removing DVC from project
dependencies, deleting DVC repository scaffolding, removing the DVC-specific audit ignore, and
updating documentation so DVC is no longer presented as an active execution path.

The active reproducibility surface is:

- the supported Colab notebook operator workflow
- `docs/execution.md` as the single execution guide
- generated manifests and reports
- packaged `.tar.zst` archives
- validation records

The implemented Dataset Build workflow already expresses the reproducible contract without DVC.
Keeping DVC installed only to preserve an unused alternative execution path makes dependency audits
noisier, requires a CVE exception, and can mislead contributors about which workflow is canonical.

Removing DVC aligns the repository with the actual operator path while preserving the historical
reason it was introduced in ADR-0003.

## Consequences

- Contributors should use the supported notebook workflow documented in `docs/execution.md` rather
  than `dvc repro` or any older CLI-first framing.
- The repository no longer carries DVC config files, DVC stages, or DVC cache expectations.
- The dependency graph no longer includes DVC's transitive `diskcache` chain.
- ADR-0003 remains useful historical context, but this ADR governs the current workflow.
- Future reproducibility work should extend the existing notebook-first execution and artifact
  contracts before adding new tool layers.

## Alternatives Considered

- Keep DVC as active tooling. Rejected because it preserves an unused workflow path and keeps the
  DVC dependency chain, including the audit burden that motivated this removal.
- Keep DVC as inactive scaffold. Rejected because inactive scaffolding still signals an active
  project surface, creates maintenance questions, and leaves contributors unsure whether `dvc
  repro` is supported.
- Remove DVC from the active workflow. Chosen because it keeps the repository simpler, removes the
  unjustified dependency and CVE ignore, and matches the validated Dataset Build operating model.

## Narrowed By

- [ADR-0016: Adopt Notebook-Only Supported Workflow And Single Execution Guide](adr-0016-adopt-notebook-only-supported-workflow-and-single-execution-guide.md)
