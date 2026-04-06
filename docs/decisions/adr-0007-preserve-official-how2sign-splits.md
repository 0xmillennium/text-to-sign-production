# ADR-0007: Preserve Official How2Sign Splits

- Status: Accepted
- Date: 2026-04-06

## Context

The downloaded How2Sign translation files and BFH keypoints are already organized into official
`train`, `val`, and `test` splits. Rebuilding or inventing new splits would weaken comparability
and make the thesis pipeline less faithful to the source data.

## Decision

Sprint 2 preserves the observed official split structure exactly. The pipeline keeps the `train`,
`val`, and `test` names unchanged and exports processed manifests with the same split boundaries.

## Consequences

- Split integrity is simpler to validate and report.
- Future modeling work can compare results against the official partitioning.
- Unmatched translation rows remain visible within their original split instead of being reassigned.
