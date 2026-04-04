# text-to-sign-production

`text-to-sign-production` is a graduation thesis repository focused on building a reproducible
research foundation for future text-to-sign work. Sprint 1 provides infrastructure only.

## Current State

- The repository is installable as a Python package.
- Tests, linting, type checks, and docs builds are automated.
- DVC is reserved for future data and model workflow stages.
- Notebooks are explicitly limited to thin runner behavior.

## Not Yet Implemented

The following are intentionally outside Sprint 1:

- text tokenization
- retrieval components
- pose-token generation
- keypoint production or filtering
- skeleton or avatar rendering

This boundary is deliberate so the repository remains honest about its current maturity.
