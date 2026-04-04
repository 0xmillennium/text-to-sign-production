"""Small deterministic helpers used for smoke validation."""


def smoke_check() -> str:
    """Return a deterministic sentinel used by smoke tests and notebooks."""

    return "t2sp-smoke-ok"
