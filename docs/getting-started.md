# Getting Started

## Prerequisites

- Python 3.11 or newer
- `git`

## Install

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

## Smoke Check

```bash
python -c "from text_to_sign_production import smoke_check; print(smoke_check())"
```

Expected output:

```text
t2sp-smoke-ok
```

## Next Steps

- Read the development setup guide before contributing.
- Use ADRs for non-trivial architectural decisions.
- Record future empirical work with the experiment log template.
