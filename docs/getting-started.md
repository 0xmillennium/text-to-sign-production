# Getting Started

## Prerequisites

- Python 3.11 or newer
- `git`

## Install

Full contributor setup:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,docs]"
```

Or install one extra set at a time:

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[docs]"
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
