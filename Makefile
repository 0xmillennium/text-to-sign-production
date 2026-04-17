PYTHON ?= python

.PHONY: install install-dev install-docs lint test docs check ci-local

install-dev:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[dev]"
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type commit-msg

install-docs:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[docs]"

install:
	$(PYTHON) -m pip install --upgrade pip setuptools wheel
	$(PYTHON) -m pip install -e ".[dev,docs]"
	$(PYTHON) -m pre_commit install --hook-type pre-commit --hook-type commit-msg

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .
	$(PYTHON) -m mypy src tests

test:
	$(PYTHON) -m pytest --cov=text_to_sign_production --cov-report=term-missing

docs:
	$(PYTHON) -m mkdocs build --strict

check: lint test docs

ci-local:
	PRE_COMMIT_HOME=$(CURDIR)/.cache/pre-commit SKIP=no-commit-to-branch sh -c 'git ls-files -z --cached --others --exclude-standard | xargs -0 -r -n 200 $(PYTHON) -m pre_commit run --show-diff-on-failure --files'
	# Skip the local editable project package, which is not published on PyPI.
	$(PYTHON) -m pip_audit --local --skip-editable --cache-dir $(CURDIR)/.cache/pip-audit
	$(PYTHON) -m pytest --cov=text_to_sign_production --cov-report=term-missing --cov-report=xml
	$(PYTHON) -m mkdocs build --strict
