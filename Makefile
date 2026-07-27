# Development shortcuts. Everything here is a thin wrapper over uv, and nothing depends on
# make itself - the underlying commands are in CONTRIBUTING.md if you would rather run them
# directly.

.PHONY: help install check test cov fmt types bench build docs docs-serve clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies and the git pre-commit hook
	uv sync --group dev
	uv run pre-commit install
	@echo "Ready. The pre-commit hook now runs on every commit."

check:  ## Run every pre-commit hook over the whole tree
	@# 'pre-commit run --all-files' only sees files git tracks, so a newly created file is
	@# skipped in silence and the run reports success without ever looking at it. 'git add
	@# -N' registers paths in the index without staging their contents, which is enough to
	@# make them visible. This is why 'make check' exists rather than just documenting the
	@# pre-commit command.
	git add --intent-to-add .
	uv run pre-commit run --all-files

test:  ## Run the test suite
	uv run pytest

cov:  ## Run the tests with coverage, enforcing the same threshold CI does
	uv run pytest --cov --cov-report=term-missing
	@echo "Note: Ray cannot be installed on macOS x86_64, so ray_backend.py is"
	@echo "uncovered there and the local figure understates CI's."

types:  ## Type-check with mypy
	uv run mypy

fmt:  ## Format and auto-fix what can be fixed
	git add --intent-to-add .
	uv run ruff format .
	uv run ruff check . --fix

bench:  ## Regenerate the algorithm comparison table in the README
	uv run python examples/benchmark.py

docs:  ## Build the documentation site, exactly as CI does
	@{ echo '<!-- Generated from CHANGELOG.md by the docs build; edit that file instead. -->'; \
	   echo; cat CHANGELOG.md; } > docs/changelog.md
	uv run --group docs mkdocs build --strict

docs-serve:  ## Serve the documentation locally with live reload
	@{ echo '<!-- Generated from CHANGELOG.md by the docs build; edit that file instead. -->'; \
	   echo; cat CHANGELOG.md; } > docs/changelog.md
	uv run --group docs mkdocs serve

build:  ## Build the sdist and wheel
	uv build

clean:  ## Remove build artefacts and caches
	rm -rf dist build site .pytest_cache .ruff_cache .mypy_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

all: check test  ## Everything CI runs
