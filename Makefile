.DEFAULT_GOAL := check
.PHONY: install fmt lint typecheck test evidence-check reproduce check

install:  ## Sync the dev environment
	uv sync --extra dev --extra serving

fmt:  ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

lint:  ## Lint and format-check (no changes)
	uv run ruff check .
	uv run ruff format --check .

typecheck:  ## Check package types
	uv run mypy llm_infer

test:  ## Run unit tests
	uv run python -m pytest

evidence-check:  ## Check committed benchmark evidence and figure regeneration
	uv run scripts/check_benchmark_evidence.py

# Each registered A/B must give the verdict docs/findings.md reports. The judge exits 1 on a
# failed gate, so an expected FAIL must exit exactly 1 (exit 2 is bad input).
reproduce: evidence-check  ## Re-derive every committed verdict from its records (no GPU)
	uv run scripts/judge_paired_ab.py assets/esme-decode-host-plan-ab-confirm-1.json assets/esme-decode-host-plan-ab-confirm-2.json
	uv run scripts/judge_paired_ab.py assets/esme-full-step-ab-1.json assets/esme-full-step-ab-2.json
	uv run scripts/judge_paired_ab.py assets/esme-spec-decode-ab-1.json assets/esme-spec-decode-ab-2.json; test $$? -eq 1

check: lint typecheck test  ## The gate: lint + format-check + types + unit tests
