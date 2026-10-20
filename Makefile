export UV_PROJECT_ENVIRONMENT ?= .venv

setup:
	uv sync --extra ml --extra ui --extra train --group dev

check: 
	uv run ruff check perception && uv run black --check perception && uv run mypy perception 

ui: 
	uv run --extra ml --extra ui perception-ui

cov:
	uv run pytest -m "not model" --cov --cov-report=term-missing --cov-report=html -q

test:
	uv run pytest -m "not model" -q

mutation-test:
	uv run mutmut run
	uv run python scripts/mutation_report.py | tee mutation-report.md

export-onnx:
	uv run --extra ml --extra export python scripts/export_onnx.py --out detr.onnx
