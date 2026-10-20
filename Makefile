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