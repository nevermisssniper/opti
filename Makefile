.PHONY: install test lint typecheck demo clean

install:
	pip install -e ".[dev]"

test:
	pytest -v

lint:
	ruff check opti tests

typecheck:
	mypy opti

demo:
	python -m opti.cli demo

demo-json:
	python -m opti.cli demo --json

check: lint test

clean:
	rm -rf build/ dist/ *.egg-info .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
