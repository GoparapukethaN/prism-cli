.PHONY: verify test lint

verify:
	./scripts/verify-local.sh

test:
	python -m pytest -q

lint:
	python -m ruff check src tests
