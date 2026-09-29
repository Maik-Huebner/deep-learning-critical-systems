PYTHON ?= python3.13

.PHONY: install quality test build audit evidence-check reproduce-evidence

install:
	$(PYTHON) -m pip install -r requirements/lock-py313.txt
	$(PYTHON) -m pip install --no-build-isolation --no-deps -e .

quality:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) scripts/audit_repository.py

evidence-check:
	$(PYTHON) scripts/reproduce_evidence.py

reproduce-evidence:
	MPLBACKEND=Agg $(PYTHON) scripts/reproduce_evidence.py --write

build:
	$(PYTHON) -m build --no-isolation
