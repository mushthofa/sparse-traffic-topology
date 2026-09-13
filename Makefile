.PHONY: audit test lint paper

audit:
	python scripts/audit_bandung.py

test:
	pytest -q

lint:
	ruff check src tests scripts

paper:
	cd paper && latexmk -pdf main.tex
