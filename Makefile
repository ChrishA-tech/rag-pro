.PHONY: install test eval ingest serve
install:
	pip install -e ".[dev]"
test:
	pytest -q
eval:
	python -m evaluation.run_eval --out evaluation/results.md
ingest:
	rag-pro ingest data/sample_docs --index .index
serve:
	rag-pro serve --index .index
