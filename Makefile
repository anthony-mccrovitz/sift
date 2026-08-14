.PHONY: help db up down ingest smoke api test eval ragas fixture bench clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

db: ## start Postgres + pgvector
	docker compose up -d db

down: ## stop everything (keeps data)
	docker compose down

reset: ## stop and DESTROY the database volume
	docker compose down -v

smoke: db ## ingest 25 documents -- use this while developing
	python -m sift.ingest --limit 25

ingest: db ## ingest the full 500-document corpus
	python -m sift.ingest --limit 500 --workers 6

api: ## run the API at http://localhost:8000/docs
	uvicorn sift.api:app --reload

test: ## unit tests (no database, no model)
	python -m pytest tests/ -q

eval: ## tier 1: retrieval gate (free, no API key)
	python -m eval.retrieval_eval

ragas: ## tier 2: RAGAS gate (needs ANTHROPIC_API_KEY or OPENAI_API_KEY)
	python -m eval.ragas_eval

fixture: ## re-export the frozen eval corpus from the current database
	python scripts/fixture.py export

bench: ## record metrics to benchmarks/history.md and refresh the README
	python scripts/record_benchmark.py --note "$(NOTE)"

clean: ## remove downloaded PDFs
	rm -rf data/raw
