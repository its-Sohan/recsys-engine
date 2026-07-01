.PHONY: help venv install download-data data lint format test train evaluate serve frontend docker-build docker-run clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv: ## Create virtual environment
	python3 -m venv .venv
	@echo "Activate with: source .venv/bin/activate"

install: ## Install dependencies
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

download-data: ## Download MovieLens 25M dataset
	python -m src.data.download

data: download-data ## Build processed data artifacts (splits, indices)
	python -m src.data.build

lint: ## Lint with ruff
	ruff check src tests

format: ## Format with ruff
	ruff format src tests
	ruff check --fix src tests

test: ## Run tests
	pytest

train: ## Train all models (baseline, SVD; NCF trained in Colab)
	python -m src.models.train --all

load-ncf: ## Load a pretrained NCF checkpoint (artifacts/ncf.pth from Colab)
	python -m src.models.train --load-ncf artifacts/ncf.pth

evaluate: ## Evaluate trained models and emit results table
	python -m src.evaluate.run

serve: ## Run FastAPI locally
	uvicorn src.serving.app:app --reload --host 0.0.0.0 --port 8000

frontend: ## Run Streamlit frontend
	streamlit run frontend/app.py

docker-build: ## Build Docker image
	docker build -t recsys-engine:latest .

docker-run: ## Run Docker container
	docker run --rm -p 8000:8000 recsys-engine:latest

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
