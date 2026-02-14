# ===========================================================
# DataFlow Pro — Makefile
# ===========================================================
# Usage: make <target>
# ===========================================================

.PHONY: help install install-dev test lint format run docker-build docker-run clean

PYTHON   ?= python3
VENV_DIR ?= .venv
PORT     ?= 8501

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Setup ─────────────────────────────────────────

install: ## Install production dependencies
	pip install -r requirements.txt

install-dev: ## Install all dependencies (production + dev)
	pip install -r requirements-dev.txt

# ── Quality ───────────────────────────────────────

lint: ## Run linter (ruff) and format check (black)
	ruff check src/ myapp.py
	black --check --diff src/ myapp.py

format: ## Auto-format code with black and fix with ruff
	black src/ myapp.py
	ruff check --fix src/ myapp.py

typecheck: ## Run mypy type checker
	mypy src/

# ── Testing ───────────────────────────────────────

test: ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	pytest tests/ -v -x

# ── Application ──────────────────────────────────

run: ## Start the Streamlit application
	streamlit run myapp.py --server.port=$(PORT)

# ── Docker ────────────────────────────────────────

docker-build: ## Build the Docker image
	docker build -t dataflow-pro .

docker-run: ## Run the Docker container
	docker run -p $(PORT):8501 --name dataflow-pro --rm dataflow-pro

# ── Cleanup ───────────────────────────────────────

clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
