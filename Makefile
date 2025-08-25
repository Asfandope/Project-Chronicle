# Magazine PDF Extractor - Development Makefile

.PHONY: help install setup test lint format clean docker dev prod

# Default target
help: ## Show this help message
	@echo "Magazine PDF Extractor - Development Commands"
	@echo "=============================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development Setup
install: ## Install all dependencies with Poetry
	poetry install

setup: install ## Complete development setup
	poetry run pre-commit install
	cp .env.example .env
	@echo "✅ Development setup complete!"
	@echo "📝 Edit .env file with your configuration"
	@echo "🐳 Run 'make docker-dev' to start services"

# Code Quality
lint: ## Run all linting and type checks
	@echo "Running linting and formatting..."
	docker-compose exec orchestrator python -m ruff check . || poetry run ruff check .
	docker-compose exec orchestrator python -m ruff format . || poetry run ruff format .
	@echo "Linting completed."

format: ## Format code with black and isort
	poetry run black .
	poetry run isort .

# Testing
test: ## Run all tests
	@echo "Running tests..."
	docker-compose exec orchestrator python -m pytest tests/ -v --tb=short || poetry run pytest -v
	@echo "Tests completed."

test-unit: ## Run unit tests only
	poetry run pytest tests/ -m "unit" -v

test-integration: ## Run integration tests only
	poetry run pytest tests/ -m "integration" -v

test-api: ## Run API tests only
	poetry run pytest tests/ -m "api" -v

test-performance: ## Run performance tests
	poetry run pytest tests/ -m "performance" -v --benchmark-only

test-coverage: ## Run tests with coverage report
	poetry run pytest --cov=services --cov=shared --cov-report=html --cov-report=term

# Docker Development
dev: ## Start development environment with Docker Compose
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "Services starting up. Use 'make logs' to follow logs."
	@echo "Services will be available at:"
	@echo "  - Orchestrator API: http://localhost:8000"
	@echo "  - Model Service: http://localhost:8001"
	@echo "  - Evaluation Service: http://localhost:8002"
	@echo "  - Grafana: http://localhost:3000 (admin/admin)"
	@echo "  - Prometheus: http://localhost:9090"
	@echo "  - Flower (Celery): http://localhost:5555"

docker-dev: ## Start development environment with Docker Compose
	docker-compose up -d postgres redis
	docker-compose up orchestrator model-service evaluation

docker-build: ## Build all Docker images
	docker-compose build

docker-down: ## Stop and remove Docker containers
	docker-compose down

docker-clean: ## Remove Docker containers, volumes, and images
	docker-compose down -v --rmi all

# Database Management
db-upgrade: ## Run database migrations
	cd services/orchestrator && poetry run alembic upgrade head
	cd services/evaluation && poetry run alembic upgrade head

db-downgrade: ## Rollback last database migration
	cd services/orchestrator && poetry run alembic downgrade -1
	cd services/evaluation && poetry run alembic downgrade -1

db-migration: ## Generate new database migration
	@read -p "Enter migration message: " msg; \
	cd services/orchestrator && poetry run alembic revision --autogenerate -m "$$msg"

db-reset: ## Reset database (WARNING: destroys all data)
	@echo "⚠️  This will destroy all data in the database!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		docker-compose down -v; \
		docker-compose up -d postgres; \
		sleep 5; \
		make db-upgrade; \
	fi

# Development Services
dev-orchestrator: ## Run orchestrator service in development mode
	cd services/orchestrator && poetry run uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000

dev-model-service: ## Run model service in development mode
	cd services/model_service && poetry run uvicorn model_service.main:app --reload --host 0.0.0.0 --port 8001

dev-evaluation: ## Run evaluation service in development mode
	cd services/evaluation && poetry run uvicorn evaluation.main:app --reload --host 0.0.0.0 --port 8002

dev-worker: ## Run Celery worker for orchestrator
	cd services/orchestrator && poetry run celery -A orchestrator.celery_app worker --loglevel=info

# Production
prod-build: ## Build production Docker images
	docker build --target production-cpu -t magazine-extractor-orchestrator .
	docker build --target production-cpu -t magazine-extractor-model-service .
	docker build --target production-cpu -t magazine-extractor-evaluation .

prod-deploy: prod-build ## Deploy to production (placeholder)
	@echo "🚀 Production deployment would go here"
	@echo "📋 This should be replaced with your actual deployment commands"

# Model Management
download-models: ## Download required ML models
	cd services/model_service && poetry run python -c "from model_service.core.model_manager import ModelManager; import asyncio; asyncio.run(ModelManager().load_models())"

# Data Management
setup-gold-sets: ## Initialize gold standard datasets (requires data)
	@echo "📊 Setting up gold standard datasets..."
	mkdir -p data/gold_sets/{economist,time,newsweek}
	@echo "📁 Created gold set directories"
	@echo "📋 Please add your annotated PDF files to data/gold_sets/{brand}/"

# Monitoring and Maintenance
logs: ## Show logs from all services
	docker-compose logs -f

logs-orchestrator: ## Show orchestrator logs
	docker-compose logs -f orchestrator

logs-model-service: ## Show model service logs
	docker-compose logs -f model-service

logs-evaluation: ## Show evaluation logs
	docker-compose logs -f evaluation

health-check: ## Check health of all services
	@echo "🏥 Checking service health..."
	@curl -f http://localhost:8000/health/ || echo "❌ Orchestrator unhealthy"
	@curl -f http://localhost:8001/health/ || echo "❌ Model Service unhealthy"
	@curl -f http://localhost:8002/health/ || echo "❌ Evaluation unhealthy"

# Cleanup
clean: ## Clean up generated files and caches
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name ".mypy_cache" -delete
	rm -rf htmlcov/
	rm -rf .coverage
	rm -f bandit-report.json

clean-all: clean docker-clean ## Deep clean including Docker resources
	docker system prune -f
	poetry cache clear --all pypi

# Documentation
docs-build: ## Build documentation (if implemented)
	@echo "📚 Documentation build would go here"

docs-serve: ## Serve documentation locally (if implemented)
	@echo "📚 Documentation server would go here"

# Configuration Management
validate-configs: ## Validate all configuration files
	poetry run python scripts/config_cli.py validate-all

validate-brand: ## Validate specific brand configuration (usage: make validate-brand BRAND=economist)
	poetry run python scripts/config_cli.py validate-brand $(BRAND)

list-brands: ## List all available brand configurations
	poetry run python scripts/config_cli.py list-brands

show-config: ## Show configuration for brand (usage: make show-config BRAND=economist)
	poetry run python scripts/config_cli.py show-config $(BRAND)

check-config-consistency: ## Check configuration consistency across brands
	poetry run python scripts/config_cli.py check-consistency

config-help: ## Show configuration CLI help
	poetry run python scripts/config_cli.py --help

# Utilities
check-deps: ## Check for dependency updates
	poetry show --outdated

security-scan: ## Run security scan
	poetry run bandit -r services/ -f json -o bandit-report.json
	poetry run safety check

# Development Utilities
shell-orchestrator: ## Open shell in orchestrator environment
	cd services/orchestrator && poetry shell

shell-model-service: ## Open shell in model service environment
	cd services/model_service && poetry shell

shell-evaluation: ## Open shell in evaluation environment
	cd services/evaluation && poetry shell

# Quick Development Workflow
quick-setup: ## Quick setup for new developers
	make install
	make format
	make lint
	make test-unit
	@echo "🎉 Quick setup complete! Ready for development."

# CI/CD Simulation
ci: ## Simulate CI pipeline locally
	make lint
	make test-coverage
	make security-scan
	make docker-build
	@echo "✅ CI pipeline simulation complete"

# Performance and Profiling
profile: ## Run performance profiling (requires implementation)
	@echo "📊 Performance profiling would go here"

benchmark: ## Run benchmarks
	poetry run pytest tests/ -m "performance" --benchmark-only --benchmark-sort=mean

# Environment Info
info: ## Show environment information
	@echo "=== Environment Information ==="
	@echo "Python version: $$(python --version)"
	@echo "Poetry version: $$(poetry --version)"
	@echo "Docker version: $$(docker --version)"
	@echo "Docker Compose version: $$(docker-compose --version)"
	@echo "Git version: $$(git --version)"
	@echo ""
	@echo "=== Project Status ==="
	@echo "Git branch: $$(git branch --show-current)"
	@echo "Git commit: $$(git rev-parse --short HEAD)"
	@echo "Working directory: $$(pwd)"
	@echo ""
	@echo "=== Services Status ==="
	@make health-check