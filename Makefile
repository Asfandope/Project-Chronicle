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
	cd evaluation_service && poetry run alembic upgrade head

db-downgrade: ## Rollback last database migration
	cd services/orchestrator && poetry run alembic downgrade -1
	cd evaluation_service && poetry run alembic downgrade -1

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
	cd evaluation_service && poetry run uvicorn evaluation_service.main:app --reload --host 0.0.0.0 --port 8002

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

# Gold Standard Dataset Management
setup-gold-sets: ## Initialize gold standard dataset directory structure
	@echo "📊 Setting up gold standard datasets..."
	mkdir -p data/gold_sets/{economist,time,newsweek,vogue}/{pdfs,ground_truth,annotations,metadata}
	@echo "📁 Created complete gold set directory structure"
	@echo "📋 Directory structure:"
	@echo "  data/gold_sets/{brand}/pdfs/        - Original PDF files"
	@echo "  data/gold_sets/{brand}/ground_truth/ - XML ground truth files" 
	@echo "  data/gold_sets/{brand}/annotations/  - Human annotations"
	@echo "  data/gold_sets/{brand}/metadata/     - File metadata"
	@echo ""
	@echo "🔧 Next steps:"
	@echo "  1. Add PDF files: make ingest-pdfs SOURCE=/path/to/pdfs BRAND=economist"
	@echo "  2. Add ground truth: make ingest-xml SOURCE=/path/to/xml BRAND=economist"
	@echo "  3. Validate dataset: make validate-gold-sets BRAND=economist"

validate-gold-sets: ## Validate gold standard datasets (usage: make validate-gold-sets BRAND=economist)
	@echo "🔍 Validating gold standard datasets..."
	@if [ -z "$(BRAND)" ]; then \
		echo "🔍 Validating all brands..."; \
		python -c "from data_management.schema_validator import DatasetValidator; import sys; \
		validator = DatasetValidator(); \
		brands = ['economist', 'time', 'newsweek', 'vogue']; \
		all_passed = True; \
		for brand in brands: \
			print(f'\\n=== Validating {brand} ==='); \
			report = validator.validate_brand_dataset(brand); \
			print(f'Files: {report.total_files}, Valid: {report.valid_files}, Rate: {report.validation_rate:.1f}%'); \
			if report.recommendations: print('Recommendations:', '\\n'.join(report.recommendations[:3])); \
			if report.validation_rate < 100: all_passed = False; \
		sys.exit(0 if all_passed else 1)"; \
	else \
		python -c "from data_management.schema_validator import DatasetValidator; import sys; \
		validator = DatasetValidator(); \
		report = validator.validate_brand_dataset('$(BRAND)'); \
		print(f'=== $(BRAND) Validation Results ==='); \
		print(f'Total Files: {report.total_files}'); \
		print(f'Valid Files: {report.valid_files}'); \
		print(f'Validation Rate: {report.validation_rate:.1f}%'); \
		print(f'Avg Quality Score: {report.average_quality_score:.3f}'); \
		if report.recommendations: print('\\nRecommendations:'); [print(f'- {r}') for r in report.recommendations]; \
		print('\\n✅ Validation passed' if report.validation_rate == 100 else '❌ Validation issues found'); \
		sys.exit(0 if report.validation_rate == 100 else 1)"; \
	fi

validate-xml: ## Validate XML ground truth files (usage: make validate-xml BRAND=economist FILE=optional_file.xml)
	@echo "🔍 Validating XML ground truth files..."
	@python -c "from data_management.schema_validator import GroundTruthSchemaValidator; \
	from pathlib import Path; \
	validator = GroundTruthSchemaValidator(); \
	brand = '$(BRAND)' or 'economist'; \
	file_arg = '$(FILE)'; \
	if file_arg: \
		result = validator.validate_xml_structure(Path(f'data/gold_sets/{brand}/ground_truth/{file_arg}')); \
		print(f'File: {file_arg}'); \
		print(f'Valid: {result.is_valid}'); \
		print(f'Quality Score: {result.quality_score:.3f}'); \
		if result.errors: print('Errors:', '\\n'.join(result.errors)); \
		if result.warnings: print('Warnings:', '\\n'.join(result.warnings)); \
	else: \
		xml_dir = Path(f'data/gold_sets/{brand}/ground_truth'); \
		xml_files = list(xml_dir.glob('*.xml')) if xml_dir.exists() else []; \
		if not xml_files: print(f'No XML files found in {xml_dir}'); \
		for xml_file in xml_files[:5]: \
			result = validator.validate_xml_structure(xml_file); \
			status = '✅' if result.is_valid else '❌'; \
			print(f'{status} {xml_file.name}: Quality {result.quality_score:.2f}');"

ingest-pdfs: ## Ingest PDF files into gold standard dataset (usage: make ingest-pdfs SOURCE=/path/to/pdfs BRAND=economist)
	@echo "📥 Ingesting PDF files..."
	@if [ -z "$(SOURCE)" ] || [ -z "$(BRAND)" ]; then \
		echo "❌ Usage: make ingest-pdfs SOURCE=/path/to/pdfs BRAND=economist"; \
		exit 1; \
	fi
	@python -c "from data_management.ingestion import DataIngestionManager; \
	from pathlib import Path; \
	manager = DataIngestionManager(); \
	report = manager.ingest_files(Path('$(SOURCE)'), '$(BRAND)', 'pdf', validate_on_ingest=True); \
	print(f'=== Ingestion Report for $(BRAND) ==='); \
	print(f'Files Processed: {report.files_processed}'); \
	print(f'Files Succeeded: {report.files_succeeded}'); \
	print(f'Success Rate: {report.success_rate:.1f}%'); \
	if report.errors: print('Errors:', '\\n'.join(report.errors[:3])); \
	if report.warnings: print('Warnings:', '\\n'.join(report.warnings[:3]));"

ingest-xml: ## Ingest XML ground truth files (usage: make ingest-xml SOURCE=/path/to/xml BRAND=economist)
	@echo "📥 Ingesting XML ground truth files..."
	@if [ -z "$(SOURCE)" ] || [ -z "$(BRAND)" ]; then \
		echo "❌ Usage: make ingest-xml SOURCE=/path/to/xml BRAND=economist"; \
		exit 1; \
	fi
	@python -c "from data_management.ingestion import DataIngestionManager; \
	from pathlib import Path; \
	manager = DataIngestionManager(); \
	report = manager.ingest_files(Path('$(SOURCE)'), '$(BRAND)', 'xml', validate_on_ingest=True); \
	print(f'=== XML Ingestion Report for $(BRAND) ==='); \
	print(f'Files Processed: {report.files_processed}'); \
	print(f'Files Succeeded: {report.files_succeeded}'); \
	print(f'Success Rate: {report.success_rate:.1f}%'); \
	if report.errors: print('Errors:', '\\n'.join(report.errors[:3])); \
	if report.warnings: print('Warnings:', '\\n'.join(report.warnings[:3]));"

gold-sets-report: ## Generate comprehensive gold standard dataset report
	@echo "📊 Generating gold standard dataset report..."
	@python -c "from data_management.schema_validator import DatasetValidator; \
	from data_management.ingestion import DataIngestionManager; \
	from datetime import datetime; \
	import json; \
	validator = DatasetValidator(); \
	ingestion = DataIngestionManager(); \
	brands = ['economist', 'time', 'newsweek', 'vogue']; \
	report = {'timestamp': datetime.now().isoformat(), 'brands': {}}; \
	print('=== GOLD STANDARD DATASETS REPORT ===\\n'); \
	for brand in brands: \
		validation_report = validator.validate_brand_dataset(brand); \
		manifest = ingestion.create_dataset_manifest(brand); \
		report['brands'][brand] = {'validation': validation_report.__dict__, 'manifest': manifest}; \
		print(f'📁 {brand.upper()}:'); \
		print(f'   Files: {validation_report.total_files} | Valid: {validation_report.valid_files} | Rate: {validation_report.validation_rate:.1f}%'); \
		if 'statistics' in manifest: print(f'   PDFs: {manifest[\"statistics\"][\"total_pdfs\"]} | XML: {manifest[\"statistics\"][\"total_ground_truth\"]}'); \
		if validation_report.recommendations: print(f'   Top Rec: {validation_report.recommendations[0]}'); \
		print(); \
	with open('gold_sets_report.json', 'w') as f: json.dump(report, f, indent=2, default=str); \
	print('📋 Detailed report saved to: gold_sets_report.json');"

create-dataset-manifest: ## Create dataset manifest for brand (usage: make create-dataset-manifest BRAND=economist)  
	@echo "📋 Creating dataset manifest..."
	@if [ -z "$(BRAND)" ]; then \
		echo "❌ Usage: make create-dataset-manifest BRAND=economist"; \
		exit 1; \
	fi
	@python -c "from data_management.ingestion import DataIngestionManager; \
	import json; \
	manager = DataIngestionManager(); \
	manifest = manager.create_dataset_manifest('$(BRAND)'); \
	print(f'=== Dataset Manifest for $(BRAND) ==='); \
	print(json.dumps(manifest, indent=2, default=str));"

# Benchmark and Evaluation Commands
benchmark-brand: ## Run benchmark evaluation for specific brand (usage: make benchmark-brand BRAND=economist)
	@echo "🎯 Running benchmark evaluation..."
	@if [ -z "$(BRAND)" ]; then \
		echo "❌ Usage: make benchmark-brand BRAND=economist"; \
		exit 1; \
	fi
	@python scripts/run_benchmarks.py $(BRAND)

benchmark-all: ## Run benchmark evaluation for all brands
	@echo "🎯 Running comprehensive benchmark evaluation..."
	@python scripts/run_benchmarks.py --all

benchmark-targets: ## List all accuracy targets and thresholds
	@echo "🎯 Accuracy targets for Project Chronicle..."
	@python scripts/run_benchmarks.py --targets

benchmark-report: ## Generate benchmark report and save to file
	@echo "📊 Generating comprehensive benchmark report..."
	@python scripts/run_benchmarks.py --all --verbose

# LayoutLM Training Commands
train-brand: ## Train LayoutLM for specific brand (usage: make train-brand BRAND=economist)
	@echo "🎯 Training LayoutLM model..."
	@if [ -z "$(BRAND)" ]; then \
		echo "❌ Usage: make train-brand BRAND=economist"; \
		exit 1; \
	fi
	@python scripts/train_$(BRAND).py

train-all: ## Train LayoutLM for all brands sequentially
	@echo "🚀 Training LayoutLM for all brands..."
	@python scripts/train_all_brands.py

train-parallel: ## Train LayoutLM for all brands in parallel
	@echo "🚀 Training LayoutLM for all brands in parallel..."
	@python scripts/train_all_brands.py --parallel

training-summary: ## Show training experiments summary
	@echo "📊 Training experiments summary..."
	@python -c "import sys; sys.path.append('.'); from data_management.experiment_tracking import ExperimentTracker, print_experiment_summary; tracker = ExperimentTracker(); print_experiment_summary(tracker)"

model-compare: ## Compare available brand models
	@echo "🔍 Comparing brand models..."
	@python data_management/brand_model_manager.py compare

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
	cd evaluation_service && poetry shell

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