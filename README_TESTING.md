# Project Chronicle - Local Testing Setup

This document explains how to get Project Chronicle running locally for testing.

## Quick Start

### Option 1: Automated Setup
```bash
python start_local.py
```

This script will:
1. Check if PostgreSQL is running (starts with Docker if needed)
2. Initialize database tables
3. Run system tests
4. Start the application

### Option 2: Manual Setup

1. **Start PostgreSQL**:
```bash
docker-compose up -d postgres redis
```

2. **Initialize Database**:
```bash
python -c "from database import init_database; init_database()"
```

3. **Run Tests**:
```bash
python test_local_setup.py
```

4. **Start Application**:
```bash
python main.py
```

## Services and Ports

- **Project Chronicle API**: http://localhost:8100 (when using docker-compose)
- **Project Chronicle API**: http://localhost:8000 (when running directly)
- **Original Magazine System**: http://localhost:8000 (orchestrator)
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379

## API Documentation

Once running, visit:
- Main API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- System status: http://localhost:8000/status

## Key Endpoints

### Parameter Management
- `GET /api/v1/parameters` - List all parameters
- `POST /api/v1/parameters` - Create parameter
- `GET /api/v1/parameters/{key}` - Get parameter value

### Evaluation Service
- `POST /api/v1/evaluation/evaluate` - Submit evaluation
- `GET /api/v1/evaluation/runs` - List evaluation runs
- `GET /api/v1/evaluation/drift` - Check for drift

### Self-Tuning
- `POST /api/v1/self-tuning/start` - Start tuning run
- `GET /api/v1/self-tuning/status` - System status
- `GET /api/v1/self-tuning/runs` - List tuning runs

### Quarantine
- `POST /api/v1/quarantine/evaluate` - Evaluate for quarantine
- `GET /api/v1/quarantine/items` - List quarantined items
- `POST /api/v1/quarantine/retry` - Retry quarantined items

## Testing Components

### 1. Synthetic Data Generation
```python
from synthetic_data import SyntheticDataGenerator
generator = SyntheticDataGenerator()
# Generate test data
```

### 2. Parameter Management
```python
from parameter_management import get_parameter, ParameterKeys
threshold = get_parameter(ParameterKeys.ACCURACY_WER_THRESHOLD)
```

### 3. Evaluation System
```python
from evaluation_service.service import EvaluationService
service = EvaluationService()
# Submit evaluation
```

### 4. Quarantine System
```python
from quarantine import quarantine_if_needed
quarantined = quarantine_if_needed(
    issue_id="test_001",
    extraction_output={"title": "..."},
    accuracy_scores={"overall": 0.85}
)
```

### 5. Self-Tuning
```python
from self_tuning import start_tuning_for_brand
tuning_id = start_tuning_for_brand("TestBrand", session)
```

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps postgres

# View PostgreSQL logs
docker-compose logs postgres

# Reset database (WARNING: deletes all data)
python -c "from database import reset_database; reset_database()"
```

### Import Errors
Make sure you're in the project root directory:
```bash
export PYTHONPATH=/path/to/Project-Chronicle
```

### Port Conflicts
If port 8000 is in use, change the port in main.py:
```python
uvicorn.run("main:app", host="0.0.0.0", port=8001)
```

## Test Data Generation

Create test evaluation data:
```python
python -c "
from database import get_db_session
from evaluation_service.service import EvaluationService
from datetime import datetime, timezone

with get_db_session() as session:
    service = EvaluationService()
    # Create test evaluation
    result = service.create_evaluation_run(
        session=session,
        brand_name='TestBrand',
        issue_id='test_001',
        extraction_output={'title': 'Test Article'},
        ground_truth={'title': 'Test Article'},
        created_by='test_user'
    )
    print(f'Created test evaluation: {result.id}')
"
```

## Next Steps

1. ✅ Verify all tests pass
2. ✅ Check API documentation at /docs
3. 🔄 Run integration tests
4. 🔄 Test with sample magazine data
5. 🔄 Verify quarantine workflow
6. 🔄 Test parameter tuning cycle

## Support

If you encounter issues:
1. Check the logs: `tail -f project_chronicle.log`
2. Verify database health: `python -c "from database import check_database_health; print(check_database_health())"`
3. Run individual component tests: `python test_local_setup.py`