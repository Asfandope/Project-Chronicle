import pytest
import asyncio
from typing import AsyncGenerator, Generator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient
import tempfile
import shutil
from pathlib import Path

# Import all services for testing
from orchestrator.main import create_app as create_orchestrator_app
from model_service.main import create_app as create_model_service_app
from evaluation.main import create_app as create_evaluation_app
from orchestrator.core.database import Base as OrchestratorBase
from evaluation.core.database import Base as EvaluationBase

# Test database URL
TEST_DATABASE_URL = "postgresql+asyncpg://postgres:postgres@localhost:5432/test_magazine_extractor"
TEST_REDIS_URL = "redis://localhost:6379/1"  # Use database 1 for tests

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(OrchestratorBase.metadata.create_all)
        await conn.run_sync(EvaluationBase.metadata.create_all)
    
    yield engine
    
    # Clean up
    async with engine.begin() as conn:
        await conn.run_sync(OrchestratorBase.metadata.drop_all)
        await conn.run_sync(EvaluationBase.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def test_db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    TestSessionLocal = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()

@pytest.fixture
def temp_directory() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
async def orchestrator_client(test_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for orchestrator service."""
    app = create_orchestrator_app()
    
    # Override database dependency
    async def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[f"orchestrator.core.database.get_db"] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def model_service_client() -> AsyncGenerator[AsyncClient, None]:
    """Create test client for model service."""
    app = create_model_service_app()
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
async def evaluation_client(test_db_session) -> AsyncGenerator[AsyncClient, None]:
    """Create test client for evaluation service."""
    app = create_evaluation_app()
    
    # Override database dependency
    async def override_get_db():
        yield test_db_session
    
    app.dependency_overrides[f"evaluation.core.database.get_db"] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def sample_pdf_content() -> bytes:
    """Generate sample PDF content for testing."""
    # This would typically be a real PDF file
    # For now, return mock PDF header
    return b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n' + b'Mock PDF content for testing' + b'\n%%EOF'

@pytest.fixture
def sample_job_data() -> dict:
    """Sample job creation data."""
    return {
        "filename": "test_magazine.pdf",
        "brand": "economist",
        "file_size": 1024000
    }

@pytest.fixture
def sample_layout_data() -> dict:
    """Sample layout analysis data."""
    return {
        "pages": {
            "1": {
                "blocks": [
                    {
                        "id": "block_1_1",
                        "type": "title",
                        "bbox": [100, 100, 400, 150],
                        "text": "Sample Article Title",
                        "confidence": 0.95
                    },
                    {
                        "id": "block_1_2",
                        "type": "body", 
                        "bbox": [100, 200, 400, 500],
                        "text": "Sample article body text...",
                        "confidence": 0.92
                    }
                ]
            }
        },
        "semantic_graph": {
            "nodes": [
                {"id": "block_1_1", "type": "title", "page": 1},
                {"id": "block_1_2", "type": "body", "page": 1}
            ],
            "edges": [
                {"from": "block_1_1", "to": "block_1_2", "relationship": "title_to_body"}
            ]
        }
    }

@pytest.fixture
def sample_gold_standard() -> dict:
    """Sample gold standard data for evaluation testing."""
    return {
        "title": "Sample Article Title",
        "body": "Sample article body text content...",
        "contributors": [
            {
                "name": "John Smith",
                "normalized_name": "Smith, John", 
                "role": "author"
            }
        ],
        "images": [
            {
                "filename": "image_001.jpg",
                "caption": "Sample image caption"
            }
        ]
    }

@pytest.fixture(scope="session")
def celery_config():
    """Celery configuration for testing."""
    return {
        'broker_url': TEST_REDIS_URL,
        'result_backend': TEST_REDIS_URL,
        'task_always_eager': True,  # Execute tasks synchronously for testing
        'task_eager_propagates': True,
    }

# Markers for different test categories
pytest_plugins = ["pytest_asyncio"]

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "integration: mark test as integration test") 
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "model: mark test as requiring ML models")