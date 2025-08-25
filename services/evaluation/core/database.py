from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import structlog

from evaluation.core.config import get_settings

logger = structlog.get_logger()
settings = get_settings()

# Sync engine for migrations
engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Async engine for application
async_engine = create_async_engine(
    settings.database_url.replace("postgresql://", "postgresql+asyncpg://"),
    echo=settings.debug,
)
AsyncSessionLocal = sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    logger.info("Initializing evaluation database")
    # Import models to ensure they're registered
    from evaluation.models import evaluation_result, gold_set, drift_event
    
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)