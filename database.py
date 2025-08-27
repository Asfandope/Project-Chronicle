"""
Database initialization and session management for Project Chronicle.

Centralizes database configuration and provides session management
for all services.
"""

import logging
import os
from contextlib import contextmanager

from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from evaluation_service.models import Base as EvaluationBase
from parameter_management.models import Base as ParameterBase
from quarantine.models import Base as QuarantineBase
from self_tuning.models import Base as SelfTuningBase

logger = logging.getLogger(__name__)


# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:password@localhost:5432/project_chronicle"
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    echo=os.getenv("SQL_ECHO", "false").lower() == "true",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=3600,  # Recycle connections every hour
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@contextmanager
def get_db_session() -> Session:
    """
    Get database session with automatic cleanup.

    Usage:
        with get_db_session() as session:
            # Use session here
            pass
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


def get_db_session_dependency():
    """
    FastAPI dependency for database sessions.

    Usage in FastAPI endpoints:
        @app.get("/endpoint")
        def endpoint(session: Session = Depends(get_db_session_dependency)):
            pass
    """
    session = SessionLocal()
    try:
        yield session
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_database():
    """Initialize database tables for all services."""
    logger.info("Initializing database tables...")

    try:
        # Test connection first
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection successful")

        # Create all tables
        logger.info("Creating evaluation service tables...")
        EvaluationBase.metadata.create_all(bind=engine)

        logger.info("Creating parameter management tables...")
        ParameterBase.metadata.create_all(bind=engine)

        logger.info("Creating self-tuning tables...")
        SelfTuningBase.metadata.create_all(bind=engine)

        logger.info("Creating quarantine tables...")
        QuarantineBase.metadata.create_all(bind=engine)

        logger.info("Database initialization completed successfully")

    except OperationalError as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("Make sure PostgreSQL is running and accessible")
        raise
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def reset_database():
    """Reset database - DROP and recreate all tables (DANGEROUS!)."""
    logger.warning("RESETTING DATABASE - ALL DATA WILL BE LOST")

    try:
        # Drop all tables
        logger.info("Dropping all tables...")
        QuarantineBase.metadata.drop_all(bind=engine)
        SelfTuningBase.metadata.drop_all(bind=engine)
        ParameterBase.metadata.drop_all(bind=engine)
        EvaluationBase.metadata.drop_all(bind=engine)

        # Recreate all tables
        logger.info("Recreating all tables...")
        init_database()

        logger.warning("Database reset completed - all previous data lost")

    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        raise


def check_database_health() -> dict:
    """Check database health and return status information."""
    try:
        with get_db_session() as session:
            # Test basic connectivity
            session.execute(text("SELECT 1"))

            # Check table existence
            tables = {
                "evaluation_runs": "evaluation_runs",
                "parameters": "parameters",
                "tuning_runs": "tuning_runs",
                "quarantine_items": "quarantine_items",
            }

            table_status = {}
            for service, table_name in tables.items():
                try:
                    result = session.execute(
                        text(f"SELECT COUNT(*) FROM {table_name}")
                    ).scalar()
                    table_status[service] = {"exists": True, "count": result}
                except Exception as e:
                    table_status[service] = {"exists": False, "error": str(e)}

            return {
                "status": "healthy",
                "connection": "successful",
                "tables": table_status,
                "database_url": DATABASE_URL.split("@")[1]
                if "@" in DATABASE_URL
                else "masked",
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "connection": "failed",
            "error": str(e),
            "database_url": DATABASE_URL.split("@")[1]
            if "@" in DATABASE_URL
            else "masked",
        }


if __name__ == "__main__":
    """Run database initialization standalone."""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        print("WARNING: This will DELETE ALL DATA in the database!")
        confirm = input("Type 'RESET' to confirm: ")
        if confirm == "RESET":
            reset_database()
        else:
            print("Reset cancelled")
    else:
        init_database()

        # Test the setup
        health = check_database_health()
        print(f"Database health check: {health}")
