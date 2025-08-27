#!/usr/bin/env python3
"""
Quick start script for Project Chronicle - bypasses Docker setup.

Use this when PostgreSQL is already running.
"""

import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up environment variables."""
    os.environ.setdefault(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/magazine_extractor",
    )
    os.environ.setdefault("LOG_LEVEL", "info")
    os.environ.setdefault("SQL_ECHO", "false")


def test_database_connection():
    """Test database connection."""
    try:
        from sqlalchemy import text

        from db_deps import get_db_session

        # Test basic connection
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        health = {"status": "healthy"}
        if health["status"] == "healthy":
            logger.info("✅ Database connection successful")
            return True
        else:
            logger.error(f"❌ Database unhealthy: {health}")
            return False
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return False


def initialize_database():
    """Initialize database."""
    try:
        # Initialize database tables by importing the models
        from db_deps import engine
        from evaluation_service.models import Base as EvaluationBase
        from parameter_management.models import Base as ParameterBase
        from quarantine.models import Base as QuarantineBase
        from self_tuning.models import Base as SelfTuningBase

        logger.info("Creating database tables...")
        EvaluationBase.metadata.create_all(bind=engine)
        ParameterBase.metadata.create_all(bind=engine)
        SelfTuningBase.metadata.create_all(bind=engine)
        QuarantineBase.metadata.create_all(bind=engine)
        logger.info("✅ Database initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False


def initialize_parameters():
    """Initialize parameter system."""
    try:
        from db_deps import get_db_session
        from parameter_management.initialization import (
            initialize_parameter_management_system,
        )

        logger.info("Initializing parameter management system...")
        with get_db_session() as session:
            results = initialize_parameter_management_system(session)
            logger.info(
                f"✅ Created {results['parameters_created']} parameters, {results['parameters_skipped']} skipped"
            )
            return True
    except Exception as e:
        logger.error(f"❌ Parameter initialization failed: {e}")
        return False


def start_application():
    """Start the application."""
    try:
        logger.info("🚀 Starting Project Chronicle...")
        logger.info("📖 API docs will be available at: http://localhost:8000/docs")
        logger.info("🏥 Health check: http://localhost:8000/health")
        logger.info("📊 System status: http://localhost:8000/status")
        logger.info("\nPress Ctrl+C to stop\n")

        import uvicorn

        from main import app

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except KeyboardInterrupt:
        logger.info("\n👋 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Application failed: {e}")
        sys.exit(1)


def main():
    """Main startup sequence."""
    logger.info("🚀 Quick Start - Project Chronicle")
    logger.info("=" * 50)

    # Setup environment
    setup_environment()

    # Test database
    if not test_database_connection():
        logger.error("Please ensure PostgreSQL is running:")
        logger.error("docker-compose up -d postgres")
        sys.exit(1)

    # Initialize database
    if not initialize_database():
        sys.exit(1)

    # Initialize parameters
    if not initialize_parameters():
        logger.warning("Parameter initialization failed, but continuing...")

    # Start application
    start_application()


if __name__ == "__main__":
    main()
