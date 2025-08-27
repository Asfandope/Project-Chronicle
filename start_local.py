#!/usr/bin/env python3
"""
Local startup script for Project Chronicle.

This script:
1. Checks if PostgreSQL is running
2. Initializes the database
3. Runs the test suite
4. Starts the application
"""

import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_postgres():
    """Check if PostgreSQL is running using Python."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="magazine_extractor",
            connect_timeout=5,
        )
        conn.close()
        return True
    except Exception:
        return False


def check_docker_postgres():
    """Check if PostgreSQL is running in Docker."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=postgres", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "postgres" in result.stdout
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def start_postgres_docker():
    """Start PostgreSQL using docker-compose."""
    logger.info("Starting PostgreSQL with docker-compose...")
    try:
        subprocess.run(
            ["docker-compose", "up", "-d", "postgres"], check=True, timeout=60
        )

        # Wait for PostgreSQL to be ready
        for i in range(30):
            if check_postgres():
                logger.info("PostgreSQL is ready!")
                return True
            logger.info(f"Waiting for PostgreSQL... ({i+1}/30)")
            time.sleep(2)

        logger.error("PostgreSQL did not start in time")
        return False

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to start PostgreSQL: {e}")
        return False


def setup_database():
    """Initialize database tables."""
    logger.info("Setting up database...")
    try:
        from database import init_database

        init_database()
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        return False


def run_tests():
    """Run the test suite."""
    logger.info("Running test suite...")
    try:
        result = subprocess.run([sys.executable, "test_local_setup.py"], timeout=120)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        logger.error("Test suite timed out")
        return False
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        return False


def start_application():
    """Start the main application."""
    logger.info("Starting Project Chronicle application...")
    try:
        # Set environment variables
        os.environ.setdefault(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/magazine_extractor",
        )
        os.environ.setdefault("LOG_LEVEL", "info")

        # Start the application
        subprocess.run([sys.executable, "main.py"])

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {e}")


def main():
    """Main startup sequence."""
    logger.info("🚀 Starting Project Chronicle local development setup...")

    # Check if PostgreSQL is running
    if check_postgres():
        logger.info("✅ PostgreSQL is already running")
    else:
        logger.info("PostgreSQL not running, checking Docker...")
        if check_docker_postgres():
            logger.info("✅ PostgreSQL running in Docker")
        else:
            logger.info("Starting PostgreSQL with Docker...")
            if not start_postgres_docker():
                logger.error("❌ Failed to start PostgreSQL")
                sys.exit(1)

    # Setup database
    if not setup_database():
        logger.error("❌ Database setup failed")
        sys.exit(1)

    # Run tests
    logger.info("Running system tests...")
    if not run_tests():
        logger.warning("⚠️  Some tests failed, but continuing...")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != "y":
            sys.exit(1)

    # Start application
    logger.info("✅ Starting application on http://localhost:8000")
    logger.info("📖 API documentation available at http://localhost:8000/docs")
    start_application()


if __name__ == "__main__":
    main()
