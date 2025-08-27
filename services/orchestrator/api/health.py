import os
from datetime import datetime, timezone
from pathlib import Path

import httpx
import redis
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from orchestrator.core.config import get_settings
from orchestrator.core.database import get_db
from orchestrator.models.job import Job
from orchestrator.utils.correlation import get_correlation_id
from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()
router = APIRouter()


@router.get("/")
async def health_check(request: Request):
    """
    Health check with dependency status.

    Returns basic health status with key dependency checks.
    Used by load balancers and monitoring systems for quick health verification.

    Returns:
    - status: healthy, degraded, or unhealthy
    - service: service name and version
    - timestamp: current timestamp
    - dependencies: status of critical dependencies
    """
    correlation_id = get_correlation_id(request)
    settings = get_settings()

    health_status = {
        "status": "healthy",
        "service": "orchestrator",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dependencies": {},
        "correlation_id": correlation_id,
    }

    # Quick database check
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            health_status["dependencies"]["database"] = "healthy"
    except Exception as e:
        logger.error(
            "Database health check failed", error=str(e), correlation_id=correlation_id
        )
        health_status["dependencies"]["database"] = "unhealthy"
        health_status["status"] = "unhealthy"

    # Quick Redis check
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_client.ping()
        health_status["dependencies"]["redis"] = "healthy"
        redis_client.close()
    except Exception as e:
        logger.error(
            "Redis health check failed", error=str(e), correlation_id=correlation_id
        )
        health_status["dependencies"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"

    # File system check
    try:
        # Check if critical directories exist and are writable
        directories_to_check = [
            settings.input_directory,
            settings.output_directory,
            settings.temp_directory,
        ]

        for directory in directories_to_check:
            dir_path = Path(directory)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = dir_path / f".health_check_{correlation_id}"
            test_file.write_text("health_check")
            test_file.unlink()

        health_status["dependencies"]["file_system"] = "healthy"

    except Exception as e:
        logger.error(
            "File system health check failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        health_status["dependencies"]["file_system"] = "unhealthy"
        health_status["status"] = "degraded"

    # Check service components
    try:
        services = getattr(request.app.state, "services", {})

        # Job queue manager
        job_queue = services.get("job_queue")
        if job_queue and hasattr(job_queue, "_running") and job_queue._running:
            health_status["dependencies"]["job_queue"] = "healthy"
        else:
            health_status["dependencies"]["job_queue"] = "unhealthy"
            health_status["status"] = "degraded"

        # File watcher
        file_watcher = services.get("file_watcher")
        if file_watcher and hasattr(file_watcher, "_running") and file_watcher._running:
            health_status["dependencies"]["file_watcher"] = "healthy"
        elif not settings.enable_file_watcher:
            health_status["dependencies"]["file_watcher"] = "disabled"
        else:
            health_status["dependencies"]["file_watcher"] = "unhealthy"
            health_status["status"] = "degraded"

    except Exception as e:
        logger.error(
            "Service components health check failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        health_status["status"] = "degraded"

    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(
            status_code=200, detail=health_status
        )  # Still serving but with issues

    return health_status


@router.get("/detailed")
async def detailed_health_check(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Comprehensive health check with detailed dependency status.

    Provides in-depth health information including:
    - All external service dependencies
    - Database connectivity and performance
    - Queue statistics and processing capacity
    - File system status and storage usage
    - Service component health

    Returns:
    - Detailed health report with metrics
    """
    correlation_id = get_correlation_id(request)
    settings = get_settings()

    health_status = {
        "status": "healthy",
        "service": "orchestrator",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "correlation_id": correlation_id,
        "dependencies": {},
        "metrics": {},
        "services": {},
    }

    # Database health with metrics
    try:
        start_time = datetime.now()
        await db.execute(text("SELECT 1"))
        db_latency = (datetime.now() - start_time).total_seconds() * 1000

        # Get job statistics
        job_counts = await db.execute(
            select(Job.overall_status, func.count(Job.id)).group_by(Job.overall_status)
        )
        job_stats = {status.value: count for status, count in job_counts}

        health_status["dependencies"]["database"] = {
            "status": "healthy",
            "latency_ms": round(db_latency, 2),
            "job_statistics": job_stats,
        }

    except Exception as e:
        logger.error(
            "Database health check failed", error=str(e), correlation_id=correlation_id
        )
        health_status["dependencies"]["database"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "unhealthy"

    # Redis/Celery health
    try:
        redis_client = redis.from_url(settings.redis_url)
        redis_info = redis_client.info()

        health_status["dependencies"]["redis"] = {
            "status": "healthy",
            "connected_clients": redis_info.get("connected_clients", 0),
            "used_memory_human": redis_info.get("used_memory_human", "unknown"),
            "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0),
        }
        redis_client.close()

    except Exception as e:
        logger.error(
            "Redis health check failed", error=str(e), correlation_id=correlation_id
        )
        health_status["dependencies"]["redis"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # External services health
    external_services = [
        ("model_service", settings.model_service_url),
        ("evaluation_service", settings.evaluation_service_url),
    ]

    for service_name, service_url in external_services:
        try:
            start_time = datetime.now()
            async with httpx.AsyncClient(
                timeout=settings.health_check_timeout
            ) as client:
                response = await client.get(f"{service_url}/health")
                latency = (datetime.now() - start_time).total_seconds() * 1000

                if response.status_code == 200:
                    health_status["dependencies"][service_name] = {
                        "status": "healthy",
                        "latency_ms": round(latency, 2),
                        "response_code": response.status_code,
                    }
                else:
                    health_status["dependencies"][service_name] = {
                        "status": "unhealthy",
                        "latency_ms": round(latency, 2),
                        "response_code": response.status_code,
                    }
                    health_status["status"] = "degraded"

        except Exception as e:
            logger.error(
                f"{service_name} health check failed",
                error=str(e),
                correlation_id=correlation_id,
            )
            health_status["dependencies"][service_name] = {
                "status": "unreachable",
                "error": str(e),
            }
            health_status["status"] = "degraded"

    # File system health with storage info
    try:
        directories_info = {}
        for dir_name, dir_path in [
            ("input", settings.input_directory),
            ("output", settings.output_directory),
            ("temp", settings.temp_directory),
            ("quarantine", settings.quarantine_directory),
        ]:
            path_obj = Path(dir_path)
            if path_obj.exists():
                stat = os.statvfs(str(path_obj))
                total_space = stat.f_frsize * stat.f_blocks
                free_space = stat.f_frsize * stat.f_available
                used_space = total_space - free_space

                directories_info[dir_name] = {
                    "path": str(path_obj),
                    "exists": True,
                    "total_space_gb": round(total_space / (1024**3), 2),
                    "free_space_gb": round(free_space / (1024**3), 2),
                    "used_space_gb": round(used_space / (1024**3), 2),
                    "usage_percentage": round((used_space / total_space) * 100, 1),
                }
            else:
                directories_info[dir_name] = {"path": str(path_obj), "exists": False}

        health_status["dependencies"]["file_system"] = {
            "status": "healthy",
            "directories": directories_info,
        }

    except Exception as e:
        logger.error(
            "File system health check failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        health_status["dependencies"]["file_system"] = {
            "status": "unhealthy",
            "error": str(e),
        }
        health_status["status"] = "degraded"

    # Service components detailed status
    try:
        services = getattr(request.app.state, "services", {})

        # Job queue manager
        job_queue = services.get("job_queue")
        if job_queue:
            queue_stats = await job_queue.get_queue_stats()
            health_status["services"]["job_queue"] = {
                "status": "healthy" if job_queue._running else "stopped",
                "statistics": queue_stats,
            }
        else:
            health_status["services"]["job_queue"] = {"status": "not_initialized"}

        # File watcher
        file_watcher = services.get("file_watcher")
        if file_watcher:
            watcher_stats = file_watcher.get_stats()
            health_status["services"]["file_watcher"] = {
                "status": "healthy" if file_watcher._running else "stopped",
                "statistics": watcher_stats,
            }
        elif settings.enable_file_watcher:
            health_status["services"]["file_watcher"] = {"status": "not_initialized"}
        else:
            health_status["services"]["file_watcher"] = {"status": "disabled"}

    except Exception as e:
        logger.error(
            "Service components detailed check failed",
            error=str(e),
            correlation_id=correlation_id,
        )
        health_status["services"]["error"] = str(e)

    # System metrics
    try:
        health_status["metrics"]["system"] = {
            "cpu_count": os.cpu_count(),
            "load_average": os.getloadavg() if hasattr(os, "getloadavg") else None,
            "process_id": os.getpid(),
        }
    except Exception as e:
        logger.error("System metrics collection failed", error=str(e))

    # Return appropriate HTTP status
    if health_status["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_status)
    elif health_status["status"] == "degraded":
        raise HTTPException(status_code=200, detail=health_status)

    return health_status


# Add missing import for AsyncSessionLocal
from orchestrator.core.database import AsyncSessionLocal
