"""
File Watcher Service for hot folder monitoring.
Automatically detects new PDF files and queues them for processing.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Set

import structlog
from orchestrator.core.config import get_settings
from orchestrator.services.job_queue_manager import JobQueueManager
from orchestrator.utils.file_utils import extract_brand_from_filename, get_file_hash
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

logger = structlog.get_logger(__name__)


class PDFFileHandler(FileSystemEventHandler):
    """
    File system event handler for PDF files.
    Processes new PDF files added to the watch directory.
    """

    def __init__(self, file_watcher: "FileWatcherService"):
        super().__init__()
        self.file_watcher = file_watcher
        self.settings = get_settings()

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            asyncio.create_task(self.file_watcher._handle_new_file(event.src_path))

    def on_moved(self, event):
        """Handle file move events (useful for atomic writes)."""
        if not event.is_directory:
            asyncio.create_task(self.file_watcher._handle_new_file(event.dest_path))


class FileWatcherService:
    """
    Service for monitoring hot folder and automatically processing new PDF files.

    Features:
    - Real-time file system monitoring
    - Duplicate detection using file hashes
    - Brand extraction from filename patterns
    - File validation and quarantine
    - Batch processing support
    - Retry logic for failed file operations
    """

    def __init__(self, watch_directory: str, job_queue_manager: JobQueueManager):
        self.settings = get_settings()
        self.watch_directory = Path(watch_directory)
        self.job_queue_manager = job_queue_manager

        # File tracking
        self._processed_files: Dict[str, Dict] = {}  # filename -> metadata
        self._file_hashes: Set[str] = set()  # Track processed file hashes
        self._processing_files: Set[str] = set()  # Currently processing files

        # Watchdog components
        self._observer: Optional[Observer] = None
        self._event_handler: Optional[PDFFileHandler] = None

        # Monitoring task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.stats = {
            "files_processed": 0,
            "files_quarantined": 0,
            "duplicates_detected": 0,
            "processing_errors": 0,
            "started_at": None,
        }

    async def start(self) -> None:
        """Start the file watcher service."""
        if self._running:
            logger.warning("File watcher already running")
            return

        # Ensure watch directory exists
        self.watch_directory.mkdir(parents=True, exist_ok=True)

        self._running = True
        self.stats["started_at"] = datetime.now(timezone.utc)

        logger.info(
            "Starting file watcher service",
            watch_directory=str(self.watch_directory),
            supported_extensions=self.settings.supported_file_extensions,
        )

        # Process existing files in directory
        await self._process_existing_files()

        # Set up file system monitoring
        self._event_handler = PDFFileHandler(self)
        self._observer = Observer()
        self._observer.schedule(
            self._event_handler, str(self.watch_directory), recursive=True
        )
        self._observer.start()

        # Start monitoring task for periodic checks
        self._monitor_task = asyncio.create_task(self._run_monitor())

        logger.info("File watcher service started")

    async def stop(self) -> None:
        """Stop the file watcher service."""
        if not self._running:
            return

        logger.info("Stopping file watcher service")
        self._running = False

        # Stop file system observer
        if self._observer:
            self._observer.stop()
            self._observer.join()

        # Cancel monitoring task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("File watcher service stopped")

    async def _process_existing_files(self) -> None:
        """Process any existing files in the watch directory."""
        logger.info("Processing existing files in watch directory")

        existing_files = []
        for ext in self.settings.supported_file_extensions:
            pattern = f"**/*{ext}"
            existing_files.extend(self.watch_directory.rglob(pattern))

        for file_path in existing_files:
            if file_path.is_file():
                await self._handle_new_file(str(file_path))

        logger.info(f"Processed {len(existing_files)} existing files")

    async def _handle_new_file(self, file_path: str) -> None:
        """
        Handle a new file detected in the watch directory.

        Args:
            file_path: Path to the new file
        """
        file_path = Path(file_path)
        filename = file_path.name

        # Skip if already processing this file
        if filename in self._processing_files:
            return

        # Check if file extension is supported
        if file_path.suffix.lower() not in self.settings.supported_file_extensions:
            logger.debug(f"Ignoring unsupported file type: {filename}")
            return

        # Add to processing set
        self._processing_files.add(filename)

        try:
            await self._process_file(file_path)
        except Exception as e:
            logger.error(
                "Error processing file", filename=filename, error=str(e), exc_info=True
            )
            self.stats["processing_errors"] += 1
        finally:
            # Remove from processing set
            self._processing_files.discard(filename)

    async def _process_file(self, file_path: Path) -> None:
        """
        Process a single PDF file.

        Args:
            file_path: Path to the PDF file
        """
        filename = file_path.name

        logger.info("Processing new file", filename=filename, path=str(file_path))

        # Wait for file to be fully written (avoid processing incomplete files)
        await self._wait_for_file_stability(file_path)

        # Validate file
        validation_result = await self._validate_file(file_path)
        if not validation_result["valid"]:
            await self._quarantine_file(file_path, validation_result["reason"])
            return

        # Check for duplicates
        file_hash = get_file_hash(file_path)
        if file_hash in self._file_hashes:
            logger.info(f"Duplicate file detected, skipping: {filename}")
            self.stats["duplicates_detected"] += 1
            return

        # Extract brand information from filename
        brand = extract_brand_from_filename(filename)

        # Get file information
        file_size = file_path.stat().st_size

        # Queue job for processing
        try:
            job_id = await self.job_queue_manager.enqueue_job(
                file_path=str(file_path),
                filename=filename,
                file_size=file_size,
                brand=brand,
                priority=self._calculate_priority(file_path, brand),
                correlation_id=f"filewatcher-{filename}",
            )

            # Track processed file
            self._processed_files[filename] = {
                "job_id": job_id,
                "file_path": str(file_path),
                "file_hash": file_hash,
                "file_size": file_size,
                "brand": brand,
                "processed_at": datetime.now(timezone.utc),
            }
            self._file_hashes.add(file_hash)
            self.stats["files_processed"] += 1

            logger.info(
                "File queued for processing",
                filename=filename,
                job_id=job_id,
                brand=brand,
                file_size=file_size,
            )

        except Exception as e:
            logger.error(
                "Failed to queue file for processing",
                filename=filename,
                error=str(e),
                exc_info=True,
            )
            await self._quarantine_file(file_path, f"Failed to queue: {str(e)}")

    async def _wait_for_file_stability(
        self, file_path: Path, timeout: int = 30
    ) -> None:
        """
        Wait for file to be stable (no longer being written to).

        Args:
            file_path: Path to the file
            timeout: Maximum wait time in seconds
        """
        initial_size = file_path.stat().st_size if file_path.exists() else 0
        stable_count = 0

        for _ in range(timeout):
            await asyncio.sleep(1)

            if not file_path.exists():
                break

            current_size = file_path.stat().st_size
            if current_size == initial_size:
                stable_count += 1
                if stable_count >= 3:  # Stable for 3 seconds
                    break
            else:
                initial_size = current_size
                stable_count = 0

    async def _validate_file(self, file_path: Path) -> Dict:
        """
        Validate a PDF file before processing.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with validation result
        """
        try:
            # Check if file exists and is readable
            if not file_path.exists():
                return {"valid": False, "reason": "File does not exist"}

            if not file_path.is_file():
                return {"valid": False, "reason": "Path is not a file"}

            # Check file size
            file_size = file_path.stat().st_size
            max_size = self.settings.max_file_size_mb * 1024 * 1024

            if file_size == 0:
                return {"valid": False, "reason": "File is empty"}

            if file_size > max_size:
                return {
                    "valid": False,
                    "reason": f"File too large: {file_size} bytes > {max_size} bytes",
                }

            # Basic PDF validation (check magic number)
            with open(file_path, "rb") as f:
                header = f.read(4)
                if header != b"%PDF":
                    return {"valid": False, "reason": "Not a valid PDF file"}

            return {"valid": True, "reason": None}

        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {str(e)}"}

    async def _quarantine_file(self, file_path: Path, reason: str) -> None:
        """
        Move file to quarantine directory.

        Args:
            file_path: Path to the file to quarantine
            reason: Reason for quarantine
        """
        quarantine_dir = Path(self.settings.quarantine_directory)
        quarantine_dir.mkdir(parents=True, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        quarantine_filename = f"{timestamp}_{file_path.name}"
        quarantine_path = quarantine_dir / quarantine_filename

        try:
            # Move file to quarantine
            file_path.rename(quarantine_path)

            # Create info file with quarantine reason
            info_path = quarantine_path.with_suffix(quarantine_path.suffix + ".info")
            with open(info_path, "w") as f:
                f.write(f"Quarantined: {datetime.now().isoformat()}\n")
                f.write(f"Original path: {file_path}\n")
                f.write(f"Reason: {reason}\n")

            self.stats["files_quarantined"] += 1

            logger.warning(
                "File quarantined",
                filename=file_path.name,
                reason=reason,
                quarantine_path=str(quarantine_path),
            )

        except Exception as e:
            logger.error(
                "Failed to quarantine file",
                filename=file_path.name,
                error=str(e),
                exc_info=True,
            )

    def _calculate_priority(self, file_path: Path, brand: Optional[str]) -> int:
        """
        Calculate processing priority for a file.

        Args:
            file_path: Path to the file
            brand: Extracted brand information

        Returns:
            Priority value (higher = more priority)
        """
        priority = 0

        # Brand-specific priority
        brand_priorities = {"economist": 10, "time": 8, "vogue": 6}
        if brand and brand.lower() in brand_priorities:
            priority += brand_priorities[brand.lower()]

        # File age priority (newer files get higher priority)
        file_age_hours = (
            datetime.now(timezone.utc)
            - datetime.fromtimestamp(file_path.stat().st_mtime, timezone.utc)
        ).total_seconds() / 3600

        if file_age_hours < 1:  # Less than 1 hour old
            priority += 5
        elif file_age_hours < 24:  # Less than 1 day old
            priority += 2

        return priority

    async def _run_monitor(self) -> None:
        """Background monitoring task for periodic maintenance."""
        logger.info("File watcher monitor started")

        while self._running:
            try:
                # Clean up old processed file records (keep last 1000)
                if len(self._processed_files) > 1000:
                    oldest_files = sorted(
                        self._processed_files.items(),
                        key=lambda x: x[1]["processed_at"],
                    )

                    # Remove oldest 100 files
                    for filename, _ in oldest_files[:100]:
                        file_info = self._processed_files.pop(filename)
                        self._file_hashes.discard(file_info["file_hash"])

                # Log statistics
                logger.info(
                    "File watcher statistics",
                    **self.stats,
                    processed_files_tracked=len(self._processed_files),
                    processing_files=len(self._processing_files),
                )

                # Wait before next monitoring cycle
                await asyncio.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(
                    "Error in file watcher monitor", error=str(e), exc_info=True
                )
                await asyncio.sleep(60)  # Wait longer on error

    def get_stats(self) -> Dict:
        """Get file watcher statistics."""
        stats = self.stats.copy()
        stats.update(
            {
                "is_running": self._running,
                "watch_directory": str(self.watch_directory),
                "processed_files_tracked": len(self._processed_files),
                "processing_files": len(self._processing_files),
                "unique_hashes": len(self._file_hashes),
            }
        )
        return stats
