"""
Utility functions and comprehensive error handling for PDF processing.
"""

import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import structlog

from .types import PDFProcessingError

logger = structlog.get_logger(__name__)


class PDFProcessingContext:
    """Context manager for PDF processing operations with error handling."""

    def __init__(
        self,
        operation: str,
        pdf_path: Optional[Path] = None,
        page_num: Optional[int] = None,
    ):
        self.operation = operation
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.start_time = None
        self.logger = logger.bind(
            operation=operation,
            pdf_path=str(pdf_path) if pdf_path else None,
            page_num=page_num,
        )

    def __enter__(self):
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        processing_time = time.time() - self.start_time if self.start_time else 0

        if exc_type is None:
            self.logger.info(
                f"Completed {self.operation}", processing_time=processing_time
            )
        else:
            self.logger.error(
                f"Failed {self.operation}",
                error=str(exc_val),
                processing_time=processing_time,
                exc_info=True,
            )

        return False  # Don't suppress exceptions


def with_error_handling(operation: str):
    """
    Decorator for adding comprehensive error handling to PDF processing functions.

    Args:
        operation: Description of the operation being performed
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract PDF path and page number from common argument patterns
            pdf_path = None
            page_num = None

            # Look for pdf_path in args or kwargs
            if args and isinstance(args[0], Path):
                pdf_path = args[0]
            elif "pdf_path" in kwargs:
                pdf_path = kwargs["pdf_path"]

            # Look for page_num in args or kwargs
            if len(args) > 1 and isinstance(args[1], int):
                page_num = args[1]
            elif "page_num" in kwargs:
                page_num = kwargs["page_num"]

            with PDFProcessingContext(operation, pdf_path, page_num):
                try:
                    return func(*args, **kwargs)
                except PDFProcessingError:
                    # Re-raise PDF processing errors as-is
                    raise
                except Exception as e:
                    # Wrap other exceptions in PDFProcessingError
                    error_msg = f"Unexpected error in {operation}: {str(e)}"
                    raise PDFProcessingError(error_msg, pdf_path, page_num) from e

        return wrapper

    return decorator


@contextmanager
def safe_pdf_operation(
    operation: str, pdf_path: Optional[Path] = None, page_num: Optional[int] = None
):
    """
    Context manager for safe PDF operations with automatic error logging.

    Args:
        operation: Description of the operation
        pdf_path: Optional PDF file path
        page_num: Optional page number
    """
    op_logger = logger.bind(
        operation=operation,
        pdf_path=str(pdf_path) if pdf_path else None,
        page_num=page_num,
    )

    start_time = time.time()
    op_logger.info(f"Starting {operation}")

    try:
        yield
        processing_time = time.time() - start_time
        op_logger.info(f"Completed {operation}", processing_time=processing_time)
    except Exception as e:
        processing_time = time.time() - start_time
        op_logger.error(
            f"Failed {operation}",
            error=str(e),
            error_type=type(e).__name__,
            processing_time=processing_time,
            exc_info=True,
        )
        raise


def validate_pdf_path(pdf_path: Union[str, Path]) -> Path:
    """
    Validate and normalize PDF path.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Validated Path object

    Raises:
        PDFProcessingError: If path is invalid
    """
    try:
        path = Path(pdf_path)

        if not path.exists():
            raise PDFProcessingError(f"PDF file does not exist: {path}")

        if not path.is_file():
            raise PDFProcessingError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise PDFProcessingError(f"File is not a PDF: {path}")

        return path

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Invalid PDF path: {str(e)}")


def validate_page_number(page_num: int, total_pages: int) -> int:
    """
    Validate page number against document page count.

    Args:
        page_num: Page number (1-indexed)
        total_pages: Total number of pages in document

    Returns:
        Validated page number

    Raises:
        PDFProcessingError: If page number is invalid
    """
    if not isinstance(page_num, int):
        raise PDFProcessingError(
            f"Page number must be an integer, got {type(page_num)}"
        )

    if page_num < 1:
        raise PDFProcessingError(f"Page number must be positive, got {page_num}")

    if page_num > total_pages:
        raise PDFProcessingError(
            f"Page number {page_num} exceeds document pages ({total_pages})"
        )

    return page_num


def validate_page_range(page_range: tuple, total_pages: int) -> tuple:
    """
    Validate page range against document page count.

    Args:
        page_range: Tuple of (start_page, end_page) 1-indexed
        total_pages: Total number of pages in document

    Returns:
        Validated page range tuple

    Raises:
        PDFProcessingError: If page range is invalid
    """
    if not isinstance(page_range, tuple) or len(page_range) != 2:
        raise PDFProcessingError("Page range must be a tuple of (start_page, end_page)")

    start_page, end_page = page_range

    validate_page_number(start_page, total_pages)
    validate_page_number(end_page, total_pages)

    if start_page > end_page:
        raise PDFProcessingError(
            f"Start page {start_page} cannot be greater than end page {end_page}"
        )

    return (start_page, end_page)


def create_output_directory(
    output_dir: Optional[Path], base_name: str = "pdf_output"
) -> Path:
    """
    Create and validate output directory.

    Args:
        output_dir: Desired output directory (None for temp directory)
        base_name: Base name for temporary directory

    Returns:
        Path to output directory

    Raises:
        PDFProcessingError: If directory creation fails
    """
    try:
        if output_dir is None:
            import tempfile

            output_dir = Path(tempfile.mkdtemp(prefix=f"{base_name}_"))
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

        # Verify directory is writable
        test_file = output_dir / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
        except Exception:
            raise PDFProcessingError(f"Output directory is not writable: {output_dir}")

        return output_dir

    except PDFProcessingError:
        raise
    except Exception as e:
        raise PDFProcessingError(f"Cannot create output directory: {str(e)}")


def safe_file_operation(
    file_path: Path, operation: str, func: Callable, *args, **kwargs
):
    """
    Safely perform file operations with proper error handling.

    Args:
        file_path: Path to file
        operation: Description of operation
        func: Function to execute
        *args, **kwargs: Arguments for function

    Returns:
        Result of function execution

    Raises:
        PDFProcessingError: If operation fails
    """
    try:
        op_logger = logger.bind(operation=operation, file_path=str(file_path))
        op_logger.debug(f"Starting {operation}")

        result = func(*args, **kwargs)

        op_logger.debug(f"Completed {operation}")
        return result

    except Exception as e:
        error_msg = f"File operation failed ({operation}): {str(e)}"
        logger.error(error_msg, file_path=str(file_path), error=str(e), exc_info=True)
        raise PDFProcessingError(error_msg)


def get_file_hash(file_path: Path, algorithm: str = "md5") -> str:
    """
    Calculate file hash for integrity checking.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm ('md5', 'sha256', etc.)

    Returns:
        Hexadecimal hash string

    Raises:
        PDFProcessingError: If hash calculation fails
    """
    try:
        import hashlib

        hasher = hashlib.new(algorithm)

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        return hasher.hexdigest()

    except Exception as e:
        raise PDFProcessingError(f"Cannot calculate file hash: {str(e)}")


def cleanup_temp_files(file_paths: List[Path]) -> int:
    """
    Clean up temporary files safely.

    Args:
        file_paths: List of file paths to delete

    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0

    for file_path in file_paths:
        try:
            if file_path.exists():
                file_path.unlink()
                deleted_count += 1
        except Exception as e:
            logger.warning(
                "Failed to delete temporary file",
                file_path=str(file_path),
                error=str(e),
            )

    return deleted_count


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage statistics
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit_index = 0

    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1

    if unit_index == 0:
        return f"{int(size)} {units[unit_index]}"
    else:
        return f"{size:.1f} {units[unit_index]}"


def log_processing_stats(
    operation: str,
    pdf_path: Path,
    pages_processed: int = 0,
    items_extracted: int = 0,
    processing_time: float = 0,
    file_size: int = 0,
    **additional_stats,
):
    """
    Log comprehensive processing statistics.

    Args:
        operation: Operation name
        pdf_path: PDF file path
        pages_processed: Number of pages processed
        items_extracted: Number of items extracted (text blocks, images, etc.)
        processing_time: Processing time in seconds
        file_size: File size in bytes
        **additional_stats: Additional statistics to log
    """
    stats = {
        "operation": operation,
        "pdf_path": str(pdf_path),
        "pdf_name": pdf_path.name,
        "pages_processed": pages_processed,
        "items_extracted": items_extracted,
        "processing_time": round(processing_time, 3),
        "file_size": file_size,
        "file_size_formatted": format_file_size(file_size),
        **additional_stats,
    }

    if processing_time > 0 and pages_processed > 0:
        stats["pages_per_second"] = round(pages_processed / processing_time, 2)

    if processing_time > 0 and items_extracted > 0:
        stats["items_per_second"] = round(items_extracted / processing_time, 2)

    logger.info("Processing statistics", **stats)


class ProgressTracker:
    """Simple progress tracker for long-running operations."""

    def __init__(self, total_items: int, operation: str = "Processing"):
        self.total_items = total_items
        self.operation = operation
        self.processed_items = 0
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.log_interval = 5.0  # Log every 5 seconds

    def update(self, increment: int = 1):
        """Update progress counter."""
        self.processed_items += increment
        current_time = time.time()

        # Log progress periodically
        if current_time - self.last_log_time >= self.log_interval:
            self.log_progress()
            self.last_log_time = current_time

    def log_progress(self):
        """Log current progress."""
        if self.total_items > 0:
            percentage = (self.processed_items / self.total_items) * 100
            elapsed_time = time.time() - self.start_time

            if self.processed_items > 0:
                estimated_total_time = elapsed_time * (
                    self.total_items / self.processed_items
                )
                remaining_time = estimated_total_time - elapsed_time
            else:
                remaining_time = 0

            logger.info(
                f"{self.operation} progress",
                processed=self.processed_items,
                total=self.total_items,
                percentage=round(percentage, 1),
                elapsed_time=round(elapsed_time, 1),
                estimated_remaining=round(remaining_time, 1),
            )

    def complete(self):
        """Mark processing as complete and log final statistics."""
        total_time = time.time() - self.start_time
        logger.info(
            f"{self.operation} completed",
            total_items=self.processed_items,
            total_time=round(total_time, 3),
            items_per_second=round(self.processed_items / total_time, 2)
            if total_time > 0
            else 0,
        )


def retry_on_error(
    max_attempts: int = 3, delay: float = 1.0, exponential_backoff: bool = True
):
    """
    Decorator for retrying operations that may fail temporarily.

    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between attempts (seconds)
        exponential_backoff: Whether to use exponential backoff
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < max_attempts - 1:  # Don't delay on last attempt
                        logger.warning(
                            f"Attempt {attempt + 1} failed, retrying in {current_delay}s",
                            error=str(e),
                            function=func.__name__,
                        )
                        time.sleep(current_delay)

                        if exponential_backoff:
                            current_delay *= 2

            # If we get here, all attempts failed
            logger.error(
                f"All {max_attempts} attempts failed",
                function=func.__name__,
                final_error=str(last_exception),
            )
            raise last_exception

        return wrapper

    return decorator
