import os
import hashlib
import re
from typing import Tuple, Dict, Optional
from pathlib import Path
import structlog

logger = structlog.get_logger()

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of file"""
    hash_sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()

def get_file_hash(file_path: Path) -> str:
    """Get SHA-256 hash of file (Path object version)"""
    return calculate_file_hash(str(file_path))

async def validate_pdf_file(file_path: str) -> Dict[str, any]:
    """
    Validate if file is a proper PDF with comprehensive checks.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        Dictionary with validation result and details
    """
    try:
        path_obj = Path(file_path)
        
        # Check if file exists
        if not path_obj.exists():
            return {"valid": False, "reason": "File does not exist"}
        
        # Check if it's a file (not directory)
        if not path_obj.is_file():
            return {"valid": False, "reason": "Path is not a file"}
        
        # Check file size
        file_size = path_obj.stat().st_size
        if file_size == 0:
            return {"valid": False, "reason": "File is empty"}
        
        # Read file header
        with open(file_path, "rb") as f:
            header = f.read(8)
        
        if not header.startswith(b'%PDF-'):
            return {"valid": False, "reason": "File does not have valid PDF header"}
        
        # Extract PDF version
        version_match = re.match(rb'%PDF-(\d\.\d)', header)
        pdf_version = version_match.group(1).decode() if version_match else "unknown"
        
        # Basic PDF structure validation
        with open(file_path, "rb") as f:
            content = f.read(min(file_size, 8192))  # Read first 8KB
            
            # Check for essential PDF elements
            if b'trailer' not in content.lower() and file_size > 1024:
                # Only check for trailer in larger files
                f.seek(max(0, file_size - 1024))  # Check end of file
                footer = f.read(1024)
                if b'trailer' not in footer.lower() and b'%%EOF' not in footer:
                    return {"valid": False, "reason": "PDF structure appears invalid"}
        
        return {
            "valid": True,
            "reason": None,
            "pdf_version": pdf_version,
            "file_size": file_size
        }
        
    except Exception as e:
        return {"valid": False, "reason": f"Error validating PDF: {str(e)}"}

def extract_brand_from_filename(filename: str) -> Optional[str]:
    """
    Extract brand information from filename using pattern matching.
    
    Args:
        filename: Original filename
        
    Returns:
        Brand name if detected, None otherwise
    """
    filename_lower = filename.lower()
    
    # Define brand patterns with priority (more specific first)
    brand_patterns = [
        (r'economist[_\-\s]', 'economist'),
        (r'time[_\-\s]magazine', 'time'),
        (r'time[_\-\s]', 'time'),
        (r'vogue[_\-\s]', 'vogue'),
        (r'newsweek[_\-\s]', 'newsweek'),
        (r'national[_\-\s]geographic', 'national_geographic'),
        (r'natgeo[_\-\s]', 'national_geographic'),
        (r'fortune[_\-\s]', 'fortune'),
        (r'forbes[_\-\s]', 'forbes'),
        (r'wired[_\-\s]', 'wired'),
        (r'atlantic[_\-\s]', 'atlantic'),
        (r'new[_\-\s]yorker', 'new_yorker'),
        (r'newyorker[_\-\s]', 'new_yorker'),
        (r'harper[_\-\s]?s?[_\-\s]?bazaar', 'harpers_bazaar'),
        (r'elle[_\-\s]', 'elle'),
        (r'cosmopolitan[_\-\s]', 'cosmopolitan'),
        (r'cosmo[_\-\s]', 'cosmopolitan'),
        (r'rolling[_\-\s]stone', 'rolling_stone'),
        (r'people[_\-\s]', 'people'),
        (r'entertainment[_\-\s]weekly', 'entertainment_weekly'),
        (r'vanity[_\-\s]fair', 'vanity_fair'),
        (r'scientific[_\-\s]american', 'scientific_american')
    ]
    
    for pattern, brand in brand_patterns:
        if re.search(pattern, filename_lower):
            logger.debug("Brand detected from filename", filename=filename, brand=brand)
            return brand
    
    # Try to extract from directory path if filename doesn't contain brand
    parent_path = Path(filename).parent.name.lower()
    for pattern, brand in brand_patterns:
        if re.search(pattern.replace('[_\\-\\s]', ''), parent_path):
            logger.debug("Brand detected from path", path=parent_path, brand=brand)
            return brand
    
    logger.debug("No brand detected from filename", filename=filename)
    return None

def extract_issue_date_from_filename(filename: str) -> Optional[str]:
    """
    Extract issue date from filename using pattern matching.
    
    Args:
        filename: Original filename
        
    Returns:
        Issue date in YYYY-MM-DD format if detected, None otherwise
    """
    # Date patterns to try (in order of preference)
    date_patterns = [
        r'(\d{4})[-_](\d{2})[-_](\d{2})',  # YYYY-MM-DD or YYYY_MM_DD
        r'(\d{4})(\d{2})(\d{2})',          # YYYYMMDD
        r'(\d{2})[-_](\d{2})[-_](\d{4})',  # MM-DD-YYYY or MM_DD_YYYY
        r'(\d{1,2})[-_](\d{1,2})[-_](\d{4})',  # M-D-YYYY or M_D_YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            
            if len(groups[0]) == 4:  # Year first
                year, month, day = groups
            else:  # Year last
                if len(groups) == 3:
                    month, day, year = groups
                else:
                    continue
            
            try:
                # Normalize to YYYY-MM-DD format
                year = int(year)
                month = int(month)
                day = int(day)
                
                # Basic validation
                if 1900 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
                    return f"{year:04d}-{month:02d}-{day:02d}"
                    
            except (ValueError, TypeError):
                continue
    
    return None

def ensure_directories_exist(directories: list):
    """Ensure all required directories exist"""
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug("Ensured directory exists", directory=directory)

def get_file_extension(filename: str) -> str:
    """Get file extension in lowercase"""
    return Path(filename).suffix.lower()

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove potentially dangerous characters
    unsafe_chars = '<>:"/\\|?*'
    for char in unsafe_chars:
        filename = filename.replace(char, '_')
    
    # Limit filename length
    name_part = Path(filename).stem[:200]  # Limit to 200 chars
    extension = Path(filename).suffix[:10]  # Limit extension to 10 chars
    
    return name_part + extension

def get_safe_filename(original_filename: str, prefix: str = None) -> str:
    """
    Generate a safe filename with optional prefix and timestamp.
    
    Args:
        original_filename: Original filename
        prefix: Optional prefix to add
        
    Returns:
        Safe filename for storage
    """
    from datetime import datetime
    
    # Sanitize the original filename
    safe_name = sanitize_filename(original_filename)
    
    # Add timestamp prefix if requested
    if prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = f"{prefix}_{timestamp}_{safe_name}"
    
    return safe_name

def is_file_locked(file_path: Path) -> bool:
    """
    Check if a file is locked (being written to).
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if file appears to be locked/in use
    """
    try:
        # Try to open file in append mode
        with open(file_path, 'a'):
            pass
        return False
    except (OSError, IOError):
        return True

def get_directory_size(directory: Path) -> int:
    """
    Get total size of all files in a directory.
    
    Args:
        directory: Path to directory
        
    Returns:
        Total size in bytes
    """
    total_size = 0
    try:
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
    except (OSError, IOError) as e:
        logger.error("Error calculating directory size", directory=str(directory), error=str(e))
    
    return total_size

def cleanup_temp_files(temp_directory: str, max_age_hours: int = 24) -> int:
    """
    Clean up old temporary files.
    
    Args:
        temp_directory: Path to temporary directory
        max_age_hours: Maximum age of files to keep in hours
        
    Returns:
        Number of files cleaned up
    """
    import time
    
    temp_path = Path(temp_directory)
    if not temp_path.exists():
        return 0
    
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    cleaned_count = 0
    
    try:
        for file_path in temp_path.rglob('*'):
            if file_path.is_file():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        cleaned_count += 1
                        logger.debug("Cleaned up temp file", file=str(file_path))
                    except OSError as e:
                        logger.warning("Failed to clean up temp file", file=str(file_path), error=str(e))
                        
    except (OSError, IOError) as e:
        logger.error("Error during temp file cleanup", directory=temp_directory, error=str(e))
    
    return cleaned_count