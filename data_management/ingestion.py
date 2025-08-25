"""
Data ingestion utilities for gold standard datasets.

Handles importing PDFs, XML ground truth files, and metadata into the
standardized dataset structure with validation and quality control.
"""

import shutil
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
import hashlib
import structlog
from PIL import Image
import PyPDF2
import fitz  # PyMuPDF

from .schema_validator import DatasetValidator, ValidationResult

logger = structlog.get_logger(__name__)


@dataclass
class FileMetadata:
    """Metadata for an ingested file."""
    filename: str
    original_path: str
    file_hash: str
    file_size: int
    ingestion_timestamp: datetime
    brand: str
    file_type: str  # 'pdf', 'xml', 'json'
    validation_status: str  # 'passed', 'failed', 'pending'
    quality_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['ingestion_timestamp'] = self.ingestion_timestamp.isoformat()
        return result


@dataclass
class IngestionReport:
    """Report of data ingestion process."""
    brand: str
    ingestion_timestamp: datetime
    files_processed: int
    files_succeeded: int
    files_failed: int
    validation_results: List[ValidationResult]
    warnings: List[str]
    errors: List[str]
    
    @property
    def success_rate(self) -> float:
        """Percentage of files successfully ingested."""
        return (self.files_succeeded / self.files_processed) * 100 if self.files_processed > 0 else 0.0


class DataIngestionManager:
    """Manages ingestion of files into gold standard dataset structure."""
    
    def __init__(self, data_root: Path = None):
        """
        Initialize data ingestion manager.
        
        Args:
            data_root: Root path for gold standard datasets
        """
        self.data_root = data_root or Path("data/gold_sets")
        self.validator = DatasetValidator(self.data_root)
        self.logger = logger.bind(component="DataIngestion")
        
        # Supported file types
        self.supported_pdf_extensions = {'.pdf'}
        self.supported_xml_extensions = {'.xml'}
        self.supported_metadata_extensions = {'.json'}
        
        # Create data root if it doesn't exist
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def ingest_files(
        self, 
        source_path: Path, 
        brand: str,
        file_type: str = "auto",
        validate_on_ingest: bool = True,
        overwrite_existing: bool = False
    ) -> IngestionReport:
        """
        Ingest files from source directory into brand dataset.
        
        Args:
            source_path: Path to source files or directory
            brand: Target brand for ingestion
            file_type: Type of files to ingest ('pdf', 'xml', 'json', 'auto')
            validate_on_ingest: Whether to validate files during ingestion
            overwrite_existing: Whether to overwrite existing files
            
        Returns:
            IngestionReport with details of the process
        """
        start_time = datetime.now()
        
        self.logger.info("Starting data ingestion",
                        source=str(source_path),
                        brand=brand,
                        file_type=file_type)
        
        # Ensure brand directory structure exists
        self._ensure_brand_structure(brand)
        
        # Collect files to ingest
        files_to_ingest = self._collect_files(source_path, file_type)
        
        if not files_to_ingest:
            return IngestionReport(
                brand=brand,
                ingestion_timestamp=start_time,
                files_processed=0,
                files_succeeded=0,
                files_failed=0,
                validation_results=[],
                warnings=[f"No files found to ingest from {source_path}"],
                errors=[]
            )
        
        # Process each file
        succeeded = 0
        failed = 0
        validation_results = []
        warnings = []
        errors = []
        
        for source_file in files_to_ingest:
            try:
                result = self._ingest_single_file(
                    source_file, 
                    brand, 
                    validate_on_ingest,
                    overwrite_existing
                )
                
                if result["success"]:
                    succeeded += 1
                    if result["validation"]:
                        validation_results.append(result["validation"])
                    if result["warnings"]:
                        warnings.extend(result["warnings"])
                else:
                    failed += 1
                    if result["errors"]:
                        errors.extend(result["errors"])
                    
            except Exception as e:
                failed += 1
                error_msg = f"Failed to ingest {source_file}: {str(e)}"
                errors.append(error_msg)
                self.logger.error("File ingestion error", file=str(source_file), error=str(e))
        
        ingestion_time = datetime.now() - start_time
        
        report = IngestionReport(
            brand=brand,
            ingestion_timestamp=start_time,
            files_processed=len(files_to_ingest),
            files_succeeded=succeeded,
            files_failed=failed,
            validation_results=validation_results,
            warnings=warnings,
            errors=errors
        )
        
        self.logger.info("Data ingestion completed",
                        brand=brand,
                        files_processed=len(files_to_ingest),
                        succeeded=succeeded,
                        failed=failed,
                        success_rate=report.success_rate,
                        duration=ingestion_time.total_seconds())
        
        return report
    
    def _ensure_brand_structure(self, brand: str) -> None:
        """Ensure brand directory structure exists."""
        brand_path = self.data_root / brand
        subdirs = ["pdfs", "ground_truth", "annotations", "metadata"]
        
        for subdir in subdirs:
            (brand_path / subdir).mkdir(parents=True, exist_ok=True)
    
    def _collect_files(self, source_path: Path, file_type: str) -> List[Path]:
        """Collect files to ingest based on type and path."""
        files = []
        
        if source_path.is_file():
            # Single file
            if self._is_supported_file(source_path, file_type):
                files.append(source_path)
        elif source_path.is_dir():
            # Directory - scan for supported files
            if file_type == "auto":
                # Collect all supported file types
                extensions = (
                    self.supported_pdf_extensions | 
                    self.supported_xml_extensions | 
                    self.supported_metadata_extensions
                )
            elif file_type == "pdf":
                extensions = self.supported_pdf_extensions
            elif file_type == "xml":
                extensions = self.supported_xml_extensions
            elif file_type == "json":
                extensions = self.supported_metadata_extensions
            else:
                self.logger.warning("Unknown file type", file_type=file_type)
                return []
            
            for ext in extensions:
                files.extend(source_path.glob(f"*{ext}"))
                files.extend(source_path.glob(f"**/*{ext}"))  # Recursive
        
        return sorted(files)
    
    def _is_supported_file(self, file_path: Path, file_type: str) -> bool:
        """Check if file is supported for ingestion."""
        ext = file_path.suffix.lower()
        
        if file_type == "auto":
            return ext in (
                self.supported_pdf_extensions | 
                self.supported_xml_extensions | 
                self.supported_metadata_extensions
            )
        elif file_type == "pdf":
            return ext in self.supported_pdf_extensions
        elif file_type == "xml":
            return ext in self.supported_xml_extensions
        elif file_type == "json":
            return ext in self.supported_metadata_extensions
        
        return False
    
    def _ingest_single_file(
        self, 
        source_file: Path, 
        brand: str,
        validate_on_ingest: bool,
        overwrite_existing: bool
    ) -> Dict[str, Any]:
        """
        Ingest a single file into the brand dataset.
        
        Returns:
            Dictionary with success status and details
        """
        result = {
            "success": False,
            "validation": None,
            "warnings": [],
            "errors": []
        }
        
        try:
            # Determine file type and target directory
            file_ext = source_file.suffix.lower()
            
            if file_ext in self.supported_pdf_extensions:
                target_subdir = "pdfs"
                detected_type = "pdf"
            elif file_ext in self.supported_xml_extensions:
                target_subdir = "ground_truth"
                detected_type = "xml"
            elif file_ext in self.supported_metadata_extensions:
                target_subdir = "metadata"
                detected_type = "json"
            else:
                result["errors"].append(f"Unsupported file type: {file_ext}")
                return result
            
            # Generate target path
            brand_path = self.data_root / brand
            target_dir = brand_path / target_subdir
            target_file = target_dir / source_file.name
            
            # Check if file already exists
            if target_file.exists() and not overwrite_existing:
                result["warnings"].append(f"File already exists, skipping: {target_file}")
                result["success"] = True
                return result
            
            # Pre-ingestion validation
            if validate_on_ingest:
                pre_validation = self._validate_file_before_ingestion(source_file, detected_type)
                if not pre_validation["valid"]:
                    result["errors"].extend(pre_validation["errors"])
                    return result
                if pre_validation["warnings"]:
                    result["warnings"].extend(pre_validation["warnings"])
            
            # Copy file to target location
            shutil.copy2(source_file, target_file)
            
            # Create file metadata
            metadata = self._create_file_metadata(
                source_file, target_file, brand, detected_type
            )
            
            # Save metadata
            metadata_file = (brand_path / "metadata" / f"{source_file.stem}_file_metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
            
            # Post-ingestion validation
            if validate_on_ingest:
                if detected_type == "xml":
                    validation = self.validator.xml_validator.validate_xml_structure(target_file)
                elif detected_type == "json":
                    validation = self.validator.metadata_validator.validate_metadata(target_file)
                else:
                    # For PDFs, do basic validation
                    validation = self._validate_pdf_file(target_file)
                
                result["validation"] = validation
                
                if not validation.is_valid:
                    result["warnings"].append(f"File ingested but failed validation: {target_file}")
            
            result["success"] = True
            self.logger.debug("File ingested successfully",
                            source=str(source_file),
                            target=str(target_file),
                            type=detected_type)
            
        except Exception as e:
            result["errors"].append(f"Ingestion failed: {str(e)}")
            self.logger.error("File ingestion failed",
                            file=str(source_file),
                            error=str(e),
                            exc_info=True)
        
        return result
    
    def _validate_file_before_ingestion(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Validate file before ingestion to catch obvious issues."""
        validation = {"valid": True, "errors": [], "warnings": []}
        
        try:
            # Check file size
            file_size = file_path.stat().st_size
            
            if file_size == 0:
                validation["valid"] = False
                validation["errors"].append("File is empty")
                return validation
            
            if file_type == "pdf":
                # Basic PDF validation
                try:
                    with open(file_path, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        page_count = len(pdf_reader.pages)
                        
                        if page_count == 0:
                            validation["errors"].append("PDF has no pages")
                            validation["valid"] = False
                        elif page_count > 50:
                            validation["warnings"].append(f"Large PDF with {page_count} pages")
                            
                except Exception as e:
                    validation["errors"].append(f"PDF validation failed: {str(e)}")
                    validation["valid"] = False
            
            elif file_type == "xml":
                # Basic XML validation
                try:
                    import xml.etree.ElementTree as ET
                    ET.parse(file_path)
                except ET.ParseError as e:
                    validation["errors"].append(f"Invalid XML structure: {str(e)}")
                    validation["valid"] = False
            
            elif file_type == "json":
                # Basic JSON validation
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json.load(f)
                except json.JSONDecodeError as e:
                    validation["errors"].append(f"Invalid JSON format: {str(e)}")
                    validation["valid"] = False
            
            # Check file size limits
            size_limits = {
                "pdf": 100 * 1024 * 1024,  # 100MB
                "xml": 10 * 1024 * 1024,   # 10MB
                "json": 1 * 1024 * 1024    # 1MB
            }
            
            if file_size > size_limits.get(file_type, 1024 * 1024):
                validation["warnings"].append(f"Large file size: {file_size / (1024*1024):.1f}MB")
                
        except Exception as e:
            validation["errors"].append(f"Pre-validation failed: {str(e)}")
            validation["valid"] = False
        
        return validation
    
    def _create_file_metadata(
        self, 
        source_file: Path, 
        target_file: Path, 
        brand: str, 
        file_type: str
    ) -> FileMetadata:
        """Create metadata for ingested file."""
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(source_file)
        
        return FileMetadata(
            filename=target_file.name,
            original_path=str(source_file),
            file_hash=file_hash,
            file_size=source_file.stat().st_size,
            ingestion_timestamp=datetime.now(),
            brand=brand,
            file_type=file_type,
            validation_status="pending",
            quality_score=0.0
        )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file for integrity checking."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _validate_pdf_file(self, pdf_path: Path) -> ValidationResult:
        """Basic validation for PDF files."""
        errors = []
        warnings = []
        metadata = {}
        
        try:
            # Use PyMuPDF for detailed PDF analysis
            doc = fitz.open(str(pdf_path))
            
            page_count = doc.page_count
            metadata["page_count"] = page_count
            
            if page_count == 0:
                errors.append("PDF contains no pages")
            
            # Check for text content
            text_pages = 0
            total_text_length = 0
            
            for page_num in range(min(5, page_count)):  # Check first 5 pages
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    text_pages += 1
                    total_text_length += len(text)
            
            doc.close()
            
            metadata["text_pages_sampled"] = text_pages
            metadata["avg_text_length"] = total_text_length / min(5, page_count) if page_count > 0 else 0
            
            if text_pages == 0:
                warnings.append("No text found in sampled pages - may be image-only PDF")
            
            # Calculate basic quality score
            quality_score = 1.0
            if text_pages < min(3, page_count):
                quality_score -= 0.3
            if total_text_length < 1000:
                quality_score -= 0.2
            
            quality_score = max(0.0, quality_score)
            
            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                quality_score=quality_score,
                metadata=metadata
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"PDF validation error: {str(e)}"],
                warnings=[],
                quality_score=0.0,
                metadata={}
            )
    
    def create_dataset_manifest(self, brand: str) -> Dict[str, Any]:
        """
        Create a manifest file describing the complete dataset.
        
        Args:
            brand: Brand name
            
        Returns:
            Dataset manifest dictionary
        """
        brand_path = self.data_root / brand
        
        if not brand_path.exists():
            return {"error": f"Brand directory does not exist: {brand}"}
        
        manifest = {
            "brand": brand,
            "created_timestamp": datetime.now().isoformat(),
            "dataset_version": "1.0",
            "files": {
                "pdfs": [],
                "ground_truth": [],
                "metadata": []
            },
            "statistics": {},
            "validation_summary": {}
        }
        
        # Collect file information
        for file_type, subdir in [("pdfs", "pdfs"), ("ground_truth", "ground_truth"), ("metadata", "metadata")]:
            file_dir = brand_path / subdir
            
            if file_dir.exists():
                files = []
                for file_path in file_dir.iterdir():
                    if file_path.is_file():
                        files.append({
                            "filename": file_path.name,
                            "size": file_path.stat().st_size,
                            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        })
                
                manifest["files"][file_type] = sorted(files, key=lambda x: x["filename"])
        
        # Calculate statistics
        manifest["statistics"] = {
            "total_pdfs": len(manifest["files"]["pdfs"]),
            "total_ground_truth": len(manifest["files"]["ground_truth"]),
            "total_metadata": len(manifest["files"]["metadata"]),
            "complete_triplets": min(
                len(manifest["files"]["pdfs"]),
                len(manifest["files"]["ground_truth"]),
                len(manifest["files"]["metadata"])
            )
        }
        
        # Run validation and add summary
        validation_report = self.validator.validate_brand_dataset(brand)
        manifest["validation_summary"] = {
            "validation_rate": validation_report.validation_rate,
            "average_quality_score": validation_report.average_quality_score,
            "total_errors": sum(len(r.errors) for r in validation_report.file_results),
            "total_warnings": sum(len(r.warnings) for r in validation_report.file_results),
            "recommendations_count": len(validation_report.recommendations)
        }
        
        # Save manifest
        manifest_path = brand_path / "dataset_manifest.json"
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        self.logger.info("Dataset manifest created",
                        brand=brand,
                        path=str(manifest_path),
                        total_files=sum(manifest["statistics"].values()))
        
        return manifest


# Utility functions for common ingestion tasks
def ingest_pdf_with_ground_truth(
    pdf_path: Path,
    xml_path: Path,
    brand: str,
    metadata: Optional[Dict[str, Any]] = None,
    data_root: Path = None
) -> Dict[str, ValidationResult]:
    """
    Convenience function to ingest a PDF with its corresponding ground truth XML.
    
    Args:
        pdf_path: Path to PDF file
        xml_path: Path to XML ground truth file
        brand: Target brand
        metadata: Optional additional metadata
        data_root: Root directory for datasets
        
    Returns:
        Dictionary with validation results for each file
    """
    ingestion_manager = DataIngestionManager(data_root)
    results = {}
    
    # Ingest PDF
    pdf_report = ingestion_manager.ingest_files(pdf_path, brand, "pdf")
    if pdf_report.validation_results:
        results["pdf"] = pdf_report.validation_results[0]
    
    # Ingest XML
    xml_report = ingestion_manager.ingest_files(xml_path, brand, "xml")
    if xml_report.validation_results:
        results["xml"] = xml_report.validation_results[0]
    
    # Create paired metadata if provided
    if metadata:
        brand_path = (data_root or Path("data/gold_sets")) / brand
        metadata_path = brand_path / "metadata" / f"{pdf_path.stem}_metadata.json"
        
        paired_metadata = {
            "dataset_info": {
                "brand": brand,
                "filename": pdf_path.name,
                "creation_date": datetime.now().isoformat(),
                "file_type": "pdf_xml_pair"
            },
            "quality_metrics": {
                "manual_validation": False,
                "annotation_quality": 0.9,  # Default
                "completeness_score": 0.9   # Default
            },
            "content_info": {
                "page_count": 1,  # Will be updated by validation
                "article_count": 1,  # Will be updated by validation  
                "layout_complexity": "standard"
            },
            "custom_metadata": metadata
        }
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(paired_metadata, f, indent=2, ensure_ascii=False)
    
    return results