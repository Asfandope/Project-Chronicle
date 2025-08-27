"""
Image extractor with deterministic naming for PDF documents.
Extracts images larger than 100x100px with comprehensive metadata.
"""

import hashlib
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import structlog
from PIL import Image

from .types import BoundingBox, ImageInfo, PDFProcessingError

logger = structlog.get_logger(__name__)


class ImageExtractionError(PDFProcessingError):
    """Exception raised when image extraction fails."""


class ImageExtractor:
    """
    Extracts images from PDF pages with deterministic naming and metadata.

    Handles both embedded images and page renderings for scanned PDFs.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        min_width: int = 100,
        min_height: int = 100,
        min_area: int = 10000,  # 100x100 = 10,000 pixels
        max_file_size_mb: int = 50,
        supported_formats: Optional[List[str]] = None,
    ):
        """
        Initialize image extractor.

        Args:
            output_dir: Directory to save extracted images
            min_width: Minimum image width in pixels
            min_height: Minimum image height in pixels
            min_area: Minimum image area in pixels
            max_file_size_mb: Maximum file size for extracted images
            supported_formats: List of supported image formats
        """
        self.output_dir = output_dir
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.supported_formats = supported_formats or [
            "PNG",
            "JPEG",
            "JPG",
            "TIFF",
            "BMP",
        ]
        self.logger = logger.bind(component="ImageExtractor")

        # Ensure output directory exists
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_from_page(
        self,
        pdf_path: Path,
        page_num: int,
        save_images: bool = True,
        include_page_render: bool = False,
    ) -> List[ImageInfo]:
        """
        Extract images from a specific page.

        Args:
            pdf_path: Path to PDF file
            page_num: Page number (1-indexed)
            save_images: Whether to save extracted images to disk
            include_page_render: Whether to include full page render as image

        Returns:
            List of ImageInfo objects

        Raises:
            ImageExtractionError: If extraction fails
        """
        try:
            self.logger.info(
                "Extracting images from page", pdf_path=str(pdf_path), page_num=page_num
            )

            doc = fitz.open(str(pdf_path))

            try:
                if page_num < 1 or page_num > doc.page_count:
                    raise ImageExtractionError(
                        f"Invalid page number: {page_num} (PDF has {doc.page_count} pages)",
                        pdf_path,
                        page_num,
                    )

                page = doc[page_num - 1]  # Convert to 0-indexed

                images = []

                # Extract embedded images
                embedded_images = self._extract_embedded_images(
                    page, pdf_path, page_num, save_images
                )
                images.extend(embedded_images)

                # Extract page render if requested (useful for scanned PDFs)
                if include_page_render:
                    page_render = self._extract_page_render(
                        page, pdf_path, page_num, save_images
                    )
                    if page_render:
                        images.append(page_render)

                self.logger.info(
                    "Image extraction completed",
                    page_num=page_num,
                    images_found=len(images),
                )

                return images

            finally:
                doc.close()

        except ImageExtractionError:
            raise
        except Exception as e:
            self.logger.error(
                "Unexpected error during image extraction",
                pdf_path=str(pdf_path),
                page_num=page_num,
                error=str(e),
                exc_info=True,
            )
            raise ImageExtractionError(
                f"Unexpected image extraction error: {str(e)}", pdf_path, page_num
            )

    def extract_from_pdf(
        self,
        pdf_path: Path,
        page_range: Optional[Tuple[int, int]] = None,
        save_images: bool = True,
        include_page_renders: bool = False,
    ) -> Dict[int, List[ImageInfo]]:
        """
        Extract images from all pages or a range of pages.

        Args:
            pdf_path: Path to PDF file
            page_range: Optional tuple of (start_page, end_page) 1-indexed
            save_images: Whether to save extracted images to disk
            include_page_renders: Whether to include full page renders

        Returns:
            Dictionary mapping page numbers to lists of ImageInfo objects

        Raises:
            ImageExtractionError: If extraction fails
        """
        start_time = time.time()

        try:
            self.logger.info(
                "Starting image extraction from PDF",
                pdf_path=str(pdf_path),
                page_range=page_range,
            )

            doc = fitz.open(str(pdf_path))

            try:
                total_pages = doc.page_count

                # Determine page range
                if page_range:
                    start_page, end_page = page_range
                    start_page = max(1, start_page)
                    end_page = min(total_pages, end_page)
                else:
                    start_page, end_page = 1, total_pages

                results = {}

                for page_num in range(start_page, end_page + 1):
                    try:
                        images = self.extract_from_page(
                            pdf_path, page_num, save_images, include_page_renders
                        )
                        results[page_num] = images

                    except Exception as e:
                        self.logger.error(
                            "Error extracting images from page",
                            page_num=page_num,
                            error=str(e),
                        )
                        results[page_num] = []  # Continue with other pages

                processing_time = time.time() - start_time
                total_images = sum(len(images) for images in results.values())

                self.logger.info(
                    "PDF image extraction completed",
                    pdf_path=str(pdf_path),
                    pages_processed=len(results),
                    total_images=total_images,
                    processing_time=processing_time,
                )

                return results

            finally:
                doc.close()

        except Exception as e:
            self.logger.error(
                "Unexpected error during PDF image extraction",
                pdf_path=str(pdf_path),
                error=str(e),
                exc_info=True,
            )
            raise ImageExtractionError(
                f"Unexpected PDF image extraction error: {str(e)}", pdf_path
            )

    def _extract_embedded_images(
        self, page: fitz.Page, pdf_path: Path, page_num: int, save_images: bool
    ) -> List[ImageInfo]:
        """Extract embedded images from a page."""
        try:
            images = []
            image_list = page.get_images()

            for img_index, img_ref in enumerate(image_list):
                try:
                    # Get image reference and extract
                    xref = img_ref[0]
                    pix = fitz.Pixmap(page.parent, xref)

                    # Skip if image is too small
                    if pix.width < self.min_width or pix.height < self.min_height:
                        pix = None
                        continue

                    if pix.width * pix.height < self.min_area:
                        pix = None
                        continue

                    # Convert CMYK to RGB if necessary
                    if pix.n - pix.alpha > 3:  # CMYK
                        pix = fitz.Pixmap(fitz.csRGB, pix)

                    # Get image data and format
                    img_data = pix.tobytes("png")
                    img_format = "PNG"  # Default to PNG for consistent output

                    # Check file size
                    if len(img_data) > self.max_file_size_bytes:
                        self.logger.warning(
                            "Image too large, skipping",
                            page_num=page_num,
                            img_index=img_index,
                            size_mb=len(img_data) / 1024 / 1024,
                        )
                        pix = None
                        continue

                    # Generate deterministic image ID and filename
                    image_id = self._generate_image_id(
                        pdf_path, page_num, img_index, pix.width, pix.height
                    )

                    # Get bounding box (approximate, as embedded images don't have exact placement)
                    bbox = self._estimate_image_bbox(
                        page, img_index, pix.width, pix.height
                    )

                    # Save image if requested
                    file_path = None
                    if save_images and self.output_dir:
                        filename = self._generate_deterministic_filename(
                            pdf_path.stem, page_num, image_id, img_format
                        )
                        file_path = self.output_dir / filename

                        with open(file_path, "wb") as f:
                            f.write(img_data)

                    # Classify image type using basic heuristics
                    is_photo, is_chart, is_diagram = self._classify_image_type(pix)

                    # Create ImageInfo object
                    image_info = ImageInfo(
                        image_id=image_id,
                        bbox=bbox,
                        width=pix.width,
                        height=pix.height,
                        format=img_format,
                        file_path=file_path or Path(),
                        file_size=len(img_data),
                        page_num=page_num,
                        confidence=1.0,  # Embedded images have full confidence
                        is_photo=is_photo,
                        is_chart=is_chart,
                        is_diagram=is_diagram,
                    )

                    images.append(image_info)
                    pix = None  # Clean up

                except Exception as e:
                    self.logger.warning(
                        "Error extracting embedded image",
                        page_num=page_num,
                        img_index=img_index,
                        error=str(e),
                    )
                    continue

            return images

        except Exception as e:
            self.logger.error(
                "Error in embedded image extraction", page_num=page_num, error=str(e)
            )
            return []

    def _extract_page_render(
        self, page: fitz.Page, pdf_path: Path, page_num: int, save_images: bool
    ) -> Optional[ImageInfo]:
        """Extract full page render as image (useful for scanned PDFs)."""
        try:
            # Render page at reasonable resolution (150 DPI)
            matrix = fitz.Matrix(150 / 72, 150 / 72)  # 72 DPI is default
            pix = page.get_pixmap(matrix=matrix)

            # Check if rendered image meets size requirements
            if pix.width < self.min_width or pix.height < self.min_height:
                pix = None
                return None

            if pix.width * pix.height < self.min_area:
                pix = None
                return None

            # Convert to RGB if necessary
            if pix.n - pix.alpha > 3:  # CMYK
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # Get image data
            img_data = pix.tobytes("png")
            img_format = "PNG"

            # Check file size
            if len(img_data) > self.max_file_size_bytes:
                self.logger.warning(
                    "Page render too large, skipping",
                    page_num=page_num,
                    size_mb=len(img_data) / 1024 / 1024,
                )
                pix = None
                return None

            # Generate deterministic image ID for page render
            image_id = self._generate_page_render_id(
                pdf_path, page_num, pix.width, pix.height
            )

            # Full page bounding box
            bbox = BoundingBox(0, 0, page.rect.width, page.rect.height)

            # Save image if requested
            file_path = None
            if save_images and self.output_dir:
                filename = f"{pdf_path.stem}_page{page_num:03d}_render.png"
                file_path = self.output_dir / filename

                with open(file_path, "wb") as f:
                    f.write(img_data)

            # Create ImageInfo object
            image_info = ImageInfo(
                image_id=image_id,
                bbox=bbox,
                width=pix.width,
                height=pix.height,
                format=img_format,
                file_path=file_path or Path(),
                file_size=len(img_data),
                page_num=page_num,
                confidence=0.9,  # Page renders have slightly lower confidence
                is_photo=False,
                is_chart=False,
                is_diagram=True,  # Page render is considered a diagram
            )

            pix = None  # Clean up
            return image_info

        except Exception as e:
            self.logger.warning(
                "Error extracting page render", page_num=page_num, error=str(e)
            )
            return None

    def _generate_image_id(
        self, pdf_path: Path, page_num: int, img_index: int, width: int, height: int
    ) -> str:
        """Generate deterministic image ID."""
        # Create content for hashing that uniquely identifies this image
        content = f"{pdf_path.name}_{page_num}_{img_index}_{width}_{height}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_page_render_id(
        self, pdf_path: Path, page_num: int, width: int, height: int
    ) -> str:
        """Generate deterministic image ID for page render."""
        content = f"{pdf_path.name}_page_render_{page_num}_{width}_{height}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def _generate_deterministic_filename(
        self, pdf_name: str, page_num: int, image_id: str, format: str
    ) -> str:
        """Generate deterministic filename for extracted image."""
        return f"{pdf_name}_page{page_num:03d}_{image_id}.{format.lower()}"

    def _estimate_image_bbox(
        self, page: fitz.Page, img_index: int, width: int, height: int
    ) -> BoundingBox:
        """
        Estimate bounding box for embedded image.

        Note: PyMuPDF doesn't always provide exact positioning for embedded images.
        This provides a reasonable estimate based on page layout.
        """
        try:
            # Try to get image blocks which may include positioning
            blocks = page.get_text("dict")

            # Look for image blocks
            image_blocks = [
                block for block in blocks.get("blocks", []) if "lines" not in block
            ]

            if img_index < len(image_blocks) and "bbox" in image_blocks[img_index]:
                x0, y0, x1, y1 = image_blocks[img_index]["bbox"]
                return BoundingBox(x0, y0, x1, y1)

            # Fallback: estimate based on page layout and image index
            page_width, page_height = page.rect.width, page.rect.height

            # Simple grid-based estimation
            cols = 2 if page_width > page_height else 1
            rows = max(1, (img_index + 1) // cols + 1)

            col = img_index % cols
            row = img_index // cols

            cell_width = page_width / cols
            cell_height = page_height / rows

            x0 = col * cell_width
            y0 = row * cell_height
            x1 = x0 + min(cell_width, width * 0.75)  # Assume 75% of cell width
            y1 = y0 + min(cell_height, height * 0.75)

            return BoundingBox(x0, y0, x1, y1)

        except Exception:
            # Ultimate fallback: center of page
            page_width, page_height = page.rect.width, page.rect.height
            img_width = min(width * 0.75, page_width * 0.5)
            img_height = min(height * 0.75, page_height * 0.5)

            x0 = (page_width - img_width) / 2
            y0 = (page_height - img_height) / 2
            x1 = x0 + img_width
            y1 = y0 + img_height

            return BoundingBox(x0, y0, x1, y1)

    def _classify_image_type(self, pix: fitz.Pixmap) -> Tuple[bool, bool, bool]:
        """
        Classify image type using basic heuristics.

        Returns:
            Tuple of (is_photo, is_chart, is_diagram)
        """
        try:
            # Convert to PIL Image for analysis
            img_data = pix.tobytes("png")
            pil_img = Image.open(BytesIO(img_data))

            # Basic color analysis
            if pil_img.mode in ["RGB", "RGBA"]:
                # Convert to RGB for consistent analysis
                if pil_img.mode == "RGBA":
                    pil_img = pil_img.convert("RGB")

                # Get color statistics
                colors = pil_img.getcolors(maxcolors=256 * 256 * 256)

                if colors:
                    unique_colors = len(colors)
                    total_pixels = pil_img.width * pil_img.height

                    # Heuristics for image classification
                    color_ratio = unique_colors / total_pixels

                    # Photos typically have many unique colors
                    if color_ratio > 0.1 and unique_colors > 1000:
                        return True, False, False  # is_photo

                    # Charts/diagrams typically have fewer, distinct colors
                    elif unique_colors < 50:
                        return False, True, False  # is_chart

                    # Diagrams are in between
                    else:
                        return False, False, True  # is_diagram

            # Default classification
            return False, False, True  # Assume diagram if unclear

        except Exception as e:
            self.logger.warning("Error classifying image type", error=str(e))
            return False, False, True  # Default to diagram

    def get_image_summary(self, images: List[ImageInfo]) -> Dict[str, Any]:
        """Get summary statistics for extracted images."""
        if not images:
            return {
                "total_images": 0,
                "total_size_bytes": 0,
                "avg_width": 0,
                "avg_height": 0,
                "formats": {},
                "types": {"photos": 0, "charts": 0, "diagrams": 0},
                "size_distribution": {"small": 0, "medium": 0, "large": 0},
            }

        total_size = sum(img.file_size for img in images)
        avg_width = sum(img.width for img in images) / len(images)
        avg_height = sum(img.height for img in images) / len(images)

        # Count formats
        formats = {}
        for img in images:
            formats[img.format] = formats.get(img.format, 0) + 1

        # Count types
        types = {
            "photos": sum(1 for img in images if img.is_photo),
            "charts": sum(1 for img in images if img.is_chart),
            "diagrams": sum(1 for img in images if img.is_diagram),
        }

        # Size distribution (based on pixel area)
        size_dist = {"small": 0, "medium": 0, "large": 0}
        for img in images:
            area = img.area_pixels
            if area < 50000:  # < 50k pixels
                size_dist["small"] += 1
            elif area < 500000:  # 50k - 500k pixels
                size_dist["medium"] += 1
            else:  # > 500k pixels
                size_dist["large"] += 1

        return {
            "total_images": len(images),
            "total_size_bytes": total_size,
            "avg_width": avg_width,
            "avg_height": avg_height,
            "formats": formats,
            "types": types,
            "size_distribution": size_dist,
        }

    def cleanup_extracted_images(self, images: List[ImageInfo]) -> int:
        """
        Clean up extracted image files.

        Args:
            images: List of ImageInfo objects with file paths to delete

        Returns:
            Number of files successfully deleted
        """
        deleted_count = 0

        for image in images:
            try:
                if image.file_path and image.file_path.exists():
                    image.file_path.unlink()
                    deleted_count += 1
            except Exception as e:
                self.logger.warning(
                    "Error deleting image file",
                    file_path=str(image.file_path),
                    error=str(e),
                )

        return deleted_count
