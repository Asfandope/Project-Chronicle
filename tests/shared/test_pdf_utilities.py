"""
Unit tests for shared PDF utilities.
"""

import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from shared.pdf import (
    BoundingBox,
    ImageExtractor,
    ImageInfo,
    MetadataExtractionError,
    MetadataExtractor,
    PageSplitError,
    PageSplitter,
    PDFValidationError,
    PDFValidator,
    TextBlock,
    TextBlockExtractor,
    TextExtractionError,
)
from shared.pdf.utils import (
    PDFProcessingError,
    ProgressTracker,
    create_output_directory,
    format_file_size,
    get_file_hash,
    validate_page_number,
    validate_page_range,
    validate_pdf_path,
)


class TestPDFValidator:
    """Test PDF validator functionality."""

    def test_init(self):
        """Test validator initialization."""
        validator = PDFValidator(max_file_size_mb=100, min_pages=1, max_pages=500)
        assert validator.max_file_size_bytes == 100 * 1024 * 1024
        assert validator.min_pages == 1
        assert validator.max_pages == 500

    def test_validate_nonexistent_file(self):
        """Test validation of non-existent file."""
        validator = PDFValidator()
        with pytest.raises(PDFValidationError, match="does not exist"):
            validator.validate(Path("/nonexistent/file.pdf"))

    @patch("fitz.open")
    def test_validate_corrupted_pdf(self, mock_fitz_open):
        """Test validation of corrupted PDF."""
        mock_fitz_open.side_effect = Exception("Cannot open PDF")

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake pdf content")

        try:
            validator = PDFValidator()
            with pytest.raises(PDFValidationError, match="Cannot open PDF"):
                validator.validate(temp_path)
        finally:
            temp_path.unlink()

    @patch("fitz.open")
    def test_validate_empty_file(self, mock_fitz_open):
        """Test validation of empty file."""
        # Create an empty file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            validator = PDFValidator()
            with pytest.raises(PDFValidationError, match="empty"):
                validator.validate(temp_path)
        finally:
            temp_path.unlink()

    @patch("fitz.open")
    def test_quick_validate_success(self, mock_fitz_open):
        """Test quick validation success."""
        # Mock a valid PDF document
        mock_doc = Mock()
        mock_doc.is_closed = False
        mock_doc.page_count = 5
        mock_doc.needs_pass = False
        mock_doc.metadata = {}
        mock_doc.pdf_version.return_value = (1, 4)
        mock_doc.is_fast_web_view = False
        mock_doc.permissions = 0xFF
        mock_fitz_open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake pdf content" * 1000)  # Make it non-empty

        try:
            validator = PDFValidator()
            result = validator.quick_validate(temp_path)
            assert result is True
        finally:
            temp_path.unlink()

    @patch("fitz.open")
    def test_validate_too_many_pages(self, mock_fitz_open):
        """Test validation of PDF with too many pages."""
        mock_doc = Mock()
        mock_doc.is_closed = False
        mock_doc.page_count = 2000  # Exceeds default max of 1000
        mock_doc.needs_pass = False
        mock_doc.metadata = {}
        mock_fitz_open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"fake pdf content" * 1000)

        try:
            validator = PDFValidator(max_pages=1000)
            pdf_info = validator.validate(temp_path)
            assert not pdf_info.is_valid
            assert any(
                "Too many pages" in error for error in pdf_info.validation_errors
            )
        finally:
            temp_path.unlink()


class TestPageSplitter:
    """Test page splitter functionality."""

    def test_init(self):
        """Test splitter initialization."""
        output_dir = Path("/tmp/test")
        splitter = PageSplitter(output_dir=output_dir, preserve_metadata=True)
        assert splitter.output_dir == output_dir
        assert splitter.preserve_metadata is True

    @patch("fitz.open")
    def test_split_to_files_invalid_pdf(self, mock_fitz_open):
        """Test splitting invalid PDF."""
        mock_fitz_open.side_effect = Exception("Cannot open PDF")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            splitter = PageSplitter()
            with pytest.raises(PageSplitError, match="Cannot open"):
                splitter.split_to_files(temp_path, validate_input=False)
        finally:
            temp_path.unlink()

    def test_get_page_info_nonexistent_file(self):
        """Test getting page info from non-existent file."""
        splitter = PageSplitter()
        with pytest.raises(PageSplitError):
            splitter.get_page_info(Path("/nonexistent/file.pdf"))


class TestTextBlockExtractor:
    """Test text block extractor functionality."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = TextBlockExtractor(
            min_confidence=0.8, min_text_length=5, merge_nearby_blocks=True
        )
        assert extractor.min_confidence == 0.8
        assert extractor.min_text_length == 5
        assert extractor.merge_nearby_blocks is True

    @patch("fitz.open")
    def test_extract_from_page_invalid_page(self, mock_fitz_open):
        """Test extracting from invalid page number."""
        mock_doc = Mock()
        mock_doc.page_count = 5
        mock_fitz_open.return_value = mock_doc

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            extractor = TextBlockExtractor()
            with pytest.raises(TextExtractionError, match="Invalid page number"):
                extractor.extract_from_page(temp_path, 10)
        finally:
            temp_path.unlink()

    def test_classify_text_type(self):
        """Test text type classification."""
        extractor = TextBlockExtractor()

        # Test title classification
        result = extractor._classify_text_type(
            "Short Bold Title", {"bold": True, "size": 16}
        )
        assert result == "title"

        # Test paragraph classification
        result = extractor._classify_text_type(
            "This is a longer text that should be classified as a paragraph.",
            {"bold": False, "size": 12},
        )
        assert result == "paragraph"

        # Test caption classification
        result = extractor._classify_text_type(
            "Photo by John Doe", {"bold": False, "size": 10}
        )
        assert result == "attribution"

    def test_merge_nearby_blocks(self):
        """Test merging of nearby text blocks."""
        extractor = TextBlockExtractor(merge_distance_threshold=15.0)

        # Create two nearby blocks that should be merged
        block1 = TextBlock(
            text="First block",
            bbox=BoundingBox(10, 10, 100, 30),
            confidence=0.9,
            page_num=1,
        )

        block2 = TextBlock(
            text="Second block",
            bbox=BoundingBox(10, 35, 100, 55),  # Close vertically
            confidence=0.9,
            page_num=1,
        )

        blocks = [block1, block2]
        merged = extractor._merge_nearby_blocks(blocks)

        assert len(merged) == 1
        assert "First block" in merged[0].text
        assert "Second block" in merged[0].text

    def test_filter_blocks(self):
        """Test filtering of text blocks."""
        extractor = TextBlockExtractor(min_confidence=0.8, min_text_length=5)

        blocks = [
            TextBlock("Good block", BoundingBox(0, 0, 100, 20), 0.9, page_num=1),
            TextBlock(
                "Low confidence", BoundingBox(0, 30, 100, 50), 0.5, page_num=1
            ),  # Too low confidence
            TextBlock("Hi", BoundingBox(0, 60, 100, 80), 0.9, page_num=1),  # Too short
            TextBlock(
                "   ", BoundingBox(0, 90, 100, 110), 0.9, page_num=1
            ),  # Just whitespace
        ]

        filtered = extractor._filter_blocks(blocks)
        assert len(filtered) == 1
        assert filtered[0].text == "Good block"


class TestImageExtractor:
    """Test image extractor functionality."""

    def test_init(self):
        """Test extractor initialization."""
        output_dir = Path("/tmp/images")
        extractor = ImageExtractor(
            output_dir=output_dir, min_width=150, min_height=150, min_area=20000
        )
        assert extractor.output_dir == output_dir
        assert extractor.min_width == 150
        assert extractor.min_height == 150
        assert extractor.min_area == 20000

    def test_generate_image_id(self):
        """Test deterministic image ID generation."""
        extractor = ImageExtractor()

        pdf_path = Path("test.pdf")
        page_num = 1
        img_index = 0
        width, height = 200, 300

        id1 = extractor._generate_image_id(pdf_path, page_num, img_index, width, height)
        id2 = extractor._generate_image_id(pdf_path, page_num, img_index, width, height)

        # Should be deterministic
        assert id1 == id2
        assert len(id1) == 16  # MD5 hash truncated to 16 chars

    def test_generate_deterministic_filename(self):
        """Test deterministic filename generation."""
        extractor = ImageExtractor()

        filename = extractor._generate_deterministic_filename(
            "test", 1, "abc123", "PNG"
        )
        assert filename == "test_page001_abc123.png"

    def test_estimate_image_bbox(self):
        """Test image bounding box estimation."""
        extractor = ImageExtractor()

        # Create a mock page
        mock_page = Mock()
        mock_page.rect.width = 612
        mock_page.rect.height = 792
        mock_page.get_text.return_value = {"blocks": []}

        bbox = extractor._estimate_image_bbox(mock_page, 0, 200, 300)

        assert isinstance(bbox, BoundingBox)
        assert bbox.x0 >= 0
        assert bbox.y0 >= 0
        assert bbox.x1 <= mock_page.rect.width
        assert bbox.y1 <= mock_page.rect.height

    def test_get_image_summary(self):
        """Test image summary generation."""
        extractor = ImageExtractor()

        images = [
            ImageInfo(
                "img1",
                BoundingBox(0, 0, 200, 300),
                200,
                300,
                "PNG",
                Path("img1.png"),
                1000,
                1,
                is_photo=True,
            ),
            ImageInfo(
                "img2",
                BoundingBox(0, 0, 100, 100),
                100,
                100,
                "JPEG",
                Path("img2.jpg"),
                500,
                1,
                is_chart=True,
            ),
        ]

        summary = extractor.get_image_summary(images)

        assert summary["total_images"] == 2
        assert summary["total_size_bytes"] == 1500
        assert summary["formats"]["PNG"] == 1
        assert summary["formats"]["JPEG"] == 1
        assert summary["types"]["photos"] == 1
        assert summary["types"]["charts"] == 1


class TestMetadataExtractor:
    """Test metadata extractor functionality."""

    def test_init(self):
        """Test extractor initialization."""
        extractor = MetadataExtractor(extract_xmp=True, analyze_structure=True)
        assert extractor.extract_xmp is True
        assert extractor.analyze_structure is True

    def test_parse_pdf_date(self):
        """Test PDF date parsing."""
        extractor = MetadataExtractor()

        # Test standard PDF date format
        date1 = extractor._parse_pdf_date("D:20231215143022")
        assert date1.year == 2023
        assert date1.month == 12
        assert date1.day == 15

        # Test invalid date
        date2 = extractor._parse_pdf_date("invalid")
        assert date2 is None

        # Test None input
        date3 = extractor._parse_pdf_date(None)
        assert date3 is None

    def test_parse_keywords(self):
        """Test keyword parsing."""
        extractor = MetadataExtractor()

        # Test semicolon-separated keywords
        keywords1 = extractor._parse_keywords("keyword1;keyword2;keyword3")
        assert keywords1 == ["keyword1", "keyword2", "keyword3"]

        # Test comma-separated keywords
        keywords2 = extractor._parse_keywords("keyword1,keyword2,keyword3")
        assert keywords2 == ["keyword1", "keyword2", "keyword3"]

        # Test single keyword
        keywords3 = extractor._parse_keywords("single_keyword")
        assert keywords3 == ["single_keyword"]

        # Test empty input
        keywords4 = extractor._parse_keywords("")
        assert keywords4 == []

    def test_clean_string(self):
        """Test string cleaning."""
        extractor = MetadataExtractor()

        assert extractor._clean_string("  test  ") == "test"
        assert extractor._clean_string("") is None
        assert extractor._clean_string(None) is None
        assert extractor._clean_string("test\\x00with\\0nulls") == "testwith nulls"

    @patch("fitz.open")
    def test_extract_metadata_error_handling(self, mock_fitz_open):
        """Test metadata extraction error handling."""
        mock_fitz_open.side_effect = Exception("Cannot open PDF")

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            extractor = MetadataExtractor()
            with pytest.raises(MetadataExtractionError):
                extractor.extract(temp_path)
        finally:
            temp_path.unlink()


class TestBoundingBox:
    """Test BoundingBox functionality."""

    def test_init(self):
        """Test bounding box initialization."""
        bbox = BoundingBox(10, 20, 100, 200)
        assert bbox.x0 == 10
        assert bbox.y0 == 20
        assert bbox.x1 == 100
        assert bbox.y1 == 200

    def test_properties(self):
        """Test bounding box properties."""
        bbox = BoundingBox(10, 20, 100, 200)

        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.area == 16200
        assert bbox.center == (55, 110)

    def test_overlaps(self):
        """Test bounding box overlap detection."""
        bbox1 = BoundingBox(10, 10, 50, 50)
        bbox2 = BoundingBox(30, 30, 70, 70)  # Overlaps
        bbox3 = BoundingBox(60, 60, 100, 100)  # No overlap

        assert bbox1.overlaps(bbox2) is True
        assert bbox1.overlaps(bbox3) is False
        assert bbox2.overlaps(bbox3) is True

    def test_intersection_area(self):
        """Test intersection area calculation."""
        bbox1 = BoundingBox(10, 10, 50, 50)
        bbox2 = BoundingBox(30, 30, 70, 70)

        intersection = bbox1.intersection_area(bbox2)
        assert intersection == 400  # 20x20 overlap


class TestUtils:
    """Test utility functions."""

    def test_validate_pdf_path(self):
        """Test PDF path validation."""
        # Test non-existent file
        with pytest.raises(PDFProcessingError, match="does not exist"):
            validate_pdf_path("/nonexistent/file.pdf")

        # Test non-PDF file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            with pytest.raises(PDFProcessingError, match="not a PDF"):
                validate_pdf_path(temp_path)
        finally:
            temp_path.unlink()

    def test_validate_page_number(self):
        """Test page number validation."""
        # Valid page number
        assert validate_page_number(1, 10) == 1
        assert validate_page_number(5, 10) == 5

        # Invalid page numbers
        with pytest.raises(PDFProcessingError, match="must be positive"):
            validate_page_number(0, 10)

        with pytest.raises(PDFProcessingError, match="exceeds document pages"):
            validate_page_number(15, 10)

        with pytest.raises(PDFProcessingError, match="must be an integer"):
            validate_page_number("5", 10)

    def test_validate_page_range(self):
        """Test page range validation."""
        # Valid range
        assert validate_page_range((1, 5), 10) == (1, 5)

        # Invalid range
        with pytest.raises(PDFProcessingError, match="must be a tuple"):
            validate_page_range([1, 5], 10)

        with pytest.raises(PDFProcessingError, match="cannot be greater"):
            validate_page_range((5, 1), 10)

    def test_create_output_directory(self):
        """Test output directory creation."""
        # Test with None (temporary directory)
        temp_dir = create_output_directory(None, "test")
        assert temp_dir.exists()
        assert temp_dir.is_dir()

        # Clean up
        shutil.rmtree(temp_dir)

        # Test with specific directory
        with tempfile.TemporaryDirectory() as temp_base:
            output_dir = Path(temp_base) / "test_output"
            result_dir = create_output_directory(output_dir)
            assert result_dir == output_dir
            assert output_dir.exists()

    def test_format_file_size(self):
        """Test file size formatting."""
        assert format_file_size(100) == "100 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1024 * 1024) == "1.0 MB"
        assert format_file_size(1024 * 1024 * 1024) == "1.0 GB"

    def test_get_file_hash(self):
        """Test file hash calculation."""
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(b"test content")

        try:
            hash1 = get_file_hash(temp_path, "md5")
            hash2 = get_file_hash(temp_path, "md5")

            assert hash1 == hash2  # Should be deterministic
            assert len(hash1) == 32  # MD5 hash length
        finally:
            temp_path.unlink()


class TestProgressTracker:
    """Test progress tracker functionality."""

    def test_init(self):
        """Test progress tracker initialization."""
        tracker = ProgressTracker(100, "Test Operation")
        assert tracker.total_items == 100
        assert tracker.operation == "Test Operation"
        assert tracker.processed_items == 0

    def test_update(self):
        """Test progress updates."""
        tracker = ProgressTracker(100)

        tracker.update(10)
        assert tracker.processed_items == 10

        tracker.update(5)
        assert tracker.processed_items == 15

    def test_complete(self):
        """Test completion logging."""
        tracker = ProgressTracker(10)
        tracker.update(10)

        # Should not raise any exceptions
        tracker.complete()


class TestTextBlock:
    """Test TextBlock functionality."""

    def test_init(self):
        """Test text block initialization."""
        bbox = BoundingBox(10, 20, 100, 120)
        block = TextBlock(
            text="Test text", bbox=bbox, confidence=0.95, font_size=12.0, page_num=1
        )

        assert block.text == "Test text"
        assert block.bbox == bbox
        assert block.confidence == 0.95
        assert block.font_size == 12.0
        assert block.page_num == 1

    def test_properties(self):
        """Test text block properties."""
        bbox = BoundingBox(10, 20, 100, 120)
        block = TextBlock("Hello world test", bbox, 0.9, page_num=1)

        assert block.word_count == 3
        assert block.char_count == 16

    def test_get_hash(self):
        """Test text block hash generation."""
        bbox = BoundingBox(10, 20, 100, 120)
        block = TextBlock("Test text", bbox, 0.9, page_num=1)

        hash1 = block.get_hash()
        hash2 = block.get_hash()

        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 32  # MD5 hash length


class TestImageInfo:
    """Test ImageInfo functionality."""

    def test_init(self):
        """Test image info initialization."""
        bbox = BoundingBox(10, 20, 110, 170)
        image = ImageInfo(
            image_id="test123",
            bbox=bbox,
            width=200,
            height=300,
            format="PNG",
            file_path=Path("test.png"),
            file_size=1000,
            page_num=1,
        )

        assert image.image_id == "test123"
        assert image.width == 200
        assert image.height == 300
        assert image.format == "PNG"

    def test_properties(self):
        """Test image info properties."""
        bbox = BoundingBox(10, 20, 110, 170)
        image = ImageInfo("test", bbox, 200, 300, "PNG", Path("test.png"), 1000, 1)

        assert image.aspect_ratio == pytest.approx(200 / 300, rel=1e-3)
        assert image.area_pixels == 60000

    def test_get_deterministic_filename(self):
        """Test deterministic filename generation."""
        bbox = BoundingBox(10, 20, 110, 170)
        image = ImageInfo("test", bbox, 200, 300, "PNG", Path("test.png"), 1000, 1)

        filename1 = image.get_deterministic_filename("document")
        filename2 = image.get_deterministic_filename("document")

        assert filename1 == filename2  # Should be deterministic
        assert filename1.startswith("document_page001_")
        assert filename1.endswith(".png")
