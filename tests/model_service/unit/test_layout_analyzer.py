from unittest.mock import Mock

import pytest
from model_service.models.layout_analyzer import LayoutAnalyzer


@pytest.mark.unit
@pytest.mark.asyncio
class TestLayoutAnalyzer:
    """Test layout analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model_manager = Mock()
        self.analyzer = LayoutAnalyzer(self.mock_model_manager)

    async def test_analyze_pdf_layout_basic(self):
        """Test basic PDF layout analysis."""
        # Mock model manager responses
        self.mock_model_manager.is_model_loaded.return_value = True
        self.mock_model_manager.get_model.return_value = Mock()

        result = await self.analyzer.analyze_pdf_layout(
            job_id="test-job-123", file_path="/path/to/test.pdf"
        )

        assert "pages" in result
        assert "semantic_graph" in result
        assert "confidence_scores" in result

        # Check basic structure
        assert isinstance(result["pages"], dict)
        assert isinstance(result["semantic_graph"], dict)
        assert isinstance(result["confidence_scores"], dict)

    async def test_analyze_pdf_layout_with_brand_config(self):
        """Test layout analysis with brand-specific configuration."""
        brand_config = {
            "layout_hints": {
                "column_count": [2, 3],
                "title_patterns": ["^[A-Z][a-z]+.*"],
            }
        }

        result = await self.analyzer.analyze_pdf_layout(
            job_id="test-job-123",
            file_path="/path/to/test.pdf",
            brand_config=brand_config,
        )

        assert result is not None
        assert "confidence_scores" in result
        assert "overall" in result["confidence_scores"]

    async def test_classify_blocks_basic(self):
        """Test block classification functionality."""
        blocks_data = {
            "pages": {
                "1": {
                    "blocks": [
                        {
                            "id": "block_1",
                            "bbox": [100, 50, 400, 100],  # Top of page - likely title
                            "text": "Article Title",
                        },
                        {
                            "id": "block_2",
                            "bbox": [
                                100,
                                200,
                                400,
                                400,
                            ],  # Middle of page - likely body
                            "text": "Article body text content...",
                        },
                    ]
                }
            }
        }

        result = await self.analyzer.classify_blocks(blocks_data)

        assert "classified_blocks" in result
        assert "confidence_scores" in result

        # Check that blocks were classified
        page_1_blocks = result["classified_blocks"]["1"]
        assert len(page_1_blocks) == 2

        # First block should be classified as title (top position)
        title_block = next(b for b in page_1_blocks if b["id"] == "block_1")
        assert title_block["type"] == "title"
        assert title_block["confidence"] > 0

        # Second block should be classified as body (middle position)
        body_block = next(b for b in page_1_blocks if b["id"] == "block_2")
        assert body_block["type"] == "body"
        assert body_block["confidence"] > 0

    def test_extract_text_features(self):
        """Test text feature extraction."""
        # Test various text characteristics
        test_cases = [
            ("Short title", {"length": 11, "word_count": 2}),
            ("UPPERCASE TEXT", {"is_uppercase": True}),
            ("Sentence with punctuation!", {"has_punctuation": True}),
            ("Capitalized sentence", {"starts_with_capital": True}),
        ]

        for text, expected_features in test_cases:
            features = self.analyzer._extract_text_features(text)

            for key, expected_value in expected_features.items():
                assert features[key] == expected_value

    def test_extract_spatial_features(self):
        """Test spatial feature extraction."""
        bbox = [100, 200, 300, 400]  # x1, y1, x2, y2
        page_dims = (600, 800)  # width, height

        features = self.analyzer._extract_spatial_features(bbox, page_dims)

        assert features["width"] == 200  # 300 - 100
        assert features["height"] == 200  # 400 - 200
        assert features["area"] == 40000  # 200 * 200
        assert features["x_center"] == 200  # (100 + 300) / 2
        assert features["y_center"] == 300  # (200 + 400) / 2
        assert features["relative_x"] == pytest.approx(0.333, rel=1e-2)  # 200 / 600
        assert features["relative_y"] == pytest.approx(0.375, rel=1e-2)  # 300 / 800
        assert features["aspect_ratio"] == 1.0  # 200 / 200

    def test_extract_spatial_features_edge_cases(self):
        """Test spatial features with edge cases."""
        # Zero height bbox
        bbox_zero_height = [100, 200, 300, 200]
        page_dims = (600, 800)

        features = self.analyzer._extract_spatial_features(bbox_zero_height, page_dims)
        assert features["height"] == 0
        assert features["aspect_ratio"] == 1.0  # Fallback for zero height

        # Very wide bbox
        bbox_wide = [0, 100, 600, 150]
        features_wide = self.analyzer._extract_spatial_features(bbox_wide, page_dims)
        assert features_wide["aspect_ratio"] == 12.0  # 600 / 50

    @pytest.mark.parametrize(
        "position,expected_type",
        [
            ((200, 50), "title"),  # Top of page
            ((200, 400), "body"),  # Middle of page
            ((200, 750), "footer"),  # Bottom of page
        ],
    )
    def test_position_based_classification(self, position, expected_type):
        """Test that block classification considers position on page."""
        x, y = position
        bbox = [x - 50, y - 25, x + 50, y + 25]

        blocks_data = {
            "pages": {
                "1": {
                    "blocks": [
                        {"id": "test_block", "bbox": bbox, "text": "Test content"}
                    ]
                }
            }
        }

        result = self.analyzer.classify_blocks(blocks_data)
        classified_block = result["classified_blocks"]["1"][0]

        assert classified_block["type"] == expected_type
