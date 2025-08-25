"""
Main synthetic data generator that orchestrates the entire generation process.

This module coordinates all the components to generate comprehensive test suites
with 100+ variants per brand for testing the magazine extraction pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import uuid

from .types import (
    BrandConfiguration, BrandStyle, GenerationConfig, TestSuite,
    GeneratedDocument, GroundTruthData, LayoutComplexity, EdgeCaseType,
    SyntheticDataError
)
from .layout_engine import LayoutEngine, MagazineStyle
from .content_factory import ContentFactory
from .variations import VariationEngine
from .pdf_renderer import PDFRenderer, RenderingOptions
from .ground_truth import GroundTruthGenerator


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Main orchestrator for synthetic magazine test data generation."""
    
    def __init__(self, generation_config: GenerationConfig):
        self.config = generation_config
        
        # Initialize core engines
        self.layout_engine = LayoutEngine()
        self.content_factory = ContentFactory()
        self.variation_engine = VariationEngine()
        self.pdf_renderer = PDFRenderer(
            RenderingOptions.create_high_quality() if self.config.pdf_dpi > 200 
            else RenderingOptions.create_test_quality()
        )
        self.ground_truth_generator = GroundTruthGenerator()
        
        # Ensure output directory exists
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        
        # Create brand configurations
        self.brand_configs = self._create_brand_configurations()
        
        logger.info(f"Initialized synthetic data generator with {len(self.brand_configs)} brands")
    
    def _create_brand_configurations(self) -> List[BrandConfiguration]:
        """Create all brand configurations."""
        brands = [
            BrandConfiguration.create_tech_magazine(),
            BrandConfiguration.create_fashion_magazine(),
            BrandConfiguration.create_news_magazine(),
            self._create_lifestyle_magazine(),
            self._create_academic_journal(),
            self._create_tabloid_newspaper()
        ]
        return brands
    
    def _create_lifestyle_magazine(self) -> BrandConfiguration:
        """Create lifestyle magazine configuration."""
        return BrandConfiguration(
            brand_name="LifestyleLiving",
            brand_style=BrandStyle.LIFESTYLE,
            primary_font="Georgia",
            secondary_font="Arial",
            title_font="Playfair Display",
            default_columns=2,
            accent_color=(0.2, 0.6, 0.4),
            typical_article_length=(400, 1200),
            image_frequency=0.5,
            decorative_title_probability=0.3
        )
    
    def _create_academic_journal(self) -> BrandConfiguration:
        """Create academic journal configuration."""
        return BrandConfiguration(
            brand_name="AcademicQuarterly",
            brand_style=BrandStyle.ACADEMIC,
            primary_font="Times New Roman",
            secondary_font="Arial",
            title_font="Times New Roman",
            default_columns=2,
            accent_color=(0.1, 0.1, 0.4),
            typical_article_length=(2000, 5000),
            image_frequency=0.15,
            complex_layout_probability=0.2
        )
    
    def _create_tabloid_newspaper(self) -> BrandConfiguration:
        """Create tabloid newspaper configuration."""
        return BrandConfiguration(
            brand_name="DailyTabloid",
            brand_style=BrandStyle.TABLOID,
            primary_font="Arial",
            secondary_font="Arial Black",
            title_font="Impact",
            default_columns=4,
            accent_color=(0.8, 0.0, 0.0),
            typical_article_length=(200, 800),
            image_frequency=0.4,
            decorative_title_probability=0.5
        )
    
    def generate_complete_test_suite(self) -> TestSuite:
        """Generate complete test suite with all brands and variations."""
        suite = TestSuite(
            suite_name=f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generation_config=self.config
        )
        
        logger.info("Starting comprehensive test suite generation")
        start_time = time.time()
        
        try:
            # Generate documents for each brand
            for brand_config in self.brand_configs:
                logger.info(f"Generating documents for brand: {brand_config.brand_name}")
                brand_documents = self._generate_brand_documents(brand_config)
                
                for doc in brand_documents:
                    suite.add_document(doc)
                
                logger.info(f"Generated {len(brand_documents)} documents for {brand_config.brand_name}")
            
            suite.generation_end_time = datetime.now()
            
            # Generate suite summary
            self._export_suite_summary(suite)
            
            end_time = time.time()
            logger.info(f"Test suite generation completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Generated {suite.total_documents} total documents")
            logger.info(f"Success rate: {suite.successful_generations / max(1, suite.total_documents):.2%}")
            
            return suite
            
        except Exception as e:
            suite.generation_end_time = datetime.now()
            logger.error(f"Test suite generation failed: {str(e)}")
            raise SyntheticDataError(f"Failed to generate test suite: {str(e)}")
    
    def _generate_brand_documents(self, brand_config: BrandConfiguration) -> List[GeneratedDocument]:
        """Generate all documents for a specific brand."""
        documents = []
        
        # Determine complexity distribution
        complexity_counts = self._calculate_complexity_distribution(self.config.documents_per_brand)
        
        # Generate documents by complexity level
        document_counter = 0
        for complexity, count in complexity_counts.items():
            for i in range(count):
                document_id = f"{brand_config.brand_name}_{complexity.value}_{document_counter:03d}"
                
                try:
                    document = self._generate_single_document(
                        document_id,
                        brand_config,
                        complexity
                    )
                    documents.append(document)
                    
                except Exception as e:
                    logger.error(f"Failed to generate document {document_id}: {str(e)}")
                    # Create failed document record
                    failed_doc = GeneratedDocument(
                        document_id=document_id,
                        brand_name=brand_config.brand_name,
                        generation_successful=False,
                        validation_errors=[str(e)]
                    )
                    documents.append(failed_doc)
                
                document_counter += 1
        
        return documents
    
    def _calculate_complexity_distribution(self, total_documents: int) -> Dict[LayoutComplexity, int]:
        """Calculate how many documents to generate for each complexity level."""
        distribution = {}
        remaining = total_documents
        
        for complexity, probability in self.config.layout_complexity_distribution.items():
            count = int(total_documents * probability)
            distribution[complexity] = count
            remaining -= count
        
        # Distribute remaining documents to ensure we hit the target
        if remaining > 0:
            complexities = list(distribution.keys())
            for i in range(remaining):
                complexity = complexities[i % len(complexities)]
                distribution[complexity] += 1
        
        return distribution
    
    def _generate_single_document(
        self,
        document_id: str,
        brand_config: BrandConfiguration,
        complexity: LayoutComplexity
    ) -> GeneratedDocument:
        """Generate a single synthetic document."""
        start_time = time.time()
        
        # Determine document parameters
        page_count = self._select_page_count()
        edge_cases = self._select_edge_cases()
        
        logger.debug(f"Generating {document_id}: {page_count} pages, complexity={complexity.value}")
        
        # Generate magazine style
        magazine_style = self._create_magazine_style(brand_config, complexity)
        
        # Generate articles
        articles = self._generate_articles_for_document(
            brand_config, 
            page_count, 
            complexity, 
            edge_cases
        )
        
        # Apply variations
        articles = self.variation_engine.apply_variations(
            articles,
            brand_config,
            complexity,
            edge_cases
        )
        
        # Create paths
        pdf_path = self.config.output_directory / f"{document_id}.pdf"
        gt_xml_path = self.config.output_directory / f"{document_id}_ground_truth.xml"
        gt_json_path = self.config.output_directory / f"{document_id}_ground_truth.json"
        
        # Generate ground truth
        ground_truth = self.ground_truth_generator.generate_ground_truth(
            articles,
            brand_config,
            self.config,
            document_id
        )
        
        success = True
        errors = []
        
        # Validate ground truth if enabled
        if self.config.validate_ground_truth:
            validation_errors = ground_truth.validate()
            if validation_errors:
                errors.extend(validation_errors)
                if validation_errors:
                    logger.warning(f"Ground truth validation errors for {document_id}: {validation_errors}")
        
        # Generate PDF if enabled
        if self.config.generate_pdfs and success:
            try:
                pdf_success = self.pdf_renderer.render_document(
                    articles,
                    brand_config,
                    pdf_path,
                    self.config
                )
                if not pdf_success:
                    success = False
                    errors.append("PDF rendering failed")
                    
            except Exception as e:
                success = False
                errors.append(f"PDF rendering error: {str(e)}")
                logger.error(f"PDF rendering failed for {document_id}: {str(e)}")
        
        # Export ground truth if enabled
        if self.config.generate_ground_truth and success:
            try:
                # Export XML
                self.ground_truth_generator.export_to_xml(ground_truth, gt_xml_path)
                
                # Export JSON for easier programmatic access
                self.ground_truth_generator.export_to_json(ground_truth, gt_json_path)
                
            except Exception as e:
                success = False
                errors.append(f"Ground truth export error: {str(e)}")
                logger.error(f"Ground truth export failed for {document_id}: {str(e)}")
        
        generation_time = time.time() - start_time
        
        # Create document record
        document = GeneratedDocument(
            document_id=document_id,
            brand_name=brand_config.brand_name,
            pdf_path=pdf_path if self.config.generate_pdfs else None,
            ground_truth_path=gt_xml_path if self.config.generate_ground_truth else None,
            ground_truth=ground_truth,
            generation_time=generation_time,
            generation_successful=success,
            validation_passed=len(errors) == 0,
            validation_errors=errors
        )
        
        return document
    
    def _select_page_count(self) -> int:
        """Select number of pages for document."""
        import random
        min_pages, max_pages = self.config.pages_per_document
        return random.randint(min_pages, max_pages)
    
    def _select_edge_cases(self) -> List[EdgeCaseType]:
        """Select edge cases for document."""
        import random
        
        if random.random() > self.config.edge_case_probability:
            return []
        
        selected_cases = []
        for edge_case, probability in self.config.edge_case_distribution.items():
            if random.random() < probability:
                selected_cases.append(edge_case)
        
        return selected_cases
    
    def _create_magazine_style(
        self,
        brand_config: BrandConfiguration,
        complexity: LayoutComplexity
    ) -> MagazineStyle:
        """Create magazine style for document."""
        from .layout_engine import MagazineStyle
        
        return MagazineStyle(
            name=f"{brand_config.brand_name}_style",
            brand_style=brand_config.brand_style,
            columns=brand_config.default_columns,
            margins=(
                brand_config.margin_top,
                brand_config.margin_right,
                brand_config.margin_bottom,
                brand_config.margin_left
            ),
            primary_font=brand_config.primary_font,
            secondary_font=brand_config.secondary_font,
            title_font=brand_config.title_font,
            primary_color=brand_config.primary_color,
            accent_color=brand_config.accent_color,
            complexity_level=complexity
        )
    
    def _generate_articles_for_document(
        self,
        brand_config: BrandConfiguration,
        page_count: int,
        complexity: LayoutComplexity,
        edge_cases: List[EdgeCaseType]
    ) -> List:
        """Generate articles for a document."""
        articles = []
        
        # Determine number of articles based on page count and complexity
        if complexity == LayoutComplexity.SIMPLE:
            articles_per_page = 1.0
        elif complexity == LayoutComplexity.MODERATE:
            articles_per_page = 1.5
        elif complexity == LayoutComplexity.COMPLEX:
            articles_per_page = 2.0
        else:  # CHAOTIC
            articles_per_page = 2.5
        
        target_articles = max(1, int(page_count * articles_per_page))
        
        # Generate articles
        current_page = 1
        for i in range(target_articles):
            article_id = f"article_{i:03d}"
            
            # Determine article page range
            if EdgeCaseType.SPLIT_ARTICLES in edge_cases and i % 3 == 0:
                # Some articles span multiple pages
                article_pages = min(2, page_count - current_page + 1)
                page_range = (current_page, current_page + article_pages - 1)
            else:
                page_range = (current_page, current_page)
            
            article = self.content_factory.generate_article(
                brand_config,
                article_id=article_id,
                complexity_level=complexity.value,
                edge_cases=edge_cases
            )
            
            # Convert to ArticleData and set page range
            from .types import ArticleData
            article_data = ArticleData(
                article_id=article_id,
                title=article["title"],
                contributors=article.get("contributors", []),
                text_elements=article.get("text_elements", []),
                image_elements=article.get("image_elements", []),
                page_range=page_range,
                is_split_article=(page_range[1] > page_range[0]),
                article_type=article.get("article_type", "feature"),
                complexity_level=complexity,
                edge_cases=edge_cases
            )
            
            articles.append(article_data)
            current_page = page_range[1] + 1
            
            if current_page > page_count:
                break
        
        return articles
    
    def _export_suite_summary(self, suite: TestSuite) -> None:
        """Export test suite summary."""
        summary = suite.get_summary()
        
        summary_path = self.config.output_directory / f"{suite.suite_name}_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Test suite summary exported to: {summary_path}")
    
    def generate_brand_focused_suite(self, brand_name: str, document_count: int = 200) -> TestSuite:
        """Generate a test suite focused on a single brand."""
        brand_config = None
        for config in self.brand_configs:
            if config.brand_name.lower() == brand_name.lower():
                brand_config = config
                break
        
        if not brand_config:
            raise SyntheticDataError(f"Brand '{brand_name}' not found")
        
        suite = TestSuite(
            suite_name=f"{brand_name}_focused_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            generation_config=self.config
        )
        
        logger.info(f"Generating focused test suite for {brand_name}")
        
        # Temporarily adjust config for this brand
        original_count = self.config.documents_per_brand
        self.config.documents_per_brand = document_count
        
        try:
            documents = self._generate_brand_documents(brand_config)
            for doc in documents:
                suite.add_document(doc)
            
            suite.generation_end_time = datetime.now()
            self._export_suite_summary(suite)
            
            return suite
            
        finally:
            # Restore original config
            self.config.documents_per_brand = original_count
    
    def validate_generation_setup(self) -> Dict[str, Any]:
        """Validate that the generation setup is working correctly."""
        logger.info("Validating synthetic data generation setup")
        
        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "engines_initialized": True,
            "output_directory_writable": self.config.output_directory.exists(),
            "brands_configured": len(self.brand_configs),
            "test_generation": {}
        }
        
        try:
            # Test single document generation
            test_brand = self.brand_configs[0]
            test_doc = self._generate_single_document(
                "validation_test",
                test_brand,
                LayoutComplexity.SIMPLE
            )
            
            validation_results["test_generation"] = {
                "success": test_doc.generation_successful,
                "has_pdf": test_doc.pdf_path and test_doc.pdf_path.exists(),
                "has_ground_truth": test_doc.ground_truth_path and test_doc.ground_truth_path.exists(),
                "validation_errors": test_doc.validation_errors
            }
            
            # Clean up test files
            if test_doc.pdf_path and test_doc.pdf_path.exists():
                test_doc.pdf_path.unlink()
            if test_doc.ground_truth_path and test_doc.ground_truth_path.exists():
                test_doc.ground_truth_path.unlink()
            
        except Exception as e:
            validation_results["test_generation"] = {
                "success": False,
                "error": str(e)
            }
        
        # Export validation results
        validation_path = self.config.output_directory / "validation_results.json"
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation results exported to: {validation_path}")
        return validation_results


def create_comprehensive_test_suite(output_dir: Path, documents_per_brand: int = 150) -> TestSuite:
    """Create a comprehensive test suite with all brands and edge cases."""
    config = GenerationConfig.create_comprehensive_test(output_dir)
    config.documents_per_brand = documents_per_brand
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_complete_test_suite()


def create_edge_case_test_suite(output_dir: Path, documents_per_brand: int = 100) -> TestSuite:
    """Create a test suite focused on edge cases."""
    config = GenerationConfig.create_edge_case_focused(output_dir)
    config.documents_per_brand = documents_per_brand
    
    generator = SyntheticDataGenerator(config)
    return generator.generate_complete_test_suite()


if __name__ == "__main__":
    # Example usage
    output_directory = Path("./test_output")
    
    # Validate setup
    config = GenerationConfig.create_comprehensive_test(output_directory)
    generator = SyntheticDataGenerator(config)
    
    validation = generator.validate_generation_setup()
    print("Validation Results:")
    print(json.dumps(validation, indent=2))
    
    # Generate small test suite
    config.documents_per_brand = 5  # Small for testing
    suite = generator.generate_complete_test_suite()
    
    print(f"Generated {suite.total_documents} test documents")
    print(f"Success rate: {suite.successful_generations / max(1, suite.total_documents):.2%}")