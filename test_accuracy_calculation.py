#!/usr/bin/env python3
"""
Test script for accuracy calculation system.

This script demonstrates and tests the PRD section 6 accuracy calculation
with the specified weights and metrics.
"""

import json
from pathlib import Path
from synthetic_data import (
    AccuracyCalculator, 
    BrandConfiguration, 
    ArticleData,
    TextElement,
    ImageElement,
    GroundTruthData
)


def create_sample_ground_truth() -> GroundTruthData:
    """Create sample ground truth data for testing."""
    
    # Create sample text elements
    title_element = TextElement(
        element_id="title_001",
        element_type="text",
        bbox=(72, 650, 540, 700),
        page_number=1,
        text_content="Revolutionary AI Technology Transforms Healthcare Industry",
        font_family="Helvetica",
        font_size=24,
        font_style="bold",
        semantic_type="title",
        reading_order=1
    )
    
    body_element1 = TextElement(
        element_id="body_001",
        element_type="text", 
        bbox=(72, 500, 540, 640),
        page_number=1,
        text_content="The healthcare industry is experiencing a revolutionary transformation through artificial intelligence. Machine learning algorithms are now capable of diagnosing diseases with unprecedented accuracy, while natural language processing helps doctors analyze patient records more efficiently.",
        font_family="Times New Roman",
        font_size=12,
        semantic_type="paragraph",
        reading_order=2
    )
    
    body_element2 = TextElement(
        element_id="body_002",
        element_type="text",
        bbox=(72, 350, 540, 490),
        page_number=1,
        text_content="Recent studies have shown that AI-powered diagnostic tools can detect early-stage cancer with 95% accuracy, significantly outperforming traditional methods. This breakthrough promises to save thousands of lives through early detection and intervention.",
        font_family="Times New Roman",
        font_size=12,
        semantic_type="paragraph",
        reading_order=3
    )
    
    byline_element = TextElement(
        element_id="byline_001",
        element_type="text",
        bbox=(72, 320, 300, 340),
        page_number=1,
        text_content="By Dr. Sarah Johnson, Medical Technology Reporter",
        font_family="Arial",
        font_size=10,
        font_style="italic",
        semantic_type="byline",
        reading_order=4
    )
    
    caption_element = TextElement(
        element_id="caption_001",
        element_type="text",
        bbox=(320, 250, 540, 280),
        page_number=1,
        text_content="AI diagnostic system analyzing medical imaging data",
        font_family="Arial",
        font_size=9,
        semantic_type="caption",
        reading_order=5
    )
    
    # Create sample image element
    image_element = ImageElement(
        element_id="image_001",
        element_type="image",
        bbox=(320, 150, 540, 240),
        page_number=1,
        alt_text="AI diagnostic system",
        width=220,
        height=90,
        dpi=300
    )
    
    # Create sample article
    article = ArticleData(
        article_id="test_article_001",
        title="Revolutionary AI Technology Transforms Healthcare Industry",
        contributors=[
            {"name": "Dr. Sarah Johnson", "role": "author", "affiliation": "Medical Technology Institute"}
        ],
        text_elements=[title_element, body_element1, body_element2, byline_element, caption_element],
        image_elements=[image_element],
        page_range=(1, 1)
    )
    
    # Create ground truth
    brand_config = BrandConfiguration.create_tech_magazine()
    
    ground_truth = GroundTruthData(
        document_id="test_doc_001",
        brand_name=brand_config.brand_name,
        generation_timestamp=datetime.now(),
        articles=[article],
        all_text_elements=[title_element, body_element1, body_element2, byline_element, caption_element],
        all_image_elements=[image_element]
    )
    
    return ground_truth


def create_perfect_extraction() -> dict:
    """Create perfect extraction that should get 100% accuracy."""
    return {
        "document_id": "test_doc_001",
        "articles": [
            {
                "article_id": "test_article_001",
                "title": "Revolutionary AI Technology Transforms Healthcare Industry",
                "text_content": "The healthcare industry is experiencing a revolutionary transformation through artificial intelligence. Machine learning algorithms are now capable of diagnosing diseases with unprecedented accuracy, while natural language processing helps doctors analyze patient records more efficiently. Recent studies have shown that AI-powered diagnostic tools can detect early-stage cancer with 95% accuracy, significantly outperforming traditional methods. This breakthrough promises to save thousands of lives through early detection and intervention.",
                "contributors": [
                    {"name": "Dr. Sarah Johnson", "role": "author"}
                ],
                "media_elements": [
                    {
                        "type": "image",
                        "bbox": (320, 150, 540, 240),
                        "width": 220,
                        "height": 90,
                        "caption": "AI diagnostic system analyzing medical imaging data"
                    }
                ]
            }
        ]
    }


def create_imperfect_extraction() -> dict:
    """Create imperfect extraction to test various accuracy scenarios."""
    return {
        "document_id": "test_doc_001", 
        "articles": [
            {
                "article_id": "test_article_001",
                "title": "Revolutionary AI Technology Transforms Healthcare",  # Missing "Industry"
                "text_content": "The healthcare industry is experiencing a revolutionary transformation through artificial intelligence. Machine learning algorithms are now capable of diagnosing diseases with unprecedented accuracy, while natural language processing helps doctors analyze patient records efficiently. Recent studies have shown that AI-powered diagnostic tools can detect early-stage cancer with 95% accuracy, significantly outperforming traditional methods.",  # Missing last sentence
                "contributors": [
                    {"name": "Sarah Johnson", "role": "author"}  # Missing "Dr." title
                ],
                "media_elements": [
                    {
                        "type": "image", 
                        "bbox": (315, 145, 545, 245),  # Slightly different bbox
                        "width": 230,
                        "height": 100,
                        "caption": "AI system analyzing medical data"  # Shorter caption
                    }
                ]
            }
        ]
    }


def create_poor_extraction() -> dict:
    """Create poor extraction with significant errors."""
    return {
        "document_id": "test_doc_001",
        "articles": [
            {
                "article_id": "test_article_001", 
                "title": "AI in Healthcare",  # Very different title
                "text_content": "Healthcare is changing with AI technology. Machine learning can diagnose diseases.",  # Much shorter, different wording
                "contributors": [
                    {"name": "Johnson", "role": "writer"}  # Wrong name and role
                ],
                "media_elements": [
                    {
                        "type": "image",
                        "bbox": (300, 100, 500, 200),  # Very different bbox
                        "width": 200,
                        "height": 100,
                        "caption": "Medical technology"  # Very different caption
                    }
                ]
            }
        ]
    }


def test_accuracy_calculation():
    """Test the accuracy calculation system."""
    
    print("Testing PRD Section 6 Accuracy Calculation System")
    print("=" * 60)
    
    # Create test data
    ground_truth = create_sample_ground_truth()
    calculator = AccuracyCalculator()
    
    # Test scenarios
    test_cases = [
        ("Perfect Extraction", create_perfect_extraction()),
        ("Imperfect Extraction", create_imperfect_extraction()),
        ("Poor Extraction", create_poor_extraction())
    ]
    
    for case_name, extracted_doc in test_cases:
        print(f"\n{case_name}")
        print("-" * 30)
        
        # Calculate accuracy
        doc_accuracy = calculator.calculate_document_accuracy(ground_truth, extracted_doc)
        
        # Print results
        print(f"Overall Weighted Accuracy: {doc_accuracy.document_weighted_accuracy:.2%}")
        print()
        print("Field Accuracies:")
        print(f"  Title (30%):       {doc_accuracy.overall_title_accuracy.accuracy:.2%}")
        print(f"  Body Text (40%):   {doc_accuracy.overall_body_text_accuracy.accuracy:.2%}")  
        print(f"  Contributors (20%): {doc_accuracy.overall_contributors_accuracy.accuracy:.2%}")
        print(f"  Media Links (10%): {doc_accuracy.overall_media_links_accuracy.accuracy:.2%}")
        
        # Show detailed breakdown for first article
        if doc_accuracy.article_accuracies:
            article_acc = doc_accuracy.article_accuracies[0]
            print()
            print("Detailed Breakdown:")
            
            # Title details
            title_details = article_acc.title_accuracy.details
            print(f"  Title Match: {'✓' if title_details.get('exact_match') else '✗'}")
            if not title_details.get('exact_match'):
                print(f"    Similarity: {title_details.get('similarity_ratio', 0):.2%}")
            
            # Body text details
            body_details = article_acc.body_text_accuracy.details
            print(f"  Body WER: {body_details.get('word_error_rate', 0):.4f} (threshold: {body_details.get('wer_threshold', 0):.4f})")
            print(f"  Meets WER threshold: {'✓' if body_details.get('meets_threshold') else '✗'}")
            
            # Contributors details
            contrib_details = article_acc.contributors_accuracy.details
            if 'match_details' in contrib_details:
                for i, match in enumerate(contrib_details['match_details']):
                    print(f"  Contributor {i+1}: {'✓' if match['is_correct'] else '✗'} (score: {match['match_score']:.2f})")
            
            # Media details
            media_details = article_acc.media_links_accuracy.details
            if 'pair_details' in media_details:
                for i, pair in enumerate(media_details['pair_details']):
                    print(f"  Media Pair {i+1}: {'✓' if pair['is_correct'] else '✗'} (score: {pair['match_score']:.2f})")
        
        print()
    
    # Test individual components
    print("\nComponent Testing")
    print("-" * 30)
    
    # Test text normalization
    from synthetic_data.accuracy_calculator import TextNormalizer
    normalizer = TextNormalizer()
    
    test_titles = [
        ("Dr. Johnson's Revolutionary AI Technology", "doctor johnsons revolutionary ai technology"),
        ("The Future of Medicine: AI & ML", "the future of medicine ai ml"),
        ("COVID-19 Response Systems", "covid 19 response systems")
    ]
    
    print("Title Normalization:")
    for original, expected in test_titles:
        normalized = normalizer.normalize_title(original)
        print(f"  '{original}' -> '{normalized}'")
        print(f"    Expected: '{expected}'")
        print(f"    Match: {'✓' if normalized == expected else '✗'}")
        print()
    
    # Test WER calculation
    from synthetic_data.accuracy_calculator import WordErrorRateCalculator
    wer_calc = WordErrorRateCalculator()
    
    test_wer_cases = [
        (["the", "quick", "brown", "fox"], ["the", "quick", "brown", "fox"], 0.0),
        (["the", "quick", "brown", "fox"], ["the", "fast", "brown", "fox"], 0.25),
        (["hello", "world"], ["hello"], 0.5),
        (["hello"], ["hello", "world"], 1.0)
    ]
    
    print("WER Calculation:")
    for ref, hyp, expected_wer in test_wer_cases:
        actual_wer = wer_calc.calculate_wer(ref, hyp)
        print(f"  Reference: {ref}")
        print(f"  Hypothesis: {hyp}")
        print(f"  WER: {actual_wer:.3f} (expected: {expected_wer:.3f}) {'✓' if abs(actual_wer - expected_wer) < 0.001 else '✗'}")
        print()


if __name__ == "__main__":
    from datetime import datetime
    test_accuracy_calculation()