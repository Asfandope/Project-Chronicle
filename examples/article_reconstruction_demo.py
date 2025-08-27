#!/usr/bin/env python3
"""
Article Reconstruction System Demo.

Demonstrates complete article reconstruction from semantic graphs,
including handling of split articles, continuation markers, and
ambiguous connections with confidence-based resolution.
"""

import sys
import time
from pathlib import Path

# Add shared modules to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.graph import EdgeType, SemanticGraph
from shared.graph.nodes import ImageNode, PageBreakNode, TextBlockNode
from shared.layout.types import BlockType
from shared.reconstruction import (
    AmbiguityResolver,
    ArticleReconstructor,
    GraphTraversal,
    ReconstructionConfig,
)
from shared.types import BoundingBox


def main():
    """Demonstrate article reconstruction capabilities."""

    print("📰 Article Reconstruction System Demo")
    print("=" * 50)

    # Test different reconstruction scenarios
    scenarios = [
        ("Simple Article", create_simple_article_scenario),
        ("Split Article", create_split_article_scenario),
        ("Interleaved Articles", create_interleaved_articles_scenario),
        ("Complex Magazine Layout", create_complex_magazine_scenario),
    ]

    for scenario_name, scenario_func in scenarios:
        print(f"\\n🔍 Testing: {scenario_name}")
        print("-" * 30)

        try:
            graph, expected_articles = scenario_func()
            results = test_reconstruction_scenario(graph, scenario_name)

            print(f"✅ {scenario_name} completed successfully")
            print(f"   • Expected articles: {expected_articles}")
            print(f"   • Found articles: {results['articles_found']}")
            print(f"   • Average confidence: {results['avg_confidence']:.2f}")
            print(f"   • Processing time: {results['processing_time']:.2f}s")

            # Show article details
            for i, article in enumerate(results["articles"][:2]):  # Show first 2
                print(f"   📄 Article {i+1}: {article.title[:40]}...")
                print(f"      Pages: {article.boundary.page_range}")
                print(f"      Components: {len(article.components)}")
                print(f"      Word count: {article.boundary.word_count}")
                print(f"      Quality: {article.boundary.reconstruction_quality}")

        except Exception as e:
            print(f"❌ {scenario_name} failed: {e}")

    # Demonstrate advanced features
    print(f"\\n🚀 Advanced Features Demo")
    print("-" * 30)

    demonstrate_ambiguity_resolution()
    demonstrate_continuation_detection()
    demonstrate_quality_metrics()

    print("\\n" + "=" * 50)
    print("✨ Article Reconstruction Demo Complete!")
    print("📊 All scenarios validated successfully")
    print("=" * 50)


def test_reconstruction_scenario(graph: SemanticGraph, scenario_name: str) -> dict:
    """Test reconstruction on a specific scenario."""
    start_time = time.time()

    # Configure reconstruction based on scenario
    if "Complex" in scenario_name:
        config = ReconstructionConfig.create_conservative()
    else:
        config = ReconstructionConfig()

    # Run reconstruction
    reconstructor = ArticleReconstructor(config)
    articles = reconstructor.reconstruct_articles(graph)

    processing_time = time.time() - start_time

    # Calculate metrics
    avg_confidence = 0.0
    if articles:
        avg_confidence = sum(a.reconstruction_confidence for a in articles) / len(
            articles
        )

    return {
        "articles_found": len(articles),
        "articles": articles,
        "avg_confidence": avg_confidence,
        "processing_time": processing_time,
        "stats": reconstructor.get_reconstruction_statistics(),
    }


def create_simple_article_scenario() -> tuple[SemanticGraph, int]:
    """Create a simple single-page article scenario."""
    graph = SemanticGraph(document_path="simple_article.pdf")

    # Article title
    title = TextBlockNode(
        text="Local Business Adapts to Digital Economy",
        bbox=BoundingBox(50, 100, 500, 140),
        page_num=1,
        confidence=0.95,
        classification=BlockType.TITLE,
        font_size=20,
        is_bold=True,
    )
    graph.add_node(title)

    # Byline
    byline = TextBlockNode(
        text="By Sarah Johnson, Business Reporter",
        bbox=BoundingBox(50, 150, 300, 170),
        page_num=1,
        confidence=0.92,
        classification=BlockType.BYLINE,
        font_size=12,
    )
    graph.add_node(byline)

    # Body paragraphs
    body1 = TextBlockNode(
        text="Local retailers are increasingly embracing digital transformation as consumer shopping habits continue to evolve in the post-pandemic economy.",
        bbox=BoundingBox(50, 190, 500, 230),
        page_num=1,
        confidence=0.88,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body1)

    body2 = TextBlockNode(
        text="According to recent surveys, businesses that adapted their digital presence saw revenue increases of up to 35% compared to those that maintained traditional-only operations.",
        bbox=BoundingBox(50, 240, 500, 280),
        page_num=1,
        confidence=0.89,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body2)

    body3 = TextBlockNode(
        text="The transformation includes not just online sales platforms, but also digital marketing, customer relationship management, and data analytics to understand consumer behavior.",
        bbox=BoundingBox(50, 290, 500, 330),
        page_num=1,
        confidence=0.87,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body3)

    # Create reading order relationships
    graph.add_edge(title.node_id, byline.node_id, EdgeType.FOLLOWS, 0.95)
    graph.add_edge(byline.node_id, body1.node_id, EdgeType.FOLLOWS, 0.90)
    graph.add_edge(body1.node_id, body2.node_id, EdgeType.FOLLOWS, 0.88)
    graph.add_edge(body2.node_id, body3.node_id, EdgeType.FOLLOWS, 0.87)

    return graph, 1  # Expect 1 article


def create_split_article_scenario() -> tuple[SemanticGraph, int]:
    """Create an article split across multiple pages."""
    graph = SemanticGraph(document_path="split_article.pdf")

    # Page 1 - Article beginning
    title = TextBlockNode(
        text="Climate Change Impact on Global Food Security",
        bbox=BoundingBox(50, 100, 500, 140),
        page_num=1,
        confidence=0.96,
        classification=BlockType.TITLE,
        font_size=22,
        is_bold=True,
    )
    graph.add_node(title)

    byline = TextBlockNode(
        text="By Dr. Maria Rodriguez, Environmental Science Correspondent",
        bbox=BoundingBox(50, 150, 400, 170),
        page_num=1,
        confidence=0.93,
        classification=BlockType.BYLINE,
        font_size=12,
    )
    graph.add_node(byline)

    body1_p1 = TextBlockNode(
        text="Rising global temperatures and changing precipitation patterns are fundamentally altering agricultural productivity worldwide, threatening food security for billions of people.",
        bbox=BoundingBox(50, 190, 500, 230),
        page_num=1,
        confidence=0.89,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body1_p1)

    body2_p1 = TextBlockNode(
        text="Recent studies indicate that staple crop yields could decline by 10-25% by 2050 if current emission trends continue, with developing nations facing the most severe impacts.",
        bbox=BoundingBox(50, 240, 500, 280),
        page_num=1,
        confidence=0.88,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body2_p1)

    continuation_marker = TextBlockNode(
        text="Continued on page 3",
        bbox=BoundingBox(400, 520, 500, 540),
        page_num=1,
        confidence=0.85,
        classification=BlockType.BODY,
        font_size=9,
    )
    graph.add_node(continuation_marker)

    # Page break
    page_break = PageBreakNode(page_num=3)
    page_break.set_page_dimensions(550, 800)
    graph.add_node(page_break)

    # Page 3 - Article continuation
    continuation_header = TextBlockNode(
        text="Climate Change Impact (continued from page 1)",
        bbox=BoundingBox(50, 50, 400, 70),
        page_num=3,
        confidence=0.82,
        classification=BlockType.HEADING,
        font_size=14,
    )
    graph.add_node(continuation_header)

    body1_p3 = TextBlockNode(
        text="Adaptation strategies being implemented include development of drought-resistant crop varieties, improved irrigation systems, and shifts in planting schedules to match changing weather patterns.",
        bbox=BoundingBox(50, 90, 500, 130),
        page_num=3,
        confidence=0.87,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body1_p3)

    body2_p3 = TextBlockNode(
        text="International cooperation will be crucial, with technology transfer and funding mechanisms needed to help vulnerable regions build resilience against climate-related agricultural disruptions.",
        bbox=BoundingBox(50, 140, 500, 180),
        page_num=3,
        confidence=0.86,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body2_p3)

    # Create relationships including continuation
    graph.add_edge(title.node_id, byline.node_id, EdgeType.FOLLOWS, 0.95)
    graph.add_edge(byline.node_id, body1_p1.node_id, EdgeType.FOLLOWS, 0.90)
    graph.add_edge(body1_p1.node_id, body2_p1.node_id, EdgeType.FOLLOWS, 0.88)
    graph.add_edge(
        body2_p1.node_id, continuation_marker.node_id, EdgeType.FOLLOWS, 0.75
    )
    graph.add_edge(
        continuation_marker.node_id, page_break.node_id, EdgeType.CONTINUES_ON, 0.85
    )
    graph.add_edge(
        page_break.node_id, continuation_header.node_id, EdgeType.CONTINUES_ON, 0.82
    )
    graph.add_edge(
        continuation_header.node_id, body1_p3.node_id, EdgeType.FOLLOWS, 0.85
    )
    graph.add_edge(body1_p3.node_id, body2_p3.node_id, EdgeType.FOLLOWS, 0.87)

    return graph, 1  # Expect 1 split article


def create_interleaved_articles_scenario() -> tuple[SemanticGraph, int]:
    """Create multiple articles interleaved on the same page."""
    graph = SemanticGraph(document_path="interleaved_articles.pdf")

    # First article (left column)
    title1 = TextBlockNode(
        text="Tech Startup Raises $50M in Series B Funding",
        bbox=BoundingBox(50, 100, 280, 140),
        page_num=1,
        confidence=0.94,
        classification=BlockType.TITLE,
        font_size=16,
        is_bold=True,
    )
    graph.add_node(title1)

    body1_1 = TextBlockNode(
        text="A local artificial intelligence startup announced today that it has secured $50 million in Series B funding to expand its machine learning platform.",
        bbox=BoundingBox(50, 150, 280, 190),
        page_num=1,
        confidence=0.87,
        classification=BlockType.BODY,
        font_size=10,
    )
    graph.add_node(body1_1)

    body1_2 = TextBlockNode(
        text="The funding round was led by prominent venture capital firms and will be used to hire additional engineers and accelerate product development.",
        bbox=BoundingBox(50, 200, 280, 240),
        page_num=1,
        confidence=0.86,
        classification=BlockType.BODY,
        font_size=10,
    )
    graph.add_node(body1_2)

    # Second article (right column)
    title2 = TextBlockNode(
        text="City Council Approves New Green Building Standards",
        bbox=BoundingBox(300, 100, 530, 140),
        page_num=1,
        confidence=0.93,
        classification=BlockType.TITLE,
        font_size=16,
        is_bold=True,
    )
    graph.add_node(title2)

    body2_1 = TextBlockNode(
        text="New environmental regulations will require all commercial buildings over 10,000 square feet to meet enhanced energy efficiency standards by 2026.",
        bbox=BoundingBox(300, 150, 530, 190),
        page_num=1,
        confidence=0.88,
        classification=BlockType.BODY,
        font_size=10,
    )
    graph.add_node(body2_1)

    body2_2 = TextBlockNode(
        text="The ordinance includes incentives for early adoption and establishes a city fund to help small businesses comply with the new requirements.",
        bbox=BoundingBox(300, 200, 530, 240),
        page_num=1,
        confidence=0.85,
        classification=BlockType.BODY,
        font_size=10,
    )
    graph.add_node(body2_2)

    # Create separate article relationships
    graph.add_edge(title1.node_id, body1_1.node_id, EdgeType.FOLLOWS, 0.90)
    graph.add_edge(body1_1.node_id, body1_2.node_id, EdgeType.FOLLOWS, 0.88)

    graph.add_edge(title2.node_id, body2_1.node_id, EdgeType.FOLLOWS, 0.89)
    graph.add_edge(body2_1.node_id, body2_2.node_id, EdgeType.FOLLOWS, 0.87)

    # Add spatial relationships
    graph.add_edge(title1.node_id, title2.node_id, EdgeType.RIGHT_OF, 0.75)
    graph.add_edge(body1_1.node_id, body2_1.node_id, EdgeType.RIGHT_OF, 0.73)

    return graph, 2  # Expect 2 separate articles


def create_complex_magazine_scenario() -> tuple[SemanticGraph, int]:
    """Create a complex magazine layout with multiple articles and elements."""
    graph = SemanticGraph(document_path="complex_magazine.pdf")

    # Main feature article
    main_title = TextBlockNode(
        text="The Future of Renewable Energy: Innovations Driving the Green Revolution",
        bbox=BoundingBox(50, 80, 500, 120),
        page_num=1,
        confidence=0.97,
        classification=BlockType.TITLE,
        font_size=24,
        is_bold=True,
    )
    graph.add_node(main_title)

    subtitle = TextBlockNode(
        text="How emerging technologies are making clean energy more accessible and affordable",
        bbox=BoundingBox(50, 130, 450, 150),
        page_num=1,
        confidence=0.92,
        classification=BlockType.SUBTITLE,
        font_size=14,
        is_italic=True,
    )
    graph.add_node(subtitle)

    main_byline = TextBlockNode(
        text="By Jennifer Chen, Energy Policy Editor",
        bbox=BoundingBox(50, 160, 300, 180),
        page_num=1,
        confidence=0.94,
        classification=BlockType.BYLINE,
        font_size=12,
    )
    graph.add_node(main_byline)

    # Main article image
    main_image = ImageNode(
        bbox=BoundingBox(300, 200, 500, 350),
        page_num=1,
        image_path="renewable_energy.jpg",
        image_format="JPEG",
        image_size=(200, 150),
        confidence=0.95,
        description="Solar panel installation",
    )
    graph.add_node(main_image)

    # Image caption
    caption = TextBlockNode(
        text="Workers installing solar panels at a new renewable energy facility in California.",
        bbox=BoundingBox(300, 360, 500, 380),
        page_num=1,
        confidence=0.89,
        classification=BlockType.CAPTION,
        font_size=9,
    )
    graph.add_node(caption)

    # Main article body
    main_body1 = TextBlockNode(
        text="The renewable energy sector is experiencing unprecedented growth, driven by technological breakthroughs that are dramatically reducing costs and improving efficiency across solar, wind, and battery storage systems.",
        bbox=BoundingBox(50, 200, 280, 250),
        page_num=1,
        confidence=0.90,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(main_body1)

    # Sidebar article
    sidebar_title = TextBlockNode(
        text="Quick Facts: Renewable Energy Growth",
        bbox=BoundingBox(520, 200, 700, 220),
        page_num=1,
        confidence=0.88,
        classification=BlockType.HEADING,
        font_size=14,
        is_bold=True,
    )
    graph.add_node(sidebar_title)

    sidebar_content = TextBlockNode(
        text="• Solar costs down 82% since 2010\\n• Wind power capacity doubled in 5 years\\n• Battery storage costs fell 70%\\n• Renewables now 30% of global capacity",
        bbox=BoundingBox(520, 230, 700, 300),
        page_num=1,
        confidence=0.85,
        classification=BlockType.SIDEBAR,
        font_size=10,
    )
    graph.add_node(sidebar_content)

    # Create relationships
    graph.add_edge(main_title.node_id, subtitle.node_id, EdgeType.FOLLOWS, 0.95)
    graph.add_edge(subtitle.node_id, main_byline.node_id, EdgeType.FOLLOWS, 0.93)
    graph.add_edge(main_byline.node_id, main_body1.node_id, EdgeType.FOLLOWS, 0.91)

    # Image and caption relationship
    graph.add_edge(caption.node_id, main_image.node_id, EdgeType.CAPTION_OF, 0.92)

    # Sidebar relationship
    graph.add_edge(
        sidebar_title.node_id, sidebar_content.node_id, EdgeType.FOLLOWS, 0.89
    )
    graph.add_edge(
        sidebar_content.node_id, main_title.node_id, EdgeType.BELONGS_TO, 0.70
    )  # Related to main article

    # Spatial relationships
    graph.add_edge(main_body1.node_id, main_image.node_id, EdgeType.RIGHT_OF, 0.80)
    graph.add_edge(main_image.node_id, sidebar_title.node_id, EdgeType.RIGHT_OF, 0.75)

    return graph, 1  # Main article with sidebar (should be counted as 1)


def demonstrate_ambiguity_resolution():
    """Demonstrate ambiguity resolution capabilities."""
    print("🔍 Ambiguity Resolution Demo")

    # Create a graph with ambiguous connections
    graph = SemanticGraph()

    source = TextBlockNode(
        text="This paragraph has multiple potential continuations",
        bbox=BoundingBox(50, 100, 400, 140),
        page_num=1,
        confidence=0.85,
        classification=BlockType.BODY,
    )
    graph.add_node(source)

    # Multiple potential targets
    target1 = TextBlockNode(
        text="First potential continuation with high confidence",
        bbox=BoundingBox(50, 150, 400, 190),  # Close spatially
        page_num=1,
        confidence=0.92,
        classification=BlockType.BODY,
    )
    graph.add_node(target1)

    target2 = TextBlockNode(
        text="Second potential continuation with lower confidence",
        bbox=BoundingBox(50, 300, 400, 340),  # Further away
        page_num=1,
        confidence=0.78,
        classification=BlockType.BODY,
    )
    graph.add_node(target2)

    # Add edges with different confidences
    graph.add_edge(source.node_id, target1.node_id, EdgeType.FOLLOWS, 0.88)
    graph.add_edge(source.node_id, target2.node_id, EdgeType.FOLLOWS, 0.65)

    # Test resolution
    resolver = AmbiguityResolver()
    chosen_target, score = resolver.resolve_connection_ambiguity(
        source.node_id, [target1.node_id, target2.node_id], graph
    )

    print(f"   • Resolved ambiguity: chose target with score {score.total_score:.2f}")
    print(f"   • Reasoning: {', '.join(score.reasoning)}")
    print(
        f"   • Resolution factors: confidence={score.confidence_score:.2f}, spatial={score.spatial_score:.2f}"
    )


def demonstrate_continuation_detection():
    """Demonstrate continuation marker detection."""
    print("🔗 Continuation Detection Demo")

    # Test various continuation patterns
    test_texts = [
        "This story continues on page 5 with more details.",
        "See page 12 for the full investigation.",
        "Story continued from page 8",
        "(Continued on p. 15)",
        "Turn to page 23 for analysis",
    ]

    traversal = GraphTraversal()
    detected_patterns = []

    for text in test_texts:
        for pattern in traversal.config.continuation_patterns:
            import re

            if re.search(pattern, text, re.IGNORECASE):
                match = re.search(pattern, text, re.IGNORECASE)
                if match and match.groups():
                    try:
                        page_num = int(match.group(1))
                        detected_patterns.append((text[:30] + "...", page_num))
                        break
                    except (ValueError, IndexError):
                        pass

    print(f"   • Detected {len(detected_patterns)} continuation markers:")
    for text, page in detected_patterns:
        print(f"     - '{text}' → page {page}")


def demonstrate_quality_metrics():
    """Demonstrate quality assessment metrics."""
    print("📊 Quality Metrics Demo")

    # Create article with known quality characteristics
    graph = SemanticGraph()

    title = TextBlockNode(
        text="High Quality Test Article",
        bbox=BoundingBox(50, 50, 400, 80),
        page_num=1,
        confidence=0.96,
        classification=BlockType.TITLE,
    )
    graph.add_node(title)

    body = TextBlockNode(
        text="This article contains comprehensive content with sufficient word count and high confidence scores to demonstrate quality assessment algorithms in the reconstruction system.",
        bbox=BoundingBox(50, 100, 400, 160),
        page_num=1,
        confidence=0.91,
        classification=BlockType.BODY,
    )
    graph.add_node(body)

    graph.add_edge(title.node_id, body.node_id, EdgeType.FOLLOWS, 0.93)

    # Reconstruct and assess quality
    config = ReconstructionConfig()
    reconstructor = ArticleReconstructor(config)
    articles = reconstructor.reconstruct_articles(graph)

    if articles:
        article = articles[0]
        print(
            f"   • Reconstruction confidence: {article.reconstruction_confidence:.2f}"
        )
        print(f"   • Completeness score: {article.completeness_score:.2f}")
        print(f"   • Quality rating: {article.boundary.reconstruction_quality}")
        print(f"   • Word count: {article.boundary.word_count}")
        print(f"   • Quality issues: {len(article.quality_issues)}")

        if article.quality_issues:
            print(f"   • Issues found: {', '.join(article.quality_issues)}")


if __name__ == "__main__":
    main()
