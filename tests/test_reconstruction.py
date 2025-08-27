#!/usr/bin/env python3
"""
Test cases for article reconstruction scenarios.

Tests include simple articles, split articles, and interleaved articles
to validate the graph traversal and reconstruction algorithms.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.graph import EdgeType, SemanticGraph
from shared.graph.nodes import PageBreakNode, TextBlockNode
from shared.layout.types import BlockType
from shared.reconstruction import (
    AmbiguityResolver,
    ArticleReconstructor,
    GraphTraversal,
    ReconstructionConfig,
)
from shared.types import BoundingBox


class TestGraphTraversal:
    """Test graph traversal algorithms."""

    def test_identify_simple_article_start(self):
        """Test identification of article start nodes in simple case."""
        # Create a simple graph with one article
        graph = SemanticGraph()

        # Add title node
        title_node = TextBlockNode(
            text="Breaking News: Test Article Title",
            bbox=BoundingBox(50, 50, 400, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
            font_size=18,
            is_bold=True,
        )
        graph.add_node(title_node)

        # Add byline node
        byline_node = TextBlockNode(
            text="By Test Reporter",
            bbox=BoundingBox(50, 90, 200, 110),
            page_num=1,
            confidence=0.90,
            classification=BlockType.BYLINE,
            font_size=12,
        )
        graph.add_node(byline_node)

        # Add body nodes
        body1_node = TextBlockNode(
            text="This is the first paragraph of our test article. It contains meaningful content that should be part of the article reconstruction.",
            bbox=BoundingBox(50, 130, 400, 170),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
            font_size=11,
        )
        graph.add_node(body1_node)

        body2_node = TextBlockNode(
            text="This is the second paragraph providing additional content and context for the article reconstruction test.",
            bbox=BoundingBox(50, 180, 400, 220),
            page_num=1,
            confidence=0.87,
            classification=BlockType.BODY,
            font_size=11,
        )
        graph.add_node(body2_node)

        # Create relationships
        graph.add_edge(title_node.node_id, byline_node.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(byline_node.node_id, body1_node.node_id, EdgeType.FOLLOWS, 0.8)
        graph.add_edge(body1_node.node_id, body2_node.node_id, EdgeType.FOLLOWS, 0.8)

        # Test article start identification
        traversal = GraphTraversal()
        article_starts = traversal.identify_article_starts(graph)

        assert len(article_starts) == 1
        assert article_starts[0] == title_node.node_id

    def test_traverse_simple_article(self):
        """Test traversal of a simple single-page article."""
        graph = self._create_simple_article_graph()

        traversal = GraphTraversal()
        article_starts = traversal.identify_article_starts(graph)

        assert len(article_starts) >= 1

        # Traverse the article
        path = traversal.traverse_article(article_starts[0], graph)

        assert path.path_length >= 3  # Title + byline + at least one body
        assert path.start_page == 1
        assert path.end_page == 1
        assert not path.spans_multiple_pages
        assert BlockType.TITLE in path.component_types
        assert BlockType.BODY in path.component_types

    def test_traverse_split_article(self):
        """Test traversal of an article split across multiple pages."""
        graph = self._create_split_article_graph()

        traversal = GraphTraversal()
        article_starts = traversal.identify_article_starts(graph)

        assert len(article_starts) >= 1

        # Traverse the split article
        path = traversal.traverse_article(article_starts[0], graph)

        assert path.path_length >= 4  # Title + body on page 1 + continuation on page 2
        assert path.start_page == 1
        assert path.end_page >= 2
        assert path.spans_multiple_pages
        assert BlockType.TITLE in path.component_types
        assert BlockType.BODY in path.component_types

    def test_identify_continuation_markers(self):
        """Test identification of continuation markers."""
        graph = SemanticGraph()

        # Create text with continuation marker
        continuation_text = (
            "This article is very long and continues on page 3 with more content."
        )
        continuation_node = TextBlockNode(
            text=continuation_text,
            bbox=BoundingBox(50, 200, 400, 240),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(continuation_node)

        traversal = GraphTraversal()

        # Test internal method for finding continuation markers
        markers = traversal._find_continuation_markers(
            type("MockPath", (), {"node_ids": [continuation_node.node_id]})(), graph
        )

        assert len(markers) >= 1
        assert markers[0].target_page == 3
        assert "continues on page" in markers[0].marker_text.lower()

    def _create_simple_article_graph(self) -> SemanticGraph:
        """Create a graph with a simple single-page article."""
        graph = SemanticGraph()

        # Title
        title = TextBlockNode(
            text="Simple Test Article",
            bbox=BoundingBox(50, 50, 350, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
            font_size=18,
        )
        graph.add_node(title)

        # Byline
        byline = TextBlockNode(
            text="By Test Author",
            bbox=BoundingBox(50, 90, 200, 110),
            page_num=1,
            confidence=0.90,
            classification=BlockType.BYLINE,
        )
        graph.add_node(byline)

        # Body paragraphs
        body1 = TextBlockNode(
            text="This is the first paragraph of the simple test article with enough content to be meaningful.",
            bbox=BoundingBox(50, 130, 400, 170),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body1)

        body2 = TextBlockNode(
            text="This is the second paragraph providing additional context and content for the article.",
            bbox=BoundingBox(50, 180, 400, 220),
            page_num=1,
            confidence=0.87,
            classification=BlockType.BODY,
        )
        graph.add_node(body2)

        # Create relationships
        graph.add_edge(title.node_id, byline.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(byline.node_id, body1.node_id, EdgeType.FOLLOWS, 0.8)
        graph.add_edge(body1.node_id, body2.node_id, EdgeType.FOLLOWS, 0.8)

        return graph

    def _create_split_article_graph(self) -> SemanticGraph:
        """Create a graph with an article split across pages."""
        graph = SemanticGraph()

        # Page 1 - Article start
        title = TextBlockNode(
            text="Split Article Across Multiple Pages",
            bbox=BoundingBox(50, 50, 400, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
            font_size=18,
        )
        graph.add_node(title)

        body1_p1 = TextBlockNode(
            text="This article starts on page one and contains significant content that will continue on the next page.",
            bbox=BoundingBox(50, 100, 400, 140),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body1_p1)

        continuation_marker = TextBlockNode(
            text="Story continues on page 2",
            bbox=BoundingBox(300, 500, 400, 520),
            page_num=1,
            confidence=0.80,
            classification=BlockType.BODY,
        )
        graph.add_node(continuation_marker)

        # Page break
        page_break = PageBreakNode(page_num=2)
        graph.add_node(page_break)

        # Page 2 - Article continuation
        continuation_header = TextBlockNode(
            text="Split Article (continued from page 1)",
            bbox=BoundingBox(50, 50, 350, 70),
            page_num=2,
            confidence=0.80,
            classification=BlockType.HEADING,
        )
        graph.add_node(continuation_header)

        body1_p2 = TextBlockNode(
            text="This is the continuation of the article from page one, providing additional important content.",
            bbox=BoundingBox(50, 80, 400, 120),
            page_num=2,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body1_p2)

        body2_p2 = TextBlockNode(
            text="This is the final paragraph of the split article, concluding the story.",
            bbox=BoundingBox(50, 130, 400, 170),
            page_num=2,
            confidence=0.87,
            classification=BlockType.BODY,
        )
        graph.add_node(body2_p2)

        # Create relationships
        graph.add_edge(title.node_id, body1_p1.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(
            body1_p1.node_id, continuation_marker.node_id, EdgeType.FOLLOWS, 0.7
        )
        graph.add_edge(
            continuation_marker.node_id, page_break.node_id, EdgeType.CONTINUES_ON, 0.8
        )
        graph.add_edge(
            page_break.node_id, continuation_header.node_id, EdgeType.CONTINUES_ON, 0.8
        )
        graph.add_edge(
            continuation_header.node_id, body1_p2.node_id, EdgeType.FOLLOWS, 0.8
        )
        graph.add_edge(body1_p2.node_id, body2_p2.node_id, EdgeType.FOLLOWS, 0.8)

        return graph


class TestAmbiguityResolver:
    """Test ambiguity resolution algorithms."""

    def test_resolve_simple_connection(self):
        """Test resolution of connection between two nodes."""
        graph = SemanticGraph()

        # Create source node
        source = TextBlockNode(
            text="Source paragraph with content",
            bbox=BoundingBox(50, 100, 400, 140),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(source)

        # Create multiple target candidates
        target1 = TextBlockNode(
            text="First potential continuation with related content",
            bbox=BoundingBox(50, 150, 400, 190),  # Close spatially
            page_num=1,
            confidence=0.88,
            classification=BlockType.BODY,
        )
        graph.add_node(target1)

        target2 = TextBlockNode(
            text="Second potential continuation",
            bbox=BoundingBox(50, 300, 400, 340),  # Further away
            page_num=1,
            confidence=0.82,
            classification=BlockType.BODY,
        )
        graph.add_node(target2)

        # Add edges with different confidences
        graph.add_edge(source.node_id, target1.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(source.node_id, target2.node_id, EdgeType.FOLLOWS, 0.6)

        # Test resolution
        resolver = AmbiguityResolver()
        chosen_target, score = resolver.resolve_connection_ambiguity(
            source.node_id, [target1.node_id, target2.node_id], graph
        )

        assert (
            chosen_target == target1.node_id
        )  # Should choose closer, higher confidence
        assert score.total_score > 0.5
        assert (
            score.spatial_score > score.semantic_score
        )  # Spatial should dominate here

    def test_resolve_path_conflicts(self):
        """Test resolution of conflicting traversal paths."""
        from shared.reconstruction.types import TraversalPath

        graph = SemanticGraph()

        # Create competing paths
        path1 = TraversalPath(
            path_id="path1",
            node_ids=["node1", "node2", "node3"],
            total_confidence=2.4,
            path_length=3,
            start_page=1,
            end_page=1,
        )
        path1.component_types = [BlockType.TITLE, BlockType.BODY, BlockType.BODY]

        path2 = TraversalPath(
            path_id="path2",
            node_ids=["node1", "node4", "node5"],
            total_confidence=2.1,
            path_length=3,
            start_page=1,
            end_page=2,
        )
        path2.component_types = [BlockType.TITLE, BlockType.BYLINE, BlockType.BODY]

        resolver = AmbiguityResolver()
        chosen_path = resolver.resolve_path_conflicts([path1, path2], graph)

        assert chosen_path.path_id == "path1"  # Higher confidence should win


class TestArticleReconstructor:
    """Test complete article reconstruction."""

    def test_reconstruct_simple_article(self):
        """Test reconstruction of a simple single-page article."""
        graph = self._create_simple_article_graph()

        config = ReconstructionConfig(min_article_words=20, min_title_confidence=0.8)

        reconstructor = ArticleReconstructor(config)
        articles = reconstructor.reconstruct_articles(graph)

        assert len(articles) == 1

        article = articles[0]
        assert "Simple Test Article" in article.title
        assert article.boundary.start_page == 1
        assert article.boundary.end_page == 1
        assert not article.boundary.is_split_article
        assert article.reconstruction_confidence > 0.7
        assert len(article.components) >= 3  # Title + byline + body
        assert article.boundary.word_count > 20

    def test_reconstruct_split_article(self):
        """Test reconstruction of an article split across pages."""
        graph = self._create_split_article_graph()

        config = ReconstructionConfig(
            min_article_words=30, min_continuation_confidence=0.6
        )

        reconstructor = ArticleReconstructor(config)
        articles = reconstructor.reconstruct_articles(graph)

        assert len(articles) == 1

        article = articles[0]
        assert "Split Article" in article.title
        assert article.boundary.start_page == 1
        assert article.boundary.end_page >= 2
        assert article.boundary.is_split_article
        assert len(article.boundary.split_pages) >= 2
        assert article.reconstruction_confidence > 0.6
        assert len(article.components) >= 4  # Multiple components across pages

    def test_reconstruct_interleaved_articles(self):
        """Test reconstruction when multiple articles are interleaved."""
        graph = self._create_interleaved_articles_graph()

        config = ReconstructionConfig(min_article_words=25, min_title_confidence=0.7)

        reconstructor = ArticleReconstructor(config)
        articles = reconstructor.reconstruct_articles(graph)

        assert len(articles) == 2  # Should find both articles

        # Check that articles are properly separated
        article_titles = [article.title for article in articles]
        assert any("First Article" in title for title in article_titles)
        assert any("Second Article" in title for title in article_titles)

        # Check that no nodes are shared between articles
        all_node_ids = set()
        for article in articles:
            for node_id in article.node_ids:
                assert node_id not in all_node_ids  # No overlap
                all_node_ids.add(node_id)

    def test_quality_filtering(self):
        """Test filtering of low-quality article reconstructions."""
        graph = self._create_low_quality_graph()

        config = ReconstructionConfig(
            min_article_words=50,  # High threshold
            min_title_confidence=0.9,  # High threshold
            require_title=True,
        )

        reconstructor = ArticleReconstructor(config)
        articles = reconstructor.reconstruct_articles(graph)

        # Should filter out low-quality articles
        assert len(articles) == 0 or all(
            article.reconstruction_confidence > 0.6 for article in articles
        )

    def _create_simple_article_graph(self) -> SemanticGraph:
        """Create a simple single-page article graph."""
        graph = SemanticGraph()

        title = TextBlockNode(
            text="Simple Test Article",
            bbox=BoundingBox(50, 50, 350, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
        )
        graph.add_node(title)

        byline = TextBlockNode(
            text="By Test Author",
            bbox=BoundingBox(50, 90, 200, 110),
            page_num=1,
            confidence=0.90,
            classification=BlockType.BYLINE,
        )
        graph.add_node(byline)

        body = TextBlockNode(
            text="This is a comprehensive test article with sufficient content to meet the word count requirements for reconstruction.",
            bbox=BoundingBox(50, 130, 400, 170),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body)

        graph.add_edge(title.node_id, byline.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(byline.node_id, body.node_id, EdgeType.FOLLOWS, 0.8)

        return graph

    def _create_split_article_graph(self) -> SemanticGraph:
        """Create an article split across multiple pages."""
        graph = SemanticGraph()

        # Page 1
        title = TextBlockNode(
            text="Split Article Across Multiple Pages",
            bbox=BoundingBox(50, 50, 400, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
        )
        graph.add_node(title)

        body1 = TextBlockNode(
            text="This article contains substantial content that spans multiple pages and includes continuation markers to help with reconstruction.",
            bbox=BoundingBox(50, 100, 400, 140),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body1)

        # Page 2
        body2 = TextBlockNode(
            text="This is the continuation of the article from the previous page, providing additional important content for the story.",
            bbox=BoundingBox(50, 80, 400, 120),
            page_num=2,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body2)

        graph.add_edge(title.node_id, body1.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(body1.node_id, body2.node_id, EdgeType.CONTINUES_ON, 0.8)

        return graph

    def _create_interleaved_articles_graph(self) -> SemanticGraph:
        """Create multiple articles interleaved on the same page."""
        graph = SemanticGraph()

        # First article
        title1 = TextBlockNode(
            text="First Article Title",
            bbox=BoundingBox(50, 50, 300, 80),
            page_num=1,
            confidence=0.95,
            classification=BlockType.TITLE,
        )
        graph.add_node(title1)

        body1 = TextBlockNode(
            text="This is the content of the first article with enough text to meet minimum requirements.",
            bbox=BoundingBox(50, 90, 300, 130),
            page_num=1,
            confidence=0.85,
            classification=BlockType.BODY,
        )
        graph.add_node(body1)

        # Second article (interleaved)
        title2 = TextBlockNode(
            text="Second Article Title",
            bbox=BoundingBox(320, 50, 500, 80),
            page_num=1,
            confidence=0.93,
            classification=BlockType.TITLE,
        )
        graph.add_node(title2)

        body2 = TextBlockNode(
            text="This is the content of the second article which is placed alongside the first article.",
            bbox=BoundingBox(320, 90, 500, 130),
            page_num=1,
            confidence=0.87,
            classification=BlockType.BODY,
        )
        graph.add_node(body2)

        # Create separate article relationships
        graph.add_edge(title1.node_id, body1.node_id, EdgeType.FOLLOWS, 0.9)
        graph.add_edge(title2.node_id, body2.node_id, EdgeType.FOLLOWS, 0.9)

        return graph

    def _create_low_quality_graph(self) -> SemanticGraph:
        """Create a graph with low-quality content."""
        graph = SemanticGraph()

        # Low confidence title
        title = TextBlockNode(
            text="Poor Quality Title",
            bbox=BoundingBox(50, 50, 200, 70),
            page_num=1,
            confidence=0.6,  # Low confidence
            classification=BlockType.TITLE,
        )
        graph.add_node(title)

        # Very short body
        body = TextBlockNode(
            text="Short content.",  # Too short
            bbox=BoundingBox(50, 80, 150, 100),
            page_num=1,
            confidence=0.5,
            classification=BlockType.BODY,
        )
        graph.add_node(body)

        graph.add_edge(title.node_id, body.node_id, EdgeType.FOLLOWS, 0.4)

        return graph


class TestReconstructionConfig:
    """Test reconstruction configuration options."""

    def test_conservative_config(self):
        """Test conservative configuration for high precision."""
        config = ReconstructionConfig.create_conservative()

        assert config.min_connection_confidence >= 0.6
        assert config.min_title_confidence >= 0.8
        assert config.min_article_words >= 100
        assert config.require_title == True

    def test_aggressive_config(self):
        """Test aggressive configuration for high recall."""
        config = ReconstructionConfig.create_aggressive()

        assert config.min_connection_confidence <= 0.3
        assert config.min_title_confidence <= 0.6
        assert config.min_article_words <= 30
        assert config.require_title == False


if __name__ == "__main__":
    # Run specific tests
    test_traversal = TestGraphTraversal()
    test_traversal.test_identify_simple_article_start()
    test_traversal.test_traverse_simple_article()
    test_traversal.test_traverse_split_article()

    test_resolver = TestAmbiguityResolver()
    test_resolver.test_resolve_simple_connection()

    test_reconstructor = TestArticleReconstructor()
    test_reconstructor.test_reconstruct_simple_article()
    test_reconstructor.test_reconstruct_split_article()
    test_reconstructor.test_reconstruct_interleaved_articles()

    print("✅ All reconstruction tests passed!")
    print("🔗 Graph traversal algorithms working correctly")
    print("🎯 Ambiguity resolution functioning properly")
    print("📰 Article reconstruction scenarios validated")
    print("📊 Simple article: Single page reconstruction")
    print("📄 Split article: Multi-page with continuations")
    print("🔀 Interleaved articles: Multiple articles on same page")
