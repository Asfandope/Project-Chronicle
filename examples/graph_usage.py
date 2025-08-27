#!/usr/bin/env python3
"""
Example usage of the semantic graph module.

This demonstrates how to create, manipulate, and visualize semantic graphs
for document analysis.
"""

import sys
from pathlib import Path

# Add shared modules to path
sys.path.append(str(Path(__file__).parent.parent))

from shared.graph import (
    EdgeType,
    GraphFactory,
    GraphVisualizer,
    ImageNode,
    SemanticGraph,
    TextBlockNode,
)
from shared.layout.types import BlockType, TextBlock
from shared.types import BoundingBox


def main():
    """Demonstrate semantic graph functionality."""

    print("🔗 Semantic Graph Example")
    print("=" * 50)

    # 1. Create a simple graph manually
    print("\\n1. Creating a simple article structure...")

    graph = SemanticGraph(document_path="example_article.pdf")

    # Add title
    title_node = TextBlockNode(
        text="Breaking News: Semantic Graphs in Document Analysis",
        bbox=BoundingBox(50, 50, 550, 100),
        page_num=1,
        confidence=0.95,
        classification=BlockType.TITLE,
        font_size=24,
        is_bold=True,
    )
    graph.add_node(title_node)

    # Add byline
    byline_node = TextBlockNode(
        text="By Dr. Research Team",
        bbox=BoundingBox(50, 110, 200, 130),
        page_num=1,
        confidence=0.90,
        classification=BlockType.BYLINE,
        font_size=12,
    )
    graph.add_node(byline_node)

    # Add body paragraphs
    body1_node = TextBlockNode(
        text="In a groundbreaking development, researchers have successfully implemented semantic graphs for document analysis. This revolutionary approach provides unprecedented insights into document structure and content relationships.",
        bbox=BoundingBox(50, 150, 550, 200),
        page_num=1,
        confidence=0.85,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body1_node)

    body2_node = TextBlockNode(
        text="The implementation uses NetworkX for graph algorithms and provides comprehensive visualization tools for debugging and analysis. Early results show significant improvements in layout understanding.",
        bbox=BoundingBox(50, 220, 550, 270),
        page_num=1,
        confidence=0.87,
        classification=BlockType.BODY,
        font_size=11,
    )
    graph.add_node(body2_node)

    # Add an image
    image_node = ImageNode(
        bbox=BoundingBox(100, 300, 400, 450),
        page_num=1,
        image_path="diagram.png",
        image_format="PNG",
        image_size=(300, 150),
        confidence=0.95,
    )
    graph.add_node(image_node)

    # Add image caption
    caption_node = TextBlockNode(
        text="Figure 1: Semantic graph visualization showing document structure",
        bbox=BoundingBox(100, 460, 400, 480),
        page_num=1,
        confidence=0.92,
        classification=BlockType.CAPTION,
        font_size=9,
    )
    graph.add_node(caption_node)

    # Create relationships
    print("\\n2. Adding semantic relationships...")

    # Reading order
    graph.add_edge(title_node.node_id, byline_node.node_id, EdgeType.FOLLOWS, 0.9)
    graph.add_edge(byline_node.node_id, body1_node.node_id, EdgeType.FOLLOWS, 0.8)
    graph.add_edge(body1_node.node_id, body2_node.node_id, EdgeType.FOLLOWS, 0.8)
    graph.add_edge(body2_node.node_id, image_node.node_id, EdgeType.FOLLOWS, 0.7)
    graph.add_edge(image_node.node_id, caption_node.node_id, EdgeType.FOLLOWS, 0.9)

    # Hierarchical relationships
    graph.add_edge(byline_node.node_id, title_node.node_id, EdgeType.BELONGS_TO, 0.8)
    graph.add_edge(body1_node.node_id, title_node.node_id, EdgeType.BELONGS_TO, 0.7)
    graph.add_edge(body2_node.node_id, title_node.node_id, EdgeType.BELONGS_TO, 0.7)

    # Caption relationship
    graph.add_edge(caption_node.node_id, image_node.node_id, EdgeType.CAPTION_OF, 0.95)

    print(f"✅ Created graph with {graph.node_count} nodes and {graph.edge_count} edges")

    # 3. Analyze the graph
    print("\\n3. Analyzing graph structure...")

    stats = graph.get_statistics()
    print(f"📊 Statistics:")
    print(f"   • Text blocks: {stats.text_block_count}")
    print(f"   • Images: {stats.image_count}")
    print(f"   • Page breaks: {stats.page_break_count}")
    print(f"   • Average confidence: {stats.avg_node_confidence:.3f}")

    # Get reading order
    reading_order = graph.get_reading_order(page_num=1)
    print(f"\\n📖 Reading order ({len(reading_order)} nodes):")
    for i, node_id in enumerate(reading_order[:3]):  # Show first 3
        node = graph.get_node(node_id)
        if node:
            node_data = node.to_graph_data()
            text_preview = (
                node_data.text[:50] + "..."
                if node_data.text and len(node_data.text) > 50
                else node_data.text or f"[{node_data.node_type.value}]"
            )
            print(f"   {i+1}. {text_preview}")

    # 4. Serialization
    print("\\n4. Testing serialization...")

    # Export to JSON
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    json_path = output_dir / "example_graph.json"
    graph.save_json(json_path)
    print(f"💾 Saved graph to {json_path}")

    # Load it back
    loaded_graph = SemanticGraph.load_json(json_path)
    print(
        f"📥 Loaded graph with {loaded_graph.node_count} nodes and {loaded_graph.edge_count} edges"
    )

    # 5. Visualization
    print("\\n5. Creating visualizations...")

    visualizer = GraphVisualizer()

    try:
        # Network diagram
        network_path = output_dir / "network_diagram.png"
        visualizer.create_network_diagram(
            graph, output_path=network_path, show_labels=True
        )
        print(f"🎨 Created network diagram: {network_path}")

        # Layout diagram
        layout_path = output_dir / "layout_diagram.png"
        visualizer.create_layout_diagram(
            graph, page_num=1, output_path=layout_path, show_coordinates=True
        )
        print(f"📐 Created layout diagram: {layout_path}")

        # Statistics report
        stats_path = output_dir / "statistics_report.png"
        visualizer.create_statistics_report(graph, output_path=stats_path)
        print(f"📈 Created statistics report: {stats_path}")

    except ImportError as e:
        print(f"⚠️  Visualization skipped (missing matplotlib): {e}")
    except Exception as e:
        print(f"⚠️  Visualization error: {e}")

    # 6. Factory patterns
    print("\\n6. Testing factory patterns...")

    # Create article structure using factory
    sample_blocks = [
        TextBlock(
            text="This is the first paragraph of our sample article.",
            bbox=BoundingBox(50, 200, 550, 230),
            block_type=BlockType.BODY,
            confidence=0.8,
            page_num=1,
        ),
        TextBlock(
            text="This is the second paragraph with more content.",
            bbox=BoundingBox(50, 250, 550, 280),
            block_type=BlockType.BODY,
            confidence=0.8,
            page_num=1,
        ),
    ]

    factory_graph = GraphFactory.create_article_structure(
        title_text="Sample Article Title",
        body_blocks=sample_blocks,
        byline="By Factory Method",
        page_num=1,
    )

    print(f"🏭 Factory created graph with {factory_graph.node_count} nodes")

    # Add image with caption using factory
    image_bbox = BoundingBox(100, 350, 300, 450)
    GraphFactory.add_image_with_caption(
        factory_graph, image_bbox, "Sample image caption from factory", page_num=1
    )

    print(f"🖼️  Added image+caption, now {factory_graph.node_count} nodes")

    print("\\n✅ Semantic graph example completed successfully!")
    print(f"\\n📁 Output files saved to: {output_dir}")


if __name__ == "__main__":
    main()
