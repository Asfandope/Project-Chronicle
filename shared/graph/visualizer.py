"""
Visualization tools for semantic graphs.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import structlog
from matplotlib.patches import FancyBboxPatch

from .graph import SemanticGraph
from .types import EdgeType, GraphError, NodeType

logger = structlog.get_logger(__name__)


class GraphVisualizer:
    """
    Visualization tools for semantic graphs.

    Creates visual representations of document structure graphs
    for debugging and analysis purposes.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize graph visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or self._get_default_config()
        self.logger = logger.bind(component="GraphVisualizer")

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default visualization configuration."""
        return {
            # Node colors by type
            "node_colors": {
                NodeType.TEXT_BLOCK: "#4A90E2",  # Blue
                NodeType.IMAGE: "#7ED321",  # Green
                NodeType.PAGE_BREAK: "#F5A623",  # Orange
            },
            # Edge colors by type
            "edge_colors": {
                EdgeType.FOLLOWS: "#333333",  # Dark gray
                EdgeType.BELONGS_TO: "#9013FE",  # Purple
                EdgeType.CONTINUES_ON: "#FF6B35",  # Red-orange
                EdgeType.CAPTION_OF: "#00BCD4",  # Cyan
            },
            # Size settings
            "node_size": {
                NodeType.TEXT_BLOCK: 300,
                NodeType.IMAGE: 400,
                NodeType.PAGE_BREAK: 200,
            },
            # Style settings
            "figure_size": (16, 12),
            "dpi": 150,
            "font_size": 8,
            "edge_width": 1.5,
            "node_alpha": 0.8,
            "edge_alpha": 0.6,
            # Layout settings
            "layout_algorithm": "spring",
            "spring_k": 2.0,
            "spring_iterations": 50,
        }

    def create_network_diagram(
        self,
        graph: SemanticGraph,
        output_path: Optional[Union[str, Path]] = None,
        show_labels: bool = True,
        filter_edges: Optional[List[EdgeType]] = None,
        highlight_nodes: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Create a network diagram of the semantic graph.

        Args:
            graph: Semantic graph to visualize
            output_path: Output file path
            show_labels: Whether to show node labels
            filter_edges: Optional list of edge types to include
            highlight_nodes: Optional list of node IDs to highlight

        Returns:
            Output path if saved, None otherwise
        """
        try:
            self.logger.debug(
                "Creating network diagram",
                graph_id=graph.graph_id[:8],
                nodes=graph.node_count,
                edges=graph.edge_count,
            )

            # Create matplotlib figure
            fig, ax = plt.subplots(
                figsize=self.config["figure_size"], dpi=self.config["dpi"]
            )

            # Filter edges if specified
            display_graph = graph.graph.copy()
            if filter_edges:
                edges_to_remove = []
                for source, target, attrs in display_graph.edges(data=True):
                    if EdgeType(attrs["edge_type"]) not in filter_edges:
                        edges_to_remove.append((source, target))
                display_graph.remove_edges_from(edges_to_remove)

            # Calculate layout
            if self.config["layout_algorithm"] == "spring":
                pos = nx.spring_layout(
                    display_graph,
                    k=self.config["spring_k"],
                    iterations=self.config["spring_iterations"],
                )
            elif self.config["layout_algorithm"] == "hierarchical":
                pos = self._create_hierarchical_layout(display_graph)
            else:
                pos = nx.kamada_kawai_layout(display_graph)

            # Draw nodes by type
            for node_type in NodeType:
                node_list = []
                node_colors = []
                node_sizes = []

                for node_id in display_graph.nodes():
                    node_attrs = display_graph.nodes[node_id]
                    if NodeType(node_attrs["node_type"]) == node_type:
                        node_list.append(node_id)

                        # Color (highlight if specified)
                        if highlight_nodes and node_id in highlight_nodes:
                            node_colors.append("#FF4444")  # Red for highlighted
                        else:
                            node_colors.append(self.config["node_colors"][node_type])

                        # Size based on confidence
                        base_size = self.config["node_size"][node_type]
                        confidence = node_attrs.get("confidence", 1.0)
                        node_sizes.append(base_size * confidence)

                if node_list:
                    nx.draw_networkx_nodes(
                        display_graph,
                        pos,
                        nodelist=node_list,
                        node_color=node_colors,
                        node_size=node_sizes,
                        alpha=self.config["node_alpha"],
                        ax=ax,
                    )

            # Draw edges by type
            for edge_type in EdgeType:
                edge_list = []
                edge_colors = []
                edge_widths = []

                for source, target, attrs in display_graph.edges(data=True):
                    if EdgeType(attrs["edge_type"]) == edge_type:
                        edge_list.append((source, target))
                        edge_colors.append(self.config["edge_colors"][edge_type])

                        # Width based on confidence
                        confidence = attrs.get("confidence", 1.0)
                        edge_widths.append(self.config["edge_width"] * confidence)

                if edge_list:
                    nx.draw_networkx_edges(
                        display_graph,
                        pos,
                        edgelist=edge_list,
                        edge_color=edge_colors,
                        width=edge_widths,
                        alpha=self.config["edge_alpha"],
                        arrows=True,
                        arrowsize=20,
                        ax=ax,
                    )

            # Draw labels if requested
            if show_labels:
                labels = {}
                for node_id in display_graph.nodes():
                    node_attrs = display_graph.nodes[node_id]

                    # Create short label
                    if "text" in node_attrs and node_attrs["text"]:
                        text = node_attrs["text"][:20]
                        if len(text) < len(node_attrs["text"]):
                            text += "..."
                        labels[node_id] = text
                    else:
                        node_type = NodeType(node_attrs["node_type"])
                        labels[node_id] = f"{node_type.value}\\n{node_id[:6]}"

                nx.draw_networkx_labels(
                    display_graph,
                    pos,
                    labels,
                    font_size=self.config["font_size"],
                    ax=ax,
                )

            # Add title and legend
            title = f"Semantic Graph: {graph.graph_id[:8]}"
            if graph.document_path:
                title += f" ({Path(graph.document_path).name})"
            ax.set_title(title, fontsize=14, fontweight="bold")

            # Create legend
            self._add_legend(ax, filter_edges)

            # Remove axes
            ax.set_axis_off()

            # Tight layout
            plt.tight_layout()

            # Save or show
            if output_path:
                output_path = Path(output_path)
                plt.savefig(
                    output_path,
                    dpi=self.config["dpi"],
                    bbox_inches="tight",
                    facecolor="white",
                )
                self.logger.info("Saved network diagram", output_path=str(output_path))
                plt.close(fig)
                return str(output_path)
            else:
                plt.show()
                return None

        except Exception as e:
            self.logger.error(
                "Error creating network diagram", error=str(e), exc_info=True
            )
            raise GraphError(f"Failed to create network diagram: {e}")

    def create_layout_diagram(
        self,
        graph: SemanticGraph,
        page_num: Optional[int] = None,
        output_path: Optional[Union[str, Path]] = None,
        show_coordinates: bool = True,
    ) -> Optional[str]:
        """
        Create a layout diagram showing spatial relationships.

        Args:
            graph: Semantic graph to visualize
            page_num: Specific page to visualize (None for all)
            output_path: Output file path
            show_coordinates: Whether to show bounding box coordinates

        Returns:
            Output path if saved, None otherwise
        """
        try:
            self.logger.debug(
                "Creating layout diagram",
                graph_id=graph.graph_id[:8],
                page_num=page_num,
            )

            # Get nodes for specified page(s)
            if page_num is not None:
                page_nodes = graph.get_nodes_by_page(page_num)
                title_suffix = f" (Page {page_num})"
            else:
                page_nodes = [graph.get_node(nid) for nid in graph.graph.nodes()]
                page_nodes = [n for n in page_nodes if n is not None]
                title_suffix = " (All Pages)"

            if not page_nodes:
                self.logger.warning("No nodes found for visualization")
                return None

            # Calculate figure bounds
            all_bboxes = [
                node.bbox if hasattr(node, "bbox") else node.to_graph_data().bbox
                for node in page_nodes
            ]

            min_x = min(bbox.x0 for bbox in all_bboxes)
            max_x = max(bbox.x1 for bbox in all_bboxes)
            min_y = min(bbox.y0 for bbox in all_bboxes)
            max_y = max(bbox.y1 for bbox in all_bboxes)

            # Create figure with correct aspect ratio
            width = max_x - min_x
            height = max_y - min_y
            aspect_ratio = width / height if height > 0 else 1

            fig_width = min(16, max(8, aspect_ratio * 10))
            fig_height = min(12, max(6, fig_width / aspect_ratio))

            fig, ax = plt.subplots(
                figsize=(fig_width, fig_height), dpi=self.config["dpi"]
            )

            # Draw nodes as rectangles
            for node in page_nodes:
                node_data = node.to_graph_data()
                bbox = node_data.bbox
                node_type = node_data.node_type

                # Get color
                color = self.config["node_colors"][node_type]

                # Create rectangle
                rect = FancyBboxPatch(
                    (bbox.x0, bbox.y0),
                    bbox.width,
                    bbox.height,
                    boxstyle="round,pad=2",
                    facecolor=color,
                    alpha=self.config["node_alpha"],
                    edgecolor="black",
                    linewidth=1,
                )
                ax.add_patch(rect)

                # Add text label
                if node_data.text and len(node_data.text.strip()) > 0:
                    text = node_data.text[:100]
                    if len(text) < len(node_data.text):
                        text += "..."

                    # Position text in center of box
                    text_x = bbox.center_x
                    text_y = bbox.center_y

                    ax.text(
                        text_x,
                        text_y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=max(6, min(10, bbox.width / 20)),
                        wrap=True,
                        clip_on=True,
                    )

                # Show coordinates if requested
                if show_coordinates:
                    coord_text = f"({bbox.x0:.0f},{bbox.y0:.0f})"
                    ax.text(
                        bbox.x0,
                        bbox.y1 + 5,
                        coord_text,
                        fontsize=6,
                        color="gray",
                        ha="left",
                        va="bottom",
                    )

            # Draw edges with arrows
            for source_id, target_id, attrs in graph.graph.edges(data=True):
                source_node = graph.get_node(source_id)
                target_node = graph.get_node(target_id)

                if source_node and target_node:
                    source_data = source_node.to_graph_data()
                    target_data = target_node.to_graph_data()

                    # Skip if not on current page
                    if page_num is not None:
                        if (
                            source_data.page_num != page_num
                            or target_data.page_num != page_num
                        ):
                            continue

                    edge_type = EdgeType(attrs["edge_type"])
                    color = self.config["edge_colors"][edge_type]

                    # Draw arrow from center to center
                    ax.annotate(
                        "",
                        xy=(target_data.bbox.center_x, target_data.bbox.center_y),
                        xytext=(source_data.bbox.center_x, source_data.bbox.center_y),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=color,
                            alpha=self.config["edge_alpha"],
                            lw=self.config["edge_width"],
                        ),
                    )

            # Set axis properties
            ax.set_xlim(min_x - 50, max_x + 50)
            ax.set_ylim(min_y - 50, max_y + 50)
            ax.invert_yaxis()  # PDF coordinates have Y=0 at top
            ax.set_aspect("equal")

            # Add title
            title = f"Layout Diagram: {graph.graph_id[:8]}{title_suffix}"
            if graph.document_path:
                title += f" ({Path(graph.document_path).name})"
            ax.set_title(title, fontsize=14, fontweight="bold")

            # Add grid
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("X Coordinate (pixels)")
            ax.set_ylabel("Y Coordinate (pixels)")

            # Create legend
            self._add_legend(ax)

            plt.tight_layout()

            # Save or show
            if output_path:
                output_path = Path(output_path)
                plt.savefig(
                    output_path,
                    dpi=self.config["dpi"],
                    bbox_inches="tight",
                    facecolor="white",
                )
                self.logger.info("Saved layout diagram", output_path=str(output_path))
                plt.close(fig)
                return str(output_path)
            else:
                plt.show()
                return None

        except Exception as e:
            self.logger.error(
                "Error creating layout diagram", error=str(e), exc_info=True
            )
            raise GraphError(f"Failed to create layout diagram: {e}")

    def create_statistics_report(
        self, graph: SemanticGraph, output_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Create a visual statistics report for the graph.

        Args:
            graph: Semantic graph to analyze
            output_path: Optional output path for saving plots

        Returns:
            Statistics dictionary
        """
        try:
            self.logger.debug("Creating statistics report", graph_id=graph.graph_id[:8])

            stats = graph.get_statistics()
            stats_dict = stats.to_dict()

            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(16, 12), dpi=self.config["dpi"]
            )

            # 1. Node type distribution (pie chart)
            node_counts = stats_dict["node_counts"]
            if node_counts["total"] > 0:
                labels = []
                sizes = []
                colors = []

                for node_type_str, count in node_counts.items():
                    if node_type_str != "total" and count > 0:
                        node_type = NodeType(node_type_str)
                        labels.append(f"{node_type.value.title()}\\n({count})")
                        sizes.append(count)
                        colors.append(self.config["node_colors"][node_type])

                ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%")
                ax1.set_title("Node Type Distribution")
            else:
                ax1.text(0.5, 0.5, "No nodes", ha="center", va="center")
                ax1.set_title("Node Type Distribution")

            # 2. Edge type distribution (bar chart)
            edge_counts = stats_dict["edge_counts"]
            if edge_counts["total"] > 0:
                edge_types = []
                edge_values = []
                edge_colors = []

                for edge_type_str, count in edge_counts.items():
                    if edge_type_str != "total" and count > 0:
                        edge_type = EdgeType(edge_type_str)
                        edge_types.append(edge_type.value.replace("_", " ").title())
                        edge_values.append(count)
                        edge_colors.append(self.config["edge_colors"][edge_type])

                bars = ax2.bar(edge_types, edge_values, color=edge_colors)
                ax2.set_title("Edge Type Distribution")
                ax2.set_ylabel("Count")
                ax2.tick_params(axis="x", rotation=45)

                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                    )
            else:
                ax2.text(0.5, 0.5, "No edges", ha="center", va="center")
                ax2.set_title("Edge Type Distribution")

            # 3. Quality metrics (gauge-style visualization)
            quality_metrics = stats_dict["quality_metrics"]

            metrics = [
                ("Avg Node\\nConfidence", quality_metrics["avg_node_confidence"]),
                ("Avg Edge\\nConfidence", quality_metrics["avg_edge_confidence"]),
                (
                    "Connectivity\\nRatio",
                    1
                    - (
                        quality_metrics["unconnected_nodes"]
                        / max(node_counts["total"], 1)
                    ),
                ),
            ]

            x_pos = range(len(metrics))
            values = [m[1] for m in metrics]
            labels = [m[0] for m in metrics]

            bars = ax3.bar(x_pos, values, color=["#4CAF50", "#2196F3", "#FF9800"])
            ax3.set_title("Quality Metrics")
            ax3.set_ylabel("Score (0-1)")
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels)
            ax3.set_ylim(0, 1)

            # Add value labels
            for bar, value in zip(bars, values):
                ax3.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.01,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                )

            # 4. Layout statistics (text summary)
            layout_stats = stats_dict["layout_stats"]

            summary_text = f"""
Graph Overview:
• Graph ID: {graph.graph_id[:12]}...
• Total Nodes: {node_counts['total']}
• Total Edges: {edge_counts['total']}
• Pages: {layout_stats['page_count']}
• Avg Blocks/Page: {layout_stats['avg_blocks_per_page']:.1f}

Quality Metrics:
• Node Confidence: {quality_metrics['avg_node_confidence']:.3f}
• Edge Confidence: {quality_metrics['avg_edge_confidence']:.3f}
• Unconnected Nodes: {quality_metrics['unconnected_nodes']}

Creation Time: {graph.creation_time.strftime('%Y-%m-%d %H:%M:%S')}
            """.strip()

            ax4.text(
                0.05,
                0.95,
                summary_text,
                transform=ax4.transAxes,
                fontsize=10,
                verticalalignment="top",
                fontfamily="monospace",
            )
            ax4.set_title("Graph Summary")
            ax4.axis("off")

            # Adjust layout
            plt.tight_layout()

            # Save or show
            if output_path:
                output_path = Path(output_path)
                plt.savefig(
                    output_path,
                    dpi=self.config["dpi"],
                    bbox_inches="tight",
                    facecolor="white",
                )
                self.logger.info(
                    "Saved statistics report", output_path=str(output_path)
                )
                plt.close(fig)
            else:
                plt.show()

            return stats_dict

        except Exception as e:
            self.logger.error(
                "Error creating statistics report", error=str(e), exc_info=True
            )
            raise GraphError(f"Failed to create statistics report: {e}")

    def export_graph_data(
        self, graph: SemanticGraph, output_path: Union[str, Path], format: str = "json"
    ) -> str:
        """
        Export graph data in various formats.

        Args:
            graph: Semantic graph to export
            output_path: Output file path
            format: Export format ("json", "graphml", "gexf")

        Returns:
            Output file path
        """
        try:
            output_path = Path(output_path)

            if format.lower() == "json":
                # Use built-in JSON serialization
                graph.save_json(output_path)

            elif format.lower() == "graphml":
                # Export as GraphML
                nx.write_graphml(graph.graph, output_path)

            elif format.lower() == "gexf":
                # Export as GEXF (Gephi format)
                nx.write_gexf(graph.graph, output_path)

            else:
                raise ValueError(f"Unsupported export format: {format}")

            self.logger.info(
                "Exported graph data",
                output_path=str(output_path),
                format=format,
                nodes=graph.node_count,
                edges=graph.edge_count,
            )

            return str(output_path)

        except Exception as e:
            self.logger.error("Error exporting graph data", error=str(e), exc_info=True)
            raise GraphError(f"Failed to export graph data: {e}")

    def _create_hierarchical_layout(
        self, graph: nx.DiGraph
    ) -> Dict[str, Tuple[float, float]]:
        """Create hierarchical layout based on page numbers and reading order."""
        pos = {}

        # Group nodes by page
        pages = {}
        for node_id in graph.nodes():
            node_attrs = graph.nodes[node_id]
            page_num = node_attrs.get("page_num", 1)
            if page_num not in pages:
                pages[page_num] = []
            pages[page_num].append(node_id)

        # Layout each page
        y_offset = 0
        for page_num in sorted(pages.keys()):
            page_nodes = pages[page_num]

            # Simple grid layout for page
            nodes_per_row = max(1, int(len(page_nodes) ** 0.5))

            for i, node_id in enumerate(page_nodes):
                x = (i % nodes_per_row) * 2
                y = y_offset + (i // nodes_per_row) * 1
                pos[node_id] = (x, y)

            y_offset += max(1, len(page_nodes) // nodes_per_row + 1) + 2

        return pos

    def _add_legend(self, ax, filter_edges: Optional[List[EdgeType]] = None):
        """Add legend to the plot."""
        legend_elements = []

        # Node type legend
        for node_type in NodeType:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=self.config["node_colors"][node_type],
                    markersize=10,
                    label=node_type.value.title(),
                )
            )

        # Edge type legend
        edge_types = filter_edges if filter_edges else list(EdgeType)
        for edge_type in edge_types:
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=self.config["edge_colors"][edge_type],
                    linewidth=2,
                    label=edge_type.value.replace("_", " ").title(),
                )
            )

        ax.legend(
            handles=legend_elements,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=10,
        )
