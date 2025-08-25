"""
Main article reconstructor combining all reconstruction components.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import time
import uuid
import structlog

from ..graph import SemanticGraph
from ..layout.types import BlockType
from .traversal import GraphTraversal
from .resolver import AmbiguityResolver
from .types import (
    ArticleBoundary, ReconstructedArticle, ReconstructionConfig,
    TraversalPath, ReconstructionError, ContinuationMarker
)


logger = structlog.get_logger(__name__)


@dataclass
class ReconstructedArticle:
    """Complete reconstructed article with all components."""
    
    # Article identification
    article_id: str
    title: str
    boundary: ArticleBoundary
    
    # Content components
    components: List[Dict[str, Any]] = field(default_factory=list)
    full_text: str = ""
    
    # Structure information
    traversal_path: Optional[TraversalPath] = None
    node_ids: List[str] = field(default_factory=list)
    
    # Quality metrics
    reconstruction_confidence: float = 1.0
    completeness_score: float = 1.0
    quality_issues: List[str] = field(default_factory=list)
    
    # Processing metadata
    processing_time: float = 0.0
    reconstruction_method: str = ""
    
    def get_content_summary(self) -> Dict[str, Any]:
        """Get summary of article content."""
        return {
            "article_id": self.article_id,
            "title": self.title,
            "word_count": len(self.full_text.split()) if self.full_text else 0,
            "component_count": len(self.components),
            "page_range": self.boundary.page_range,
            "confidence": self.reconstruction_confidence,
            "completeness": self.completeness_score,
            "has_issues": len(self.quality_issues) > 0
        }


class ArticleReconstructor:
    """
    Main article reconstructor for semantic graphs.
    
    Orchestrates the complete article reconstruction process including
    start node identification, graph traversal, ambiguity resolution,
    and boundary determination.
    """
    
    def __init__(self, config: Optional[ReconstructionConfig] = None):
        """
        Initialize article reconstructor.
        
        Args:
            config: Reconstruction configuration
        """
        self.config = config or ReconstructionConfig()
        
        # Initialize sub-components
        self.traversal = GraphTraversal(self.config)
        self.resolver = AmbiguityResolver(self.config)
        
        self.logger = logger.bind(component="ArticleReconstructor")
        
        # Processing statistics
        self.stats = {
            "articles_reconstructed": 0,
            "total_processing_time": 0.0,
            "average_confidence": 0.0,
            "split_articles_found": 0,
            "ambiguities_resolved": 0
        }
        
        self.logger.info("Initialized article reconstructor")
    
    def reconstruct_articles(
        self,
        graph: SemanticGraph,
        max_articles: Optional[int] = None
    ) -> List[ReconstructedArticle]:
        """
        Reconstruct all articles from a semantic graph.
        
        Args:
            graph: Semantic graph to process
            max_articles: Maximum number of articles to reconstruct
            
        Returns:
            List of reconstructed articles
        """
        try:
            start_time = time.time()
            
            self.logger.info(
                "Starting article reconstruction",
                graph_nodes=graph.node_count,
                graph_edges=graph.edge_count
            )
            
            # Step 1: Identify article start nodes
            article_starts = self.traversal.identify_article_starts(graph)
            
            if max_articles:
                article_starts = article_starts[:max_articles]
            
            self.logger.debug("Identified article starts", count=len(article_starts))
            
            # Step 2: Reconstruct each article
            reconstructed_articles = []
            visited_nodes: Set[str] = set()
            
            for start_node_id in article_starts:
                if start_node_id in visited_nodes:
                    continue  # Already part of another article
                
                try:
                    article = self._reconstruct_single_article(
                        start_node_id, graph, visited_nodes
                    )
                    
                    if article and self._meets_quality_criteria(article):
                        reconstructed_articles.append(article)
                        
                        # Mark nodes as visited
                        visited_nodes.update(article.node_ids)
                        
                        self.stats["articles_reconstructed"] += 1
                        
                except Exception as e:
                    self.logger.warning(
                        "Error reconstructing article",
                        start_node=start_node_id[:8],
                        error=str(e)
                    )
            
            # Step 3: Post-process and validate articles
            reconstructed_articles = self._post_process_articles(
                reconstructed_articles, graph
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["total_processing_time"] += processing_time
            
            if reconstructed_articles:
                avg_confidence = sum(a.reconstruction_confidence for a in reconstructed_articles) / len(reconstructed_articles)
                self.stats["average_confidence"] = avg_confidence
            
            self.logger.info(
                "Article reconstruction completed",
                articles_found=len(reconstructed_articles),
                processing_time=processing_time,
                avg_confidence=self.stats["average_confidence"]
            )
            
            return reconstructed_articles
            
        except Exception as e:
            self.logger.error("Error in article reconstruction", error=str(e), exc_info=True)
            raise ReconstructionError(f"Failed to reconstruct articles: {e}")
    
    def _reconstruct_single_article(
        self,
        start_node_id: str,
        graph: SemanticGraph,
        visited_nodes: Set[str]
    ) -> Optional[ReconstructedArticle]:
        """Reconstruct a single article starting from a title node."""
        try:
            article_start_time = time.time()
            
            # Traverse the graph to collect article components
            path = self.traversal.traverse_article(start_node_id, graph, visited_nodes.copy())
            
            if path.path_length < self.config.min_article_components:
                return None
            
            # Create article boundary
            boundary = self._create_article_boundary(path, graph)
            
            # Extract article components
            components = self._extract_components(path, graph)
            
            # Build full text
            full_text = self._build_full_text(components)
            
            # Calculate quality metrics
            reconstruction_confidence = self._calculate_reconstruction_confidence(path, components)
            completeness_score = self._calculate_completeness_score(path, components)
            quality_issues = self._identify_quality_issues(path, components)
            
            # Create reconstructed article
            article = ReconstructedArticle(
                article_id=str(uuid.uuid4()),
                title=boundary.title,
                boundary=boundary,
                components=components,
                full_text=full_text,
                traversal_path=path,
                node_ids=path.node_ids,
                reconstruction_confidence=reconstruction_confidence,
                completeness_score=completeness_score,
                quality_issues=quality_issues,
                processing_time=time.time() - article_start_time,
                reconstruction_method="graph_traversal"
            )
            
            return article
            
        except Exception as e:
            self.logger.warning("Error reconstructing single article", error=str(e))
            return None
    
    def _create_article_boundary(self, path: TraversalPath, graph: SemanticGraph) -> ArticleBoundary:
        """Create article boundary from traversal path."""
        try:
            # Get title from start node
            start_node = graph.get_node(path.node_ids[0])
            title = "Untitled Article"
            byline = None
            
            if start_node:
                start_data = start_node.to_graph_data()
                if start_data.text:
                    title = start_data.text.strip()
            
            # Look for byline in early components
            for node_id in path.node_ids[1:4]:  # Check first few nodes
                node = graph.get_node(node_id)
                if node:
                    node_data = node.to_graph_data()
                    if node_data.classification == BlockType.BYLINE and node_data.text:
                        byline = node_data.text.strip()
                        break
            
            # Count continuation markers and split pages
            continuation_markers = []
            split_pages = []
            
            if path.spans_multiple_pages:
                # This is a simplified detection - in reality would parse the actual text
                split_pages = list(range(path.start_page, path.end_page + 1))
            
            # Calculate quality scores
            completeness_score = min(1.0, path.average_confidence)
            confidence_score = path.average_confidence
            
            # Determine quality level
            if confidence_score >= 0.9 and completeness_score >= 0.9:
                quality = "high"
            elif confidence_score >= 0.7 and completeness_score >= 0.7:
                quality = "medium"
            else:
                quality = "low"
            
            # Calculate word count and reading time
            word_count = 0
            for node_id in path.node_ids:
                node = graph.get_node(node_id)
                if node:
                    node_data = node.to_graph_data()
                    if node_data.text:
                        word_count += len(node_data.text.split())
            
            reading_time = word_count / self.config.words_per_minute
            
            return ArticleBoundary(
                article_id=str(uuid.uuid4()),
                title=title,
                start_page=path.start_page,
                end_page=path.end_page,
                total_pages=path.end_page - path.start_page + 1,
                start_node_id=path.node_ids[0],
                end_node_id=path.node_ids[-1],
                component_count=path.path_length,
                continuation_markers=continuation_markers,
                is_split_article=path.spans_multiple_pages,
                split_pages=split_pages,
                completeness_score=completeness_score,
                confidence_score=confidence_score,
                reconstruction_quality=quality,
                byline=byline,
                word_count=word_count,
                estimated_reading_time=reading_time
            )
            
        except Exception as e:
            self.logger.error("Error creating article boundary", error=str(e))
            raise ReconstructionError(f"Failed to create article boundary: {e}")
    
    def _extract_components(self, path: TraversalPath, graph: SemanticGraph) -> List[Dict[str, Any]]:
        """Extract components from traversal path."""
        components = []
        
        try:
            for i, node_id in enumerate(path.node_ids):
                node = graph.get_node(node_id)
                if not node:
                    continue
                
                node_data = node.to_graph_data()
                
                component = {
                    "node_id": node_id,
                    "component_index": i,
                    "text": node_data.text or "",
                    "block_type": node_data.classification.value if node_data.classification else "unknown",
                    "confidence": node_data.confidence,
                    "page_num": node_data.page_num,
                    "bbox": node_data.bbox.to_dict() if node_data.bbox else None,
                    "metadata": node_data.metadata
                }
                
                # Add edge information if not first node
                if i > 0 and i < len(path.edge_types):
                    component["connection_type"] = path.edge_types[i-1].value
                
                components.append(component)
            
            return components
            
        except Exception as e:
            self.logger.error("Error extracting components", error=str(e))
            return []
    
    def _build_full_text(self, components: List[Dict[str, Any]]) -> str:
        """Build full article text from components."""
        try:
            text_parts = []
            
            for component in components:
                text = component.get("text", "").strip()
                if text:
                    # Add appropriate spacing based on block type
                    block_type = component.get("block_type", "")
                    
                    if block_type in ["title", "subtitle", "heading"]:
                        text_parts.append(f"\n\n{text}\n")
                    elif block_type == "byline":
                        text_parts.append(f"{text}\n")
                    else:
                        text_parts.append(f"{text}\n\n")
            
            return "".join(text_parts).strip()
            
        except Exception as e:
            self.logger.warning("Error building full text", error=str(e))
            return ""
    
    def _calculate_reconstruction_confidence(
        self,
        path: TraversalPath,
        components: List[Dict[str, Any]]
    ) -> float:
        """Calculate overall reconstruction confidence."""
        try:
            # Base confidence from path
            path_confidence = path.average_confidence
            
            # Component quality factor
            if components:
                component_confidences = [c.get("confidence", 0.0) for c in components]
                component_confidence = sum(component_confidences) / len(component_confidences)
            else:
                component_confidence = 0.0
            
            # Path completeness factor
            completeness_factor = min(1.0, path.path_length / 10)  # Normalize to 10 components
            
            # Combine factors
            overall_confidence = (
                path_confidence * 0.4 +
                component_confidence * 0.4 +
                completeness_factor * 0.2
            )
            
            return min(1.0, overall_confidence)
            
        except Exception:
            return 0.5
    
    def _calculate_completeness_score(
        self,
        path: TraversalPath,
        components: List[Dict[str, Any]]
    ) -> float:
        """Calculate article completeness score."""
        try:
            score = 0.0
            
            # Check for essential components
            block_types = [c.get("block_type", "") for c in components]
            
            if "title" in block_types:
                score += 0.3
            if "body" in block_types:
                score += 0.4
            if block_types.count("body") >= 2:
                score += 0.2  # Multiple body paragraphs
            
            # Length factor
            total_words = sum(len(c.get("text", "").split()) for c in components)
            if total_words >= self.config.min_article_words:
                score += 0.1
            
            return min(1.0, score)
            
        except Exception:
            return 0.5
    
    def _identify_quality_issues(
        self,
        path: TraversalPath,
        components: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify quality issues with the reconstruction."""
        issues = []
        
        try:
            # Check for missing title
            block_types = [c.get("block_type", "") for c in components]
            if "title" not in block_types:
                issues.append("missing_title")
            
            # Check for insufficient content
            total_words = sum(len(c.get("text", "").split()) for c in components)
            if total_words < self.config.min_article_words:
                issues.append("insufficient_content")
            
            # Check for low confidence
            if path.average_confidence < 0.6:
                issues.append("low_confidence")
            
            # Check for fragmented path
            if len(path.ambiguous_connections) > 2:
                issues.append("fragmented_path")
            
            # Check for page gaps
            if path.spans_multiple_pages:
                page_range = path.end_page - path.start_page + 1
                if page_range > 3:  # Suspiciously large gap
                    issues.append("large_page_gap")
            
            return issues
            
        except Exception as e:
            self.logger.warning("Error identifying quality issues", error=str(e))
            return ["analysis_error"]
    
    def _meets_quality_criteria(self, article: ReconstructedArticle) -> bool:
        """Check if article meets minimum quality criteria."""
        try:
            # Minimum confidence threshold
            if article.reconstruction_confidence < 0.4:
                return False
            
            # Minimum word count
            word_count = len(article.full_text.split()) if article.full_text else 0
            if word_count < self.config.min_article_words:
                return False
            
            # Require title if configured
            if self.config.require_title and "missing_title" in article.quality_issues:
                return False
            
            # Filter out suspected advertisements
            if (self.config.filter_advertisements and 
                any("advertisement" in c.get("block_type", "") for c in article.components)):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _post_process_articles(
        self,
        articles: List[ReconstructedArticle],
        graph: SemanticGraph
    ) -> List[ReconstructedArticle]:
        """Post-process reconstructed articles."""
        try:
            # Sort by confidence and completeness
            articles.sort(
                key=lambda a: (a.reconstruction_confidence, a.completeness_score),
                reverse=True
            )
            
            # Remove duplicate articles (based on content similarity)
            deduplicated = self._remove_duplicate_articles(articles)
            
            # Merge split article parts if detected
            merged = self._merge_split_articles(deduplicated, graph)
            
            return merged
            
        except Exception as e:
            self.logger.warning("Error in post-processing", error=str(e))
            return articles
    
    def _remove_duplicate_articles(self, articles: List[ReconstructedArticle]) -> List[ReconstructedArticle]:
        """Remove duplicate articles based on content similarity."""
        if len(articles) <= 1:
            return articles
        
        unique_articles = []
        
        for article in articles:
            is_duplicate = False
            
            for existing in unique_articles:
                if self._are_duplicate_articles(article, existing):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_articles.append(article)
        
        return unique_articles
    
    def _are_duplicate_articles(self, article1: ReconstructedArticle, article2: ReconstructedArticle) -> bool:
        """Check if two articles are duplicates."""
        # Simple duplicate detection based on title similarity
        title1 = article1.title.lower().strip()
        title2 = article2.title.lower().strip()
        
        if title1 == title2:
            return True
        
        # Check for substantial overlap in node IDs
        nodes1 = set(article1.node_ids)
        nodes2 = set(article2.node_ids)
        
        overlap = len(nodes1.intersection(nodes2))
        union = len(nodes1.union(nodes2))
        
        overlap_ratio = overlap / union if union > 0 else 0
        
        return overlap_ratio > 0.7  # 70% overlap suggests duplication
    
    def _merge_split_articles(
        self,
        articles: List[ReconstructedArticle],
        graph: SemanticGraph
    ) -> List[ReconstructedArticle]:
        """Merge articles that are parts of split articles."""
        # This would implement sophisticated merging logic
        # For now, return articles as-is
        return articles
    
    def get_reconstruction_statistics(self) -> Dict[str, Any]:
        """Get reconstruction statistics."""
        return {
            "articles_reconstructed": self.stats["articles_reconstructed"],
            "total_processing_time": self.stats["total_processing_time"],
            "average_confidence": self.stats["average_confidence"],
            "split_articles_found": self.stats["split_articles_found"],
            "traversal_stats": self.traversal._visited_nodes,
            "resolution_stats": self.resolver.get_resolution_statistics()
        }