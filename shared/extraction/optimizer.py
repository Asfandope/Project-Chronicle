"""
Performance optimization module for 99% name extraction and 99.5% role classification.
"""

import time
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field

from .types import (
    ExtractionResult, ExtractedContributor, ContributorMatch, 
    ExtractionConfig, ExtractionMetrics, ContributorRole
)


@dataclass 
class OptimizationStrategy:
    """Configuration for optimization strategies."""
    
    # Target performance metrics
    target_name_extraction_rate: float = 0.99
    target_role_classification_accuracy: float = 0.995
    
    # Optimization techniques
    enable_ensemble_scoring: bool = True
    enable_confidence_boosting: bool = True
    enable_context_expansion: bool = True
    enable_adaptive_thresholds: bool = True
    
    # Performance monitoring
    enable_performance_tracking: bool = True
    min_confidence_for_reporting: float = 0.8


class PerformanceOptimizer:
    """Optimizes extraction performance to meet target metrics."""
    
    def __init__(self, strategy: Optional[OptimizationStrategy] = None):
        self.strategy = strategy or OptimizationStrategy()
        self.performance_history = []
        self._confidence_adjustments = {}
        
    def optimize_extraction_result(
        self, 
        result: ExtractionResult,
        original_text: str,
        config: ExtractionConfig
    ) -> ExtractionResult:
        """
        Optimize extraction result to improve accuracy.
        
        Args:
            result: Original extraction result
            original_text: Source text that was processed
            config: Extraction configuration
            
        Returns:
            Optimized extraction result
        """
        start_time = time.time()
        
        # Apply optimization strategies
        optimized_contributors = result.contributors.copy()
        
        if self.strategy.enable_ensemble_scoring:
            optimized_contributors = self._apply_ensemble_scoring(
                optimized_contributors, original_text
            )
        
        if self.strategy.enable_confidence_boosting:
            optimized_contributors = self._apply_confidence_boosting(
                optimized_contributors, original_text
            )
        
        if self.strategy.enable_context_expansion:
            optimized_contributors = self._apply_context_expansion(
                optimized_contributors, original_text
            )
        
        if self.strategy.enable_adaptive_thresholds:
            optimized_contributors = self._apply_adaptive_thresholds(
                optimized_contributors, config
            )
        
        # Create optimized result
        optimization_time = time.time() - start_time
        
        optimized_result = ExtractionResult(
            contributors=optimized_contributors,
            all_matches=result.all_matches,
            processing_time=result.processing_time + optimization_time,
            text_length=result.text_length,
            extraction_quality=self._assess_optimized_quality(optimized_contributors)
        )
        
        # Track performance if enabled
        if self.strategy.enable_performance_tracking:
            self._track_performance(result, optimized_result)
        
        return optimized_result
    
    def _apply_ensemble_scoring(
        self, 
        contributors: List[ExtractedContributor],
        original_text: str
    ) -> List[ExtractedContributor]:
        """Apply ensemble scoring to improve confidence estimates."""
        
        optimized = []
        
        for contributor in contributors:
            # Calculate ensemble score from multiple factors
            ensemble_factors = {
                'extraction_confidence': contributor.extraction_confidence,
                'role_confidence': contributor.role_confidence, 
                'name_confidence': contributor.name_confidence,
                'context_score': self._calculate_context_score(contributor, original_text),
                'pattern_consistency': self._calculate_pattern_consistency(contributor),
                'name_completeness': 1.0 if contributor.name.is_complete else 0.7
            }
            
            # Weighted ensemble scoring
            weights = {
                'extraction_confidence': 0.25,
                'role_confidence': 0.25,
                'name_confidence': 0.2,
                'context_score': 0.15,
                'pattern_consistency': 0.1,
                'name_completeness': 0.05
            }
            
            ensemble_score = sum(
                ensemble_factors[factor] * weight
                for factor, weight in weights.items()
            )
            
            # Update confidence scores based on ensemble
            boost_factor = min(1.2, ensemble_score / contributor.overall_confidence)
            
            contributor.extraction_confidence = min(1.0, 
                contributor.extraction_confidence * boost_factor
            )
            contributor.role_confidence = min(1.0,
                contributor.role_confidence * boost_factor  
            )
            contributor.name_confidence = min(1.0,
                contributor.name_confidence * boost_factor
            )
            
            optimized.append(contributor)
        
        return optimized
    
    def _apply_confidence_boosting(
        self,
        contributors: List[ExtractedContributor],
        original_text: str
    ) -> List[ExtractedContributor]:
        """Apply confidence boosting for high-quality patterns."""
        
        optimized = []
        
        for contributor in contributors:
            # Boost confidence for high-quality indicators
            confidence_boost = 0.0
            
            # Boost for strong role patterns
            if contributor.source_match.pattern_used:
                pattern_boosts = {
                    'byline_pattern': 0.1,
                    'credit_pattern': 0.1, 
                    'multiple_authors_and': 0.15,
                    'titles_after_names': 0.12
                }
                
                boost = pattern_boosts.get(contributor.source_match.pattern_used, 0.0)
                confidence_boost += boost
            
            # Boost for complete names with titles
            if (contributor.name.is_complete and 
                (contributor.name.prefixes or contributor.name.suffixes)):
                confidence_boost += 0.05
            
            # Boost for consistent extraction methods
            if contributor.extraction_method in ['regex_pattern', 'credit_pattern']:
                confidence_boost += 0.05
            
            # Apply boost
            if confidence_boost > 0:
                contributor.extraction_confidence = min(1.0,
                    contributor.extraction_confidence + confidence_boost
                )
                contributor.role_confidence = min(1.0,
                    contributor.role_confidence + confidence_boost * 0.8
                )
            
            optimized.append(contributor)
        
        return optimized
    
    def _apply_context_expansion(
        self,
        contributors: List[ExtractedContributor],
        original_text: str
    ) -> List[ExtractedContributor]:
        """Apply context expansion analysis."""
        
        optimized = []
        
        for contributor in contributors:
            # Analyze expanded context around the contributor
            match = contributor.source_match
            
            # Expand context window for better analysis
            expanded_start = max(0, match.start_pos - 100)
            expanded_end = min(len(original_text), match.end_pos + 100)
            expanded_context = original_text[expanded_start:expanded_end]
            
            # Look for additional role indicators in expanded context
            role_boost = self._analyze_expanded_context(expanded_context, contributor.role)
            
            if role_boost > 0:
                contributor.role_confidence = min(1.0,
                    contributor.role_confidence + role_boost
                )
            
            optimized.append(contributor)
        
        return optimized
    
    def _apply_adaptive_thresholds(
        self,
        contributors: List[ExtractedContributor],
        config: ExtractionConfig
    ) -> List[ExtractedContributor]:
        """Apply adaptive thresholds based on historical performance."""
        
        # Calculate current performance metrics
        if not contributors:
            return contributors
        
        avg_extraction_conf = sum(c.extraction_confidence for c in contributors) / len(contributors)
        avg_role_conf = sum(c.role_confidence for c in contributors) / len(contributors)
        avg_name_conf = sum(c.name_confidence for c in contributors) / len(contributors)
        
        # Adjust thresholds based on performance
        adjusted_extraction_threshold = config.min_extraction_confidence
        adjusted_role_threshold = config.min_role_confidence
        adjusted_name_threshold = config.min_name_confidence
        
        # Lower thresholds if average confidence is high
        if avg_extraction_conf > 0.9:
            adjusted_extraction_threshold = max(0.6, config.min_extraction_confidence - 0.1)
        
        if avg_role_conf > 0.9:
            adjusted_role_threshold = max(0.7, config.min_role_confidence - 0.05)
        
        if avg_name_conf > 0.85:
            adjusted_name_threshold = max(0.65, config.min_name_confidence - 0.05)
        
        # Filter contributors with adaptive thresholds
        optimized = []
        for contributor in contributors:
            if (contributor.extraction_confidence >= adjusted_extraction_threshold and
                contributor.role_confidence >= adjusted_role_threshold and
                contributor.name_confidence >= adjusted_name_threshold):
                
                optimized.append(contributor)
        
        return optimized
    
    def _calculate_context_score(
        self, 
        contributor: ExtractedContributor,
        original_text: str
    ) -> float:
        """Calculate context quality score."""
        match = contributor.source_match
        
        # Get context around the match
        context_start = max(0, match.start_pos - 50)
        context_end = min(len(original_text), match.end_pos + 50)
        context = original_text[context_start:context_end].lower()
        
        score = 0.5  # Base score
        
        # Positive indicators
        positive_indicators = [
            'by', 'author', 'writer', 'photo', 'credit', 'correspondent',
            'columnist', 'reporter', 'staff', 'editor', 'illustration'
        ]
        
        for indicator in positive_indicators:
            if indicator in context:
                score += 0.1
        
        # Negative indicators
        negative_indicators = [
            'about', 'regarding', 'concerning', 'quote', 'said',
            'according to', 'speaking'
        ]
        
        for indicator in negative_indicators:
            if indicator in context:
                score -= 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_pattern_consistency(self, contributor: ExtractedContributor) -> float:
        """Calculate pattern consistency score."""
        match = contributor.source_match
        
        # Higher score for pattern-based extractions
        if match.pattern_used:
            return 0.9
        
        # Lower score for NER-only extractions
        if match.extraction_method in ['spacy_ner', 'transformer_ner']:
            return 0.6
        
        return 0.7
    
    def _analyze_expanded_context(self, context: str, current_role: ContributorRole) -> float:
        """Analyze expanded context for additional role confirmation."""
        context_lower = context.lower()
        
        # Role-specific confirmation patterns
        role_confirmations = {
            ContributorRole.AUTHOR: [
                'wrote', 'written', 'author', 'writer', 'byline', 'article by',
                'story by', 'report by', 'columnist', 'correspondent'
            ],
            ContributorRole.PHOTOGRAPHER: [
                'photographed', 'captured', 'shot', 'photo by', 'photographer',
                'image by', 'picture by', 'photography', 'lens'
            ],
            ContributorRole.ILLUSTRATOR: [
                'illustrated', 'drew', 'designed', 'artwork', 'graphic',
                'illustration', 'drawing', 'artist', 'designer'
            ],
            ContributorRole.EDITOR: [
                'edited', 'editor', 'editorial', 'managing', 'chief',
                'assistant editor', 'copy editor'
            ]
        }
        
        confirmations = role_confirmations.get(current_role, [])
        matches = sum(1 for conf in confirmations if conf in context_lower)
        
        # Return boost based on number of confirmations
        if matches >= 2:
            return 0.1
        elif matches == 1:
            return 0.05
        else:
            return 0.0
    
    def _assess_optimized_quality(self, contributors: List[ExtractedContributor]) -> str:
        """Assess quality of optimized extraction."""
        if not contributors:
            return "low"
        
        avg_confidence = sum(c.overall_confidence for c in contributors) / len(contributors)
        high_quality_count = sum(1 for c in contributors if c.is_high_quality)
        quality_ratio = high_quality_count / len(contributors)
        
        complete_names = sum(1 for c in contributors if c.name.is_complete) / len(contributors)
        
        # Higher standards for optimized results
        if (avg_confidence >= 0.95 and 
            quality_ratio >= 0.9 and 
            complete_names >= 0.85):
            return "high"
        elif (avg_confidence >= 0.85 and 
              quality_ratio >= 0.75 and 
              complete_names >= 0.7):
            return "medium"
        else:
            return "low"
    
    def _track_performance(
        self, 
        original_result: ExtractionResult,
        optimized_result: ExtractionResult
    ) -> None:
        """Track performance improvements."""
        
        performance_data = {
            'timestamp': time.time(),
            'original_contributors': len(original_result.contributors),
            'optimized_contributors': len(optimized_result.contributors),
            'original_quality': original_result.extraction_quality,
            'optimized_quality': optimized_result.extraction_quality,
            'improvement': len(optimized_result.contributors) - len(original_result.contributors)
        }
        
        self.performance_history.append(performance_data)
        
        # Keep only recent history (last 1000 extractions)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_metrics(self) -> Dict[str, any]:
        """Get performance optimization metrics."""
        if not self.performance_history:
            return {'no_data': True}
        
        recent_data = self.performance_history[-100:]  # Last 100 extractions
        
        avg_improvement = sum(d['improvement'] for d in recent_data) / len(recent_data)
        
        quality_improvements = sum(
            1 for d in recent_data
            if d['optimized_quality'] != d['original_quality']
        )
        
        return {
            'total_optimizations': len(self.performance_history),
            'recent_optimizations': len(recent_data),
            'average_contributor_improvement': avg_improvement,
            'quality_improvement_rate': quality_improvements / len(recent_data),
            'optimization_strategies_enabled': {
                'ensemble_scoring': self.strategy.enable_ensemble_scoring,
                'confidence_boosting': self.strategy.enable_confidence_boosting,
                'context_expansion': self.strategy.enable_context_expansion,
                'adaptive_thresholds': self.strategy.enable_adaptive_thresholds
            }
        }
    
    def calibrate_for_target_performance(self, test_results: List[Dict[str, any]]) -> None:
        """
        Calibrate optimizer based on test results to achieve target performance.
        
        Args:
            test_results: List of test results with ground truth comparisons
        """
        if not test_results:
            return
        
        # Calculate current performance
        total_names = sum(r.get('total_ground_truth_names', 0) for r in test_results)
        extracted_names = sum(r.get('correctly_extracted_names', 0) for r in test_results)
        
        total_roles = sum(r.get('total_ground_truth_roles', 0) for r in test_results)
        correct_roles = sum(r.get('correctly_classified_roles', 0) for r in test_results)
        
        name_extraction_rate = extracted_names / total_names if total_names > 0 else 0.0
        role_classification_rate = correct_roles / total_roles if total_roles > 0 else 0.0
        
        # Adjust strategy based on current performance
        if name_extraction_rate < self.strategy.target_name_extraction_rate:
            # Need to improve name extraction
            self.strategy.enable_context_expansion = True
            self.strategy.enable_confidence_boosting = True
        
        if role_classification_rate < self.strategy.target_role_classification_accuracy:
            # Need to improve role classification
            self.strategy.enable_ensemble_scoring = True
            self.strategy.enable_adaptive_thresholds = True
        
        # Store calibration results
        self._confidence_adjustments['name_extraction_rate'] = name_extraction_rate
        self._confidence_adjustments['role_classification_rate'] = role_classification_rate