"""
Accuracy calculation system implementing PRD section 6 requirements.

This module calculates extraction accuracy with weighted scoring:
- Title match: 30% weight (exact match after normalization)
- Body text: 40% weight (WER < 0.1%)
- Contributors: 20% weight (name + role correct)
- Media links: 10% weight (correct image-caption pairs)
"""

import re
import string
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import difflib
import unicodedata

from .types import (
    GroundTruthData, ArticleData, TextElement, ImageElement,
    SyntheticDataError
)


@dataclass
class FieldAccuracy:
    """Accuracy metrics for a specific field."""
    field_name: str
    correct: int = 0
    total: int = 0
    accuracy: float = 0.0
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}
    
    @property
    def accuracy_percentage(self) -> float:
        """Get accuracy as percentage."""
        return self.accuracy * 100.0


@dataclass
class ArticleAccuracy:
    """Complete accuracy metrics for an article."""
    article_id: str
    title_accuracy: FieldAccuracy
    body_text_accuracy: FieldAccuracy
    contributors_accuracy: FieldAccuracy
    media_links_accuracy: FieldAccuracy
    weighted_overall_accuracy: float = 0.0
    
    def __post_init__(self):
        self._calculate_weighted_accuracy()
    
    def _calculate_weighted_accuracy(self):
        """Calculate weighted overall accuracy based on PRD weights."""
        weights = {
            'title': 0.30,
            'body_text': 0.40,
            'contributors': 0.20,
            'media_links': 0.10
        }
        
        self.weighted_overall_accuracy = (
            self.title_accuracy.accuracy * weights['title'] +
            self.body_text_accuracy.accuracy * weights['body_text'] +
            self.contributors_accuracy.accuracy * weights['contributors'] +
            self.media_links_accuracy.accuracy * weights['media_links']
        )


@dataclass
class DocumentAccuracy:
    """Complete accuracy metrics for a document."""
    document_id: str
    article_accuracies: List[ArticleAccuracy]
    overall_title_accuracy: FieldAccuracy
    overall_body_text_accuracy: FieldAccuracy
    overall_contributors_accuracy: FieldAccuracy
    overall_media_links_accuracy: FieldAccuracy
    document_weighted_accuracy: float = 0.0
    
    def __post_init__(self):
        self._calculate_document_accuracy()
    
    def _calculate_document_accuracy(self):
        """Calculate document-level weighted accuracy."""
        if not self.article_accuracies:
            self.document_weighted_accuracy = 0.0
            return
        
        # Average the weighted accuracies across all articles
        total_weighted = sum(
            article.weighted_overall_accuracy 
            for article in self.article_accuracies
        )
        self.document_weighted_accuracy = total_weighted / len(self.article_accuracies)


class TextNormalizer:
    """Normalizes text for comparison."""
    
    def __init__(self):
        # Punctuation to remove for title comparison
        self.punctuation_translator = str.maketrans('', '', string.punctuation)
        
        # Common title words that should be lowercase
        self.articles_prepositions = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'within'
        }
    
    def normalize_title(self, title: str) -> str:
        """Normalize title for exact matching."""
        if not title:
            return ""
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', title)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Remove punctuation except hyphens and apostrophes
        normalized = re.sub(r'[^\w\s\'-]', '', normalized)
        
        # Handle common abbreviations
        normalized = re.sub(r'\bdr\.?\b', 'doctor', normalized)
        normalized = re.sub(r'\bmr\.?\b', 'mister', normalized)
        normalized = re.sub(r'\bms\.?\b', 'miss', normalized)
        normalized = re.sub(r'\bprof\.?\b', 'professor', normalized)
        
        return normalized
    
    def normalize_body_text(self, text: str) -> List[str]:
        """Normalize body text and return word tokens for WER calculation."""
        if not text:
            return []
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', text)
        
        # Convert to lowercase
        normalized = normalized.lower()
        
        # Remove extra whitespace and normalize punctuation
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'[""\u2018\u2019\u201A\u201C\u201D\u201E]', '"', normalized)
        normalized = re.sub(r'[–—]', '-', normalized)
        
        # Tokenize into words (removing punctuation)
        words = re.findall(r'\b\w+\b', normalized)
        
        return words
    
    def normalize_contributor_name(self, name: str) -> str:
        """Normalize contributor name."""
        if not name:
            return ""
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFKD', name)
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        # Convert to title case
        normalized = normalized.title()
        
        # Handle common abbreviations and suffixes
        normalized = re.sub(r'\bDr\.?\b', 'Dr.', normalized)
        normalized = re.sub(r'\bMr\.?\b', 'Mr.', normalized)
        normalized = re.sub(r'\bMs\.?\b', 'Ms.', normalized)
        normalized = re.sub(r'\bProf\.?\b', 'Prof.', normalized)
        normalized = re.sub(r'\bJr\.?\b', 'Jr.', normalized)
        normalized = re.sub(r'\bSr\.?\b', 'Sr.', normalized)
        
        return normalized


class WordErrorRateCalculator:
    """Calculates Word Error Rate (WER) for body text accuracy."""
    
    def calculate_wer(self, reference: List[str], hypothesis: List[str]) -> float:
        """Calculate Word Error Rate between reference and hypothesis."""
        if not reference and not hypothesis:
            return 0.0
        if not reference:
            return 1.0 if hypothesis else 0.0
        if not hypothesis:
            return 1.0
        
        # Use edit distance (Levenshtein) to calculate WER
        edit_distance = self._edit_distance(reference, hypothesis)
        wer = edit_distance / len(reference)
        
        return min(1.0, wer)  # Cap at 1.0
    
    def _edit_distance(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate edit distance between two sequences."""
        m, n = len(seq1), len(seq2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )
        
        return dp[m][n]


class AccuracyCalculator:
    """Main accuracy calculator implementing PRD section 6 requirements."""
    
    def __init__(self):
        self.text_normalizer = TextNormalizer()
        self.wer_calculator = WordErrorRateCalculator()
        
        # PRD weight constants
        self.WEIGHTS = {
            'title': 0.30,
            'body_text': 0.40,
            'contributors': 0.20,
            'media_links': 0.10
        }
        
        # WER threshold for body text accuracy
        self.WER_THRESHOLD = 0.001  # 0.1% as specified in PRD
    
    def calculate_article_accuracy(
        self,
        ground_truth_article: ArticleData,
        extracted_article: Dict[str, Any]
    ) -> ArticleAccuracy:
        """Calculate accuracy for a single article."""
        
        # Calculate title accuracy
        title_accuracy = self._calculate_title_accuracy(
            ground_truth_article.title,
            extracted_article.get('title', '')
        )
        
        # Calculate body text accuracy
        body_text_accuracy = self._calculate_body_text_accuracy(
            ground_truth_article,
            extracted_article.get('text_content', '')
        )
        
        # Calculate contributors accuracy
        contributors_accuracy = self._calculate_contributors_accuracy(
            ground_truth_article.contributors,
            extracted_article.get('contributors', [])
        )
        
        # Calculate media links accuracy
        media_links_accuracy = self._calculate_media_links_accuracy(
            ground_truth_article,
            extracted_article.get('media_elements', [])
        )
        
        return ArticleAccuracy(
            article_id=ground_truth_article.article_id,
            title_accuracy=title_accuracy,
            body_text_accuracy=body_text_accuracy,
            contributors_accuracy=contributors_accuracy,
            media_links_accuracy=media_links_accuracy
        )
    
    def _calculate_title_accuracy(
        self,
        ground_truth_title: str,
        extracted_title: str
    ) -> FieldAccuracy:
        """Calculate title accuracy with exact match after normalization."""
        
        gt_normalized = self.text_normalizer.normalize_title(ground_truth_title)
        ex_normalized = self.text_normalizer.normalize_title(extracted_title)
        
        is_exact_match = gt_normalized == ex_normalized
        
        details = {
            'ground_truth_normalized': gt_normalized,
            'extracted_normalized': ex_normalized,
            'exact_match': is_exact_match,
            'similarity_ratio': difflib.SequenceMatcher(
                None, gt_normalized, ex_normalized
            ).ratio()
        }
        
        return FieldAccuracy(
            field_name='title',
            correct=1 if is_exact_match else 0,
            total=1,
            accuracy=1.0 if is_exact_match else 0.0,
            details=details
        )
    
    def _calculate_body_text_accuracy(
        self,
        ground_truth_article: ArticleData,
        extracted_text: str
    ) -> FieldAccuracy:
        """Calculate body text accuracy using WER < 0.1%."""
        
        # Collect all body text from ground truth
        body_elements = [
            elem for elem in ground_truth_article.text_elements
            if elem.semantic_type in ['paragraph', 'body', 'text']
        ]
        
        if not body_elements:
            return FieldAccuracy(
                field_name='body_text',
                correct=1 if not extracted_text else 0,
                total=1,
                accuracy=1.0 if not extracted_text else 0.0,
                details={'reason': 'no_ground_truth_body_text'}
            )
        
        # Combine all body text
        ground_truth_text = ' '.join(elem.text_content for elem in body_elements)
        
        # Normalize and tokenize
        gt_tokens = self.text_normalizer.normalize_body_text(ground_truth_text)
        ex_tokens = self.text_normalizer.normalize_body_text(extracted_text)
        
        # Calculate WER
        wer = self.wer_calculator.calculate_wer(gt_tokens, ex_tokens)
        
        # Check if WER meets threshold
        meets_threshold = wer <= self.WER_THRESHOLD
        
        details = {
            'word_error_rate': wer,
            'wer_threshold': self.WER_THRESHOLD,
            'meets_threshold': meets_threshold,
            'ground_truth_word_count': len(gt_tokens),
            'extracted_word_count': len(ex_tokens),
            'character_similarity': difflib.SequenceMatcher(
                None, ground_truth_text, extracted_text
            ).ratio()
        }
        
        return FieldAccuracy(
            field_name='body_text',
            correct=1 if meets_threshold else 0,
            total=1,
            accuracy=1.0 if meets_threshold else 0.0,
            details=details
        )
    
    def _calculate_contributors_accuracy(
        self,
        ground_truth_contributors: List[Dict[str, Any]],
        extracted_contributors: List[Dict[str, Any]]
    ) -> FieldAccuracy:
        """Calculate contributors accuracy (name + role correct)."""
        
        if not ground_truth_contributors:
            return FieldAccuracy(
                field_name='contributors',
                correct=1 if not extracted_contributors else 0,
                total=1,
                accuracy=1.0 if not extracted_contributors else 0.0,
                details={'reason': 'no_ground_truth_contributors'}
            )
        
        correct_matches = 0
        total_contributors = len(ground_truth_contributors)
        match_details = []
        
        for gt_contrib in ground_truth_contributors:
            gt_name = self.text_normalizer.normalize_contributor_name(
                gt_contrib.get('name', '')
            )
            gt_role = gt_contrib.get('role', '').lower().strip()
            
            # Find best match in extracted contributors
            best_match = None
            best_score = 0.0
            
            for ex_contrib in extracted_contributors:
                ex_name = self.text_normalizer.normalize_contributor_name(
                    ex_contrib.get('name', '')
                )
                ex_role = ex_contrib.get('role', '').lower().strip()
                
                # Calculate match score
                name_match = gt_name == ex_name
                role_match = gt_role == ex_role
                
                if name_match and role_match:
                    best_score = 1.0
                    best_match = ex_contrib
                    break
                elif name_match:
                    name_similarity = 1.0
                    role_similarity = difflib.SequenceMatcher(None, gt_role, ex_role).ratio()
                    score = (name_similarity + role_similarity) / 2
                    if score > best_score:
                        best_score = score
                        best_match = ex_contrib
                else:
                    name_similarity = difflib.SequenceMatcher(None, gt_name, ex_name).ratio()
                    role_similarity = difflib.SequenceMatcher(None, gt_role, ex_role).ratio()
                    score = (name_similarity + role_similarity) / 2
                    if score > best_score:
                        best_score = score
                        best_match = ex_contrib
            
            # Consider it correct if both name and role match exactly
            is_correct = best_score == 1.0
            if is_correct:
                correct_matches += 1
            
            match_details.append({
                'ground_truth': {'name': gt_name, 'role': gt_role},
                'best_match': best_match,
                'match_score': best_score,
                'is_correct': is_correct
            })
        
        accuracy = correct_matches / total_contributors if total_contributors > 0 else 1.0
        
        return FieldAccuracy(
            field_name='contributors',
            correct=correct_matches,
            total=total_contributors,
            accuracy=accuracy,
            details={
                'match_details': match_details,
                'extracted_count': len(extracted_contributors)
            }
        )
    
    def _calculate_media_links_accuracy(
        self,
        ground_truth_article: ArticleData,
        extracted_media: List[Dict[str, Any]]
    ) -> FieldAccuracy:
        """Calculate media links accuracy (correct image-caption pairs)."""
        
        # Get ground truth image-caption pairs
        gt_image_elements = ground_truth_article.image_elements
        
        if not gt_image_elements:
            return FieldAccuracy(
                field_name='media_links',
                correct=1 if not extracted_media else 0,
                total=1,
                accuracy=1.0 if not extracted_media else 0.0,
                details={'reason': 'no_ground_truth_media'}
            )
        
        correct_pairs = 0
        total_pairs = len(gt_image_elements)
        pair_details = []
        
        # Find captions associated with each image
        for gt_image in gt_image_elements:
            gt_caption = self._find_associated_caption(gt_image, ground_truth_article)
            
            # Find best matching extracted media element
            best_match = None
            best_score = 0.0
            
            for ex_media in extracted_media:
                # Compare image properties and caption
                image_score = self._compare_image_properties(gt_image, ex_media)
                
                if gt_caption:
                    ex_caption = ex_media.get('caption', '')
                    caption_score = difflib.SequenceMatcher(
                        None, 
                        gt_caption.lower().strip(),
                        ex_caption.lower().strip()
                    ).ratio()
                else:
                    caption_score = 1.0 if not ex_media.get('caption') else 0.0
                
                # Combined score (image properties 60%, caption 40%)
                combined_score = image_score * 0.6 + caption_score * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = ex_media
            
            # Consider correct if score > 0.8
            is_correct = best_score > 0.8
            if is_correct:
                correct_pairs += 1
            
            pair_details.append({
                'ground_truth_image_id': gt_image.element_id,
                'ground_truth_caption': gt_caption,
                'best_match': best_match,
                'match_score': best_score,
                'is_correct': is_correct
            })
        
        accuracy = correct_pairs / total_pairs if total_pairs > 0 else 1.0
        
        return FieldAccuracy(
            field_name='media_links',
            correct=correct_pairs,
            total=total_pairs,
            accuracy=accuracy,
            details={
                'pair_details': pair_details,
                'extracted_media_count': len(extracted_media)
            }
        )
    
    def _find_associated_caption(
        self,
        image_element: ImageElement,
        article: ArticleData
    ) -> Optional[str]:
        """Find caption text associated with an image element."""
        
        # Look for caption elements near the image
        caption_elements = [
            elem for elem in article.text_elements
            if elem.semantic_type == 'caption' and elem.page_number == image_element.page_number
        ]
        
        if not caption_elements:
            return None
        
        # Find closest caption by distance
        image_center_x = (image_element.bbox[0] + image_element.bbox[2]) / 2
        image_center_y = (image_element.bbox[1] + image_element.bbox[3]) / 2
        
        closest_caption = None
        min_distance = float('inf')
        
        for caption in caption_elements:
            caption_center_x = (caption.bbox[0] + caption.bbox[2]) / 2
            caption_center_y = (caption.bbox[1] + caption.bbox[3]) / 2
            
            distance = ((image_center_x - caption_center_x) ** 2 + 
                       (image_center_y - caption_center_y) ** 2) ** 0.5
            
            if distance < min_distance:
                min_distance = distance
                closest_caption = caption
        
        return closest_caption.text_content if closest_caption else None
    
    def _compare_image_properties(
        self,
        gt_image: ImageElement,
        extracted_media: Dict[str, Any]
    ) -> float:
        """Compare image properties for matching."""
        
        # Compare bounding box (normalized)
        gt_bbox = gt_image.bbox
        ex_bbox = extracted_media.get('bbox')
        
        if not ex_bbox:
            return 0.0
        
        # Calculate IoU (Intersection over Union)
        iou = self._calculate_bbox_iou(gt_bbox, ex_bbox)
        
        # Compare dimensions if available
        dimension_score = 1.0
        if 'width' in extracted_media and 'height' in extracted_media:
            gt_aspect = gt_image.width / max(1, gt_image.height)
            ex_aspect = extracted_media['width'] / max(1, extracted_media['height'])
            aspect_diff = abs(gt_aspect - ex_aspect) / max(gt_aspect, ex_aspect, 1)
            dimension_score = max(0.0, 1.0 - aspect_diff)
        
        # Combined score
        return (iou * 0.7 + dimension_score * 0.3)
    
    def _calculate_bbox_iou(
        self,
        bbox1: Tuple[float, float, float, float],
        bbox2: Tuple[float, float, float, float]
    ) -> float:
        """Calculate Intersection over Union for bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / max(union_area, 1e-6)
    
    def calculate_document_accuracy(
        self,
        ground_truth: GroundTruthData,
        extracted_document: Dict[str, Any]
    ) -> DocumentAccuracy:
        """Calculate accuracy for an entire document."""
        
        extracted_articles = extracted_document.get('articles', [])
        article_accuracies = []
        
        # Calculate accuracy for each article
        for gt_article in ground_truth.articles:
            # Find matching extracted article by ID or title
            ex_article = self._find_matching_article(gt_article, extracted_articles)
            
            if ex_article:
                accuracy = self.calculate_article_accuracy(gt_article, ex_article)
                article_accuracies.append(accuracy)
            else:
                # No match found - zero accuracy
                zero_accuracy = ArticleAccuracy(
                    article_id=gt_article.article_id,
                    title_accuracy=FieldAccuracy('title', 0, 1, 0.0),
                    body_text_accuracy=FieldAccuracy('body_text', 0, 1, 0.0),
                    contributors_accuracy=FieldAccuracy('contributors', 0, 1, 0.0),
                    media_links_accuracy=FieldAccuracy('media_links', 0, 1, 0.0)
                )
                article_accuracies.append(zero_accuracy)
        
        # Calculate overall field accuracies
        overall_title = self._aggregate_field_accuracy(article_accuracies, 'title_accuracy')
        overall_body = self._aggregate_field_accuracy(article_accuracies, 'body_text_accuracy')
        overall_contributors = self._aggregate_field_accuracy(article_accuracies, 'contributors_accuracy')
        overall_media = self._aggregate_field_accuracy(article_accuracies, 'media_links_accuracy')
        
        return DocumentAccuracy(
            document_id=ground_truth.document_id,
            article_accuracies=article_accuracies,
            overall_title_accuracy=overall_title,
            overall_body_text_accuracy=overall_body,
            overall_contributors_accuracy=overall_contributors,
            overall_media_links_accuracy=overall_media
        )
    
    def _find_matching_article(
        self,
        gt_article: ArticleData,
        extracted_articles: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Find matching extracted article for ground truth article."""
        
        # Try exact ID match first
        for ex_article in extracted_articles:
            if ex_article.get('article_id') == gt_article.article_id:
                return ex_article
        
        # Try title similarity match
        gt_title_normalized = self.text_normalizer.normalize_title(gt_article.title)
        best_match = None
        best_similarity = 0.0
        
        for ex_article in extracted_articles:
            ex_title_normalized = self.text_normalizer.normalize_title(
                ex_article.get('title', '')
            )
            
            similarity = difflib.SequenceMatcher(
                None, gt_title_normalized, ex_title_normalized
            ).ratio()
            
            if similarity > best_similarity and similarity > 0.7:
                best_similarity = similarity
                best_match = ex_article
        
        return best_match
    
    def _aggregate_field_accuracy(
        self,
        article_accuracies: List[ArticleAccuracy],
        field_attr: str
    ) -> FieldAccuracy:
        """Aggregate field accuracy across all articles."""
        
        if not article_accuracies:
            return FieldAccuracy(
                field_name=field_attr.replace('_accuracy', ''),
                correct=0,
                total=0,
                accuracy=0.0
            )
        
        total_correct = sum(
            getattr(acc, field_attr).correct 
            for acc in article_accuracies
        )
        total_count = sum(
            getattr(acc, field_attr).total 
            for acc in article_accuracies
        )
        
        overall_accuracy = total_correct / max(1, total_count)
        
        return FieldAccuracy(
            field_name=field_attr.replace('_accuracy', ''),
            correct=total_correct,
            total=total_count,
            accuracy=overall_accuracy
        )


def create_accuracy_report(
    document_accuracy: DocumentAccuracy,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Create detailed accuracy report."""
    
    report = {
        'document_id': document_accuracy.document_id,
        'timestamp': str(datetime.now()),
        'weighted_overall_accuracy': {
            'percentage': round(document_accuracy.document_weighted_accuracy * 100, 2),
            'score': document_accuracy.document_weighted_accuracy
        },
        'field_accuracies': {
            'title': {
                'weight': 30,
                'accuracy_percentage': round(document_accuracy.overall_title_accuracy.accuracy_percentage, 2),
                'correct': document_accuracy.overall_title_accuracy.correct,
                'total': document_accuracy.overall_title_accuracy.total
            },
            'body_text': {
                'weight': 40,
                'accuracy_percentage': round(document_accuracy.overall_body_text_accuracy.accuracy_percentage, 2),
                'correct': document_accuracy.overall_body_text_accuracy.correct,
                'total': document_accuracy.overall_body_text_accuracy.total
            },
            'contributors': {
                'weight': 20,
                'accuracy_percentage': round(document_accuracy.overall_contributors_accuracy.accuracy_percentage, 2),
                'correct': document_accuracy.overall_contributors_accuracy.correct,
                'total': document_accuracy.overall_contributors_accuracy.total
            },
            'media_links': {
                'weight': 10,
                'accuracy_percentage': round(document_accuracy.overall_media_links_accuracy.accuracy_percentage, 2),
                'correct': document_accuracy.overall_media_links_accuracy.correct,
                'total': document_accuracy.overall_media_links_accuracy.total
            }
        },
        'article_details': [
            {
                'article_id': acc.article_id,
                'weighted_accuracy': round(acc.weighted_overall_accuracy * 100, 2),
                'field_accuracies': {
                    'title': round(acc.title_accuracy.accuracy_percentage, 2),
                    'body_text': round(acc.body_text_accuracy.accuracy_percentage, 2),
                    'contributors': round(acc.contributors_accuracy.accuracy_percentage, 2),
                    'media_links': round(acc.media_links_accuracy.accuracy_percentage, 2)
                }
            }
            for acc in document_accuracy.article_accuracies
        ]
    }
    
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    
    return report