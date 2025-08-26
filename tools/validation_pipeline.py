#!/usr/bin/env python3
"""
Validation Pipeline for Gold Standard Datasets

This tool provides comprehensive validation and quality control for 
gold standard magazine extraction datasets including:
- Schema validation
- Quality metrics calculation
- Performance benchmarking
- Dataset integrity checks
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import hashlib
import difflib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Results from validation checks"""
    check_name: str
    passed: bool
    score: float
    issues: List[str]
    details: Dict[str, Any]
    timestamp: str

@dataclass
class QualityMetrics:
    """Quality metrics for dataset assessment"""
    ocr_accuracy: float
    layout_classification_accuracy: float
    article_reconstruction_completeness: float
    boundary_detection_accuracy: float
    contributor_extraction_accuracy: float
    overall_score: float
    
@dataclass
class BenchmarkResult:
    """Performance benchmark results"""
    processing_time: float
    memory_usage: float
    throughput_pages_per_minute: float
    accuracy_scores: Dict[str, float]
    error_count: int

class ValidationPipeline:
    """Main validation pipeline for gold standard datasets"""
    
    def __init__(self, base_path: str = "data/gold_sets"):
        self.base_path = Path(base_path)
        self.schema_version = "v1.0"
        
        # Quality thresholds from requirements
        self.quality_thresholds = {
            "ocr_accuracy_born_digital": 0.9995,  # WER < 0.0005
            "ocr_accuracy_scanned": 0.985,        # WER < 0.015  
            "layout_classification": 0.995,       # >99.5%
            "article_reconstruction": 0.98,       # >98%
            "boundary_detection": 0.95,           # >95%
            "contributor_extraction": 0.90        # >90%
        }
        
        # Validation checks registry
        self.validation_checks = {
            "schema_compliance": self._validate_schema_compliance,
            "data_integrity": self._validate_data_integrity,
            "content_quality": self._validate_content_quality,
            "annotation_consistency": self._validate_annotation_consistency,
            "cross_references": self._validate_cross_references
        }
    
    def validate_dataset(self, brand: Optional[str] = None, 
                        issue_id: Optional[str] = None) -> Dict[str, Any]:
        """Run complete validation pipeline on dataset"""
        logger.info(f"Starting validation for {brand or 'all brands'}")
        
        start_time = datetime.now()
        results = {
            "validation_timestamp": start_time.isoformat(),
            "brand": brand,
            "issue_id": issue_id,
            "overall_status": "pending",
            "validation_results": {},
            "quality_metrics": {},
            "summary": {}
        }
        
        # Determine files to validate
        files_to_validate = self._get_validation_files(brand, issue_id)
        
        if not files_to_validate:
            results["overall_status"] = "no_files"
            return results
        
        # Run validation checks
        all_passed = True
        total_score = 0.0
        check_count = 0
        
        for file_path in files_to_validate:
            file_results = {}
            
            for check_name, check_func in self.validation_checks.items():
                try:
                    validation_result = check_func(file_path)
                    file_results[check_name] = asdict(validation_result)
                    
                    if not validation_result.passed:
                        all_passed = False
                    
                    total_score += validation_result.score
                    check_count += 1
                    
                except Exception as e:
                    logger.error(f"Validation check {check_name} failed for {file_path}: {str(e)}")
                    file_results[check_name] = {
                        "passed": False,
                        "score": 0.0,
                        "issues": [f"Check failed: {str(e)}"],
                        "details": {},
                        "timestamp": datetime.now().isoformat()
                    }
                    all_passed = False
            
            results["validation_results"][str(file_path)] = file_results
        
        # Calculate overall metrics
        results["overall_status"] = "passed" if all_passed else "failed"
        results["quality_metrics"] = self._calculate_quality_metrics(results["validation_results"])
        results["summary"] = {
            "total_files": len(files_to_validate),
            "passed_files": sum(1 for f in results["validation_results"].values() 
                               if all(check["passed"] for check in f.values())),
            "average_score": total_score / max(1, check_count),
            "validation_duration": (datetime.now() - start_time).total_seconds()
        }
        
        return results
    
    def _get_validation_files(self, brand: Optional[str] = None, 
                             issue_id: Optional[str] = None) -> List[Path]:
        """Get list of files to validate"""
        files = []
        
        brands_to_check = [brand] if brand else ["economist", "time", "newsweek", "vogue"]
        
        for brand_name in brands_to_check:
            brand_path = self.base_path / brand_name / "ground_truth"
            
            if not brand_path.exists():
                continue
            
            for xml_file in brand_path.glob("*.xml"):
                if issue_id is None or issue_id in xml_file.name:
                    files.append(xml_file)
        
        return files
    
    def _validate_schema_compliance(self, xml_path: Path) -> ValidationResult:
        """Validate XML schema compliance"""
        issues = []
        details = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Check root element
            if root.tag != "magazine_extraction":
                issues.append(f"Invalid root element: {root.tag}")
            
            # Check schema version
            schema_version = root.get("schema_version")
            if schema_version != self.schema_version:
                issues.append(f"Schema version mismatch: {schema_version} vs {self.schema_version}")
            
            # Check required attributes
            required_attrs = ["brand", "schema_version"]
            for attr in required_attrs:
                if not root.get(attr):
                    issues.append(f"Missing required attribute: {attr}")
            
            # Check required sections
            required_sections = ["metadata", "pages", "articles"]
            for section in required_sections:
                if root.find(section) is None:
                    issues.append(f"Missing required section: {section}")
            
            # Detailed validation of sections
            self._validate_metadata_section(root.find("metadata"), issues, details)
            self._validate_pages_section(root.find("pages"), issues, details)
            self._validate_articles_section(root.find("articles"), issues, details)
            
            # Calculate compliance score
            score = max(0.0, 1.0 - (len(issues) * 0.1))  # -0.1 per issue
            
            return ValidationResult(
                check_name="schema_compliance",
                passed=len(issues) == 0,
                score=score,
                issues=issues,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
        except ET.ParseError as e:
            return ValidationResult(
                check_name="schema_compliance", 
                passed=False,
                score=0.0,
                issues=[f"XML parsing error: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            return ValidationResult(
                check_name="schema_compliance",
                passed=False,
                score=0.0,
                issues=[f"Validation error: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_metadata_section(self, metadata_elem: Optional[ET.Element], 
                                  issues: List[str], details: Dict[str, Any]):
        """Validate metadata section"""
        if metadata_elem is None:
            return
        
        required_fields = ["brand", "issue_id", "schema_version", "total_pages"]
        metadata_details = {}
        
        for field in required_fields:
            element = metadata_elem.find(field)
            if element is None:
                issues.append(f"Missing metadata field: {field}")
            else:
                metadata_details[field] = element.text
        
        details["metadata"] = metadata_details
    
    def _validate_pages_section(self, pages_elem: Optional[ET.Element],
                               issues: List[str], details: Dict[str, Any]):
        """Validate pages section"""
        if pages_elem is None:
            return
        
        pages_details = {"page_count": 0, "block_count": 0, "page_numbers": []}
        page_numbers = set()
        
        for page in pages_elem.findall("page"):
            page_num = page.get("number")
            if not page_num:
                issues.append("Page missing number attribute")
            elif page_num in page_numbers:
                issues.append(f"Duplicate page number: {page_num}")
            else:
                page_numbers.add(page_num)
                pages_details["page_numbers"].append(int(page_num))
            
            pages_details["page_count"] += 1
            
            # Validate blocks
            blocks = page.find("blocks")
            if blocks is None:
                issues.append(f"Page {page_num} missing blocks section")
            else:
                block_count = len(blocks.findall("block"))
                pages_details["block_count"] += block_count
                
                if block_count == 0:
                    issues.append(f"Page {page_num} has no blocks")
        
        details["pages"] = pages_details
    
    def _validate_articles_section(self, articles_elem: Optional[ET.Element],
                                  issues: List[str], details: Dict[str, Any]):
        """Validate articles section"""
        if articles_elem is None:
            return
        
        articles_details = {"article_count": 0, "total_contributors": 0}
        article_ids = set()
        
        for article in articles_elem.findall("article"):
            article_id = article.get("id")
            if not article_id:
                issues.append("Article missing id attribute")
            elif article_id in article_ids:
                issues.append(f"Duplicate article id: {article_id}")
            else:
                article_ids.add(article_id)
            
            articles_details["article_count"] += 1
            
            # Check required elements
            if article.find("title") is None:
                issues.append(f"Article {article_id} missing title")
            
            # Count contributors
            contributors = article.find("contributors")
            if contributors is not None:
                contrib_count = len(contributors.findall("contributor"))
                articles_details["total_contributors"] += contrib_count
        
        details["articles"] = articles_details
    
    def _validate_data_integrity(self, xml_path: Path) -> ValidationResult:
        """Validate data integrity and consistency"""
        issues = []
        details = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Collect all block IDs from pages
            all_block_ids = set()
            pages = root.find("pages")
            if pages is not None:
                for page in pages.findall("page"):
                    blocks = page.find("blocks")
                    if blocks is not None:
                        for block in blocks.findall("block"):
                            block_id = block.get("id")
                            if block_id:
                                if block_id in all_block_ids:
                                    issues.append(f"Duplicate block ID: {block_id}")
                                all_block_ids.add(block_id)
            
            # Check article block references
            referenced_blocks = set()
            articles = root.find("articles")
            if articles is not None:
                for article in articles.findall("article"):
                    article_id = article.get("id")
                    
                    # Check all block reference sections
                    for section_name in ["title_blocks", "body_blocks", "byline_blocks", "caption_blocks"]:
                        section = article.find(section_name)
                        if section is not None:
                            for block_ref in section.findall("block_id"):
                                block_id = block_ref.text
                                if block_id:
                                    referenced_blocks.add(block_id)
                                    if block_id not in all_block_ids:
                                        issues.append(f"Article {article_id} references non-existent block: {block_id}")
            
            # Check for unreferenced blocks (warnings, not errors)
            unreferenced = all_block_ids - referenced_blocks
            if unreferenced:
                details["unreferenced_blocks"] = list(unreferenced)
            
            details["integrity_stats"] = {
                "total_blocks": len(all_block_ids),
                "referenced_blocks": len(referenced_blocks),
                "unreferenced_blocks": len(unreferenced)
            }
            
            score = max(0.0, 1.0 - (len(issues) * 0.05))
            
            return ValidationResult(
                check_name="data_integrity",
                passed=len(issues) == 0,
                score=score,
                issues=issues,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="data_integrity",
                passed=False,
                score=0.0,
                issues=[f"Integrity check failed: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_content_quality(self, xml_path: Path) -> ValidationResult:
        """Validate content quality metrics"""
        issues = []
        details = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            quality_stats = {
                "empty_blocks": 0,
                "short_text_blocks": 0,
                "missing_bboxes": 0,
                "invalid_bboxes": 0,
                "articles_without_content": 0,
                "total_blocks": 0,
                "total_articles": 0
            }
            
            # Analyze blocks
            pages = root.find("pages")
            if pages is not None:
                for page in pages.findall("page"):
                    blocks = page.find("blocks")
                    if blocks is not None:
                        for block in blocks.findall("block"):
                            quality_stats["total_blocks"] += 1
                            
                            # Check text content
                            text_elem = block.find("text")
                            if text_elem is None or not text_elem.text:
                                quality_stats["empty_blocks"] += 1
                                issues.append(f"Block {block.get('id')} has empty text")
                            elif len(text_elem.text.strip()) < 3:
                                quality_stats["short_text_blocks"] += 1
                            
                            # Check bounding box
                            bbox = block.find("bbox")
                            if bbox is None:
                                quality_stats["missing_bboxes"] += 1
                                issues.append(f"Block {block.get('id')} missing bbox")
                            else:
                                # Validate bbox coordinates
                                try:
                                    x = float(bbox.get("x", 0))
                                    y = float(bbox.get("y", 0))
                                    width = float(bbox.get("width", 0))
                                    height = float(bbox.get("height", 0))
                                    
                                    if width <= 0 or height <= 0:
                                        quality_stats["invalid_bboxes"] += 1
                                        issues.append(f"Block {block.get('id')} has invalid bbox dimensions")
                                        
                                except (ValueError, TypeError):
                                    quality_stats["invalid_bboxes"] += 1
                                    issues.append(f"Block {block.get('id')} has non-numeric bbox values")
            
            # Analyze articles
            articles = root.find("articles")
            if articles is not None:
                for article in articles.findall("article"):
                    quality_stats["total_articles"] += 1
                    
                    # Check if article has content blocks
                    has_content = False
                    for section_name in ["title_blocks", "body_blocks"]:
                        section = article.find(section_name)
                        if section is not None and section.findall("block_id"):
                            has_content = True
                            break
                    
                    if not has_content:
                        quality_stats["articles_without_content"] += 1
                        issues.append(f"Article {article.get('id')} has no content blocks")
            
            # Calculate quality score
            total_checks = (quality_stats["total_blocks"] * 3 +  # text, bbox, dimensions
                          quality_stats["total_articles"])      # content
            failed_checks = (quality_stats["empty_blocks"] + 
                           quality_stats["missing_bboxes"] + 
                           quality_stats["invalid_bboxes"] + 
                           quality_stats["articles_without_content"])
            
            score = max(0.0, 1.0 - (failed_checks / max(1, total_checks)))
            
            details["quality_stats"] = quality_stats
            
            return ValidationResult(
                check_name="content_quality",
                passed=len(issues) == 0,
                score=score,
                issues=issues,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="content_quality",
                passed=False,
                score=0.0,
                issues=[f"Quality check failed: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_annotation_consistency(self, xml_path: Path) -> ValidationResult:
        """Validate annotation consistency"""
        issues = []
        details = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            consistency_stats = {
                "block_type_distribution": Counter(),
                "contributor_role_distribution": Counter(),
                "inconsistent_naming": 0,
                "confidence_issues": 0
            }
            
            # Analyze block types
            pages = root.find("pages")
            if pages is not None:
                for page in pages.findall("page"):
                    blocks = page.find("blocks")
                    if blocks is not None:
                        for block in blocks.findall("block"):
                            block_type = block.get("type", "unknown")
                            consistency_stats["block_type_distribution"][block_type] += 1
                            
                            # Check confidence values
                            confidence = block.get("confidence")
                            if confidence:
                                try:
                                    conf_val = float(confidence)
                                    if not (0.0 <= conf_val <= 1.0):
                                        consistency_stats["confidence_issues"] += 1
                                        issues.append(f"Block {block.get('id')} has confidence out of range: {conf_val}")
                                except ValueError:
                                    consistency_stats["confidence_issues"] += 1
                                    issues.append(f"Block {block.get('id')} has invalid confidence value: {confidence}")
            
            # Analyze contributor consistency
            articles = root.find("articles")
            if articles is not None:
                contributor_names = []
                for article in articles.findall("article"):
                    contributors = article.find("contributors")
                    if contributors is not None:
                        for contributor in contributors.findall("contributor"):
                            name_elem = contributor.find("name")
                            role_elem = contributor.find("role")
                            
                            if name_elem is not None and name_elem.text:
                                contributor_names.append(name_elem.text)
                            
                            if role_elem is not None and role_elem.text:
                                role = role_elem.text
                                consistency_stats["contributor_role_distribution"][role] += 1
                
                # Check for potential name inconsistencies (similar names)
                for i, name1 in enumerate(contributor_names):
                    for j, name2 in enumerate(contributor_names[i+1:], i+1):
                        similarity = difflib.SequenceMatcher(None, name1.lower(), name2.lower()).ratio()
                        if 0.8 < similarity < 1.0:  # Similar but not identical
                            consistency_stats["inconsistent_naming"] += 1
                            issues.append(f"Potentially inconsistent names: '{name1}' vs '{name2}'")
            
            # Calculate consistency score
            total_issues = (consistency_stats["confidence_issues"] + 
                          consistency_stats["inconsistent_naming"])
            score = max(0.0, 1.0 - (total_issues * 0.1))
            
            details["consistency_stats"] = dict(consistency_stats)
            # Convert Counter objects to regular dicts for JSON serialization
            details["consistency_stats"]["block_type_distribution"] = dict(consistency_stats["block_type_distribution"])
            details["consistency_stats"]["contributor_role_distribution"] = dict(consistency_stats["contributor_role_distribution"])
            
            return ValidationResult(
                check_name="annotation_consistency",
                passed=len(issues) == 0,
                score=score,
                issues=issues,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="annotation_consistency",
                passed=False,
                score=0.0,
                issues=[f"Consistency check failed: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_cross_references(self, xml_path: Path) -> ValidationResult:
        """Validate cross-references between elements"""
        issues = []
        details = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Build reference maps
            all_blocks = {}
            all_contributors = {}
            all_images = {}
            
            pages = root.find("pages")
            if pages is not None:
                for page in pages.findall("page"):
                    blocks = page.find("blocks")
                    if blocks is not None:
                        for block in blocks.findall("block"):
                            block_id = block.get("id")
                            if block_id:
                                all_blocks[block_id] = {
                                    "type": block.get("type"),
                                    "page": page.get("number")
                                }
            
            # Validate article references
            articles = root.find("articles")
            if articles is not None:
                for article in articles.findall("article"):
                    article_id = article.get("id")
                    
                    # Check block references
                    for section_name in ["title_blocks", "body_blocks", "byline_blocks", "caption_blocks"]:
                        section = article.find(section_name)
                        if section is not None:
                            for block_ref in section.findall("block_id"):
                                block_id = block_ref.text
                                if block_id and block_id not in all_blocks:
                                    issues.append(f"Article {article_id} references non-existent block: {block_id}")
                    
                    # Validate contributors
                    contributors = article.find("contributors")
                    if contributors is not None:
                        for contributor in contributors.findall("contributor"):
                            contrib_id = contributor.get("id")
                            if contrib_id:
                                all_contributors[contrib_id] = contributor
                    
                    # Validate images
                    images = article.find("images")
                    if images is not None:
                        for image in images.findall("image"):
                            img_id = image.get("id")
                            if img_id:
                                all_images[img_id] = image
            
            details["reference_stats"] = {
                "total_blocks": len(all_blocks),
                "total_contributors": len(all_contributors),
                "total_images": len(all_images),
                "broken_references": len([issue for issue in issues if "non-existent" in issue])
            }
            
            score = max(0.0, 1.0 - (len(issues) * 0.1))
            
            return ValidationResult(
                check_name="cross_references",
                passed=len(issues) == 0,
                score=score,
                issues=issues,
                details=details,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            return ValidationResult(
                check_name="cross_references",
                passed=False,
                score=0.0,
                issues=[f"Cross-reference check failed: {str(e)}"],
                details={},
                timestamp=datetime.now().isoformat()
            )
    
    def _calculate_quality_metrics(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall quality metrics from validation results"""
        
        # Aggregate scores across all files
        total_scores = defaultdict(list)
        
        for file_path, file_results in validation_results.items():
            for check_name, check_result in file_results.items():
                total_scores[check_name].append(check_result["score"])
        
        # Calculate averages
        avg_scores = {}
        for check_name, scores in total_scores.items():
            avg_scores[check_name] = sum(scores) / len(scores) if scores else 0.0
        
        # Map to quality metrics
        quality_metrics = {
            "schema_compliance": avg_scores.get("schema_compliance", 0.0),
            "data_integrity": avg_scores.get("data_integrity", 0.0),
            "content_quality": avg_scores.get("content_quality", 0.0),
            "annotation_consistency": avg_scores.get("annotation_consistency", 0.0),
            "cross_references": avg_scores.get("cross_references", 0.0),
            "overall_score": sum(avg_scores.values()) / len(avg_scores) if avg_scores else 0.0
        }
        
        return quality_metrics
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 output_path: str):
        """Generate comprehensive validation report"""
        
        with open(output_path, 'w') as f:
            json.dump(validation_results, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def validate_against_thresholds(self, quality_metrics: Dict[str, Any]) -> Dict[str, bool]:
        """Check quality metrics against defined thresholds"""
        
        threshold_checks = {}
        
        # Map metrics to thresholds  
        threshold_mapping = {
            "schema_compliance": 0.98,
            "data_integrity": 0.95,
            "content_quality": 0.90,
            "annotation_consistency": 0.85,
            "overall_score": 0.90
        }
        
        for metric, threshold in threshold_mapping.items():
            score = quality_metrics.get(metric, 0.0)
            threshold_checks[metric] = score >= threshold
        
        return threshold_checks

def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Validation Pipeline for Gold Standard Datasets")
    parser.add_argument("--brand", help="Brand to validate (default: all)")
    parser.add_argument("--issue-id", help="Specific issue ID to validate")
    parser.add_argument("--output", help="Output path for validation report")
    parser.add_argument("--base-path", default="data/gold_sets", help="Base path for datasets")
    parser.add_argument("--threshold-check", action="store_true", 
                       help="Check against quality thresholds")
    
    args = parser.parse_args()
    
    pipeline = ValidationPipeline(args.base_path)
    results = pipeline.validate_dataset(args.brand, args.issue_id)
    
    if args.output:
        pipeline.generate_validation_report(results, args.output)
    else:
        print(json.dumps(results, indent=2))
    
    if args.threshold_check:
        threshold_results = pipeline.validate_against_thresholds(results["quality_metrics"])
        print("\nThreshold Validation:")
        for metric, passed in threshold_results.items():
            status = "PASS" if passed else "FAIL"
            print(f"  {metric}: {status}")

if __name__ == "__main__":
    main()