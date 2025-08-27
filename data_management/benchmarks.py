"""
Evaluation benchmarks and accuracy targets for magazine extraction pipeline.

Defines specific accuracy targets for each component and provides benchmarking
infrastructure to measure performance against gold standard datasets.
"""

import json
import statistics
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .schema_validator import DatasetValidator

logger = structlog.get_logger(__name__)


class ComponentType(Enum):
    """Types of pipeline components for benchmarking."""

    LAYOUT_ANALYSIS = "layout_analysis"
    OCR_PROCESSING = "ocr_processing"
    ARTICLE_RECONSTRUCTION = "article_reconstruction"
    END_TO_END = "end_to_end"


class AccuracyLevel(Enum):
    """Accuracy level classifications."""

    PRODUCTION = "production"
    ACCEPTABLE = "acceptable"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILING = "failing"


@dataclass
class AccuracyTarget:
    """Defines accuracy targets for a specific metric."""

    metric_name: str
    production_threshold: float
    acceptable_threshold: float
    improvement_threshold: float
    unit: str = "percentage"  # percentage, ratio, wer, etc.
    description: str = ""

    def classify_score(self, score: float) -> AccuracyLevel:
        """Classify a score according to the thresholds."""
        if score >= self.production_threshold:
            return AccuracyLevel.PRODUCTION
        elif score >= self.acceptable_threshold:
            return AccuracyLevel.ACCEPTABLE
        elif score >= self.improvement_threshold:
            return AccuracyLevel.NEEDS_IMPROVEMENT
        else:
            return AccuracyLevel.FAILING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Result of running a benchmark."""

    component: ComponentType
    brand: str
    metric_name: str
    measured_value: float
    target: AccuracyTarget
    accuracy_level: AccuracyLevel
    timestamp: datetime
    sample_size: int
    processing_time: float
    metadata: Dict[str, Any]

    @property
    def passes_production(self) -> bool:
        """Whether this result meets production standards."""
        return self.accuracy_level == AccuracyLevel.PRODUCTION

    @property
    def is_acceptable(self) -> bool:
        """Whether this result is acceptable for deployment."""
        return self.accuracy_level in [
            AccuracyLevel.PRODUCTION,
            AccuracyLevel.ACCEPTABLE,
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        data["component"] = self.component.value
        data["accuracy_level"] = self.accuracy_level.value
        return data


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite results."""

    brand: str
    suite_timestamp: datetime
    component_results: Dict[ComponentType, List[BenchmarkResult]]
    overall_metrics: Dict[str, float]
    summary: Dict[str, Any]

    @property
    def overall_production_ready(self) -> bool:
        """Whether the overall system meets production standards."""
        return all(
            any(result.passes_production for result in results)
            for results in self.component_results.values()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "brand": self.brand,
            "suite_timestamp": self.suite_timestamp.isoformat(),
            "component_results": {
                component.value: [result.to_dict() for result in results]
                for component, results in self.component_results.items()
            },
            "overall_metrics": self.overall_metrics,
            "summary": self.summary,
            "overall_production_ready": self.overall_production_ready,
        }


class AccuracyTargetRegistry:
    """Registry of accuracy targets for all pipeline components."""

    def __init__(self):
        self.targets = self._initialize_targets()
        self.logger = logger.bind(component="AccuracyTargets")

    def _initialize_targets(self) -> Dict[ComponentType, Dict[str, AccuracyTarget]]:
        """Initialize all accuracy targets based on PRD requirements."""
        return {
            ComponentType.LAYOUT_ANALYSIS: {
                "block_classification_accuracy": AccuracyTarget(
                    metric_name="block_classification_accuracy",
                    production_threshold=99.5,
                    acceptable_threshold=98.0,
                    improvement_threshold=95.0,
                    unit="percentage",
                    description="Accuracy of layout block type classification (title, body, caption, etc.)",
                ),
                "block_boundary_accuracy": AccuracyTarget(
                    metric_name="block_boundary_accuracy",
                    production_threshold=98.0,
                    acceptable_threshold=95.0,
                    improvement_threshold=90.0,
                    unit="percentage",
                    description="Accuracy of text block boundary detection",
                ),
                "reading_order_accuracy": AccuracyTarget(
                    metric_name="reading_order_accuracy",
                    production_threshold=97.0,
                    acceptable_threshold=94.0,
                    improvement_threshold=90.0,
                    unit="percentage",
                    description="Accuracy of reading order determination",
                ),
                "column_detection_accuracy": AccuracyTarget(
                    metric_name="column_detection_accuracy",
                    production_threshold=96.0,
                    acceptable_threshold=92.0,
                    improvement_threshold=88.0,
                    unit="percentage",
                    description="Accuracy of column layout detection",
                ),
            },
            ComponentType.OCR_PROCESSING: {
                "wer_born_digital": AccuracyTarget(
                    metric_name="wer_born_digital",
                    production_threshold=0.0005,  # Lower is better for WER
                    acceptable_threshold=0.001,
                    improvement_threshold=0.002,
                    unit="wer",
                    description="Word Error Rate for born-digital PDFs (target: <0.05%)",
                ),
                "wer_scanned": AccuracyTarget(
                    metric_name="wer_scanned",
                    production_threshold=0.015,  # Lower is better for WER
                    acceptable_threshold=0.025,
                    improvement_threshold=0.035,
                    unit="wer",
                    description="Word Error Rate for scanned PDFs (target: <1.5%)",
                ),
                "character_accuracy": AccuracyTarget(
                    metric_name="character_accuracy",
                    production_threshold=99.8,
                    acceptable_threshold=99.5,
                    improvement_threshold=99.0,
                    unit="percentage",
                    description="Character-level recognition accuracy",
                ),
                "confidence_calibration": AccuracyTarget(
                    metric_name="confidence_calibration",
                    production_threshold=0.95,
                    acceptable_threshold=0.90,
                    improvement_threshold=0.85,
                    unit="correlation",
                    description="Correlation between confidence scores and actual accuracy",
                ),
            },
            ComponentType.ARTICLE_RECONSTRUCTION: {
                "article_boundary_accuracy": AccuracyTarget(
                    metric_name="article_boundary_accuracy",
                    production_threshold=98.0,
                    acceptable_threshold=95.0,
                    improvement_threshold=90.0,
                    unit="percentage",
                    description="Accuracy of article start/end boundary detection",
                ),
                "article_completeness": AccuracyTarget(
                    metric_name="article_completeness",
                    production_threshold=97.0,
                    acceptable_threshold=94.0,
                    improvement_threshold=90.0,
                    unit="percentage",
                    description="Percentage of article content successfully reconstructed",
                ),
                "cross_page_linking": AccuracyTarget(
                    metric_name="cross_page_linking",
                    production_threshold=95.0,
                    acceptable_threshold=90.0,
                    improvement_threshold=85.0,
                    unit="percentage",
                    description="Accuracy of linking article parts across pages",
                ),
                "contributor_extraction": AccuracyTarget(
                    metric_name="contributor_extraction",
                    production_threshold=94.0,
                    acceptable_threshold=90.0,
                    improvement_threshold=85.0,
                    unit="percentage",
                    description="Accuracy of extracting author/contributor information",
                ),
            },
            ComponentType.END_TO_END: {
                "overall_extraction_accuracy": AccuracyTarget(
                    metric_name="overall_extraction_accuracy",
                    production_threshold=96.0,
                    acceptable_threshold=92.0,
                    improvement_threshold=88.0,
                    unit="percentage",
                    description="End-to-end pipeline extraction accuracy",
                ),
                "processing_speed": AccuracyTarget(
                    metric_name="processing_speed",
                    production_threshold=0.5,  # pages per second
                    acceptable_threshold=0.3,
                    improvement_threshold=0.2,
                    unit="pages_per_second",
                    description="Processing speed in pages per second",
                ),
                "memory_efficiency": AccuracyTarget(
                    metric_name="memory_efficiency",
                    production_threshold=500,  # MB per document
                    acceptable_threshold=750,
                    improvement_threshold=1000,
                    unit="mb_per_document",
                    description="Memory usage per document processed",
                ),
            },
        }

    def get_target(
        self, component: ComponentType, metric: str
    ) -> Optional[AccuracyTarget]:
        """Get accuracy target for a specific component and metric."""
        return self.targets.get(component, {}).get(metric)

    def get_component_targets(
        self, component: ComponentType
    ) -> Dict[str, AccuracyTarget]:
        """Get all targets for a component."""
        return self.targets.get(component, {})

    def list_all_targets(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """List all targets in a serializable format."""
        return {
            component.value: {
                metric: target.to_dict() for metric, target in targets.items()
            }
            for component, targets in self.targets.items()
        }


class BenchmarkEvaluator:
    """Evaluates system performance against accuracy targets."""

    def __init__(self, data_root: Path = None):
        """
        Initialize benchmark evaluator.

        Args:
            data_root: Root path to gold standard datasets
        """
        self.data_root = data_root or Path("data/gold_sets")
        self.target_registry = AccuracyTargetRegistry()
        self.validator = DatasetValidator(self.data_root)
        self.logger = logger.bind(component="BenchmarkEvaluator")

    def evaluate_brand_dataset_quality(self, brand: str) -> BenchmarkSuite:
        """
        Evaluate dataset quality against benchmarks.

        Args:
            brand: Brand to evaluate

        Returns:
            Complete benchmark suite results
        """
        start_time = datetime.now()

        self.logger.info("Starting benchmark evaluation", brand=brand)

        # Validate dataset to get baseline metrics
        validation_report = self.validator.validate_brand_dataset(brand)

        component_results = {}

        # Evaluate layout analysis readiness
        layout_results = self._evaluate_layout_readiness(brand, validation_report)
        component_results[ComponentType.LAYOUT_ANALYSIS] = layout_results

        # Evaluate OCR readiness
        ocr_results = self._evaluate_ocr_readiness(brand, validation_report)
        component_results[ComponentType.OCR_PROCESSING] = ocr_results

        # Evaluate reconstruction readiness
        reconstruction_results = self._evaluate_reconstruction_readiness(
            brand, validation_report
        )
        component_results[ComponentType.ARTICLE_RECONSTRUCTION] = reconstruction_results

        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(component_results)

        # Generate summary
        summary = self._generate_summary(brand, component_results, overall_metrics)

        suite = BenchmarkSuite(
            brand=brand,
            suite_timestamp=start_time,
            component_results=component_results,
            overall_metrics=overall_metrics,
            summary=summary,
        )

        evaluation_time = (datetime.now() - start_time).total_seconds()

        self.logger.info(
            "Benchmark evaluation completed",
            brand=brand,
            evaluation_time=evaluation_time,
            production_ready=suite.overall_production_ready,
        )

        return suite

    def _evaluate_layout_readiness(
        self, brand: str, validation_report
    ) -> List[BenchmarkResult]:
        """Evaluate layout analysis component readiness."""
        results = []

        # Dataset quality as proxy for layout classification accuracy
        quality_score = validation_report.average_quality_score * 100
        target = self.target_registry.get_target(
            ComponentType.LAYOUT_ANALYSIS, "block_classification_accuracy"
        )

        if target:
            result = BenchmarkResult(
                component=ComponentType.LAYOUT_ANALYSIS,
                brand=brand,
                metric_name="block_classification_accuracy",
                measured_value=quality_score,
                target=target,
                accuracy_level=target.classify_score(quality_score),
                timestamp=datetime.now(),
                sample_size=validation_report.total_files,
                processing_time=0.0,  # Placeholder
                metadata={
                    "validation_rate": validation_report.validation_rate,
                    "total_files": validation_report.total_files,
                    "evaluation_method": "dataset_quality_proxy",
                },
            )
            results.append(result)

        return results

    def _evaluate_ocr_readiness(
        self, brand: str, validation_report
    ) -> List[BenchmarkResult]:
        """Evaluate OCR processing component readiness."""
        results = []

        # Use dataset completeness as OCR readiness proxy
        if validation_report.coverage_metrics:
            completeness = (
                validation_report.coverage_metrics.get("xml_metadata_coverage", 0) * 100
            )

            # Evaluate against character accuracy target (using completeness as proxy)
            target = self.target_registry.get_target(
                ComponentType.OCR_PROCESSING, "character_accuracy"
            )
            if target:
                result = BenchmarkResult(
                    component=ComponentType.OCR_PROCESSING,
                    brand=brand,
                    metric_name="character_accuracy",
                    measured_value=completeness,
                    target=target,
                    accuracy_level=target.classify_score(completeness),
                    timestamp=datetime.now(),
                    sample_size=validation_report.total_files,
                    processing_time=0.0,
                    metadata={
                        "coverage_metrics": validation_report.coverage_metrics,
                        "evaluation_method": "dataset_completeness_proxy",
                    },
                )
                results.append(result)

        return results

    def _evaluate_reconstruction_readiness(
        self, brand: str, validation_report
    ) -> List[BenchmarkResult]:
        """Evaluate article reconstruction component readiness."""
        results = []

        # Use validation rate as reconstruction accuracy proxy
        reconstruction_accuracy = validation_report.validation_rate
        target = self.target_registry.get_target(
            ComponentType.ARTICLE_RECONSTRUCTION, "article_boundary_accuracy"
        )

        if target:
            result = BenchmarkResult(
                component=ComponentType.ARTICLE_RECONSTRUCTION,
                brand=brand,
                metric_name="article_boundary_accuracy",
                measured_value=reconstruction_accuracy,
                target=target,
                accuracy_level=target.classify_score(reconstruction_accuracy),
                timestamp=datetime.now(),
                sample_size=validation_report.total_files,
                processing_time=0.0,
                metadata={
                    "validation_details": {
                        "valid_files": validation_report.valid_files,
                        "invalid_files": validation_report.invalid_files,
                    },
                    "evaluation_method": "validation_rate_proxy",
                },
            )
            results.append(result)

        return results

    def _calculate_overall_metrics(
        self, component_results: Dict[ComponentType, List[BenchmarkResult]]
    ) -> Dict[str, float]:
        """Calculate overall system metrics."""
        all_scores = []
        production_count = 0
        total_count = 0

        for results in component_results.values():
            for result in results:
                all_scores.append(result.measured_value)
                total_count += 1
                if result.passes_production:
                    production_count += 1

        return {
            "average_score": statistics.mean(all_scores) if all_scores else 0.0,
            "median_score": statistics.median(all_scores) if all_scores else 0.0,
            "production_rate": (production_count / total_count * 100)
            if total_count > 0
            else 0.0,
            "component_coverage": len(component_results),
            "total_metrics_evaluated": total_count,
        }

    def _generate_summary(
        self,
        brand: str,
        component_results: Dict[ComponentType, List[BenchmarkResult]],
        overall_metrics: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate human-readable summary."""

        # Count results by accuracy level
        level_counts = {level.value: 0 for level in AccuracyLevel}
        component_status = {}

        for component, results in component_results.items():
            component_production_ready = any(r.passes_production for r in results)
            component_status[component.value] = {
                "production_ready": component_production_ready,
                "result_count": len(results),
                "best_score": max(r.measured_value for r in results)
                if results
                else 0.0,
            }

            for result in results:
                level_counts[result.accuracy_level.value] += 1

        # Generate recommendations
        recommendations = []

        if overall_metrics["production_rate"] < 100:
            recommendations.append(
                "Improve metrics that don't meet production standards"
            )

        if overall_metrics["component_coverage"] < 4:
            recommendations.append("Add benchmarks for missing pipeline components")

        if level_counts["failing"] > 0:
            recommendations.append(
                f"Address {level_counts['failing']} failing metrics immediately"
            )

        return {
            "brand": brand,
            "overall_production_ready": overall_metrics["production_rate"] == 100,
            "component_status": component_status,
            "accuracy_level_distribution": level_counts,
            "top_performing_component": max(
                component_status.items(), key=lambda x: x[1]["best_score"]
            )[0]
            if component_status
            else None,
            "recommendations": recommendations,
            "dataset_ready_for_training": overall_metrics["production_rate"] >= 75,
            "benchmark_coverage": f"{len(component_results)}/4 components",
        }

    def save_benchmark_report(
        self, suite: BenchmarkSuite, output_path: Optional[Path] = None
    ) -> Path:
        """Save benchmark results to JSON file."""
        if output_path is None:
            output_path = Path(
                f"benchmark_report_{suite.brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(suite.to_dict(), f, indent=2, ensure_ascii=False)

        self.logger.info("Benchmark report saved", path=str(output_path))
        return output_path

    def generate_all_brands_report(self) -> Dict[str, Any]:
        """Generate benchmark report for all available brands."""
        brands = ["economist", "time", "newsweek", "vogue"]

        all_results = {}
        overall_summary = {
            "timestamp": datetime.now().isoformat(),
            "brands_evaluated": len(brands),
            "production_ready_brands": 0,
            "average_production_rate": 0.0,
            "brand_summaries": {},
        }

        production_rates = []

        for brand in brands:
            try:
                suite = self.evaluate_brand_dataset_quality(brand)
                all_results[brand] = suite.to_dict()

                if suite.overall_production_ready:
                    overall_summary["production_ready_brands"] += 1

                production_rates.append(suite.overall_metrics["production_rate"])
                overall_summary["brand_summaries"][brand] = suite.summary

            except Exception as e:
                self.logger.error("Brand evaluation failed", brand=brand, error=str(e))
                all_results[brand] = {"error": str(e)}

        if production_rates:
            overall_summary["average_production_rate"] = statistics.mean(
                production_rates
            )

        return {
            "overall_summary": overall_summary,
            "brand_results": all_results,
            "accuracy_targets": self.target_registry.list_all_targets(),
        }


# Utility functions for CLI usage
def run_brand_benchmark(brand: str) -> None:
    """CLI utility to run benchmark for a single brand."""
    evaluator = BenchmarkEvaluator()

    print(f"🎯 Running benchmark evaluation for {brand}...")
    suite = evaluator.evaluate_brand_dataset_quality(brand)

    print(f"\n=== {brand.upper()} BENCHMARK RESULTS ===")
    print(
        f"Overall Production Ready: {'✅ YES' if suite.overall_production_ready else '❌ NO'}"
    )
    print(f"Production Rate: {suite.overall_metrics['production_rate']:.1f}%")
    print(f"Average Score: {suite.overall_metrics['average_score']:.2f}")
    print(f"Components Evaluated: {suite.overall_metrics['component_coverage']}/4")

    print("\n📊 Component Status:")
    for component, status in suite.summary["component_status"].items():
        status_icon = "✅" if status["production_ready"] else "❌"
        print(f"  {status_icon} {component}: {status['best_score']:.1f} (Best Score)")

    if suite.summary["recommendations"]:
        print("\n🔧 Recommendations:")
        for rec in suite.summary["recommendations"]:
            print(f"  - {rec}")

    # Save report
    report_path = evaluator.save_benchmark_report(suite)
    print(f"\n📋 Detailed report saved: {report_path}")


def run_all_brands_benchmark() -> None:
    """CLI utility to run benchmark for all brands."""
    evaluator = BenchmarkEvaluator()

    print("🎯 Running benchmark evaluation for all brands...")
    report = evaluator.generate_all_brands_report()

    print("\n=== ALL BRANDS BENCHMARK SUMMARY ===")
    summary = report["overall_summary"]
    print(f"Brands Evaluated: {summary['brands_evaluated']}")
    print(
        f"Production Ready: {summary['production_ready_brands']}/{summary['brands_evaluated']}"
    )
    print(f"Average Production Rate: {summary['average_production_rate']:.1f}%")

    print("\n📊 Brand Summary:")
    for brand, brand_summary in summary["brand_summaries"].items():
        ready = "✅" if brand_summary["overall_production_ready"] else "❌"
        status = (
            "Training Ready"
            if brand_summary.get("dataset_ready_for_training", False)
            else "Needs Work"
        )
        print(f"  {ready} {brand}: {status}")

    # Save comprehensive report
    output_path = Path(
        f"all_brands_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n📋 Comprehensive report saved: {output_path}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        brand = sys.argv[1]
        run_brand_benchmark(brand)
    else:
        run_all_brands_benchmark()
