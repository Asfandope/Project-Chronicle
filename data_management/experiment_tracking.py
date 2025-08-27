"""
Experiment tracking and model management system for LayoutLM fine-tuning.

Tracks training experiments, model performance, and provides comparison
utilities for brand-specific model optimization.
"""

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a training experiment."""

    experiment_id: str
    brand: str
    model_name: str
    timestamp: datetime

    # Training parameters
    learning_rate: float
    batch_size: int
    num_epochs: int
    max_sequence_length: int

    # Model parameters
    num_labels: int
    warmup_steps: int
    weight_decay: float

    # Additional metadata
    description: Optional[str] = None
    tags: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResults:
    """Results from a training experiment."""

    experiment_id: str

    # Training metrics
    training_loss: float
    validation_accuracy: float
    validation_f1: float
    training_time_seconds: float

    # Model paths
    model_path: str
    config_path: str

    # Detailed metrics
    classification_report: Dict[str, Any]
    per_label_metrics: Dict[str, Dict[str, float]]

    # Production readiness
    production_ready: bool
    accuracy_level: str  # "production", "acceptable", "needs_improvement"

    # System info
    device_used: str
    memory_usage_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentTracker:
    """Tracks and manages LayoutLM training experiments."""

    def __init__(self, experiments_dir: Path = None):
        """
        Initialize experiment tracker.

        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = experiments_dir or Path("experiments")
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.experiments_db = self.experiments_dir / "experiments.json"
        self.logger = logger.bind(component="ExperimentTracker")

        # Load existing experiments
        self.experiments = self._load_experiments()

        self.logger.info(
            "Initialized experiment tracker",
            experiments_dir=str(self.experiments_dir),
            existing_experiments=len(self.experiments),
        )

    def _load_experiments(self) -> Dict[str, Dict[str, Any]]:
        """Load existing experiments from database."""
        if not self.experiments_db.exists():
            return {}

        try:
            with open(self.experiments_db, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning("Error loading experiments database", error=str(e))
            return {}

    def _save_experiments(self):
        """Save experiments database."""
        try:
            # Convert datetime objects to strings for JSON serialization
            def json_serializer(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return str(obj)

            with open(self.experiments_db, "w") as f:
                json.dump(self.experiments, f, indent=2, default=json_serializer)
        except Exception as e:
            self.logger.error("Error saving experiments database", error=str(e))

    def create_experiment(
        self,
        brand: str,
        config: Dict[str, Any],
        description: str = None,
        tags: List[str] = None,
    ) -> str:
        """
        Create a new experiment.

        Args:
            brand: Magazine brand
            config: Training configuration
            description: Experiment description
            tags: Optional tags

        Returns:
            Experiment ID
        """
        experiment_id = (
            f"{brand}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )

        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            brand=brand,
            model_name=config.get("model_name", "microsoft/layoutlmv3-base"),
            timestamp=datetime.now(),
            learning_rate=config.get("learning_rate", 2e-5),
            batch_size=config.get("batch_size", 4),
            num_epochs=config.get("num_epochs", 10),
            max_sequence_length=config.get("max_sequence_length", 512),
            num_labels=config.get("num_labels", 13),
            warmup_steps=config.get("warmup_steps", 500),
            weight_decay=config.get("weight_decay", 0.01),
            description=description,
            tags=tags or [],
        )

        self.experiments[experiment_id] = {
            "config": experiment_config.to_dict(),
            "status": "created",
            "results": None,
        }

        self._save_experiments()

        self.logger.info(
            "Experiment created",
            experiment_id=experiment_id,
            brand=brand,
            description=description,
        )

        return experiment_id

    def start_experiment(self, experiment_id: str):
        """Mark experiment as started."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = "running"
            self.experiments[experiment_id]["start_time"] = datetime.now().isoformat()
            self._save_experiments()

            self.logger.info("Experiment started", experiment_id=experiment_id)

    def complete_experiment(
        self, experiment_id: str, results: Dict[str, Any], model_path: str
    ):
        """
        Complete an experiment with results.

        Args:
            experiment_id: Experiment ID
            results: Training results
            model_path: Path to saved model
        """
        if experiment_id not in self.experiments:
            self.logger.error("Experiment not found", experiment_id=experiment_id)
            return

        # Determine production readiness
        accuracy = results.get("eval_metrics", {}).get("eval_accuracy", 0.0)
        if isinstance(accuracy, dict):
            accuracy = accuracy.get("eval_accuracy", 0.0)

        production_ready = accuracy >= 0.995
        if accuracy >= 0.995:
            accuracy_level = "production"
        elif accuracy >= 0.98:
            accuracy_level = "acceptable"
        else:
            accuracy_level = "needs_improvement"

        experiment_results = ExperimentResults(
            experiment_id=experiment_id,
            training_loss=results.get("training_loss", 0.0),
            validation_accuracy=accuracy,
            validation_f1=results.get("eval_metrics", {}).get("eval_f1", 0.0),
            training_time_seconds=results.get("training_time", 0.0),
            model_path=model_path,
            config_path=str(Path(model_path) / "training_config.json"),
            classification_report=results.get("classification_report", {}),
            per_label_metrics=results.get("per_label_metrics", {}),
            production_ready=production_ready,
            accuracy_level=accuracy_level,
            device_used=results.get("device", "unknown"),
        )

        self.experiments[experiment_id]["status"] = "completed"
        self.experiments[experiment_id]["results"] = experiment_results.to_dict()
        self.experiments[experiment_id]["end_time"] = datetime.now().isoformat()

        self._save_experiments()

        self.logger.info(
            "Experiment completed",
            experiment_id=experiment_id,
            accuracy=accuracy,
            production_ready=production_ready,
        )

    def fail_experiment(self, experiment_id: str, error: str):
        """Mark experiment as failed."""
        if experiment_id in self.experiments:
            self.experiments[experiment_id]["status"] = "failed"
            self.experiments[experiment_id]["error"] = error
            self.experiments[experiment_id]["end_time"] = datetime.now().isoformat()
            self._save_experiments()

            self.logger.error(
                "Experiment failed", experiment_id=experiment_id, error=error
            )

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get experiment details."""
        return self.experiments.get(experiment_id)

    def list_experiments(
        self, brand: Optional[str] = None, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            brand: Filter by brand
            status: Filter by status

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if brand:
            experiments = [
                exp for exp in experiments if exp["config"]["brand"] == brand
            ]

        if status:
            experiments = [exp for exp in experiments if exp["status"] == status]

        # Sort by timestamp (newest first)
        def get_timestamp(exp):
            timestamp = exp["config"]["timestamp"]
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return timestamp

        experiments.sort(key=get_timestamp, reverse=True)

        return experiments

    def get_best_model(self, brand: str) -> Optional[Dict[str, Any]]:
        """Get best performing model for a brand."""
        brand_experiments = self.list_experiments(brand=brand, status="completed")

        if not brand_experiments:
            return None

        # Find experiment with highest accuracy
        best_experiment = max(
            brand_experiments,
            key=lambda x: x.get("results", {}).get("validation_accuracy", 0.0),
        )

        return best_experiment

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple experiments.

        Args:
            experiment_ids: List of experiment IDs to compare

        Returns:
            DataFrame with comparison data
        """
        comparison_data = []

        for exp_id in experiment_ids:
            experiment = self.experiments.get(exp_id)
            if not experiment:
                continue

            config = experiment["config"]
            results = experiment.get("results", {})

            row = {
                "experiment_id": exp_id,
                "brand": config["brand"],
                "learning_rate": config["learning_rate"],
                "batch_size": config["batch_size"],
                "num_epochs": config["num_epochs"],
                "training_loss": results.get("training_loss", None),
                "validation_accuracy": results.get("validation_accuracy", None),
                "validation_f1": results.get("validation_f1", None),
                "production_ready": results.get("production_ready", False),
                "accuracy_level": results.get("accuracy_level", "unknown"),
                "status": experiment["status"],
            }
            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def generate_summary_report(self, brand: Optional[str] = None) -> Dict[str, Any]:
        """Generate summary report of experiments."""
        experiments = self.list_experiments(brand=brand)

        total_experiments = len(experiments)
        completed_experiments = len(
            [e for e in experiments if e["status"] == "completed"]
        )
        failed_experiments = len([e for e in experiments if e["status"] == "failed"])

        production_ready_count = 0
        best_accuracy = 0.0
        brand_stats = {}

        for experiment in experiments:
            if experiment["status"] != "completed":
                continue

            exp_brand = experiment["config"]["brand"]
            if exp_brand not in brand_stats:
                brand_stats[exp_brand] = {
                    "count": 0,
                    "production_ready": 0,
                    "best_accuracy": 0.0,
                }

            brand_stats[exp_brand]["count"] += 1

            results = experiment.get("results", {})
            accuracy = results.get("validation_accuracy", 0.0)

            if results.get("production_ready", False):
                production_ready_count += 1
                brand_stats[exp_brand]["production_ready"] += 1

            if accuracy > best_accuracy:
                best_accuracy = accuracy

            if accuracy > brand_stats[exp_brand]["best_accuracy"]:
                brand_stats[exp_brand]["best_accuracy"] = accuracy

        return {
            "total_experiments": total_experiments,
            "completed_experiments": completed_experiments,
            "failed_experiments": failed_experiments,
            "production_ready_count": production_ready_count,
            "production_ready_rate": production_ready_count
            / max(completed_experiments, 1),
            "best_overall_accuracy": best_accuracy,
            "brand_statistics": brand_stats,
            "report_timestamp": datetime.now().isoformat(),
        }

    def export_results(self, output_path: Path) -> Path:
        """Export all experiment results to JSON."""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "total_experiments": len(self.experiments),
            "experiments": self.experiments,
            "summary": self.generate_summary_report(),
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

        self.logger.info("Experiments exported", output_path=str(output_path))
        return output_path


# CLI utility functions
def print_experiment_summary(tracker: ExperimentTracker, brand: str = None):
    """Print a formatted experiment summary."""
    report = tracker.generate_summary_report(brand)

    print("\n" + "=" * 60)
    print("📊 EXPERIMENT SUMMARY REPORT")
    print("=" * 60)

    if brand:
        print(f"Brand: {brand.upper()}")

    print(f"Total Experiments: {report['total_experiments']}")
    print(f"Completed: {report['completed_experiments']}")
    print(f"Failed: {report['failed_experiments']}")
    print(
        f"Production Ready: {report['production_ready_count']} ({report['production_ready_rate']*100:.1f}%)"
    )
    print(f"Best Accuracy: {report['best_overall_accuracy']*100:.2f}%")

    print("\n📈 Brand Performance:")
    for brand_name, stats in report["brand_statistics"].items():
        ready_rate = stats["production_ready"] / max(stats["count"], 1) * 100
        print(
            f"  {brand_name}: {stats['count']} experiments, "
            f"{stats['production_ready']} ready ({ready_rate:.1f}%), "
            f"best: {stats['best_accuracy']*100:.2f}%"
        )


if __name__ == "__main__":
    # Example usage
    import sys

    tracker = ExperimentTracker()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            brand = sys.argv[2] if len(sys.argv) > 2 else None
            experiments = tracker.list_experiments(brand=brand)
            print(f"Found {len(experiments)} experiments")
            for exp in experiments[:5]:  # Show latest 5
                print(f"  {exp['config']['experiment_id']}: {exp['status']}")

        elif command == "summary":
            brand = sys.argv[2] if len(sys.argv) > 2 else None
            print_experiment_summary(tracker, brand)

        elif command == "best":
            brand = sys.argv[2] if len(sys.argv) > 2 else "economist"
            best = tracker.get_best_model(brand)
            if best:
                results = best.get("results", {})
                print(
                    f"Best {brand} model: {results.get('validation_accuracy', 0)*100:.2f}% accuracy"
                )
                print(f"Model path: {results.get('model_path', 'N/A')}")
            else:
                print(f"No completed experiments found for {brand}")

    else:
        print("Usage: python experiment_tracking.py <list|summary|best> [brand]")
