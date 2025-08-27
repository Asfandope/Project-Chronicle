#!/usr/bin/env python3
"""
Train LayoutLM models for all magazine brands with experiment tracking.

Executes brand-specific fine-tuning for all supported magazines and
provides comprehensive experiment tracking and comparison.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

from data_management.experiment_tracking import (
    ExperimentTracker,
    print_experiment_summary,
)
from data_management.model_training import LayoutLMTrainer, create_training_config

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


def train_single_brand(brand: str, experiment_tracker: ExperimentTracker) -> dict:
    """
    Train LayoutLM for a single brand.

    Args:
        brand: Magazine brand to train
        experiment_tracker: Experiment tracker instance

    Returns:
        Training results dictionary
    """
    print(f"🎯 Starting LayoutLM training for {brand}")
    start_time = time.time()

    try:
        # Create brand-specific configuration
        config = create_training_config(brand)

        # Create experiment
        experiment_id = experiment_tracker.create_experiment(
            brand=brand,
            config=config.to_dict(),
            description=f"LayoutLM fine-tuning for {brand} magazine",
            tags=["layoutlm", "fine-tuning", brand],
        )

        experiment_tracker.start_experiment(experiment_id)

        logger.info("Training started", brand=brand, experiment_id=experiment_id)

        # Initialize trainer
        trainer = LayoutLMTrainer(config)

        # Load training data
        num_examples = trainer.load_training_data()
        if num_examples == 0:
            raise ValueError(f"No training data found for {brand}")

        print(f"📚 Loaded {num_examples} training examples for {brand}")

        # Prepare model and start training
        trainer.prepare_model_and_processor()
        results = trainer.train()

        # Calculate training time
        training_time = time.time() - start_time
        results["training_time"] = training_time
        results["device"] = "cuda" if trainer.model.device.type == "cuda" else "cpu"

        # Run detailed evaluation
        detailed_metrics = trainer.evaluate_model()
        results["classification_report"] = detailed_metrics["classification_report"]
        results["per_label_metrics"] = detailed_metrics

        # Complete experiment
        experiment_tracker.complete_experiment(
            experiment_id=experiment_id,
            results=results,
            model_path=results["model_path"],
        )

        # Determine success level
        accuracy = detailed_metrics["accuracy"]
        if accuracy >= 0.995:
            success_level = "🟢 PRODUCTION READY"
        elif accuracy >= 0.98:
            success_level = "🟡 ACCEPTABLE"
        else:
            success_level = "🔴 NEEDS IMPROVEMENT"

        print(f"✅ {brand} training completed: {success_level} ({accuracy*100:.2f}%)")

        return {
            "brand": brand,
            "success": True,
            "experiment_id": experiment_id,
            "accuracy": accuracy,
            "training_time": training_time,
            "model_path": results["model_path"],
            "results": results,
        }

    except Exception as e:
        logger.error("Training failed", brand=brand, error=str(e), exc_info=True)

        # Mark experiment as failed
        if "experiment_id" in locals():
            experiment_tracker.fail_experiment(experiment_id, str(e))

        print(f"❌ {brand} training failed: {e}")

        return {
            "brand": brand,
            "success": False,
            "error": str(e),
            "training_time": time.time() - start_time,
        }


def main():
    """Main training orchestration."""
    parser = argparse.ArgumentParser(
        description="Train LayoutLM models for all magazine brands",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Train all brands sequentially
  %(prog)s --brands economist time  # Train specific brands
  %(prog)s --parallel               # Train all brands in parallel
  %(prog)s --summary                # Show training summary only
        """,
    )

    parser.add_argument(
        "--brands",
        nargs="+",
        choices=["economist", "time", "newsweek", "vogue"],
        default=["economist", "time", "newsweek", "vogue"],
        help="Brands to train (default: all)",
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Train brands in parallel (faster but more memory intensive)",
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show experiment summary without training",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum parallel workers (default: 2)",
    )

    args = parser.parse_args()

    # Initialize experiment tracker
    experiment_tracker = ExperimentTracker()

    if args.summary:
        print_experiment_summary(experiment_tracker)
        return

    print("🚀 PROJECT CHRONICLE - LAYOUTLM BRAND FINE-TUNING")
    print("=" * 60)
    print(f"📋 Training brands: {', '.join(args.brands)}")
    print(f"⚡ Mode: {'Parallel' if args.parallel else 'Sequential'}")

    if args.parallel and len(args.brands) > 1:
        print(f"👥 Max workers: {args.max_workers}")

    print()

    start_time = time.time()
    results = []

    if args.parallel and len(args.brands) > 1:
        # Parallel training
        print("🔄 Training brands in parallel...")

        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            # Submit training jobs
            future_to_brand = {
                executor.submit(train_single_brand, brand, experiment_tracker): brand
                for brand in args.brands
            }

            # Collect results
            for future in as_completed(future_to_brand):
                brand = future_to_brand[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"❌ {brand} training failed with exception: {e}")
                    results.append({"brand": brand, "success": False, "error": str(e)})

    else:
        # Sequential training
        print("🔄 Training brands sequentially...")

        for brand in args.brands:
            result = train_single_brand(brand, experiment_tracker)
            results.append(result)

    # Calculate total time
    total_time = time.time() - start_time

    # Display final results
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETED - FINAL RESULTS")
    print("=" * 60)

    successful_brands = []
    failed_brands = []
    production_ready_brands = []

    for result in results:
        brand = result["brand"]
        if result["success"]:
            successful_brands.append(brand)
            accuracy = result["accuracy"]
            training_time = result["training_time"]

            status_emoji = (
                "🟢" if accuracy >= 0.995 else "🟡" if accuracy >= 0.98 else "🔴"
            )
            status_text = (
                "PRODUCTION"
                if accuracy >= 0.995
                else "ACCEPTABLE"
                if accuracy >= 0.98
                else "NEEDS WORK"
            )

            print(
                f"{status_emoji} {brand.upper()}: {accuracy*100:.2f}% accuracy - {status_text} ({training_time:.1f}s)"
            )

            if accuracy >= 0.995:
                production_ready_brands.append(brand)
        else:
            failed_brands.append(brand)
            print(f"❌ {brand.upper()}: FAILED - {result.get('error', 'Unknown error')}")

    print(f"\n📊 Summary:")
    print(f"   Total brands: {len(results)}")
    print(f"   Successful: {len(successful_brands)}")
    print(f"   Failed: {len(failed_brands)}")
    print(f"   Production ready: {len(production_ready_brands)}")
    print(f"   Total training time: {total_time:.1f}s")

    if production_ready_brands:
        print(f"\n✅ Production-ready models: {', '.join(production_ready_brands)}")

    if failed_brands:
        print(
            f"\n⚠️  Failed brands (manual intervention needed): {', '.join(failed_brands)}"
        )

    # Show experiment summary
    print()
    print_experiment_summary(experiment_tracker)

    # Export results
    export_path = Path(f"training_results_{int(time.time())}.json")
    experiment_tracker.export_results(export_path)
    print(f"\n📋 Detailed results exported to: {export_path}")

    # Return appropriate exit code
    return 0 if not failed_brands else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
