#!/usr/bin/env python3
"""
Train a single "generalist" LayoutLM model on all available brand data.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_management.model_training import LayoutLMTrainer, create_training_config
from data_management.experiment_tracking import ExperimentTracker
import structlog

logger = structlog.get_logger(__name__)

def main():
    """Train a generalist LayoutLM model."""
    brand = "generalist"
    print(f"🎯 Training a single 'generalist' LayoutLM model on all brand data")

    # Use a generic configuration, but feel free to tune this
    config = create_training_config(
        brand=brand,
        num_epochs=15, # Train for longer on the diverse dataset
        output_dir=f"models/fine_tuned/{brand}"
    )

    logger.info("Generalist training configuration created", **config.to_dict())
    
    # Initialize tracker and trainer
    tracker = ExperimentTracker()
    trainer = LayoutLMTrainer(config)
    
    # Load data from ALL brand directories
    data_root = Path("data/gold_sets/")
    all_brands = [d.name for d in data_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"📚 Loading training data from brands: {', '.join(all_brands)}")
    total_examples = 0
    for brand_dir in all_brands:
        brand_data_dir = data_root / brand_dir / "ground_truth"
        if brand_data_dir.exists():
            num_examples = trainer.load_training_data(brand_data_dir)
            total_examples += num_examples
    
    if total_examples == 0:
        print("❌ No training data found across all brands. Cannot train generalist model.")
        return 1

    print(f"✅ Loaded a total of {total_examples} training examples")
    
    # Create experiment
    exp_id = tracker.create_experiment(
        brand=brand,
        config=config.to_dict(),
        description="Generalist model trained on all available brand data.",
        tags=["generalist", "layoutlm", "multi-brand"]
    )
    tracker.start_experiment(exp_id)
    
    # Train
    print("🚀 Starting fine-tuning for the generalist model...")
    results = trainer.train()
    
    # Complete experiment
    tracker.complete_experiment(
        experiment_id=exp_id,
        results=results,
        model_path=results["model_path"]
    )
    
    print(f"🎉 GENERALIST MODEL TRAINING COMPLETE")
    print(f"📍 Model saved to: {results['model_path']}")
    print(f"🎯 Final accuracy: {results['eval_metrics'].get('eval_accuracy', 'N/A')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())