#!/usr/bin/env python3
"""
Fine-tune LayoutLM for Vogue magazine.

Optimized hyperparameters and training configuration for Vogue's
fashion-focused, highly visual layout and artistic content.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import structlog

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


def main():
    """Train LayoutLM for Vogue magazine."""
    print("🎯 Training LayoutLM for Vogue magazine")

    # Create Vogue-specific training configuration
    config = create_training_config(
        brand="vogue",
        learning_rate=2.5e-5,  # Slightly higher for complex fashion layouts
        batch_size=4,  # Standard batch size
        num_epochs=15,  # More epochs for artistic/complex layouts
        warmup_steps=600,  # Longer warmup for stability
        weight_decay=0.012,  # Slightly higher regularization for varied content
        # Vogue-specific settings
        max_sequence_length=480,  # Shorter for visual-heavy content
        early_stopping_patience=5,  # More patience for complex fashion content
        eval_steps=80,  # Frequent evaluation for artistic layouts
        save_steps=480,  # Regular checkpointing
        # Output configuration
        output_dir="models/fine_tuned/vogue",
    )

    logger.info(
        "Vogue training configuration created",
        learning_rate=config.learning_rate,
        epochs=config.num_epochs,
        batch_size=config.batch_size,
    )

    # Initialize trainer
    trainer = LayoutLMTrainer(config)

    try:
        # Load Vogue gold standard data
        print("📚 Loading Vogue training data...")
        num_examples = trainer.load_training_data()

        if num_examples == 0:
            print("❌ No training data found. Ensure gold standard data exists.")
            return 1

        print(f"✅ Loaded {num_examples} training examples")

        # Prepare model and processor
        print("🤖 Initializing LayoutLM model...")
        trainer.prepare_model_and_processor()

        # Start training
        print("🚀 Starting LayoutLM fine-tuning for Vogue...")
        results = trainer.train()

        # Display results
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED FOR VOGUE MAGAZINE")
        print("=" * 60)
        print(f"📍 Model saved to: {results['model_path']}")
        print(f"📊 Training loss: {results['training_loss']:.4f}")

        eval_metrics = results["eval_metrics"]
        if "eval_accuracy" in eval_metrics:
            accuracy = eval_metrics["eval_accuracy"]
            print(f"🎯 Evaluation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if "eval_f1" in eval_metrics:
            f1 = eval_metrics["eval_f1"]
            print(f"📈 F1 Score: {f1:.4f}")

        print("\n🔍 Running detailed evaluation...")
        detailed_metrics = trainer.evaluate_model()

        print(f"📊 Final Metrics:")
        print(
            f"   Accuracy: {detailed_metrics['accuracy']:.4f} ({detailed_metrics['accuracy']*100:.2f}%)"
        )
        print(f"   F1 Score: {detailed_metrics['f1_weighted']:.4f}")
        print(f"   Samples: {detailed_metrics['num_samples']}")

        # Check if we meet production targets
        if detailed_metrics["accuracy"] >= 0.995:
            print("✅ PRODUCTION READY: Model exceeds 99.5% accuracy target!")
        elif detailed_metrics["accuracy"] >= 0.98:
            print("⚠️  ACCEPTABLE: Model meets 98% accuracy threshold")
        else:
            print("❌ NEEDS IMPROVEMENT: Model below 98% accuracy threshold")

        print("\n📋 Model ready for integration with Vogue pipeline")
        return 0

    except Exception as e:
        logger.error("Training failed", error=str(e), exc_info=True)
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
