#!/usr/bin/env python3
"""
Demonstration of the complete LayoutLM fine-tuning and brand-aware system.

This script showcases:
1. Training infrastructure
2. Experiment tracking
3. Brand model management
4. Integration with layout classifier
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_management.brand_model_manager import BrandModelManager
from data_management.experiment_tracking import (
    ExperimentTracker,
    print_experiment_summary,
)
from data_management.model_training import LayoutLMTrainer, create_training_config
from shared.layout.layoutlm import LayoutLMClassifier


def demo_training_system():
    """Demonstrate the training system capabilities."""
    print("🎯 LAYOUTLM FINE-TUNING SYSTEM DEMONSTRATION")
    print("=" * 60)

    # 1. Show available training data
    print("\n📚 Available Training Data:")
    for brand in ["economist", "time", "newsweek", "vogue"]:
        data_dir = Path(f"data/gold_sets/{brand}/ground_truth")
        if data_dir.exists():
            xml_files = list(data_dir.glob("*.xml"))
            print(f"   {brand:10}: {len(xml_files)} XML files")
        else:
            print(f"   {brand:10}: No data directory")

    # 2. Show training configuration capabilities
    print(f"\n⚙️  Training Configuration Examples:")
    for brand in ["economist", "time", "newsweek", "vogue"]:
        config = create_training_config(brand)
        print(
            f"   {brand:10}: LR={config.learning_rate}, Epochs={config.num_epochs}, Batch={config.batch_size}"
        )

    # 3. Show experiment tracking
    print(f"\n📊 Experiment Tracking:")
    tracker = ExperimentTracker()

    # Create sample experiments to show tracking capability
    for brand in ["economist", "time"]:
        config = create_training_config(brand).to_dict()
        exp_id = tracker.create_experiment(
            brand=brand,
            config=config,
            description=f"Demo experiment for {brand}",
            tags=["demo", "layoutlm", brand],
        )
        print(f"   Created experiment: {exp_id}")

    # Show experiment summary
    print_experiment_summary(tracker)

    # 4. Show brand model management
    print(f"\n🤖 Brand Model Management:")
    try:
        manager = BrandModelManager()
        available_brands = manager.get_available_brands()
        print(f"   Available brand models: {available_brands}")

        if available_brands:
            print(f"   Model comparison:")
            comparison = manager.get_model_performance_comparison()
            for model_key, info in comparison.items():
                brand = info.get("brand", "base")
                fine_tuned = "✓" if info.get("is_fine_tuned") else "✗"
                accuracy = (
                    f"{info.get('accuracy', 0)*100:.1f}%"
                    if info.get("accuracy")
                    else "N/A"
                )
                print(
                    f"     {brand:12} | Fine-tuned: {fine_tuned} | Accuracy: {accuracy}"
                )
        else:
            print(f"   No fine-tuned models found (run training first)")

    except Exception as e:
        print(f"   Brand model management: {e}")

    # 5. Show integration with LayoutLM classifier
    print(f"\n🔧 LayoutLM Classifier Integration:")
    try:
        classifier = LayoutLMClassifier(use_brand_models=True)
        model_info = classifier.get_model_info()

        print(f"   Brand models available: {model_info.get('use_brand_models', False)}")
        if model_info.get("available_brands"):
            print(f"   Available brands: {model_info['available_brands']}")

        # Try loading a brand model
        if model_info.get("available_brands"):
            test_brand = model_info["available_brands"][0]
            success = classifier.switch_brand_model(test_brand)
            print(f"   Switched to {test_brand} model: {'✓' if success else '✗'}")

            if success:
                updated_info = classifier.get_model_info()
                print(f"   Current model: {updated_info.get('model_name', 'N/A')}")
                print(f"   Is fine-tuned: {updated_info.get('is_fine_tuned', False)}")

    except Exception as e:
        print(f"   LayoutLM classifier: {e}")


def demo_training_workflow():
    """Demonstrate a complete training workflow."""
    print(f"\n🚀 COMPLETE TRAINING WORKFLOW DEMO")
    print("=" * 40)

    # Use economist as example
    brand = "economist"

    print(f"1. Creating training configuration for {brand}...")
    config = create_training_config(brand)
    print(f"   ✅ Config: {config.learning_rate} LR, {config.num_epochs} epochs")

    print(f"2. Initializing trainer...")
    trainer = LayoutLMTrainer(config)
    print(f"   ✅ Trainer initialized")

    print(f"3. Loading training data...")
    try:
        num_examples = trainer.load_training_data()
        print(f"   ✅ Loaded {num_examples} training examples")
    except Exception as e:
        print(f"   ❌ Data loading failed: {e}")
        return

    print(f"4. Checking training readiness...")
    if num_examples > 0:
        print(f"   ✅ Ready for training ({num_examples} examples)")
        print(f"   📝 Would train with:")
        print(f"      - Learning rate: {config.learning_rate}")
        print(f"      - Batch size: {config.batch_size}")
        print(f"      - Epochs: {config.num_epochs}")
        print(
            f"      - Device: {trainer.device if hasattr(trainer, 'device') else 'auto'}"
        )
    else:
        print(f"   ❌ Not ready - no training examples")

    print(f"\n💡 To run actual training:")
    print(f"   python scripts/train_{brand}.py")
    print(f"   # or")
    print(f"   make train-brand BRAND={brand}")
    print(f"   # or for all brands:")
    print(f"   python scripts/train_all_brands.py")


def demo_makefile_commands():
    """Show available Makefile commands."""
    print(f"\n📋 AVAILABLE MAKEFILE COMMANDS")
    print("=" * 35)

    commands = [
        ("train-brand BRAND=economist", "Train specific brand"),
        ("train-all", "Train all brands sequentially"),
        ("train-parallel", "Train all brands in parallel"),
        ("training-summary", "Show training experiments summary"),
        ("model-compare", "Compare available brand models"),
        ("benchmark-all", "Run benchmarks on all datasets"),
        ("validate-gold-sets", "Validate gold standard datasets"),
    ]

    print("Training commands:")
    for cmd, desc in commands[:5]:
        print(f"   make {cmd:30} # {desc}")

    print(f"\nEvaluation commands:")
    for cmd, desc in commands[5:]:
        print(f"   make {cmd:30} # {desc}")


def main():
    """Main demonstration function."""
    try:
        demo_training_system()
        demo_training_workflow()
        demo_makefile_commands()

        print(f"\n🎉 DEMONSTRATION COMPLETE")
        print("=" * 30)
        print("The LayoutLM fine-tuning system is ready for:")
        print("✅ Brand-specific model training")
        print("✅ Experiment tracking and comparison")
        print("✅ Intelligent model loading and switching")
        print("✅ Integration with existing layout pipeline")
        print("✅ Production-ready deployment")

        print(f"\nNext steps:")
        print("1. Run 'make train-all' to train all brand models")
        print("2. Use 'make benchmark-all' to evaluate performance")
        print("3. Models will automatically be used by the layout classifier")

    except KeyboardInterrupt:
        print(f"\n⚠️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
