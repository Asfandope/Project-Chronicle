"""
Brand-aware model manager for loading fine-tuned LayoutLM models.

Manages loading and switching between base models and brand-specific
fine-tuned models with automatic fallback and performance monitoring.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
import torch
from transformers import (
    AutoTokenizer,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)

logger = structlog.get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""

    model_name: str
    brand: Optional[str]
    model_path: str
    is_fine_tuned: bool
    accuracy: Optional[float]
    load_time_seconds: float
    memory_usage_mb: Optional[float]
    device: str


class BrandModelManager:
    """
    Manages brand-specific LayoutLM models with automatic fallback.

    Provides intelligent model loading, caching, and switching based on
    brand requirements and model availability.
    """

    def __init__(
        self,
        base_model_name: str = "microsoft/layoutlmv3-large",
        fine_tuned_models_dir: Path = None,
        device: Optional[str] = None,
    ):
        """
        Initialize brand model manager.

        Args:
            base_model_name: Base LayoutLM model name
            fine_tuned_models_dir: Directory containing fine-tuned models
            device: Device to use for models
        """
        self.base_model_name = base_model_name
        self.fine_tuned_models_dir = fine_tuned_models_dir or Path("models/fine_tuned")

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Model storage
        self.loaded_models: Dict[str, Any] = {}
        self.processors: Dict[str, LayoutLMv3Processor] = {}
        self.tokenizers: Dict[str, AutoTokenizer] = {}
        self.model_info: Dict[str, ModelInfo] = {}

        # Brand model mapping
        self.brand_models: Dict[str, str] = {}
        self._discover_brand_models()

        self.logger = logger.bind(
            component="BrandModelManager", device=device, base_model=base_model_name
        )

        self.logger.info(
            "Initialized brand model manager",
            fine_tuned_dir=str(self.fine_tuned_models_dir),
            discovered_brands=list(self.brand_models.keys()),
        )

    def _discover_brand_models(self):
        """Discover available brand-specific models."""
        if not self.fine_tuned_models_dir.exists():
            return

        for brand_dir in self.fine_tuned_models_dir.iterdir():
            if brand_dir.is_dir():
                # Check if it contains a valid model
                model_files = ["config.json", "pytorch_model.bin"]
                if all((brand_dir / file).exists() for file in model_files):
                    self.brand_models[brand_dir.name] = str(brand_dir)

                    self.logger.debug(
                        "Discovered brand model",
                        brand=brand_dir.name,
                        path=str(brand_dir),
                    )

    def get_available_brands(self) -> List[str]:
        """Get list of brands with available fine-tuned models."""
        return list(self.brand_models.keys())

    def has_brand_model(self, brand: str) -> bool:
        """Check if a brand-specific model is available."""
        return brand in self.brand_models

    def load_base_model(self) -> str:
        """
        Load base LayoutLM model.

        Returns:
            Model key for accessing the loaded model
        """
        model_key = "base"

        if model_key in self.loaded_models:
            self.logger.debug("Base model already loaded")
            return model_key

        self.logger.info("Loading base model", model=self.base_model_name)

        try:
            import time

            start_time = time.time()

            # Load processor
            processor = LayoutLMv3Processor.from_pretrained(
                self.base_model_name, apply_ocr=False
            )
            self.processors[model_key] = processor

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizers[model_key] = tokenizer

            # Load model
            model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.base_model_name, num_labels=13  # Standard number of layout labels
            )

            # Move to device
            model.to(self.device)
            model.eval()

            self.loaded_models[model_key] = model

            load_time = time.time() - start_time

            # Store model info
            self.model_info[model_key] = ModelInfo(
                model_name=self.base_model_name,
                brand=None,
                model_path=self.base_model_name,
                is_fine_tuned=False,
                accuracy=None,
                load_time_seconds=load_time,
                memory_usage_mb=self._estimate_model_memory(model),
                device=self.device,
            )

            self.logger.info(
                "Base model loaded successfully",
                load_time=load_time,
                device=self.device,
            )

            return model_key

        except Exception as e:
            self.logger.error("Failed to load base model", error=str(e))
            raise

    def load_brand_model(self, brand: str) -> str:
        """
        Load brand-specific fine-tuned model.

        Args:
            brand: Brand name

        Returns:
            Model key for accessing the loaded model
        """
        model_key = f"brand_{brand}"

        if model_key in self.loaded_models:
            self.logger.debug("Brand model already loaded", brand=brand)
            return model_key

        if brand not in self.brand_models:
            self.logger.warning("No fine-tuned model for brand", brand=brand)
            # Check if generalist model exists before falling back to base
            generalist_model_path = self.fine_tuned_models_dir / "generalist"
            if generalist_model_path.exists() and "generalist" in self.brand_models:
                self.logger.info("Using generalist model as fallback", brand=brand)
                return self.load_brand_model("generalist")
            else:
                self.logger.info(
                    "No generalist model available, using base model", brand=brand
                )
                return self.load_base_model()

        model_path = Path(self.brand_models[brand])
        self.logger.info("Loading brand model", brand=brand, path=str(model_path))

        try:
            import time

            start_time = time.time()

            # Load processor
            processor = LayoutLMv3Processor.from_pretrained(
                str(model_path), apply_ocr=False
            )
            self.processors[model_key] = processor

            # Load tokenizer (might be in model path or use base)
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
            self.tokenizers[model_key] = tokenizer

            # Load fine-tuned model
            model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_path))

            # Move to device
            model.to(self.device)
            model.eval()

            self.loaded_models[model_key] = model

            load_time = time.time() - start_time

            # Try to load training metrics for accuracy info
            accuracy = self._load_model_accuracy(model_path)

            # Store model info
            self.model_info[model_key] = ModelInfo(
                model_name=f"{brand}_fine_tuned",
                brand=brand,
                model_path=str(model_path),
                is_fine_tuned=True,
                accuracy=accuracy,
                load_time_seconds=load_time,
                memory_usage_mb=self._estimate_model_memory(model),
                device=self.device,
            )

            self.logger.info(
                "Brand model loaded successfully",
                brand=brand,
                load_time=load_time,
                accuracy=accuracy,
                device=self.device,
            )

            return model_key

        except Exception as e:
            self.logger.error("Failed to load brand model", brand=brand, error=str(e))
            # Fallback to base model
            self.logger.info("Falling back to base model", brand=brand)
            return self.load_base_model()

    def _load_model_accuracy(self, model_path: Path) -> Optional[float]:
        """Load model accuracy from training config or experiment data."""
        try:
            # Try training config first
            config_path = model_path / "training_config.json"
            if config_path.exists():
                with open(config_path, "r") as f:
                    json.load(f)
                    # This would need to be enhanced based on actual config structure
                    return None

            # Could add experiment tracking integration here
            return None

        except Exception:
            return None

    def _estimate_model_memory(self, model) -> Optional[float]:
        """Estimate model memory usage in MB."""
        try:
            param_count = sum(p.numel() for p in model.parameters())
            # Rough estimate: 4 bytes per parameter (float32)
            memory_mb = param_count * 4 / (1024 * 1024)
            return memory_mb
        except Exception:
            return None

    def get_model_for_brand(self, brand: str) -> tuple:
        """
        Get model, processor, and tokenizer for a brand.

        Args:
            brand: Brand name

        Returns:
            Tuple of (model, processor, tokenizer, model_info)
        """
        # Load brand-specific model if available, otherwise base model
        model_key = self.load_brand_model(brand)

        return (
            self.loaded_models[model_key],
            self.processors[model_key],
            self.tokenizers[model_key],
            self.model_info[model_key],
        )

    def unload_model(self, brand: Optional[str] = None):
        """Unload a specific model or all models."""
        if brand is None:
            # Unload all models
            self.logger.info("Unloading all models")

            if self.device == "cuda":
                torch.cuda.empty_cache()

            self.loaded_models.clear()
            self.processors.clear()
            self.tokenizers.clear()
            self.model_info.clear()

        else:
            # Unload specific brand model
            model_key = f"brand_{brand}"
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
                del self.processors[model_key]
                del self.tokenizers[model_key]
                del self.model_info[model_key]

                if self.device == "cuda":
                    torch.cuda.empty_cache()

                self.logger.info("Unloaded brand model", brand=brand)

    def get_model_performance_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get performance comparison of loaded models."""
        comparison = {}

        for model_key, info in self.model_info.items():
            comparison[model_key] = {
                "brand": info.brand,
                "is_fine_tuned": info.is_fine_tuned,
                "accuracy": info.accuracy,
                "load_time_seconds": info.load_time_seconds,
                "memory_usage_mb": info.memory_usage_mb,
                "device": info.device,
            }

        return comparison

    def get_recommended_model(self, brand: str) -> str:
        """Get recommended model key for a brand."""
        # Prefer fine-tuned if available and performs well
        if self.has_brand_model(brand):
            brand_model_key = f"brand_{brand}"
            if brand_model_key in self.model_info:
                info = self.model_info[brand_model_key]
                # Use fine-tuned model if accuracy is good or unknown
                if info.accuracy is None or info.accuracy >= 0.98:
                    return brand_model_key

        # Fall back to base model
        return "base"

    def reload_brand_models(self):
        """Rediscover and reload brand models (useful after training)."""
        self.logger.info("Reloading brand models")

        # Clear current brand model mappings
        self.brand_models.clear()

        # Rediscover models
        self._discover_brand_models()

        # Unload any previously loaded brand models that no longer exist
        to_unload = []
        for model_key in self.loaded_models.keys():
            if model_key.startswith("brand_"):
                brand = model_key.replace("brand_", "")
                if brand not in self.brand_models:
                    to_unload.append(model_key)

        for model_key in to_unload:
            brand = model_key.replace("brand_", "")
            self.unload_model(brand)

        self.logger.info(
            "Brand models reloaded", available_brands=list(self.brand_models.keys())
        )


# CLI utilities
def main():
    """CLI interface for brand model manager."""
    import sys

    manager = BrandModelManager()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            print("Available brand models:")
            brands = manager.get_available_brands()
            if brands:
                for brand in brands:
                    print(f"  - {brand}")
            else:
                print("  No fine-tuned models found")

        elif command == "load":
            if len(sys.argv) > 2:
                brand = sys.argv[2]
                print(f"Loading model for {brand}...")
                model_key = manager.load_brand_model(brand)
                info = manager.model_info[model_key]
                print(f"Loaded: {info.model_name} ({info.device})")
                if info.accuracy:
                    print(f"Accuracy: {info.accuracy*100:.2f}%")
            else:
                print("Usage: python brand_model_manager.py load <brand>")

        elif command == "compare":
            print("Loading all models for comparison...")
            for brand in manager.get_available_brands():
                manager.load_brand_model(brand)
            manager.load_base_model()

            comparison = manager.get_model_performance_comparison()
            print("\nModel Performance Comparison:")
            print("=" * 60)
            for model_key, perf in comparison.items():
                brand = perf["brand"] or "base"
                fine_tuned = "✓" if perf["is_fine_tuned"] else "✗"
                accuracy = f"{perf['accuracy']*100:.1f}%" if perf["accuracy"] else "N/A"
                memory = (
                    f"{perf['memory_usage_mb']:.1f}MB"
                    if perf["memory_usage_mb"]
                    else "N/A"
                )

                print(
                    f"{brand:12} | Fine-tuned: {fine_tuned} | Accuracy: {accuracy:6} | Memory: {memory:8}"
                )

    else:
        print("Usage: python brand_model_manager.py <list|load|compare> [brand]")


if __name__ == "__main__":
    main()
