"""
LayoutLM fine-tuning infrastructure for magazine-specific document understanding.

This module provides brand-specific fine-tuning capabilities for LayoutLMv3 models
using gold standard datasets. Includes training pipeline, dataset preparation,
and experiment tracking.
"""

import json
import yaml
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from xml.etree import ElementTree as ET

# ML Dependencies  
from torch.utils.data import Dataset, DataLoader
from transformers import (
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.models.layoutlmv3 import LayoutLMv3FeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Image processing
from PIL import Image

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for LayoutLM training."""
    brand: str
    model_name: str = "microsoft/layoutlmv3-base"
    output_dir: str = "models/fine_tuned"
    
    # Training hyperparameters
    learning_rate: float = 2e-5
    batch_size: int = 4
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Validation settings
    eval_steps: int = 100
    save_steps: int = 500
    logging_steps: int = 50
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Data settings
    max_sequence_length: int = 512
    test_size: float = 0.2
    validation_size: float = 0.1
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainingExample:
    """Single training example for LayoutLM."""
    document_id: str
    tokens: List[str]
    bboxes: List[List[int]]  # [x0, y0, x1, y1] normalized to 1000
    labels: List[int]
    brand: str
    
    def __len__(self) -> int:
        return len(self.tokens)


@dataclass 
class TrainingMetrics:
    """Training metrics tracking."""
    epoch: int
    train_loss: float
    eval_loss: Optional[float] = None
    eval_accuracy: Optional[float] = None
    eval_f1: Optional[float] = None
    learning_rate: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self, default=str)


class MagazineLayoutDataset(Dataset):
    """PyTorch dataset for magazine layout analysis."""
    
    def __init__(
        self,
        examples: List[TrainingExample],
        processor: LayoutLMv3Processor,
        max_length: int = 512
    ):
        self.examples = examples
        self.processor = processor
        self.max_length = max_length
        self.logger = logger.bind(component="MagazineLayoutDataset")
        
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        example = self.examples[idx]
        
        # Create synthetic image (LayoutLM requires image input)
        image = Image.new("RGB", (1000, 1000), "white")
        
        # Prepare inputs for LayoutLM
        encoding = self.processor(
            image,
            example.tokens,
            boxes=example.bboxes,
            word_labels=example.labels,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Flatten tensors (remove batch dimension)
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "bbox": encoding["bbox"].flatten(),
            "pixel_values": encoding["pixel_values"].flatten() if "pixel_values" in encoding else None,
            "labels": encoding["labels"].flatten()
        }


class LayoutLMTrainer:
    """LayoutLM fine-tuning trainer for magazine brands."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.brand_config = self._load_brand_config()
        self.logger = logger.bind(component="LayoutLMTrainer", brand=config.brand)
        
        # Set random seeds for reproducibility
        self._set_seeds(config.random_seed)
        
        # Initialize components
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.label_mapping = self._create_label_mapping()
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Training state
        self.training_examples = []
        self.metrics_history = []
        
        # Output directories
        self.output_dir = Path(config.output_dir) / config.brand
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Initialized LayoutLM trainer", 
                        output_dir=str(self.output_dir),
                        num_labels=len(self.label_mapping))
    
    def _set_seeds(self, seed: int):
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _load_brand_config(self) -> Dict[str, Any]:
        """Load brand-specific configuration."""
        config_path = Path(f"configs/brands/{self.config.brand}.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}
    
    def _create_label_mapping(self) -> Dict[str, int]:
        """Create mapping from block types to label IDs."""
        return {
            "title": 0,
            "subtitle": 1, 
            "heading": 2,
            "body": 3,
            "caption": 4,
            "header": 5,
            "footer": 6,
            "byline": 7,
            "quote": 8,
            "sidebar": 9,
            "advertisement": 10,
            "page_number": 11,
            "unknown": 12
        }
    
    def load_training_data(self, data_dir: Optional[Path] = None) -> int:
        """
        Load training data from gold standard XML files.
        
        Args:
            data_dir: Directory containing gold standard data
            
        Returns:
            Number of training examples loaded
        """
        if data_dir is None:
            data_dir = Path(f"data/gold_sets/{self.config.brand}/ground_truth")
        
        self.logger.info("Loading training data", data_dir=str(data_dir))
        
        xml_files = list(data_dir.glob("*.xml"))
        self.training_examples = []
        
        for xml_file in xml_files:
            examples = self._parse_xml_to_examples(xml_file)
            self.training_examples.extend(examples)
            
        self.logger.info("Training data loaded", 
                        num_examples=len(self.training_examples),
                        num_files=len(xml_files))
        
        return len(self.training_examples)
    
    def _parse_xml_to_examples(self, xml_file: Path) -> List[TrainingExample]:
        """Parse XML file to training examples."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            examples = []
            document_id = root.get("document_id", xml_file.stem)
            
            for article in root.findall("article"):
                article_id = article.get("id", "unknown")
                
                # Collect tokens and labels from article elements
                tokens = []
                bboxes = []
                labels = []
                
                # Process title
                title_elem = article.find("title")
                if title_elem is not None:
                    title_tokens = title_elem.text.split() if title_elem.text else []
                    tokens.extend(title_tokens)
                    # Estimate bounding box (would be real data in production)
                    bbox = [50, 50, 500, 100]  # Placeholder bbox
                    bboxes.extend([bbox] * len(title_tokens))
                    labels.extend([self.label_mapping["title"]] * len(title_tokens))
                
                # Process body paragraphs
                for body_elem in article.findall("body"):
                    body_text = body_elem.text or ""
                    body_tokens = body_text.split()
                    tokens.extend(body_tokens)
                    
                    # Estimate bounding box
                    bbox = [50, 150, 500, 200]  # Placeholder bbox
                    bboxes.extend([bbox] * len(body_tokens))
                    labels.extend([self.label_mapping["body"]] * len(body_tokens))
                
                # Process contributors (bylines)
                contributors = article.find("contributors")
                if contributors is not None:
                    for contrib in contributors.findall("contributor"):
                        name = contrib.get("name", "")
                        if name:
                            contrib_tokens = name.split()
                            tokens.extend(contrib_tokens)
                            
                            bbox = [50, 120, 300, 140]  # Placeholder bbox
                            bboxes.extend([bbox] * len(contrib_tokens))
                            labels.extend([self.label_mapping["byline"]] * len(contrib_tokens))
                
                # Process image captions
                images = article.find("images")
                if images is not None:
                    for img in images.findall("image"):
                        caption_elem = img.find("caption")
                        if caption_elem is not None and caption_elem.text:
                            caption_tokens = caption_elem.text.split()
                            tokens.extend(caption_tokens)
                            
                            bbox = [50, 300, 400, 320]  # Placeholder bbox
                            bboxes.extend([bbox] * len(caption_tokens))
                            labels.extend([self.label_mapping["caption"]] * len(caption_tokens))
                
                # Create training example if we have tokens
                if tokens:
                    example = TrainingExample(
                        document_id=f"{document_id}_{article_id}",
                        tokens=tokens[:self.config.max_sequence_length],  # Truncate if too long
                        bboxes=bboxes[:self.config.max_sequence_length],
                        labels=labels[:self.config.max_sequence_length],
                        brand=self.config.brand
                    )
                    examples.append(example)
            
            return examples
            
        except Exception as e:
            self.logger.error("Error parsing XML file", file=str(xml_file), error=str(e))
            return []
    
    def prepare_model_and_processor(self):
        """Initialize LayoutLM model and processor."""
        self.logger.info("Loading LayoutLM model", model=self.config.model_name)
        
        try:
            # Load processor
            self.processor = LayoutLMv3Processor.from_pretrained(
                self.config.model_name,
                apply_ocr=False
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Load model
            self.model = LayoutLMv3ForTokenClassification.from_pretrained(
                self.config.model_name,
                num_labels=len(self.label_mapping),
                id2label=self.id_to_label,
                label2id=self.label_mapping
            )
            
            self.logger.info("Model and processor loaded successfully")
            
        except Exception as e:
            self.logger.error("Error loading model", error=str(e))
            raise
    
    def create_datasets(self) -> Tuple[MagazineLayoutDataset, MagazineLayoutDataset]:
        """Create train and validation datasets."""
        if not self.training_examples:
            raise ValueError("No training examples loaded. Call load_training_data() first.")
        
        # Split data
        train_examples, val_examples = train_test_split(
            self.training_examples,
            test_size=self.config.validation_size,
            random_state=self.config.random_seed,
            stratify=[ex.brand for ex in self.training_examples]
        )
        
        # Create datasets
        train_dataset = MagazineLayoutDataset(train_examples, self.processor, self.config.max_sequence_length)
        val_dataset = MagazineLayoutDataset(val_examples, self.processor, self.config.max_sequence_length)
        
        self.logger.info("Datasets created",
                        train_size=len(train_dataset),
                        val_size=len(val_dataset))
        
        return train_dataset, val_dataset
    
    def train(self) -> Dict[str, Any]:
        """Execute training loop."""
        if not self.model or not self.processor:
            self.prepare_model_and_processor()
        
        # Create datasets
        train_dataset, val_dataset = self.create_datasets()
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            learning_rate=self.config.learning_rate,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            remove_unused_columns=False
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            ]
        )
        
        self.logger.info("Starting training", 
                        train_size=len(train_dataset),
                        val_size=len(val_dataset),
                        epochs=self.config.num_epochs)
        
        # Train the model
        training_result = trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.processor.save_pretrained(str(self.output_dir))
        
        # Save training configuration
        config_path = self.output_dir / "training_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Generate final evaluation
        final_metrics = trainer.evaluate()
        
        self.logger.info("Training completed",
                        train_loss=training_result.training_loss,
                        eval_metrics=final_metrics)
        
        return {
            "training_loss": training_result.training_loss,
            "eval_metrics": final_metrics,
            "model_path": str(self.output_dir),
            "config": self.config.to_dict()
        }
    
    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove padding and special tokens  
        true_predictions = []
        true_labels = []
        
        for prediction, label in zip(predictions, labels):
            # Remove padding (-100 labels)
            valid_indices = label != -100
            true_predictions.extend(prediction[valid_indices])
            true_labels.extend(label[valid_indices])
        
        accuracy = accuracy_score(true_labels, true_predictions)
        f1 = f1_score(true_labels, true_predictions, average='weighted')
        
        return {
            "accuracy": accuracy,
            "f1": f1
        }
    
    def evaluate_model(self, model_path: Optional[Path] = None) -> Dict[str, Any]:
        """Evaluate trained model."""
        if model_path:
            # Load saved model
            model = LayoutLMv3ForTokenClassification.from_pretrained(str(model_path))
            processor = LayoutLMv3Processor.from_pretrained(str(model_path))
        else:
            model = self.model
            processor = self.processor
        
        if not model or not processor:
            raise ValueError("No model available for evaluation")
        
        # Create evaluation dataset
        _, val_dataset = self.create_datasets()
        
        model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in DataLoader(val_dataset, batch_size=self.config.batch_size):
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                # Remove padding
                for pred, label in zip(predictions, batch["labels"]):
                    valid_mask = label != -100
                    all_predictions.extend(pred[valid_mask].tolist())
                    all_labels.extend(label[valid_mask].tolist())
        
        # Calculate detailed metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Classification report
        label_names = [self.id_to_label[i] for i in range(len(self.id_to_label))]
        report = classification_report(
            all_labels, all_predictions, 
            target_names=label_names, 
            output_dict=True
        )
        
        metrics = {
            "accuracy": accuracy,
            "f1_weighted": f1,
            "classification_report": report,
            "num_samples": len(all_labels)
        }
        
        self.logger.info("Model evaluation completed",
                        accuracy=accuracy,
                        f1_score=f1,
                        num_samples=len(all_labels))
        
        return metrics


def create_training_config(
    brand: str,
    **kwargs
) -> TrainingConfig:
    """Create training configuration with brand-specific defaults."""
    
    # Brand-specific hyperparameter optimization
    brand_defaults = {
        "economist": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 12,
            "warmup_steps": 500,
        },
        "time": {
            "learning_rate": 1.5e-5,
            "batch_size": 4,
            "num_epochs": 10,
            "warmup_steps": 400,
        },
        "newsweek": {
            "learning_rate": 2e-5,
            "batch_size": 4,
            "num_epochs": 10,
            "warmup_steps": 450,
        },
        "vogue": {
            "learning_rate": 2.5e-5,
            "batch_size": 4,
            "num_epochs": 15,  # More epochs for fashion content
            "warmup_steps": 600,
        }
    }
    
    # Get brand defaults
    defaults = brand_defaults.get(brand, brand_defaults["economist"])
    defaults.update(kwargs)
    
    return TrainingConfig(brand=brand, **defaults)


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python model_training.py <brand>")
        sys.exit(1)
    
    brand = sys.argv[1]
    
    # Create training configuration
    config = create_training_config(brand)
    
    # Initialize trainer
    trainer = LayoutLMTrainer(config)
    
    # Load data and train
    trainer.load_training_data()
    results = trainer.train()
    
    print(f"Training completed for {brand}")
    print(f"Model saved to: {results['model_path']}")
    print(f"Final accuracy: {results['eval_metrics'].get('eval_accuracy', 'N/A')}")