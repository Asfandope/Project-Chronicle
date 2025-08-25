from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, pipeline
import structlog
from pathlib import Path

from model_service.core.config import get_settings

logger = structlog.get_logger()

class ModelManager:
    """Manages loading and caching of ML models"""
    
    def __init__(self):
        self.settings = get_settings()
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.logger = logger.bind(component="model_manager")
        
        # Ensure model cache directory exists
        Path(self.settings.model_cache_dir).mkdir(parents=True, exist_ok=True)
    
    async def load_models(self):
        """Load all required models"""
        self.logger.info("Loading models", device=self.settings.device)
        
        try:
            # Load LayoutLM for layout analysis
            await self._load_layout_model()
            
            # Load NER model for contributor extraction
            await self._load_ner_model()
            
            self.logger.info("All models loaded successfully")
            
        except Exception as e:
            self.logger.error("Failed to load models", error=str(e))
            raise
    
    async def _load_layout_model(self):
        """Load LayoutLM model for document layout analysis"""
        self.logger.info("Loading layout model", model=self.settings.layout_model_name)
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.settings.layout_model_name,
                cache_dir=self.settings.model_cache_dir
            )
            self.tokenizers['layout'] = tokenizer
            
            # Load model
            model = AutoModel.from_pretrained(
                self.settings.layout_model_name,
                cache_dir=self.settings.model_cache_dir
            )
            
            # Move to appropriate device
            if self.settings.device == "cuda" and torch.cuda.is_available():
                model = model.cuda()
            
            model.eval()  # Set to evaluation mode
            self.models['layout'] = model
            
            self.logger.info("Layout model loaded successfully")
            
        except Exception as e:
            self.logger.error("Failed to load layout model", error=str(e))
            raise
    
    async def _load_ner_model(self):
        """Load NER model for contributor name extraction"""
        self.logger.info("Loading NER model", model=self.settings.ner_model_name)
        
        try:
            # Create NER pipeline
            ner_pipeline = pipeline(
                "ner",
                model=self.settings.ner_model_name,
                tokenizer=self.settings.ner_model_name,
                device=0 if self.settings.device == "cuda" and torch.cuda.is_available() else -1,
                model_kwargs={"cache_dir": self.settings.model_cache_dir}
            )
            
            self.models['ner'] = ner_pipeline
            
            self.logger.info("NER model loaded successfully")
            
        except Exception as e:
            self.logger.error("Failed to load NER model", error=str(e))
            raise
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get a loaded model by name"""
        return self.models.get(model_name)
    
    def get_tokenizer(self, tokenizer_name: str) -> Optional[Any]:
        """Get a loaded tokenizer by name"""
        return self.tokenizers.get(tokenizer_name)
    
    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded"""
        return model_name in self.models
    
    async def unload_models(self):
        """Unload all models to free memory"""
        self.logger.info("Unloading models")
        
        # Clear CUDA cache if using GPU
        if self.settings.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.models.clear()
        self.tokenizers.clear()
        
        self.logger.info("All models unloaded")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "device": self.settings.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "model_cache_dir": self.settings.model_cache_dir
        }