from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import yaml
import os
import structlog

logger = structlog.get_logger()
router = APIRouter()

@router.get("/brands")
async def list_brand_configs() -> Dict[str, Any]:
    """List available brand configurations"""
    configs_dir = "configs/brands"
    brand_configs = {}
    
    if not os.path.exists(configs_dir):
        return {"brands": {}}
    
    for filename in os.listdir(configs_dir):
        if filename.endswith('.yaml') or filename.endswith('.yml'):
            brand_name = filename.rsplit('.', 1)[0]
            try:
                with open(os.path.join(configs_dir, filename), 'r') as f:
                    config = yaml.safe_load(f)
                brand_configs[brand_name] = config
            except Exception as e:
                logger.error("Failed to load brand config", brand=brand_name, error=str(e))
    
    return {"brands": brand_configs}

@router.get("/brands/{brand_name}")
async def get_brand_config(brand_name: str) -> Dict[str, Any]:
    """Get configuration for a specific brand"""
    config_path = f"configs/brands/{brand_name}.yaml"
    
    if not os.path.exists(config_path):
        raise HTTPException(status_code=404, detail=f"Brand configuration '{brand_name}' not found")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error("Failed to load brand config", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load configuration")

@router.put("/brands/{brand_name}")
async def update_brand_config(brand_name: str, config: Dict[str, Any]) -> Dict[str, str]:
    """Update configuration for a specific brand"""
    config_path = f"configs/brands/{brand_name}.yaml"
    
    # Ensure configs directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info("Updated brand config", brand=brand_name)
        return {"message": f"Configuration for '{brand_name}' updated successfully"}
    except Exception as e:
        logger.error("Failed to update brand config", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to update configuration")

@router.get("/processing")
async def get_processing_config() -> Dict[str, Any]:
    """Get global processing configuration"""
    config_path = "configs/processing.yaml"
    
    if not os.path.exists(config_path):
        # Return default configuration
        return {
            "ocr": {
                "engine": "tesseract",
                "confidence_threshold": 0.7,
                "preprocessing": {
                    "deskew": True,
                    "denoise": True,
                    "enhance_contrast": True
                }
            },
            "layout_analysis": {
                "model": "layoutlm-v3",
                "confidence_threshold": 0.8,
                "block_types": ["title", "body", "caption", "pullquote", "header", "footer", "ad"]
            },
            "article_reconstruction": {
                "min_block_confidence": 0.7,
                "spatial_threshold_pixels": 50,
                "jump_detection_patterns": ["continued on", "from page"]
            },
            "quality_thresholds": {
                "overall_accuracy": 0.999,
                "title_accuracy": 0.995,
                "body_accuracy": 0.999,
                "contributor_accuracy": 0.99
            }
        }
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error("Failed to load processing config", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to load processing configuration")