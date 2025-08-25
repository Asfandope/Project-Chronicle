from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import structlog

from evaluation.core.database import get_db
from evaluation.models.gold_set_manager import GoldSetManager

logger = structlog.get_logger()
router = APIRouter()

@router.get("/brands")
async def list_brand_gold_sets(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """List available gold sets for all brands"""
    try:
        manager = GoldSetManager(db)
        gold_sets = await manager.list_all_gold_sets()
        
        return {
            "gold_sets": gold_sets["sets"],
            "total_brands": gold_sets["total_brands"],
            "total_issues": gold_sets["total_issues"]
        }
        
    except Exception as e:
        logger.error("Failed to list gold sets", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list gold sets: {str(e)}")

@router.get("/brand/{brand_name}")
async def get_brand_gold_set(
    brand_name: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Get gold set information for a specific brand"""
    try:
        manager = GoldSetManager(db)
        gold_set = await manager.get_brand_gold_set(brand_name)
        
        return {
            "brand": brand_name,
            "gold_set": gold_set["set_info"],
            "statistics": gold_set["stats"],
            "coverage": gold_set["coverage"]
        }
        
    except Exception as e:
        logger.error("Failed to get brand gold set", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get gold set: {str(e)}")

@router.post("/brand/{brand_name}/upload")
async def upload_gold_set_issue(
    brand_name: str,
    file: UploadFile = File(...),
    metadata: Dict[str, Any] = None,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Upload a new gold standard issue for a brand"""
    try:
        manager = GoldSetManager(db)
        result = await manager.upload_gold_issue(
            brand_name=brand_name,
            file=file,
            metadata=metadata or {}
        )
        
        return {
            "uploaded": result["success"],
            "issue_id": result["issue_id"],
            "validation_results": result["validation"],
            "coverage_impact": result["coverage_impact"]
        }
        
    except Exception as e:
        logger.error("Gold set upload failed", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.post("/brand/{brand_name}/generate-synthetic")
async def generate_synthetic_variants(
    brand_name: str,
    generation_config: Dict[str, Any],
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Generate synthetic variants of gold set issues"""
    try:
        manager = GoldSetManager(db)
        result = await manager.generate_synthetic_variants(
            brand_name=brand_name,
            config=generation_config
        )
        
        return {
            "generated_variants": result["variants_count"],
            "base_issues_used": result["base_count"],
            "augmentation_methods": result["methods_used"],
            "quality_scores": result["quality_scores"]
        }
        
    except Exception as e:
        logger.error("Synthetic generation failed", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Synthetic generation failed: {str(e)}")

@router.get("/brand/{brand_name}/validate")
async def validate_gold_set_coverage(
    brand_name: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Validate gold set coverage for a brand"""
    try:
        manager = GoldSetManager(db)
        validation = await manager.validate_coverage(brand_name)
        
        return {
            "brand": brand_name,
            "coverage_valid": validation["valid"],
            "coverage_gaps": validation["gaps"],
            "recommendations": validation["recommendations"],
            "minimum_requirements": validation["requirements"]
        }
        
    except Exception as e:
        logger.error("Gold set validation failed", brand=brand_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@router.delete("/brand/{brand_name}/issue/{issue_id}")
async def delete_gold_set_issue(
    brand_name: str,
    issue_id: str,
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """Delete a gold set issue"""
    try:
        manager = GoldSetManager(db)
        result = await manager.delete_gold_issue(brand_name, issue_id)
        
        return {
            "deleted": result["success"],
            "impact_on_coverage": result["coverage_impact"],
            "recommendations": result["recommendations"]
        }
        
    except Exception as e:
        logger.error("Gold set deletion failed", 
                    brand=brand_name, 
                    issue_id=issue_id, 
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@router.get("/requirements")
async def get_gold_set_requirements() -> Dict[str, Any]:
    """Get gold set requirements and guidelines"""
    from evaluation.core.config import get_settings
    settings = get_settings()
    
    return {
        "minimum_issues_per_brand": settings.min_gold_issues_per_brand,
        "required_coverage": {
            "standard_layouts": 0.7,
            "edge_cases": 0.2, 
            "extreme_cases": 0.1
        },
        "synthetic_augmentation_factor": settings.synthetic_augmentation_factor,
        "holdout_percentage": settings.holdout_percentage,
        "supported_formats": ["PDF"],
        "annotation_requirements": [
            "Article boundaries",
            "Title text",
            "Body text", 
            "Contributors with roles",
            "Image captions",
            "Layout classifications"
        ]
    }