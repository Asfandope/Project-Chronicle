"""
FastAPI endpoints for parameter management.

This module provides REST API endpoints for managing parameters,
versions, overrides, and performing rollback operations.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Union
from fastapi import FastAPI, HTTPException, Depends, status, Query, Path
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import desc, and_
import uuid

from .models import (
    Parameter, ParameterVersion, ParameterOverride, ParameterChangeRequest,
    ParameterSnapshot, ParameterAuditLog, ParameterTemplate,
    ParameterType, ParameterScope, ParameterStatus,
    create_parameter_tables
)
from .service import (
    ParameterService, ParameterValidator,
    ParameterUpdateRequest, ParameterOverrideRequest, RollbackRequest,
    ParameterValidationError, ParameterNotFoundError, ParameterVersionError
)


# Pydantic models for API
class ParameterCreateRequest(BaseModel):
    """Request to create a new parameter."""
    key: str = Field(..., pattern=r'^[a-zA-Z][a-zA-Z0-9_\.]*$', max_length=255)
    name: str = Field(..., max_length=255)
    description: Optional[str] = None
    parameter_type: ParameterType
    category: str = Field(..., max_length=100)
    subcategory: Optional[str] = Field(None, max_length=100)
    data_type: str = Field(..., pattern=r'^(string|integer|float|boolean|json|array|object)$')
    default_value: Any
    validation_rules: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = []
    impact_level: str = Field('medium', pattern=r'^(low|medium|high|critical)$')
    requires_restart: bool = False


class ParameterUpdateRequestAPI(BaseModel):
    """API request to update a parameter value."""
    new_value: Any
    change_reason: str = Field(..., min_length=1, max_length=1000)
    version_number: Optional[str] = Field(None, max_length=50)
    requires_approval: bool = True
    auto_activate: bool = False


class ParameterOverrideRequestAPI(BaseModel):
    """API request to create a parameter override."""
    override_value: Any
    scope: ParameterScope
    scope_identifier: Optional[str] = Field(None, max_length=255)
    conditions: Optional[Dict[str, Any]] = None
    priority: int = Field(100, ge=1, le=1000)
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    change_reason: str = Field("", max_length=1000)
    
    @validator('valid_until')
    def valid_until_after_valid_from(cls, v, values):
        if v and values.get('valid_from') and v <= values['valid_from']:
            raise ValueError('valid_until must be after valid_from')
        return v


class RollbackRequestAPI(BaseModel):
    """API request to rollback parameters."""
    target_version_id: Optional[str] = None
    target_snapshot_id: Optional[str] = None
    parameter_keys: Optional[List[str]] = None
    rollback_reason: str = Field(..., min_length=1, max_length=1000)
    
    @validator('target_version_id', 'target_snapshot_id')
    def at_least_one_target(cls, v, values):
        if not v and not values.get('target_version_id') and not values.get('target_snapshot_id'):
            raise ValueError('Either target_version_id or target_snapshot_id must be provided')
        return v


class SnapshotCreateRequest(BaseModel):
    """Request to create a parameter snapshot."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    snapshot_type: str = Field('manual', pattern=r'^(manual|scheduled|pre_deployment|pre_change)$')


class ParameterResponse(BaseModel):
    """Response model for parameter data."""
    id: str
    key: str
    name: str
    description: Optional[str]
    parameter_type: str
    category: str
    subcategory: Optional[str]
    data_type: str
    default_value: Any
    current_value: Any
    is_active: bool
    impact_level: str
    requires_restart: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ParameterVersionResponse(BaseModel):
    """Response model for parameter version data."""
    id: str
    parameter_id: str
    version_number: str
    value: Any
    status: str
    created_by: str
    created_at: datetime
    change_reason: Optional[str]
    activated_at: Optional[datetime]
    activated_by: Optional[str]

    class Config:
        from_attributes = True


class ParameterOverrideResponse(BaseModel):
    """Response model for parameter override data."""
    id: str
    parameter_id: str
    scope: str
    scope_identifier: Optional[str]
    override_value: Any
    priority: int
    valid_from: datetime
    valid_until: Optional[datetime]
    is_active: bool
    created_by: str
    created_at: datetime

    class Config:
        from_attributes = True


class SnapshotResponse(BaseModel):
    """Response model for parameter snapshot data."""
    id: str
    name: str
    description: Optional[str]
    snapshot_type: str
    created_by: str
    created_at: datetime
    parameter_count: int
    override_count: int
    restore_count: int
    last_restored_at: Optional[datetime]

    class Config:
        from_attributes = True


class PaginatedResponse(BaseModel):
    """Generic paginated response."""
    items: List[Any]
    total_count: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


# Parameter management API
def create_parameter_management_api(
    db_session_factory,
    prefix: str = "/api/v1/parameters"
) -> FastAPI:
    """Create the parameter management FastAPI application."""
    
    app = FastAPI(
        title="Parameter Management API",
        description="Centralized parameter management with versioning and rollback",
        version="1.0.0"
    )
    
    parameter_service = ParameterService()
    
    # Dependency to get database session
    def get_db() -> Session:
        db = db_session_factory()
        try:
            yield db
        finally:
            db.close()
    
    # Dependency to get current user (placeholder)
    def get_current_user() -> str:
        # In a real implementation, this would extract user from JWT token
        return "system_user"
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.utcnow()}
    
    # Parameter CRUD endpoints
    @app.post(f"{prefix}", response_model=ParameterResponse)
    async def create_parameter(
        request: ParameterCreateRequest,
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
    ):
        """Create a new parameter."""
        try:
            parameter = parameter_service.create_parameter(
                session=db,
                key=request.key,
                name=request.name,
                description=request.description,
                parameter_type=request.parameter_type,
                category=request.category,
                data_type=request.data_type,
                default_value=request.default_value,
                created_by=current_user,
                validation_rules=request.validation_rules,
                subcategory=request.subcategory,
                tags=request.tags,
                impact_level=request.impact_level,
                requires_restart=request.requires_restart
            )
            
            current_value = parameter_service.get_parameter_value(db, parameter.key)
            
            return ParameterResponse(
                id=str(parameter.id),
                key=parameter.key,
                name=parameter.name,
                description=parameter.description,
                parameter_type=parameter.parameter_type.value,
                category=parameter.category,
                subcategory=parameter.subcategory,
                data_type=parameter.data_type,
                default_value=parameter.default_value,
                current_value=current_value,
                is_active=parameter.is_active,
                impact_level=parameter.impact_level,
                requires_restart=parameter.requires_restart,
                created_at=parameter.created_at,
                updated_at=parameter.updated_at
            )
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except ParameterValidationError as e:
            raise HTTPException(status_code=400, detail=f"Validation error: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    @app.get(f"{prefix}", response_model=PaginatedResponse)
    async def list_parameters(
        page: int = Query(1, ge=1),
        page_size: int = Query(20, ge=1, le=100),
        category: Optional[str] = Query(None),
        parameter_type: Optional[ParameterType] = Query(None),
        search: Optional[str] = Query(None),
        db: Session = Depends(get_db)
    ):
        """List parameters with filtering and pagination."""
        
        query = db.query(Parameter).filter(Parameter.is_active == True)
        
        if category:
            query = query.filter(Parameter.category == category)
        
        if parameter_type:
            query = query.filter(Parameter.parameter_type == parameter_type)
        
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                (Parameter.key.ilike(search_term)) |
                (Parameter.name.ilike(search_term)) |
                (Parameter.description.ilike(search_term))
            )
        
        total_count = query.count()
        
        # Apply pagination
        offset = (page - 1) * page_size
        parameters = query.order_by(Parameter.category, Parameter.key).offset(offset).limit(page_size).all()
        
        # Get current values for each parameter
        parameter_responses = []
        for param in parameters:
            try:
                current_value = parameter_service.get_parameter_value(db, param.key)
                parameter_responses.append(ParameterResponse(
                    id=str(param.id),
                    key=param.key,
                    name=param.name,
                    description=param.description,
                    parameter_type=param.parameter_type.value,
                    category=param.category,
                    subcategory=param.subcategory,
                    data_type=param.data_type,
                    default_value=param.default_value,
                    current_value=current_value,
                    is_active=param.is_active,
                    impact_level=param.impact_level,
                    requires_restart=param.requires_restart,
                    created_at=param.created_at,
                    updated_at=param.updated_at
                ))
            except Exception as e:
                # Skip parameters with errors
                continue
        
        total_pages = (total_count + page_size - 1) // page_size
        
        return PaginatedResponse(
            items=parameter_responses,
            total_count=total_count,
            page=page,
            page_size=page_size,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_previous=page > 1
        )
    
    @app.get(f"{prefix}/{{parameter_key}}", response_model=ParameterResponse)
    async def get_parameter(
        parameter_key: str,
        brand: Optional[str] = Query(None),
        db: Session = Depends(get_db)
    ):
        """Get a specific parameter with its current value."""
        
        parameter = db.query(Parameter).filter(
            Parameter.key == parameter_key,
            Parameter.is_active == True
        ).first()
        
        if not parameter:
            raise HTTPException(status_code=404, detail=f"Parameter '{parameter_key}' not found")
        
        try:
            current_value = parameter_service.get_parameter_value(db, parameter_key, brand)
            
            return ParameterResponse(
                id=str(parameter.id),
                key=parameter.key,
                name=parameter.name,
                description=parameter.description,
                parameter_type=parameter.parameter_type.value,
                category=parameter.category,
                subcategory=parameter.subcategory,
                data_type=parameter.data_type,
                default_value=parameter.default_value,
                current_value=current_value,
                is_active=parameter.is_active,
                impact_level=parameter.impact_level,
                requires_restart=parameter.requires_restart,
                created_at=parameter.created_at,
                updated_at=parameter.updated_at
            )
            
        except ParameterNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    @app.put(f"{prefix}/{{parameter_key}}/value", response_model=ParameterVersionResponse)
    async def update_parameter_value(
        parameter_key: str,
        request: ParameterUpdateRequestAPI,
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
    ):
        """Update a parameter value."""
        
        try:
            update_request = ParameterUpdateRequest(
                parameter_key=parameter_key,
                new_value=request.new_value,
                change_reason=request.change_reason,
                created_by=current_user,
                version_number=request.version_number,
                requires_approval=request.requires_approval,
                auto_activate=request.auto_activate
            )
            
            version = parameter_service.update_parameter_value(db, update_request)
            
            if not version:
                raise HTTPException(
                    status_code=202,
                    detail="Change request created - approval required"
                )
            
            return ParameterVersionResponse(
                id=str(version.id),
                parameter_id=str(version.parameter_id),
                version_number=version.version_number,
                value=version.value,
                status=version.status.value,
                created_by=version.created_by,
                created_at=version.created_at,
                change_reason=version.change_reason,
                activated_at=version.activated_at,
                activated_by=version.activated_by
            )
            
        except ParameterNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ParameterValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    # Parameter version endpoints
    @app.get(f"{prefix}/{{parameter_key}}/versions", response_model=List[ParameterVersionResponse])
    async def get_parameter_versions(
        parameter_key: str,
        limit: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db)
    ):
        """Get version history for a parameter."""
        
        parameter = db.query(Parameter).filter(Parameter.key == parameter_key).first()
        if not parameter:
            raise HTTPException(status_code=404, detail=f"Parameter '{parameter_key}' not found")
        
        versions = (db.query(ParameterVersion)
                   .filter(ParameterVersion.parameter_id == parameter.id)
                   .order_by(desc(ParameterVersion.created_at))
                   .limit(limit)
                   .all())
        
        return [
            ParameterVersionResponse(
                id=str(version.id),
                parameter_id=str(version.parameter_id),
                version_number=version.version_number,
                value=version.value,
                status=version.status.value,
                created_by=version.created_by,
                created_at=version.created_at,
                change_reason=version.change_reason,
                activated_at=version.activated_at,
                activated_by=version.activated_by
            )
            for version in versions
        ]
    
    # Parameter override endpoints
    @app.post(f"{prefix}/{{parameter_key}}/overrides", response_model=ParameterOverrideResponse)
    async def create_parameter_override(
        parameter_key: str,
        request: ParameterOverrideRequestAPI,
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
    ):
        """Create a parameter override."""
        
        try:
            override_request = ParameterOverrideRequest(
                parameter_key=parameter_key,
                override_value=request.override_value,
                scope=request.scope,
                scope_identifier=request.scope_identifier,
                conditions=request.conditions,
                priority=request.priority,
                valid_from=request.valid_from,
                valid_until=request.valid_until,
                change_reason=request.change_reason,
                created_by=current_user
            )
            
            override = parameter_service.create_parameter_override(db, override_request)
            
            return ParameterOverrideResponse(
                id=str(override.id),
                parameter_id=str(override.parameter_id),
                scope=override.scope.value,
                scope_identifier=override.scope_identifier,
                override_value=override.override_value,
                priority=override.priority,
                valid_from=override.valid_from,
                valid_until=override.valid_until,
                is_active=override.is_active,
                created_by=override.created_by,
                created_at=override.created_at
            )
            
        except ParameterNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except ParameterValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    @app.get(f"{prefix}/{{parameter_key}}/overrides", response_model=List[ParameterOverrideResponse])
    async def get_parameter_overrides(
        parameter_key: str,
        scope: Optional[ParameterScope] = Query(None),
        scope_identifier: Optional[str] = Query(None),
        active_only: bool = Query(True),
        db: Session = Depends(get_db)
    ):
        """Get overrides for a parameter."""
        
        parameter = db.query(Parameter).filter(Parameter.key == parameter_key).first()
        if not parameter:
            raise HTTPException(status_code=404, detail=f"Parameter '{parameter_key}' not found")
        
        query = db.query(ParameterOverride).filter(ParameterOverride.parameter_id == parameter.id)
        
        if active_only:
            current_time = datetime.now(timezone.utc)
            query = query.filter(
                ParameterOverride.is_active == True,
                ParameterOverride.valid_from <= current_time,
                (ParameterOverride.valid_until.is_(None)) |
                (ParameterOverride.valid_until > current_time)
            )
        
        if scope:
            query = query.filter(ParameterOverride.scope == scope)
        
        if scope_identifier:
            query = query.filter(ParameterOverride.scope_identifier == scope_identifier)
        
        overrides = query.order_by(desc(ParameterOverride.priority), desc(ParameterOverride.created_at)).all()
        
        return [
            ParameterOverrideResponse(
                id=str(override.id),
                parameter_id=str(override.parameter_id),
                scope=override.scope.value,
                scope_identifier=override.scope_identifier,
                override_value=override.override_value,
                priority=override.priority,
                valid_from=override.valid_from,
                valid_until=override.valid_until,
                is_active=override.is_active,
                created_by=override.created_by,
                created_at=override.created_at
            )
            for override in overrides
        ]
    
    # Configuration endpoints
    @app.get(f"{prefix}/config/{{category}}")
    async def get_category_configuration(
        category: str,
        brand: Optional[str] = Query(None),
        include_metadata: bool = Query(False),
        db: Session = Depends(get_db)
    ):
        """Get complete configuration for a category."""
        
        try:
            config = parameter_service.get_parameter_configuration(
                session=db,
                category=category,
                brand=brand,
                include_overrides=include_metadata
            )
            
            return {
                "category": category,
                "brand": brand,
                "parameters": config,
                "timestamp": datetime.utcnow()
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    # Snapshot endpoints
    @app.post("/snapshots", response_model=SnapshotResponse)
    async def create_snapshot(
        request: SnapshotCreateRequest,
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
    ):
        """Create a parameter snapshot."""
        
        try:
            snapshot = parameter_service.create_snapshot(
                session=db,
                name=request.name,
                description=request.description,
                created_by=current_user,
                snapshot_type=request.snapshot_type
            )
            
            return SnapshotResponse(
                id=str(snapshot.id),
                name=snapshot.name,
                description=snapshot.description,
                snapshot_type=snapshot.snapshot_type,
                created_by=snapshot.created_by,
                created_at=snapshot.created_at,
                parameter_count=len(snapshot.parameters_data),
                override_count=len(snapshot.overrides_data or {}),
                restore_count=snapshot.restore_count,
                last_restored_at=snapshot.last_restored_at
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    @app.get("/snapshots", response_model=List[SnapshotResponse])
    async def list_snapshots(
        limit: int = Query(20, ge=1, le=100),
        db: Session = Depends(get_db)
    ):
        """List parameter snapshots."""
        
        snapshots = (db.query(ParameterSnapshot)
                    .order_by(desc(ParameterSnapshot.created_at))
                    .limit(limit)
                    .all())
        
        return [
            SnapshotResponse(
                id=str(snapshot.id),
                name=snapshot.name,
                description=snapshot.description,
                snapshot_type=snapshot.snapshot_type,
                created_by=snapshot.created_by,
                created_at=snapshot.created_at,
                parameter_count=len(snapshot.parameters_data),
                override_count=len(snapshot.overrides_data or {}),
                restore_count=snapshot.restore_count,
                last_restored_at=snapshot.last_restored_at
            )
            for snapshot in snapshots
        ]
    
    # Rollback endpoints
    @app.post("/rollback")
    async def rollback_parameters(
        request: RollbackRequestAPI,
        db: Session = Depends(get_db),
        current_user: str = Depends(get_current_user)
    ):
        """Rollback parameters to previous version or snapshot."""
        
        try:
            rollback_request = RollbackRequest(
                target_version_id=request.target_version_id,
                target_snapshot_id=request.target_snapshot_id,
                parameter_keys=request.parameter_keys,
                rollback_reason=request.rollback_reason,
                created_by=current_user
            )
            
            rolled_back_versions = parameter_service.rollback_parameter(db, rollback_request)
            
            return {
                "rollback_successful": True,
                "versions_rolled_back": len(rolled_back_versions),
                "rollback_timestamp": datetime.utcnow(),
                "rollback_reason": request.rollback_reason
            }
            
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
    
    return app


# Convenience function to mount on existing FastAPI app
def mount_parameter_management(
    app: FastAPI,
    db_session_factory,
    prefix: str = "/api/v1/parameters"
):
    """Mount parameter management API on existing FastAPI app."""
    
    param_app = create_parameter_management_api(db_session_factory, prefix)
    app.mount("/parameters", param_app)