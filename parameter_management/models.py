"""
Parameter management database models.

This module defines the database schema for centralized parameter management
with versioning, brand-specific overrides, and rollback capabilities.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, JSON, Boolean, Text,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Enum
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID, JSONB
import uuid
import enum

Base = declarative_base()


class ParameterType(enum.Enum):
    """Types of parameters that can be managed."""
    THRESHOLD = "threshold"           # Numeric thresholds (accuracy, drift, etc.)
    PROMPT = "prompt"                # Text prompts and templates
    RULE = "rule"                    # Business rules and conditions
    MODEL_CONFIG = "model_config"    # Model configuration parameters
    FEATURE_FLAG = "feature_flag"    # Feature toggles
    PROCESSING_CONFIG = "processing_config"  # Processing parameters
    UI_CONFIG = "ui_config"          # User interface configuration


class ParameterScope(enum.Enum):
    """Scope of parameter application."""
    GLOBAL = "global"               # Applied to all brands/contexts
    BRAND_SPECIFIC = "brand_specific"  # Applied to specific brand
    SYSTEM = "system"               # System-level configuration
    USER = "user"                   # User-specific overrides


class ParameterStatus(enum.Enum):
    """Status of parameter versions."""
    DRAFT = "draft"                 # Being developed/tested
    ACTIVE = "active"               # Currently in use
    DEPRECATED = "deprecated"       # No longer recommended
    ARCHIVED = "archived"           # Removed from active use


class Parameter(Base):
    """Master parameter definitions."""
    
    __tablename__ = "parameters"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Parameter identification
    key = Column(String(255), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Parameter classification
    parameter_type = Column(Enum(ParameterType), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)  # e.g., "accuracy", "extraction", "ui"
    subcategory = Column(String(100), index=True)  # e.g., "drift_detection", "title_extraction"
    
    # Validation and constraints
    data_type = Column(String(50), nullable=False)  # string, integer, float, boolean, json, array
    validation_rules = Column(JSONB)  # JSON schema for validation
    default_value = Column(JSONB)  # Default value as JSON
    
    # Metadata
    tags = Column(JSONB)  # Array of tags for organization
    documentation_url = Column(String(500))
    impact_level = Column(String(20), default='medium')  # low, medium, high, critical
    requires_restart = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    is_system_managed = Column(Boolean, default=False)  # Cannot be modified via UI
    
    # Relationships
    versions = relationship("ParameterVersion", back_populates="parameter", cascade="all, delete-orphan")
    overrides = relationship("ParameterOverride", back_populates="parameter", cascade="all, delete-orphan")
    change_requests = relationship("ParameterChangeRequest", back_populates="parameter", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_parameters_type_category', 'parameter_type', 'category'),
        Index('idx_parameters_active', 'is_active'),
    )


class ParameterVersion(Base):
    """Version history for parameter values."""
    
    __tablename__ = "parameter_versions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parameter_id = Column(UUID(as_uuid=True), ForeignKey("parameters.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Version information
    version_number = Column(String(50), nullable=False)  # e.g., "1.2.3", "2024.01.15-1"
    status = Column(Enum(ParameterStatus), nullable=False, default=ParameterStatus.DRAFT, index=True)
    
    # Value and metadata
    value = Column(JSONB, nullable=False)  # The actual parameter value
    value_hash = Column(String(64), index=True)  # SHA-256 hash of value for deduplication
    
    # Change tracking
    created_by = Column(String(100), nullable=False)  # User ID or system identifier
    change_reason = Column(Text)  # Why this change was made
    change_type = Column(String(50), default='manual')  # manual, automated, migration, rollback
    
    # Activation tracking
    activated_at = Column(DateTime(timezone=True))
    activated_by = Column(String(100))
    deactivated_at = Column(DateTime(timezone=True))
    deactivated_by = Column(String(100))
    
    # Validation and testing
    validation_status = Column(String(20), default='pending')  # pending, passed, failed
    validation_errors = Column(JSONB)
    test_results = Column(JSONB)  # Results from automated testing
    
    # Rollback information
    rolled_back_from_version = Column(UUID(as_uuid=True), ForeignKey("parameter_versions.id"))
    rolled_back_at = Column(DateTime(timezone=True))
    rolled_back_by = Column(String(100))
    rollback_reason = Column(Text)
    
    # Relationships
    parameter = relationship("Parameter", back_populates="versions")
    rollback_source = relationship("ParameterVersion", remote_side=[id])
    
    __table_args__ = (
        Index('idx_parameter_versions_param_id', 'parameter_id'),
        Index('idx_parameter_versions_status', 'status'),
        Index('idx_parameter_versions_created_at', 'created_at'),
        Index('idx_parameter_versions_value_hash', 'value_hash'),
        UniqueConstraint('parameter_id', 'version_number', name='uq_parameter_version_number'),
    )


class ParameterOverride(Base):
    """Brand-specific and context-specific parameter overrides."""
    
    __tablename__ = "parameter_overrides"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parameter_id = Column(UUID(as_uuid=True), ForeignKey("parameters.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Override scope
    scope = Column(Enum(ParameterScope), nullable=False, index=True)
    scope_identifier = Column(String(255), index=True)  # Brand name, user ID, etc.
    
    # Override conditions
    conditions = Column(JSONB)  # JSON conditions for when this override applies
    priority = Column(Integer, default=100)  # Higher number = higher priority
    
    # Override value
    override_value = Column(JSONB, nullable=False)
    value_hash = Column(String(64), index=True)
    
    # Validity period
    valid_from = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    valid_until = Column(DateTime(timezone=True))  # NULL means indefinite
    
    # Change tracking
    created_by = Column(String(100), nullable=False)
    updated_by = Column(String(100))
    change_reason = Column(Text)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False, index=True)
    
    # Relationships
    parameter = relationship("Parameter", back_populates="overrides")
    
    __table_args__ = (
        Index('idx_parameter_overrides_param_scope', 'parameter_id', 'scope', 'scope_identifier'),
        Index('idx_parameter_overrides_priority', 'priority'),
        Index('idx_parameter_overrides_validity', 'valid_from', 'valid_until'),
        Index('idx_parameter_overrides_active', 'is_active'),
    )


class ParameterChangeRequest(Base):
    """Change requests for parameter modifications with approval workflow."""
    
    __tablename__ = "parameter_change_requests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    parameter_id = Column(UUID(as_uuid=True), ForeignKey("parameters.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Request information
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=False)
    justification = Column(Text)  # Why this change is needed
    
    # Proposed changes
    current_value = Column(JSONB)  # Current value being changed
    proposed_value = Column(JSONB, nullable=False)  # Proposed new value
    change_type = Column(String(50), nullable=False)  # create, update, delete, override
    
    # Scope of change
    affects_scope = Column(String(50), default='global')  # global, brand_specific, etc.
    affected_brands = Column(JSONB)  # Array of brand names affected
    estimated_impact = Column(Text)  # Description of expected impact
    
    # Approval workflow
    status = Column(String(20), default='pending', nullable=False, index=True)  # pending, approved, rejected, implemented
    requested_by = Column(String(100), nullable=False)
    reviewed_by = Column(String(100))
    approved_by = Column(String(100))
    
    # Timestamps
    submitted_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reviewed_at = Column(DateTime(timezone=True))
    approved_at = Column(DateTime(timezone=True))
    implemented_at = Column(DateTime(timezone=True))
    
    # Review information
    review_comments = Column(Text)
    approval_comments = Column(Text)
    rejection_reason = Column(Text)
    
    # Implementation tracking
    implemented_version_id = Column(UUID(as_uuid=True), ForeignKey("parameter_versions.id"))
    implementation_notes = Column(Text)
    
    # Risk assessment
    risk_level = Column(String(20), default='medium')  # low, medium, high, critical
    requires_testing = Column(Boolean, default=True)
    requires_approval = Column(Boolean, default=True)
    can_auto_rollback = Column(Boolean, default=False)
    
    # Relationships
    parameter = relationship("Parameter", back_populates="change_requests")
    implemented_version = relationship("ParameterVersion")
    
    __table_args__ = (
        Index('idx_parameter_change_requests_status', 'status'),
        Index('idx_parameter_change_requests_requested_by', 'requested_by'),
        Index('idx_parameter_change_requests_created_at', 'created_at'),
    )


class ParameterAuditLog(Base):
    """Comprehensive audit log for all parameter operations."""
    
    __tablename__ = "parameter_audit_log"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # What was changed
    parameter_id = Column(UUID(as_uuid=True), ForeignKey("parameters.id"))
    parameter_key = Column(String(255), index=True)  # Denormalized for performance
    
    # Action information
    action = Column(String(50), nullable=False, index=True)  # create, update, delete, activate, deactivate, override, rollback
    object_type = Column(String(50), nullable=False)  # parameter, version, override, change_request
    object_id = Column(UUID(as_uuid=True))  # ID of the object being modified
    
    # Change details
    old_value = Column(JSONB)
    new_value = Column(JSONB)
    changes = Column(JSONB)  # Detailed diff of changes
    
    # Context information
    user_id = Column(String(100), nullable=False, index=True)
    user_agent = Column(String(500))
    ip_address = Column(String(45))
    session_id = Column(String(255))
    
    # Source and reason
    source = Column(String(50), default='manual')  # manual, api, automation, migration
    reason = Column(Text)
    
    # Request context
    request_id = Column(String(255))  # For tracing across systems
    correlation_id = Column(String(255))  # For grouping related changes
    
    # Additional metadata
    additional_metadata = Column(JSONB)
    
    __table_args__ = (
        Index('idx_parameter_audit_log_param_id', 'parameter_id'),
        Index('idx_parameter_audit_log_action', 'action'),
        Index('idx_parameter_audit_log_user_id', 'user_id'),
        Index('idx_parameter_audit_log_created_at', 'created_at'),
    )


class ParameterTemplate(Base):
    """Templates for common parameter patterns and configurations."""
    
    __tablename__ = "parameter_templates"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Template identification
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    category = Column(String(100), nullable=False, index=True)
    
    # Template definition
    template_definition = Column(JSONB, nullable=False)  # JSON schema defining the template
    default_values = Column(JSONB)  # Default values for template parameters
    validation_rules = Column(JSONB)  # Additional validation rules
    
    # Usage metadata
    usage_count = Column(Integer, default=0)
    last_used_at = Column(DateTime(timezone=True))
    
    # Template metadata
    tags = Column(JSONB)
    author = Column(String(100))
    version = Column(String(50), default='1.0')
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_system_template = Column(Boolean, default=False)
    
    __table_args__ = (
        Index('idx_parameter_templates_category', 'category'),
        Index('idx_parameter_templates_active', 'is_active'),
    )


class ParameterSnapshot(Base):
    """Point-in-time snapshots of all parameter values for rollback purposes."""
    
    __tablename__ = "parameter_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    
    # Snapshot metadata
    name = Column(String(255), nullable=False)
    description = Column(Text)
    snapshot_type = Column(String(50), default='manual')  # manual, scheduled, pre_deployment, pre_change
    
    # Snapshot data
    parameters_data = Column(JSONB, nullable=False)  # Complete parameter state
    overrides_data = Column(JSONB)  # Active overrides at snapshot time
    snapshot_metadata = Column(JSONB)  # Additional snapshot metadata
    
    # Creation context
    created_by = Column(String(100), nullable=False)
    trigger_event = Column(String(100))  # What triggered this snapshot
    related_change_request_id = Column(UUID(as_uuid=True), ForeignKey("parameter_change_requests.id"))
    
    # Validation
    is_validated = Column(Boolean, default=False)
    validation_errors = Column(JSONB)
    
    # Usage tracking
    restore_count = Column(Integer, default=0)
    last_restored_at = Column(DateTime(timezone=True))
    last_restored_by = Column(String(100))
    
    # Relationships
    related_change_request = relationship("ParameterChangeRequest")
    
    __table_args__ = (
        Index('idx_parameter_snapshots_created_at', 'created_at'),
        Index('idx_parameter_snapshots_type', 'snapshot_type'),
        Index('idx_parameter_snapshots_created_by', 'created_by'),
    )


# Utility functions for parameter management
def create_parameter_tables(engine):
    """Create all parameter management tables."""
    Base.metadata.create_all(engine)


def get_active_parameter_value(
    session: Session,
    parameter_key: str,
    brand: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None
) -> Any:
    """
    Get the current active value for a parameter, considering overrides.
    
    Args:
        session: Database session
        parameter_key: The parameter key to lookup
        brand: Brand name for brand-specific overrides
        context: Additional context for conditional overrides
    
    Returns:
        The resolved parameter value
    """
    from sqlalchemy.orm import joinedload
    
    # Get the parameter with its versions and overrides
    parameter = (session.query(Parameter)
                .options(
                    joinedload(Parameter.versions),
                    joinedload(Parameter.overrides)
                )
                .filter(Parameter.key == parameter_key, Parameter.is_active == True)
                .first())
    
    if not parameter:
        raise ValueError(f"Parameter '{parameter_key}' not found or inactive")
    
    # Get the active version
    active_version = (session.query(ParameterVersion)
                     .filter(
                         ParameterVersion.parameter_id == parameter.id,
                         ParameterVersion.status == ParameterStatus.ACTIVE
                     )
                     .order_by(ParameterVersion.created_at.desc())
                     .first())
    
    if not active_version:
        # Fall back to default value if no active version
        base_value = parameter.default_value
    else:
        base_value = active_version.value
    
    # Check for applicable overrides
    current_time = datetime.now(timezone.utc)
    
    overrides_query = (session.query(ParameterOverride)
                      .filter(
                          ParameterOverride.parameter_id == parameter.id,
                          ParameterOverride.is_active == True,
                          ParameterOverride.valid_from <= current_time
                      )
                      .filter(
                          (ParameterOverride.valid_until.is_(None)) |
                          (ParameterOverride.valid_until > current_time)
                      ))
    
    # Filter by brand if specified
    if brand:
        overrides_query = overrides_query.filter(
            ((ParameterOverride.scope == ParameterScope.BRAND_SPECIFIC) &
             (ParameterOverride.scope_identifier == brand)) |
            (ParameterOverride.scope == ParameterScope.GLOBAL)
        )
    else:
        overrides_query = overrides_query.filter(
            ParameterOverride.scope == ParameterScope.GLOBAL
        )
    
    # Get overrides ordered by priority (highest first)
    applicable_overrides = overrides_query.order_by(ParameterOverride.priority.desc()).all()
    
    # Apply overrides in priority order
    final_value = base_value
    
    for override in applicable_overrides:
        # Check if override conditions are met
        if override.conditions:
            # TODO: Implement condition evaluation logic
            # For now, apply all overrides
            pass
        
        final_value = override.override_value
        break  # Use the highest priority override
    
    return final_value


def get_parameter_history(
    session: Session,
    parameter_key: str,
    limit: int = 50
) -> List[ParameterVersion]:
    """Get version history for a parameter."""
    
    parameter = session.query(Parameter).filter(Parameter.key == parameter_key).first()
    if not parameter:
        return []
    
    return (session.query(ParameterVersion)
           .filter(ParameterVersion.parameter_id == parameter.id)
           .order_by(ParameterVersion.created_at.desc())
           .limit(limit)
           .all())


def create_parameter_snapshot(
    session: Session,
    name: str,
    description: str,
    created_by: str,
    snapshot_type: str = 'manual'
) -> ParameterSnapshot:
    """Create a point-in-time snapshot of all parameters."""
    
    # Collect all active parameters and their values
    parameters_data = {}
    overrides_data = {}
    
    # Get all active parameters
    parameters = session.query(Parameter).filter(Parameter.is_active == True).all()
    
    for param in parameters:
        # Get active version
        active_version = (session.query(ParameterVersion)
                         .filter(
                             ParameterVersion.parameter_id == param.id,
                             ParameterVersion.status == ParameterStatus.ACTIVE
                         )
                         .first())
        
        parameters_data[param.key] = {
            'parameter_id': str(param.id),
            'value': active_version.value if active_version else param.default_value,
            'version_id': str(active_version.id) if active_version else None,
            'version_number': active_version.version_number if active_version else None
        }
        
        # Get active overrides
        current_time = datetime.now(timezone.utc)
        active_overrides = (session.query(ParameterOverride)
                           .filter(
                               ParameterOverride.parameter_id == param.id,
                               ParameterOverride.is_active == True,
                               ParameterOverride.valid_from <= current_time,
                               (ParameterOverride.valid_until.is_(None)) |
                               (ParameterOverride.valid_until > current_time)
                           )
                           .all())
        
        if active_overrides:
            overrides_data[param.key] = [
                {
                    'override_id': str(override.id),
                    'scope': override.scope.value,
                    'scope_identifier': override.scope_identifier,
                    'override_value': override.override_value,
                    'priority': override.priority
                }
                for override in active_overrides
            ]
    
    # Create snapshot
    snapshot = ParameterSnapshot(
        name=name,
        description=description,
        snapshot_type=snapshot_type,
        parameters_data=parameters_data,
        overrides_data=overrides_data,
        created_by=created_by,
        metadata={
            'parameter_count': len(parameters_data),
            'override_count': len(overrides_data),
            'created_timestamp': datetime.now(timezone.utc).isoformat()
        }
    )
    
    session.add(snapshot)
    session.commit()
    
    return snapshot