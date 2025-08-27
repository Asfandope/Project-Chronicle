"""
Parameter management service with business logic and validation.

This module provides the core service layer for parameter management,
including validation, versioning, and rollback operations.
"""

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import and_, desc, or_
from sqlalchemy.orm import Session

from .models import (
    Parameter,
    ParameterAuditLog,
    ParameterChangeRequest,
    ParameterOverride,
    ParameterScope,
    ParameterSnapshot,
    ParameterStatus,
    ParameterType,
    ParameterVersion,
    create_parameter_snapshot,
    get_active_parameter_value,
)

logger = logging.getLogger(__name__)


class ParameterValidationError(Exception):
    """Raised when parameter validation fails."""


class ParameterNotFoundError(Exception):
    """Raised when a parameter is not found."""


class ParameterVersionError(Exception):
    """Raised when parameter versioning operations fail."""


@dataclass
class ParameterUpdateRequest:
    """Request to update a parameter value."""

    parameter_key: str
    new_value: Any
    change_reason: str
    created_by: str
    version_number: Optional[str] = None
    requires_approval: bool = True
    auto_activate: bool = False


@dataclass
class ParameterOverrideRequest:
    """Request to create a parameter override."""

    parameter_key: str
    override_value: Any
    scope: ParameterScope
    scope_identifier: Optional[str] = None
    conditions: Optional[Dict[str, Any]] = None
    priority: int = 100
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    change_reason: str = ""
    created_by: str = ""


@dataclass
class RollbackRequest:
    """Request to rollback parameters."""

    target_version_id: Optional[str] = None
    target_snapshot_id: Optional[str] = None
    parameter_keys: Optional[List[str]] = None  # If None, rollback all
    rollback_reason: str = ""
    created_by: str = ""


class ParameterValidator:
    """Validates parameter values and constraints."""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".ParameterValidator")

    def validate_parameter_value(
        self, parameter: Parameter, value: Any
    ) -> Tuple[bool, List[str]]:
        """
        Validate a parameter value against its constraints.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        try:
            # Basic type validation
            if not self._validate_data_type(parameter.data_type, value):
                errors.append(
                    f"Value does not match expected type '{parameter.data_type}'"
                )

            # JSON schema validation if defined
            if parameter.validation_rules:
                schema_errors = self._validate_json_schema(
                    parameter.validation_rules, value
                )
                errors.extend(schema_errors)

            # Type-specific validations
            if parameter.parameter_type == ParameterType.THRESHOLD:
                threshold_errors = self._validate_threshold(value)
                errors.extend(threshold_errors)
            elif parameter.parameter_type == ParameterType.PROMPT:
                prompt_errors = self._validate_prompt(value)
                errors.extend(prompt_errors)
            elif parameter.parameter_type == ParameterType.RULE:
                rule_errors = self._validate_rule(value)
                errors.extend(rule_errors)

        except Exception as e:
            errors.append(f"Validation error: {str(e)}")

        return len(errors) == 0, errors

    def _validate_data_type(self, expected_type: str, value: Any) -> bool:
        """Validate value matches expected data type."""

        type_validators = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "float": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "json": lambda v: True,  # JSON can be any serializable type
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        validator = type_validators.get(expected_type)
        if not validator:
            return True  # Unknown type, skip validation

        return validator(value)

    def _validate_json_schema(self, schema: Dict[str, Any], value: Any) -> List[str]:
        """Validate value against JSON schema."""
        errors = []

        try:
            # Import jsonschema if available
            import jsonschema

            jsonschema.validate(value, schema)
        except ImportError:
            # jsonschema not available, skip validation
            self.logger.warning("jsonschema package not available for validation")
        except jsonschema.ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        except Exception as e:
            errors.append(f"Schema validation error: {str(e)}")

        return errors

    def _validate_threshold(self, value: Any) -> List[str]:
        """Validate threshold-type parameters."""
        errors = []

        if isinstance(value, (int, float)):
            if value < 0 or value > 1:
                errors.append("Threshold values must be between 0 and 1")
        elif isinstance(value, dict):
            # Handle threshold objects with multiple values
            for key, val in value.items():
                if isinstance(val, (int, float)) and (val < 0 or val > 1):
                    errors.append(f"Threshold '{key}' must be between 0 and 1")

        return errors

    def _validate_prompt(self, value: Any) -> List[str]:
        """Validate prompt-type parameters."""
        errors = []

        if not isinstance(value, str):
            errors.append("Prompt values must be strings")
        elif len(value.strip()) == 0:
            errors.append("Prompt values cannot be empty")
        elif len(value) > 10000:
            errors.append("Prompt values cannot exceed 10,000 characters")

        return errors

    def _validate_rule(self, value: Any) -> List[str]:
        """Validate rule-type parameters."""
        errors = []

        if isinstance(value, dict):
            # Validate rule structure
            if "condition" not in value:
                errors.append("Rules must have a 'condition' field")
            if "action" not in value:
                errors.append("Rules must have an 'action' field")
        elif isinstance(value, str):
            # Simple rule validation
            if len(value.strip()) == 0:
                errors.append("Rule values cannot be empty")

        return errors


class ParameterService:
    """Core parameter management service."""

    def __init__(self):
        self.validator = ParameterValidator()
        self.logger = logging.getLogger(__name__ + ".ParameterService")

    def get_parameter_value(
        self,
        session: Session,
        parameter_key: str,
        brand: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Get the current effective value for a parameter.

        This is the main method used by the application to get parameter values.
        It handles all override logic and returns the final effective value.
        """
        try:
            value = get_active_parameter_value(session, parameter_key, brand, context)

            # Log parameter access for auditing
            self._log_parameter_access(session, parameter_key, brand, context, value)

            return value

        except Exception as e:
            self.logger.error(
                f"Error getting parameter value for '{parameter_key}': {str(e)}"
            )
            raise ParameterNotFoundError(
                f"Parameter '{parameter_key}' not found: {str(e)}"
            )

    def create_parameter(
        self,
        session: Session,
        key: str,
        name: str,
        description: str,
        parameter_type: ParameterType,
        category: str,
        data_type: str,
        default_value: Any,
        created_by: str,
        validation_rules: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Parameter:
        """Create a new parameter definition."""

        # Check if parameter already exists
        existing = session.query(Parameter).filter(Parameter.key == key).first()
        if existing:
            raise ValueError(f"Parameter with key '{key}' already exists")

        # Validate default value
        temp_param = Parameter(
            key=key,
            parameter_type=parameter_type,
            data_type=data_type,
            validation_rules=validation_rules,
        )

        is_valid, errors = self.validator.validate_parameter_value(
            temp_param, default_value
        )
        if not is_valid:
            raise ParameterValidationError(
                f"Invalid default value: {'; '.join(errors)}"
            )

        # Create parameter
        parameter = Parameter(
            key=key,
            name=name,
            description=description,
            parameter_type=parameter_type,
            category=category,
            data_type=data_type,
            default_value=default_value,
            validation_rules=validation_rules,
            **kwargs,
        )

        session.add(parameter)
        session.flush()

        # Create initial version
        self._create_parameter_version(
            session=session,
            parameter=parameter,
            value=default_value,
            version_number="1.0.0",
            created_by=created_by,
            change_reason="Initial parameter creation",
            auto_activate=True,
        )

        # Log creation
        self._log_parameter_change(
            session=session,
            parameter_id=parameter.id,
            parameter_key=key,
            action="create",
            object_type="parameter",
            object_id=parameter.id,
            new_value=default_value,
            user_id=created_by,
            reason="Parameter creation",
        )

        session.commit()

        self.logger.info(f"Created parameter '{key}' with default value")
        return parameter

    def update_parameter_value(
        self, session: Session, request: ParameterUpdateRequest
    ) -> ParameterVersion:
        """Update a parameter value with versioning."""

        # Get parameter
        parameter = (
            session.query(Parameter)
            .filter(Parameter.key == request.parameter_key, Parameter.is_active == True)
            .first()
        )

        if not parameter:
            raise ParameterNotFoundError(
                f"Parameter '{request.parameter_key}' not found"
            )

        # Validate new value
        is_valid, errors = self.validator.validate_parameter_value(
            parameter, request.new_value
        )
        if not is_valid:
            raise ParameterValidationError(
                f"Invalid parameter value: {'; '.join(errors)}"
            )

        # Check if value actually changed
        current_value = self.get_parameter_value(session, request.parameter_key)
        if self._values_equal(current_value, request.new_value):
            self.logger.info(
                f"Parameter '{request.parameter_key}' value unchanged, skipping update"
            )
            return None

        # Generate version number if not provided
        if not request.version_number:
            request.version_number = self._generate_next_version_number(
                session, parameter
            )

        # Create change request if approval required
        if request.requires_approval and parameter.impact_level in ["high", "critical"]:
            change_request = self._create_change_request(
                session=session,
                parameter=parameter,
                proposed_value=request.new_value,
                requested_by=request.created_by,
                change_reason=request.change_reason,
            )

            self.logger.info(
                f"Created change request {change_request.id} for parameter '{request.parameter_key}'"
            )
            return None

        # Create new version
        version = self._create_parameter_version(
            session=session,
            parameter=parameter,
            value=request.new_value,
            version_number=request.version_number,
            created_by=request.created_by,
            change_reason=request.change_reason,
            auto_activate=request.auto_activate,
        )

        session.commit()

        self.logger.info(
            f"Updated parameter '{request.parameter_key}' to version {request.version_number}"
        )
        return version

    def create_parameter_override(
        self, session: Session, request: ParameterOverrideRequest
    ) -> ParameterOverride:
        """Create a parameter override for specific scopes."""

        # Get parameter
        parameter = (
            session.query(Parameter)
            .filter(Parameter.key == request.parameter_key, Parameter.is_active == True)
            .first()
        )

        if not parameter:
            raise ParameterNotFoundError(
                f"Parameter '{request.parameter_key}' not found"
            )

        # Validate override value
        is_valid, errors = self.validator.validate_parameter_value(
            parameter, request.override_value
        )
        if not is_valid:
            raise ParameterValidationError(
                f"Invalid override value: {'; '.join(errors)}"
            )

        # Set validity period
        valid_from = request.valid_from or datetime.now(timezone.utc)

        # Create override
        override = ParameterOverride(
            parameter_id=parameter.id,
            scope=request.scope,
            scope_identifier=request.scope_identifier,
            override_value=request.override_value,
            conditions=request.conditions,
            priority=request.priority,
            valid_from=valid_from,
            valid_until=request.valid_until,
            created_by=request.created_by,
            change_reason=request.change_reason,
            value_hash=self._calculate_value_hash(request.override_value),
        )

        session.add(override)
        session.flush()

        # Log override creation
        self._log_parameter_change(
            session=session,
            parameter_id=parameter.id,
            parameter_key=request.parameter_key,
            action="create_override",
            object_type="override",
            object_id=override.id,
            new_value=request.override_value,
            user_id=request.created_by,
            reason=request.change_reason,
            metadata={
                "scope": request.scope.value,
                "scope_identifier": request.scope_identifier,
                "priority": request.priority,
            },
        )

        session.commit()

        self.logger.info(
            f"Created override for parameter '{request.parameter_key}' "
            f"with scope {request.scope.value}:{request.scope_identifier}"
        )

        return override

    def rollback_parameter(
        self, session: Session, request: RollbackRequest
    ) -> List[ParameterVersion]:
        """Rollback parameters to previous versions or snapshot."""

        rolled_back_versions = []

        if request.target_snapshot_id:
            # Rollback from snapshot
            rolled_back_versions = self._rollback_from_snapshot(session, request)
        elif request.target_version_id:
            # Rollback single parameter to specific version
            rolled_back_versions = self._rollback_to_version(session, request)
        else:
            raise ValueError(
                "Either target_version_id or target_snapshot_id must be specified"
            )

        session.commit()

        self.logger.info(f"Rolled back {len(rolled_back_versions)} parameter versions")
        return rolled_back_versions

    def create_snapshot(
        self,
        session: Session,
        name: str,
        description: str,
        created_by: str,
        snapshot_type: str = "manual",
    ) -> ParameterSnapshot:
        """Create a snapshot of current parameter state."""

        snapshot = create_parameter_snapshot(
            session=session,
            name=name,
            description=description,
            created_by=created_by,
            snapshot_type=snapshot_type,
        )

        self.logger.info(
            f"Created parameter snapshot '{name}' with {len(snapshot.parameters_data)} parameters"
        )
        return snapshot

    def get_parameter_configuration(
        self,
        session: Session,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        include_overrides: bool = True,
    ) -> Dict[str, Any]:
        """Get complete parameter configuration for a category/brand."""

        query = session.query(Parameter).filter(Parameter.is_active == True)

        if category:
            query = query.filter(Parameter.category == category)

        parameters = query.all()

        config = {}

        for param in parameters:
            try:
                value = self.get_parameter_value(session, param.key, brand)
                config[param.key] = {
                    "value": value,
                    "type": param.parameter_type.value,
                    "category": param.category,
                    "description": param.description,
                }

                if include_overrides:
                    # Get applicable overrides
                    overrides = self._get_applicable_overrides(
                        session, param.key, brand
                    )
                    if overrides:
                        config[param.key]["overrides"] = [
                            {
                                "scope": override.scope.value,
                                "scope_identifier": override.scope_identifier,
                                "value": override.override_value,
                                "priority": override.priority,
                            }
                            for override in overrides
                        ]

            except Exception as e:
                self.logger.warning(
                    f"Error getting value for parameter '{param.key}': {str(e)}"
                )
                continue

        return config

    def _create_parameter_version(
        self,
        session: Session,
        parameter: Parameter,
        value: Any,
        version_number: str,
        created_by: str,
        change_reason: str,
        auto_activate: bool = False,
    ) -> ParameterVersion:
        """Create a new parameter version."""

        # Deactivate current active version if auto-activating
        if auto_activate:
            current_active = (
                session.query(ParameterVersion)
                .filter(
                    ParameterVersion.parameter_id == parameter.id,
                    ParameterVersion.status == ParameterStatus.ACTIVE,
                )
                .first()
            )

            if current_active:
                current_active.status = ParameterStatus.DEPRECATED
                current_active.deactivated_at = datetime.now(timezone.utc)
                current_active.deactivated_by = created_by

        # Create new version
        version = ParameterVersion(
            parameter_id=parameter.id,
            version_number=version_number,
            value=value,
            value_hash=self._calculate_value_hash(value),
            created_by=created_by,
            change_reason=change_reason,
            status=ParameterStatus.ACTIVE if auto_activate else ParameterStatus.DRAFT,
        )

        if auto_activate:
            version.activated_at = datetime.now(timezone.utc)
            version.activated_by = created_by

        session.add(version)
        session.flush()

        return version

    def _create_change_request(
        self,
        session: Session,
        parameter: Parameter,
        proposed_value: Any,
        requested_by: str,
        change_reason: str,
    ) -> ParameterChangeRequest:
        """Create a parameter change request."""

        current_value = self.get_parameter_value(session, parameter.key)

        change_request = ParameterChangeRequest(
            parameter_id=parameter.id,
            title=f"Update {parameter.name}",
            description=change_reason,
            current_value=current_value,
            proposed_value=proposed_value,
            change_type="update",
            requested_by=requested_by,
            risk_level=parameter.impact_level,
        )

        session.add(change_request)
        session.flush()

        return change_request

    def _rollback_from_snapshot(
        self, session: Session, request: RollbackRequest
    ) -> List[ParameterVersion]:
        """Rollback parameters from a snapshot."""

        snapshot = (
            session.query(ParameterSnapshot)
            .filter(ParameterSnapshot.id == request.target_snapshot_id)
            .first()
        )

        if not snapshot:
            raise ValueError(f"Snapshot {request.target_snapshot_id} not found")

        rolled_back_versions = []

        for param_key, param_data in snapshot.parameters_data.items():
            if request.parameter_keys and param_key not in request.parameter_keys:
                continue

            parameter = (
                session.query(Parameter).filter(Parameter.key == param_key).first()
            )
            if not parameter:
                self.logger.warning(
                    f"Parameter '{param_key}' from snapshot not found, skipping"
                )
                continue

            # Create rollback version
            version_number = self._generate_next_version_number(session, parameter)

            version = self._create_parameter_version(
                session=session,
                parameter=parameter,
                value=param_data["value"],
                version_number=version_number,
                created_by=request.created_by,
                change_reason=f"Rollback to snapshot '{snapshot.name}': {request.rollback_reason}",
                auto_activate=True,
            )

            # Mark as rollback
            version.rolled_back_from_version = param_data.get("version_id")
            version.rolled_back_at = datetime.now(timezone.utc)
            version.rolled_back_by = request.created_by
            version.rollback_reason = request.rollback_reason

            rolled_back_versions.append(version)

        # Update snapshot usage
        snapshot.restore_count += 1
        snapshot.last_restored_at = datetime.now(timezone.utc)
        snapshot.last_restored_by = request.created_by

        return rolled_back_versions

    def _rollback_to_version(
        self, session: Session, request: RollbackRequest
    ) -> List[ParameterVersion]:
        """Rollback single parameter to specific version."""

        target_version = (
            session.query(ParameterVersion)
            .filter(ParameterVersion.id == request.target_version_id)
            .first()
        )

        if not target_version:
            raise ValueError(f"Version {request.target_version_id} not found")

        parameter = target_version.parameter

        # Create rollback version
        version_number = self._generate_next_version_number(session, parameter)

        rollback_version = self._create_parameter_version(
            session=session,
            parameter=parameter,
            value=target_version.value,
            version_number=version_number,
            created_by=request.created_by,
            change_reason=f"Rollback to version {target_version.version_number}: {request.rollback_reason}",
            auto_activate=True,
        )

        # Mark as rollback
        rollback_version.rolled_back_from_version = target_version.id
        rollback_version.rolled_back_at = datetime.now(timezone.utc)
        rollback_version.rolled_back_by = request.created_by
        rollback_version.rollback_reason = request.rollback_reason

        return [rollback_version]

    def _generate_next_version_number(
        self, session: Session, parameter: Parameter
    ) -> str:
        """Generate the next version number for a parameter."""

        latest_version = (
            session.query(ParameterVersion)
            .filter(ParameterVersion.parameter_id == parameter.id)
            .order_by(desc(ParameterVersion.created_at))
            .first()
        )

        if not latest_version:
            return "1.0.0"

        # Simple semantic versioning
        try:
            major, minor, patch = map(int, latest_version.version_number.split("."))
            return f"{major}.{minor}.{patch + 1}"
        except ValueError:
            # Fallback to timestamp-based versioning
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            return f"auto.{timestamp}"

    def _calculate_value_hash(self, value: Any) -> str:
        """Calculate SHA-256 hash of parameter value."""
        value_str = json.dumps(value, sort_keys=True, default=str)
        return hashlib.sha256(value_str.encode()).hexdigest()

    def _values_equal(self, value1: Any, value2: Any) -> bool:
        """Compare two parameter values for equality."""
        return self._calculate_value_hash(value1) == self._calculate_value_hash(value2)

    def _get_applicable_overrides(
        self, session: Session, parameter_key: str, brand: Optional[str] = None
    ) -> List[ParameterOverride]:
        """Get all applicable overrides for a parameter."""

        parameter = (
            session.query(Parameter).filter(Parameter.key == parameter_key).first()
        )
        if not parameter:
            return []

        current_time = datetime.now(timezone.utc)

        query = session.query(ParameterOverride).filter(
            ParameterOverride.parameter_id == parameter.id,
            ParameterOverride.is_active == True,
            ParameterOverride.valid_from <= current_time,
            or_(
                ParameterOverride.valid_until.is_(None),
                ParameterOverride.valid_until > current_time,
            ),
        )

        if brand:
            query = query.filter(
                or_(
                    and_(
                        ParameterOverride.scope == ParameterScope.BRAND_SPECIFIC,
                        ParameterOverride.scope_identifier == brand,
                    ),
                    ParameterOverride.scope == ParameterScope.GLOBAL,
                )
            )
        else:
            query = query.filter(ParameterOverride.scope == ParameterScope.GLOBAL)

        return query.order_by(desc(ParameterOverride.priority)).all()

    def _log_parameter_access(
        self,
        session: Session,
        parameter_key: str,
        brand: Optional[str],
        context: Optional[Dict[str, Any]],
        value: Any,
    ):
        """Log parameter access for auditing (sampling to avoid too much data)."""

        # Sample parameter access logging (e.g., only log 1% of accesses)
        import random

        if random.random() > 0.01:  # 1% sampling rate
            return

        try:
            self._log_parameter_change(
                session=session,
                parameter_id=None,
                parameter_key=parameter_key,
                action="access",
                object_type="parameter",
                object_id=None,
                new_value=None,
                user_id="system",
                reason="Parameter value accessed",
                metadata={
                    "brand": brand,
                    "context": context,
                    "resolved_value_hash": self._calculate_value_hash(value),
                },
            )
            session.commit()
        except Exception as e:
            # Don't fail parameter access due to logging errors
            self.logger.warning(f"Failed to log parameter access: {str(e)}")

    def _log_parameter_change(
        self,
        session: Session,
        parameter_id: Optional[str],
        parameter_key: str,
        action: str,
        object_type: str,
        object_id: Optional[str],
        user_id: str,
        reason: str,
        old_value: Any = None,
        new_value: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log parameter changes for audit trail."""

        log_entry = ParameterAuditLog(
            parameter_id=parameter_id,
            parameter_key=parameter_key,
            action=action,
            object_type=object_type,
            object_id=object_id,
            old_value=old_value,
            new_value=new_value,
            user_id=user_id,
            reason=reason,
            source="api",
            metadata=metadata or {},
        )

        session.add(log_entry)


# Global parameter service instance
parameter_service = ParameterService()
