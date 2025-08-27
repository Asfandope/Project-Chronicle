"""
Parameter management initialization utilities.

This module provides functions to initialize the parameter management system
with default parameters and perform initial setup.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy.orm import Session

from .default_parameters import (
    BRAND_SPECIFIC_OVERRIDES,
    get_all_default_parameters,
)
from .models import (
    Parameter,
    ParameterAuditLog,
    ParameterOverride,
    ParameterScope,
    ParameterSnapshot,
    ParameterStatus,
    ParameterVersion,
)
from .service import ParameterService

logger = logging.getLogger(__name__)


class ParameterInitializer:
    """Handles initialization of the parameter management system."""

    def __init__(self):
        self.parameter_service = ParameterService()
        self.logger = logging.getLogger(__name__ + ".ParameterInitializer")

    def initialize_parameter_system(
        self, session: Session, force_recreate: bool = False, skip_existing: bool = True
    ) -> Dict[str, any]:
        """
        Initialize the complete parameter management system.

        Args:
            session: Database session
            force_recreate: Whether to recreate existing parameters
            skip_existing: Whether to skip parameters that already exist

        Returns:
            Dictionary with initialization results
        """
        results = {
            "parameters_created": 0,
            "parameters_skipped": 0,
            "parameters_failed": 0,
            "overrides_created": 0,
            "overrides_failed": 0,
            "errors": [],
            "warnings": [],
            "timestamp": datetime.now(timezone.utc),
        }

        self.logger.info("Starting parameter system initialization")

        try:
            # Step 1: Create default parameters
            param_results = self._create_default_parameters(
                session, force_recreate, skip_existing
            )
            results["parameters_created"] = param_results["created"]
            results["parameters_skipped"] = param_results["skipped"]
            results["parameters_failed"] = param_results["failed"]
            results["errors"].extend(param_results["errors"])
            results["warnings"].extend(param_results["warnings"])

            # Step 2: Create brand-specific overrides
            override_results = self._create_brand_overrides(session, skip_existing)
            results["overrides_created"] = override_results["created"]
            results["overrides_failed"] = override_results["failed"]
            results["errors"].extend(override_results["errors"])

            # Step 3: Create initial snapshot
            if results["parameters_created"] > 0:
                try:
                    snapshot = self.parameter_service.create_snapshot(
                        session=session,
                        name="Initial System Setup",
                        description="Initial parameter configuration after system setup",
                        created_by="system_initializer",
                        snapshot_type="system_initialization",
                    )
                    results["initial_snapshot_id"] = str(snapshot.id)
                    self.logger.info(f"Created initial system snapshot: {snapshot.id}")

                except Exception as e:
                    results["warnings"].append(
                        f"Failed to create initial snapshot: {str(e)}"
                    )

            session.commit()

            self.logger.info(
                f"Parameter system initialization completed. "
                f"Created {results['parameters_created']} parameters, "
                f"{results['overrides_created']} overrides."
            )

        except Exception as e:
            session.rollback()
            error_msg = f"Parameter system initialization failed: {str(e)}"
            results["errors"].append(error_msg)
            self.logger.error(error_msg)
            raise

        return results

    def _create_default_parameters(
        self, session: Session, force_recreate: bool, skip_existing: bool
    ) -> Dict[str, any]:
        """Create all default parameters."""

        results = {
            "created": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
            "warnings": [],
        }

        all_params = get_all_default_parameters()
        self.logger.info(f"Creating {len(all_params)} default parameters")

        for param_key, param_config in all_params.items():
            try:
                # Check if parameter already exists
                existing_param = (
                    session.query(Parameter).filter(Parameter.key == param_key).first()
                )

                if existing_param:
                    if skip_existing:
                        results["skipped"] += 1
                        continue
                    elif force_recreate:
                        # Delete existing parameter and recreate
                        session.delete(existing_param)
                        session.flush()
                        self.logger.info(f"Removed existing parameter: {param_key}")
                    else:
                        results["skipped"] += 1
                        continue

                # Create parameter
                parameter = self.parameter_service.create_parameter(
                    session=session,
                    key=param_key,
                    name=param_config["name"],
                    description=param_config["description"],
                    parameter_type=param_config["parameter_type"],
                    category=self._extract_category_from_key(param_key),
                    data_type=param_config["data_type"],
                    default_value=param_config["default_value"],
                    created_by="system_initializer",
                    validation_rules=param_config.get("validation_rules"),
                    subcategory=self._extract_subcategory_from_key(param_key),
                    impact_level=param_config.get("impact_level", "medium"),
                    requires_restart=param_config.get("requires_restart", False),
                )

                results["created"] += 1
                self.logger.debug(f"Created parameter: {param_key}")

            except Exception as e:
                error_msg = f"Failed to create parameter {param_key}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed"] += 1
                self.logger.error(error_msg)

                # Continue with other parameters
                session.rollback()
                continue

        return results

    def _create_brand_overrides(
        self, session: Session, skip_existing: bool
    ) -> Dict[str, any]:
        """Create brand-specific parameter overrides."""

        results = {"created": 0, "failed": 0, "errors": []}

        for brand_name, overrides in BRAND_SPECIFIC_OVERRIDES.items():
            self.logger.info(
                f"Creating {len(overrides)} overrides for brand: {brand_name}"
            )

            for param_key, override_config in overrides.items():
                try:
                    # Check if parameter exists
                    parameter = (
                        session.query(Parameter)
                        .filter(Parameter.key == param_key)
                        .first()
                    )

                    if not parameter:
                        # Create the parameter first if it doesn't exist
                        parameter = self.parameter_service.create_parameter(
                            session=session,
                            key=param_key,
                            name=override_config["name"],
                            description=override_config["description"],
                            parameter_type=override_config["parameter_type"],
                            category=self._extract_category_from_key(param_key),
                            data_type=override_config["data_type"],
                            default_value=override_config["default_value"],
                            created_by="system_initializer",
                        )
                        self.logger.info(f"Created parameter for override: {param_key}")

                    # Check if override already exists
                    existing_override = (
                        session.query(ParameterOverride)
                        .filter(
                            ParameterOverride.parameter_id == parameter.id,
                            ParameterOverride.scope == ParameterScope.BRAND_SPECIFIC,
                            ParameterOverride.scope_identifier == brand_name,
                        )
                        .first()
                    )

                    if existing_override and skip_existing:
                        continue

                    # Create override
                    from .service import ParameterOverrideRequest

                    override_request = ParameterOverrideRequest(
                        parameter_key=param_key,
                        override_value=override_config["default_value"],
                        scope=ParameterScope.BRAND_SPECIFIC,
                        scope_identifier=brand_name,
                        priority=100,
                        change_reason=f"Initial brand-specific override for {brand_name}",
                        created_by="system_initializer",
                    )

                    override = self.parameter_service.create_parameter_override(
                        session, override_request
                    )

                    results["created"] += 1
                    self.logger.debug(
                        f"Created override for {param_key} -> {brand_name}"
                    )

                except Exception as e:
                    error_msg = f"Failed to create override {param_key} for {brand_name}: {str(e)}"
                    results["errors"].append(error_msg)
                    results["failed"] += 1
                    self.logger.error(error_msg)

                    # Continue with other overrides
                    session.rollback()
                    continue

        return results

    def _extract_category_from_key(self, param_key: str) -> str:
        """Extract category from parameter key."""
        return param_key.split(".")[0] if "." in param_key else "general"

    def _extract_subcategory_from_key(self, param_key: str) -> Optional[str]:
        """Extract subcategory from parameter key."""
        parts = param_key.split(".")
        return parts[1] if len(parts) > 2 else None

    def validate_parameter_system(self, session: Session) -> Dict[str, any]:
        """Validate the parameter system setup."""

        validation_results = {
            "total_parameters": 0,
            "active_parameters": 0,
            "parameters_with_versions": 0,
            "total_overrides": 0,
            "active_overrides": 0,
            "brands_with_overrides": set(),
            "categories": {},
            "critical_parameters": [],
            "missing_parameters": [],
            "validation_errors": [],
            "warnings": [],
        }

        try:
            # Count parameters
            validation_results["total_parameters"] = session.query(Parameter).count()
            validation_results["active_parameters"] = (
                session.query(Parameter).filter(Parameter.is_active == True).count()
            )

            # Count parameters with active versions
            validation_results["parameters_with_versions"] = (
                session.query(Parameter)
                .join(ParameterVersion)
                .filter(ParameterVersion.status == ParameterStatus.ACTIVE)
                .count()
            )

            # Count overrides
            validation_results["total_overrides"] = session.query(
                ParameterOverride
            ).count()
            validation_results["active_overrides"] = (
                session.query(ParameterOverride)
                .filter(ParameterOverride.is_active == True)
                .count()
            )

            # Get brands with overrides
            brand_overrides = (
                session.query(ParameterOverride.scope_identifier)
                .filter(
                    ParameterOverride.scope == ParameterScope.BRAND_SPECIFIC,
                    ParameterOverride.is_active == True,
                )
                .distinct()
                .all()
            )

            validation_results["brands_with_overrides"] = {
                brand[0] for brand in brand_overrides if brand[0]
            }

            # Count by category
            parameters = (
                session.query(Parameter).filter(Parameter.is_active == True).all()
            )

            for param in parameters:
                category = param.category
                if category not in validation_results["categories"]:
                    validation_results["categories"][category] = 0
                validation_results["categories"][category] += 1

                # Check for critical parameters
                if param.impact_level == "critical":
                    validation_results["critical_parameters"].append(param.key)

            # Check for missing expected parameters
            expected_params = set(get_all_default_parameters().keys())
            existing_params = {param.key for param in parameters}
            validation_results["missing_parameters"] = list(
                expected_params - existing_params
            )

            # Validate parameter values can be retrieved
            for param in parameters[:10]:  # Sample validation
                try:
                    value = self.parameter_service.get_parameter_value(
                        session, param.key
                    )
                    if value is None and param.default_value is not None:
                        validation_results["warnings"].append(
                            f"Parameter {param.key} returned None but has default value"
                        )
                except Exception as e:
                    validation_results["validation_errors"].append(
                        f"Failed to retrieve parameter {param.key}: {str(e)}"
                    )

        except Exception as e:
            validation_results["validation_errors"].append(
                f"Validation failed: {str(e)}"
            )

        return validation_results

    def reset_parameter_system(
        self, session: Session, confirm_reset: bool = False
    ) -> Dict[str, any]:
        """Reset the entire parameter system (dangerous operation)."""

        if not confirm_reset:
            raise ValueError("Reset operation requires explicit confirmation")

        results = {
            "parameters_deleted": 0,
            "versions_deleted": 0,
            "overrides_deleted": 0,
            "snapshots_deleted": 0,
            "audit_logs_deleted": 0,
            "errors": [],
        }

        self.logger.warning("PERFORMING COMPLETE PARAMETER SYSTEM RESET")

        try:
            # Delete in correct order to handle foreign key constraints

            # Delete audit logs
            audit_count = session.query(ParameterAuditLog).count()
            session.query(ParameterAuditLog).delete()
            results["audit_logs_deleted"] = audit_count

            # Delete snapshots
            snapshot_count = session.query(ParameterSnapshot).count()
            session.query(ParameterSnapshot).delete()
            results["snapshots_deleted"] = snapshot_count

            # Delete overrides
            override_count = session.query(ParameterOverride).count()
            session.query(ParameterOverride).delete()
            results["overrides_deleted"] = override_count

            # Delete versions
            version_count = session.query(ParameterVersion).count()
            session.query(ParameterVersion).delete()
            results["versions_deleted"] = version_count

            # Delete parameters
            param_count = session.query(Parameter).count()
            session.query(Parameter).delete()
            results["parameters_deleted"] = param_count

            session.commit()

            self.logger.warning(
                f"Parameter system reset completed. Deleted: "
                f"{results['parameters_deleted']} parameters, "
                f"{results['versions_deleted']} versions, "
                f"{results['overrides_deleted']} overrides, "
                f"{results['snapshots_deleted']} snapshots, "
                f"{results['audit_logs_deleted']} audit logs"
            )

        except Exception as e:
            session.rollback()
            error_msg = f"Parameter system reset failed: {str(e)}"
            results["errors"].append(error_msg)
            self.logger.error(error_msg)
            raise

        return results


def initialize_parameter_management_system(
    session: Session, force_recreate: bool = False, skip_existing: bool = True
) -> Dict[str, any]:
    """
    Convenience function to initialize the parameter management system.

    This is the main entry point for setting up the parameter system.
    """
    initializer = ParameterInitializer()
    return initializer.initialize_parameter_system(
        session, force_recreate, skip_existing
    )


def validate_parameter_system(session: Session) -> Dict[str, any]:
    """Convenience function to validate parameter system setup."""
    initializer = ParameterInitializer()
    return initializer.validate_parameter_system(session)


def reset_parameter_system(session: Session, confirm: bool = False) -> Dict[str, any]:
    """Convenience function to reset parameter system (dangerous)."""
    initializer = ParameterInitializer()
    return initializer.reset_parameter_system(session, confirm)
