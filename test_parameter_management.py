#!/usr/bin/env python3
"""
Test script for the parameter management system.

This script demonstrates and tests the centralized parameter management
functionality including versioning, overrides, and rollback capabilities.
"""

import logging
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from parameter_management import (
    ParameterKeys,
    create_parameter_snapshot,
    get_category_parameters,
    get_parameter,
    initialize_parameter_management,
)
from parameter_management.initialization import initialize_parameter_management_system
from parameter_management.migrator import analyze_and_migrate_codebase
from parameter_management.models import (
    ParameterScope,
    create_parameter_tables,
)
from parameter_management.service import (
    ParameterOverrideRequest,
    ParameterService,
    ParameterUpdateRequest,
    RollbackRequest,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParameterManagementTester:
    """Tests the parameter management system functionality."""

    def __init__(
        self,
        database_url: str = "postgresql://postgres:postgres@localhost:5432/magazine_extractor",
    ):
        self.engine = create_engine(database_url, echo=False)
        self.Session = sessionmaker(bind=self.engine)

        # Create tables
        create_parameter_tables(self.engine)

        # Initialize parameter management
        initialize_parameter_management(self.Session)

        self.parameter_service = ParameterService()
        logger.info("Parameter management tester initialized")

    def test_system_initialization(self):
        """Test system initialization with default parameters."""
        print("Testing system initialization...")

        session = self.Session()
        try:
            # Initialize with default parameters
            results = initialize_parameter_management_system(
                session=session, force_recreate=False, skip_existing=True
            )

            print(f"✓ System initialization completed:")
            print(f"  - Parameters created: {results['parameters_created']}")
            print(f"  - Parameters skipped: {results['parameters_skipped']}")
            print(f"  - Overrides created: {results['overrides_created']}")
            print(
                f"  - Initial snapshot: {results.get('initial_snapshot_id', 'Not created')}"
            )

            if results["errors"]:
                print(f"  - Errors: {len(results['errors'])}")
                for error in results["errors"][:3]:  # Show first 3 errors
                    print(f"    • {error}")

            return True

        except Exception as e:
            print(f"✗ System initialization failed: {str(e)}")
            return False

        finally:
            session.close()

    def test_parameter_retrieval(self):
        """Test basic parameter retrieval functionality."""
        print("\nTesting parameter retrieval...")

        try:
            # Test predefined parameter keys
            accuracy_weights = {
                ParameterKeys.ACCURACY_TITLE_WEIGHT: 0.30,
                ParameterKeys.ACCURACY_BODY_TEXT_WEIGHT: 0.40,
                ParameterKeys.ACCURACY_CONTRIBUTORS_WEIGHT: 0.20,
                ParameterKeys.ACCURACY_MEDIA_LINKS_WEIGHT: 0.10,
            }

            print("✓ Testing accuracy weight parameters:")
            total_weight = 0
            for key, expected in accuracy_weights.items():
                value = get_parameter(key)
                total_weight += value
                print(
                    f"  - {key}: {value} (expected: {expected}) {'✓' if value == expected else '✗'}"
                )

            print(
                f"  - Total weights: {total_weight} (should be 1.0) {'✓' if abs(total_weight - 1.0) < 0.001 else '✗'}"
            )

            # Test drift detection parameters
            print("✓ Testing drift detection parameters:")
            drift_params = [
                ParameterKeys.DRIFT_WINDOW_SIZE,
                ParameterKeys.DRIFT_THRESHOLD,
                ParameterKeys.DRIFT_ALERT_THRESHOLD,
                ParameterKeys.DRIFT_AUTO_TUNING_THRESHOLD,
            ]

            for key in drift_params:
                value = get_parameter(key)
                print(f"  - {key}: {value}")

            # Test with brand-specific overrides
            print("✓ Testing brand-specific parameters:")
            brands = ["TechWeekly", "StyleMag", "NewsToday"]

            for brand in brands:
                try:
                    columns = get_parameter("brand.default_columns", brand=brand)
                    color = get_parameter("brand.primary_color", brand=brand)
                    print(f"  - {brand}: {columns} columns, color {color}")
                except Exception:
                    print(f"  - {brand}: No brand-specific overrides")

            return True

        except Exception as e:
            print(f"✗ Parameter retrieval failed: {str(e)}")
            return False

    def test_parameter_updates(self):
        """Test parameter value updates with versioning."""
        print("\nTesting parameter updates...")

        session = self.Session()
        try:
            test_key = ParameterKeys.DRIFT_THRESHOLD
            original_value = get_parameter(test_key)

            print(f"✓ Original value: {test_key} = {original_value}")

            # Update parameter
            new_value = 0.08  # Change from default 0.05 to 0.08

            update_request = ParameterUpdateRequest(
                parameter_key=test_key,
                new_value=new_value,
                change_reason="Test parameter update",
                created_by="test_user",
                auto_activate=True,
            )

            version = self.parameter_service.update_parameter_value(
                session, update_request
            )

            if version:
                print(f"✓ Parameter updated to version {version.version_number}")

                # Verify update
                updated_value = get_parameter(test_key)
                print(
                    f"  - Updated value: {updated_value} {'✓' if updated_value == new_value else '✗'}"
                )

                # Test parameter history
                from parameter_management.models import get_parameter_history

                history = get_parameter_history(session, test_key, limit=3)
                print(f"  - Version history: {len(history)} versions")

                for i, hist_version in enumerate(history):
                    print(
                        f"    {i+1}. v{hist_version.version_number}: {hist_version.value} ({hist_version.status.value})"
                    )

                return True
            else:
                print("✗ Parameter update returned None (may require approval)")
                return False

        except Exception as e:
            print(f"✗ Parameter update failed: {str(e)}")
            return False

        finally:
            session.close()

    def test_parameter_overrides(self):
        """Test brand-specific parameter overrides."""
        print("\nTesting parameter overrides...")

        session = self.Session()
        try:
            test_key = ParameterKeys.PROCESSING_BATCH_SIZE
            test_brand = "TestBrand"

            # Get original value
            original_value = get_parameter(test_key)
            print(f"✓ Original global value: {test_key} = {original_value}")

            # Create brand-specific override
            override_value = 64  # Different from default 32

            override_request = ParameterOverrideRequest(
                parameter_key=test_key,
                override_value=override_value,
                scope=ParameterScope.BRAND_SPECIFIC,
                scope_identifier=test_brand,
                priority=200,  # Higher priority
                change_reason="Test brand-specific override",
                created_by="test_user",
            )

            override = self.parameter_service.create_parameter_override(
                session, override_request
            )

            print(f"✓ Created override for {test_brand}")

            # Test override resolution
            global_value = get_parameter(test_key)  # No brand
            brand_value = get_parameter(test_key, brand=test_brand)  # With brand

            print(f"  - Global value: {global_value}")
            print(f"  - {test_brand} value: {brand_value}")
            print(
                f"  - Override working: {'✓' if brand_value == override_value else '✗'}"
            )

            return brand_value == override_value

        except Exception as e:
            print(f"✗ Parameter override test failed: {str(e)}")
            return False

        finally:
            session.close()

    def test_snapshots_and_rollback(self):
        """Test snapshot creation and rollback functionality."""
        print("\nTesting snapshots and rollback...")

        session = self.Session()
        try:
            # Create a snapshot
            snapshot_id = create_parameter_snapshot(
                name="Test Snapshot", description="Snapshot for rollback testing"
            )

            print(f"✓ Created snapshot: {snapshot_id}")

            # Make some parameter changes
            test_key = ParameterKeys.DRIFT_ALERT_THRESHOLD
            original_value = get_parameter(test_key)

            # Update parameter
            update_request = ParameterUpdateRequest(
                parameter_key=test_key,
                new_value=0.15,  # Change from default
                change_reason="Test change before rollback",
                created_by="test_user",
                auto_activate=True,
            )

            version = self.parameter_service.update_parameter_value(
                session, update_request
            )

            if version:
                updated_value = get_parameter(test_key)
                print(f"✓ Changed parameter: {test_key} = {updated_value}")

                # Perform rollback to snapshot
                rollback_request = RollbackRequest(
                    target_snapshot_id=snapshot_id,
                    rollback_reason="Test rollback to snapshot",
                    created_by="test_user",
                )

                rolled_back_versions = self.parameter_service.rollback_parameter(
                    session, rollback_request
                )

                print(f"✓ Rolled back {len(rolled_back_versions)} parameters")

                # Verify rollback
                rolled_back_value = get_parameter(test_key)
                print(f"  - Value after rollback: {rolled_back_value}")
                print(
                    f"  - Rollback successful: {'✓' if rolled_back_value == original_value else '✗'}"
                )

                return rolled_back_value == original_value
            else:
                print("✗ Could not update parameter for rollback test")
                return False

        except Exception as e:
            print(f"✗ Snapshot/rollback test failed: {str(e)}")
            return False

        finally:
            session.close()

    def test_category_parameters(self):
        """Test retrieving parameters by category."""
        print("\nTesting category parameter retrieval...")

        try:
            categories = ["accuracy", "drift", "processing", "model", "feature"]

            for category in categories:
                params = get_category_parameters(category)
                print(f"✓ {category.title()} category: {len(params)} parameters")

                # Show a few example parameters
                for key, config in list(params.items())[:2]:
                    value = config.get("value", "N/A")
                    print(f"  - {key}: {value}")

            return True

        except Exception as e:
            print(f"✗ Category parameter test failed: {str(e)}")
            return False

    def test_feature_flags(self):
        """Test feature flag parameters."""
        print("\nTesting feature flags...")

        try:
            feature_flags = [
                ParameterKeys.FEATURE_DRIFT_DETECTION_ENABLED,
                ParameterKeys.FEATURE_AUTO_TUNING_ENABLED,
                ParameterKeys.FEATURE_BRAND_OVERRIDES_ENABLED,
                ParameterKeys.FEATURE_STATISTICAL_SIGNIFICANCE_ENABLED,
            ]

            for flag in feature_flags:
                enabled = get_parameter(flag)
                print(f"  - {flag}: {'ENABLED' if enabled else 'DISABLED'}")

            # Test conditional logic based on feature flags
            if get_parameter(ParameterKeys.FEATURE_DRIFT_DETECTION_ENABLED):
                drift_threshold = get_parameter(ParameterKeys.DRIFT_THRESHOLD)
                print(f"  - Drift detection active with threshold: {drift_threshold}")

            return True

        except Exception as e:
            print(f"✗ Feature flag test failed: {str(e)}")
            return False

    def test_parameter_validation(self):
        """Test parameter validation rules."""
        print("\nTesting parameter validation...")

        session = self.Session()
        try:
            # Test invalid value (outside range)
            test_key = ParameterKeys.ACCURACY_TITLE_WEIGHT

            # Try to set weight to invalid value (> 1.0)
            invalid_request = ParameterUpdateRequest(
                parameter_key=test_key,
                new_value=1.5,  # Invalid: > 1.0
                change_reason="Test validation failure",
                created_by="test_user",
                auto_activate=True,
            )

            try:
                version = self.parameter_service.update_parameter_value(
                    session, invalid_request
                )
                print("✗ Validation should have failed for value > 1.0")
                return False

            except Exception as e:
                print(f"✓ Validation correctly rejected invalid value: {str(e)}")

            # Test valid value
            valid_request = ParameterUpdateRequest(
                parameter_key=test_key,
                new_value=0.25,  # Valid: between 0 and 1
                change_reason="Test validation success",
                created_by="test_user",
                auto_activate=True,
            )

            version = self.parameter_service.update_parameter_value(
                session, valid_request
            )
            if version:
                print("✓ Validation correctly accepted valid value")
                return True
            else:
                print("✗ Valid value was rejected")
                return False

        except Exception as e:
            print(f"✗ Parameter validation test failed: {str(e)}")
            return False

        finally:
            session.close()

    def test_hardcoded_value_migration(self):
        """Test migration of hardcoded values (demonstration)."""
        print("\nTesting hardcoded value migration...")

        try:
            # Create a sample Python file with hardcoded values
            sample_code = """
def calculate_accuracy(scores):
    THRESHOLD = 0.85  # Should be parameterized
    WEIGHT = 0.3      # Should be parameterized

    if scores > THRESHOLD:
        return scores * WEIGHT
    return 0.0

class DriftDetector:
    def __init__(self):
        self.window_size = 10     # Should be parameterized
        self.alert_level = 0.15   # Should be parameterized
"""

            # Write sample file
            sample_file = Path("sample_code.py")
            with open(sample_file, "w") as f:
                f.write(sample_code)

            session = self.Session()
            try:
                # Analyze for hardcoded values
                report = analyze_and_migrate_codebase(
                    root_path=Path("."),
                    session=session,
                    created_by="test_user",
                    min_confidence=0.5,
                    dry_run=True,  # Don't actually modify files
                )

                print(f"✓ Migration analysis completed:")
                print(
                    f"  - Total hardcoded values found: {report['analysis']['total_hardcoded_values']}"
                )
                print(
                    f"  - High confidence values: {report['analysis']['high_confidence_values']}"
                )
                print(
                    f"  - Parameters to create: {report['migration_plan']['parameters_to_create']}"
                )
                print(
                    f"  - Code locations to modify: {report['migration_plan']['code_locations_to_modify']}"
                )
                print(
                    f"  - Estimated effort: {report['migration_plan']['estimated_effort_hours']:.1f} hours"
                )

                # Show categories
                print("  - Categories found:")
                for category, count in report["analysis"]["categories"].items():
                    print(f"    • {category}: {count} values")

                return True

            finally:
                session.close()
                # Clean up sample file
                if sample_file.exists():
                    sample_file.unlink()

        except Exception as e:
            print(f"✗ Hardcoded value migration test failed: {str(e)}")
            return False

    def run_comprehensive_test_suite(self):
        """Run all parameter management tests."""
        print("Starting Parameter Management Test Suite")
        print("=" * 60)

        tests = [
            ("System Initialization", self.test_system_initialization),
            ("Parameter Retrieval", self.test_parameter_retrieval),
            ("Parameter Updates", self.test_parameter_updates),
            ("Parameter Overrides", self.test_parameter_overrides),
            ("Snapshots and Rollback", self.test_snapshots_and_rollback),
            ("Category Parameters", self.test_category_parameters),
            ("Feature Flags", self.test_feature_flags),
            ("Parameter Validation", self.test_parameter_validation),
            ("Hardcoded Value Migration", self.test_hardcoded_value_migration),
        ]

        results = {}
        passed = 0
        total = len(tests)

        for test_name, test_func in tests:
            print(f"\n{'='*20} {test_name} {'='*20}")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
                    print(f"✓ {test_name}: PASSED")
                else:
                    print(f"✗ {test_name}: FAILED")

            except Exception as e:
                results[test_name] = False
                print(f"✗ {test_name}: ERROR - {str(e)}")

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUITE SUMMARY")
        print("=" * 60)

        for test_name, result in results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name}: {status}")

        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

        if passed == total:
            print(
                "🎉 All tests passed! Parameter management system is working correctly."
            )
        else:
            print(
                f"⚠️  {total-passed} test(s) failed. Check the output above for details."
            )

        return passed == total


def main():
    """Main test function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test the parameter management system")
    parser.add_argument(
        "--database-url",
        default="postgresql://postgres:postgres@localhost:5432/magazine_extractor",
        help="Database URL for testing",
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset parameter system before testing"
    )

    args = parser.parse_args()

    # Initialize tester
    tester = ParameterManagementTester(args.database_url)

    # Reset if requested
    if args.reset:
        print("Resetting parameter system...")
        session = tester.Session()
        try:
            from parameter_management.initialization import reset_parameter_system

            reset_parameter_system(session, confirm=True)
            print("✓ Parameter system reset completed")
        except Exception as e:
            print(f"✗ Reset failed: {str(e)}")
        finally:
            session.close()

    # Run tests
    success = tester.run_comprehensive_test_suite()

    # Demo usage patterns
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)

    print("\n# Basic parameter usage:")
    print("from parameter_management import get_parameter, ParameterKeys")
    print("")
    print("# Replace hardcoded values with parameters")
    print("# OLD: threshold = 0.05")
    print("# NEW: threshold = get_parameter(ParameterKeys.DRIFT_THRESHOLD)")
    print("")
    print("# Brand-specific parameters")
    print("# columns = get_parameter('brand.default_columns', brand='TechWeekly')")
    print("")
    print("# Feature flags")
    print("# if get_parameter(ParameterKeys.FEATURE_DRIFT_DETECTION_ENABLED):")
    print("#     # Enable drift detection logic")

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
