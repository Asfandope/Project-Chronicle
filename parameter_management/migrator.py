"""
Parameter migration utilities for converting hardcoded values to managed parameters.

This module provides tools to identify hardcoded values in the codebase
and migrate them to the centralized parameter management system.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy.orm import Session

from .models import ParameterType
from .service import ParameterService

logger = logging.getLogger(__name__)


@dataclass
class HardcodedValue:
    """Represents a hardcoded value found in code."""

    file_path: str
    line_number: int
    column: int
    value: Any
    value_type: str
    context: str  # The surrounding code context
    suggested_parameter_key: str
    suggested_category: str
    confidence: float  # How confident we are this should be a parameter


@dataclass
class MigrationPlan:
    """Plan for migrating hardcoded values to parameters."""

    hardcoded_values: List[HardcodedValue]
    parameter_definitions: List[Dict[str, Any]]
    code_replacements: List[Dict[str, Any]]
    estimated_effort_hours: float


class CodeAnalyzer:
    """Analyzes code to find hardcoded values that should be parameters."""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".CodeAnalyzer")

        # Patterns that suggest a value should be parameterized
        self.parameter_indicators = {
            "threshold": r"\b(threshold|limit|max|min|cutoff|boundary)\b",
            "accuracy": r"\b(accuracy|precision|recall|f1|score)\b",
            "drift": r"\b(drift|degradation|drop|deviation)\b",
            "model": r"\b(model|algorithm|engine|predictor)\b",
            "processing": r"\b(batch_size|timeout|retry|interval)\b",
            "ui": r"\b(title|label|message|template|format)\b",
        }

        # File patterns to analyze
        self.python_files = ["**/*.py"]
        self.config_files = ["**/*.json", "**/*.yaml", "**/*.yml", "**/*.toml"]

        # Values to ignore (too generic or not suitable for parameters)
        self.ignore_values = {
            0,
            1,
            -1,
            True,
            False,
            None,
            "",
            "utf-8",
            "localhost",
            "http",
            "https",
        }

    def analyze_codebase(self, root_path: Path) -> List[HardcodedValue]:
        """Analyze entire codebase for hardcoded values."""

        hardcoded_values = []

        # Analyze Python files
        for pattern in self.python_files:
            for file_path in root_path.glob(pattern):
                if self._should_analyze_file(file_path):
                    values = self._analyze_python_file(file_path)
                    hardcoded_values.extend(values)

        # Analyze configuration files
        for pattern in self.config_files:
            for file_path in root_path.glob(pattern):
                if self._should_analyze_file(file_path):
                    values = self._analyze_config_file(file_path)
                    hardcoded_values.extend(values)

        # Sort by confidence score
        hardcoded_values.sort(key=lambda x: x.confidence, reverse=True)

        self.logger.info(
            f"Found {len(hardcoded_values)} potential parameters in codebase"
        )
        return hardcoded_values

    def _should_analyze_file(self, file_path: Path) -> bool:
        """Check if file should be analyzed."""

        # Skip common directories that shouldn't be analyzed
        skip_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            ".venv",
            "venv",
            "env",
            "build",
            "dist",
        }

        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            return False

        # Skip test files (they often have hardcoded values that are OK)
        if "test" in file_path.name.lower():
            return False

        # Skip migration files themselves
        if "migration" in file_path.name.lower():
            return False

        return True

    def _analyze_python_file(self, file_path: Path) -> List[HardcodedValue]:
        """Analyze a Python file for hardcoded values."""

        hardcoded_values = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                tree = ast.parse(content)

            analyzer = PythonASTAnalyzer(file_path, content)
            analyzer.visit(tree)
            hardcoded_values.extend(analyzer.hardcoded_values)

        except Exception as e:
            self.logger.warning(f"Error analyzing {file_path}: {str(e)}")

        return hardcoded_values

    def _analyze_config_file(self, file_path: Path) -> List[HardcodedValue]:
        """Analyze a configuration file for hardcoded values."""

        hardcoded_values = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_path.suffix.lower() == ".json":
                data = json.loads(content)
                values = self._extract_from_json(data, file_path, content)
                hardcoded_values.extend(values)

        except Exception as e:
            self.logger.warning(f"Error analyzing config file {file_path}: {str(e)}")

        return hardcoded_values

    def _extract_from_json(
        self, data: Any, file_path: Path, content: str, path: str = ""
    ) -> List[HardcodedValue]:
        """Extract hardcoded values from JSON data."""

        hardcoded_values = []

        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key

                if self._is_likely_parameter(key, value):
                    # Try to find line number in content
                    line_number = self._find_line_number_in_content(content, str(value))

                    hardcoded_value = HardcodedValue(
                        file_path=str(file_path),
                        line_number=line_number,
                        column=0,
                        value=value,
                        value_type=type(value).__name__,
                        context=f"{key}: {value}",
                        suggested_parameter_key=self._suggest_parameter_key(
                            key, current_path
                        ),
                        suggested_category=self._suggest_category(key),
                        confidence=self._calculate_confidence(key, value),
                    )

                    hardcoded_values.append(hardcoded_value)

                # Recurse into nested structures
                nested_values = self._extract_from_json(
                    value, file_path, content, current_path
                )
                hardcoded_values.extend(nested_values)

        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                nested_values = self._extract_from_json(
                    item, file_path, content, current_path
                )
                hardcoded_values.extend(nested_values)

        return hardcoded_values

    def _is_likely_parameter(self, key: str, value: Any) -> bool:
        """Check if a key-value pair is likely to be a parameter."""

        # Skip if value is in ignore list
        if value in self.ignore_values:
            return False

        # Check for parameter indicators in key name
        key_lower = key.lower()

        for category, pattern in self.parameter_indicators.items():
            if re.search(pattern, key_lower):
                return True

        # Check for specific patterns
        if any(
            pattern in key_lower
            for pattern in [
                "weight",
                "factor",
                "ratio",
                "rate",
                "size",
                "count",
                "duration",
            ]
        ):
            return True

        return False

    def _suggest_parameter_key(self, key: str, path: str) -> str:
        """Suggest a parameter key based on the original key and path."""

        # Clean up the path to create a parameter key
        key_parts = []

        # Add path components
        path_parts = (
            path.replace(".", "_").replace("[", "_").replace("]", "").split("_")
        )
        key_parts.extend([part for part in path_parts if part])

        # Join with underscores and clean up
        suggested_key = "_".join(key_parts).lower()
        suggested_key = re.sub(r"[^a-z0-9_]", "_", suggested_key)
        suggested_key = re.sub(r"_+", "_", suggested_key)
        suggested_key = suggested_key.strip("_")

        return suggested_key

    def _suggest_category(self, key: str) -> str:
        """Suggest a parameter category based on the key."""

        key_lower = key.lower()

        for category, pattern in self.parameter_indicators.items():
            if re.search(pattern, key_lower):
                return category

        # Default categories based on common patterns
        if any(word in key_lower for word in ["ui", "display", "format", "template"]):
            return "ui"
        elif any(
            word in key_lower for word in ["process", "batch", "timeout", "retry"]
        ):
            return "processing"
        elif any(word in key_lower for word in ["model", "algorithm", "prediction"]):
            return "model"
        else:
            return "general"

    def _calculate_confidence(self, key: str, value: Any) -> float:
        """Calculate confidence that this should be a parameter."""

        confidence = 0.0
        key_lower = key.lower()

        # Higher confidence for parameter-like names
        for category, pattern in self.parameter_indicators.items():
            if re.search(pattern, key_lower):
                confidence += 0.3

        # Higher confidence for numeric values
        if isinstance(value, (int, float)) and value not in self.ignore_values:
            confidence += 0.2

        # Higher confidence for thresholds and limits
        if any(word in key_lower for word in ["threshold", "limit", "max", "min"]):
            confidence += 0.3

        # Higher confidence for configuration-like values
        if any(word in key_lower for word in ["config", "setting", "param", "option"]):
            confidence += 0.2

        return min(1.0, confidence)

    def _find_line_number_in_content(self, content: str, search_value: str) -> int:
        """Find approximate line number where value appears in content."""

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if search_value in line:
                return i + 1

        return 1  # Default to line 1 if not found


class PythonASTAnalyzer(ast.NodeVisitor):
    """AST visitor to find hardcoded values in Python code."""

    def __init__(self, file_path: Path, content: str):
        self.file_path = file_path
        self.content = content
        self.lines = content.split("\n")
        self.hardcoded_values = []

        # Context tracking
        self.current_function = None
        self.current_class = None

    def visit_Constant(self, node):
        """Visit constant values (literals)."""

        # Skip certain types of constants
        if self._should_skip_constant(node.value):
            self.generic_visit(node)
            return

        # Get context information
        context = self._get_context(node.lineno)
        confidence = self._calculate_confidence_for_constant(node.value, context)

        if confidence > 0.3:  # Only include likely parameters
            hardcoded_value = HardcodedValue(
                file_path=str(self.file_path),
                line_number=node.lineno,
                column=node.col_offset,
                value=node.value,
                value_type=type(node.value).__name__,
                context=context,
                suggested_parameter_key=self._suggest_key_from_context(context),
                suggested_category=self._suggest_category_from_context(context),
                confidence=confidence,
            )

            self.hardcoded_values.append(hardcoded_value)

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Track current function context."""
        old_function = self.current_function
        self.current_function = node.name

        self.generic_visit(node)

        self.current_function = old_function

    def visit_ClassDef(self, node):
        """Track current class context."""
        old_class = self.current_class
        self.current_class = node.name

        self.generic_visit(node)

        self.current_class = old_class

    def _should_skip_constant(self, value: Any) -> bool:
        """Check if constant should be skipped."""

        # Skip common values that are rarely parameters
        skip_values = {
            0,
            1,
            -1,
            2,
            True,
            False,
            None,
            "",
            " ",
            "\n",
            "\t",
            "utf-8",
            "ascii",
            "localhost",
            "127.0.0.1",
            "GET",
            "POST",
            "PUT",
            "DELETE",
            "http",
            "https",
        }

        if value in skip_values:
            return True

        # Skip very long strings (likely not parameters)
        if isinstance(value, str) and len(value) > 100:
            return True

        # Skip strings that look like file paths
        if isinstance(value, str) and ("/" in value or "\\" in value):
            return True

        return False

    def _get_context(self, line_number: int) -> str:
        """Get code context around a line number."""

        if 1 <= line_number <= len(self.lines):
            line = self.lines[line_number - 1].strip()

            # Include function/class context if available
            context_parts = []

            if self.current_class:
                context_parts.append(f"class {self.current_class}")

            if self.current_function:
                context_parts.append(f"def {self.current_function}")

            context_parts.append(line)

            return " | ".join(context_parts)

        return ""

    def _calculate_confidence_for_constant(self, value: Any, context: str) -> float:
        """Calculate confidence that a constant should be a parameter."""

        confidence = 0.0
        context_lower = context.lower()

        # Higher confidence for numeric values in threshold-like contexts
        if isinstance(value, (int, float)):
            if any(
                word in context_lower
                for word in ["threshold", "limit", "max", "min", "cutoff", "boundary"]
            ):
                confidence += 0.4

            # Values between 0 and 1 are often thresholds
            if 0 < value < 1:
                confidence += 0.3

            # Common configuration values
            if value in [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 0.9, 0.95]:
                confidence += 0.2

        # Higher confidence for strings that look like configuration
        elif isinstance(value, str):
            if any(
                word in context_lower
                for word in ["template", "format", "message", "prompt", "query"]
            ):
                confidence += 0.4

            # API endpoints, error messages, etc.
            if value.startswith(("/", "http", "Error:", "Warning:")):
                confidence += 0.3

        # Context-based confidence
        if any(
            word in context_lower
            for word in ["config", "setting", "param", "option", "default"]
        ):
            confidence += 0.2

        return min(1.0, confidence)

    def _suggest_key_from_context(self, context: str) -> str:
        """Suggest parameter key from code context."""

        # Extract variable names and meaningful words
        words = re.findall(r"\b[a-z_][a-z0-9_]*\b", context.lower())

        # Filter out common words
        meaningful_words = [
            word
            for word in words
            if word
            not in {"def", "class", "if", "for", "in", "and", "or", "not", "is", "the"}
        ]

        # Take first few meaningful words
        key_parts = meaningful_words[:3]

        if not key_parts:
            return "unknown_parameter"

        return "_".join(key_parts)

    def _suggest_category_from_context(self, context: str) -> str:
        """Suggest parameter category from code context."""

        context_lower = context.lower()

        # Category patterns
        if any(
            word in context_lower
            for word in ["threshold", "accuracy", "score", "metric"]
        ):
            return "accuracy"
        elif any(word in context_lower for word in ["drift", "degradation", "drop"]):
            return "drift"
        elif any(
            word in context_lower for word in ["model", "predictor", "classifier"]
        ):
            return "model"
        elif any(
            word in context_lower for word in ["batch", "timeout", "retry", "interval"]
        ):
            return "processing"
        elif any(
            word in context_lower
            for word in ["message", "template", "format", "prompt"]
        ):
            return "ui"
        else:
            return "general"


class ParameterMigrator:
    """Migrates hardcoded values to centralized parameters."""

    def __init__(self, parameter_service: ParameterService):
        self.parameter_service = parameter_service
        self.logger = logging.getLogger(__name__ + ".ParameterMigrator")

    def create_migration_plan(
        self, hardcoded_values: List[HardcodedValue], min_confidence: float = 0.5
    ) -> MigrationPlan:
        """Create a migration plan from hardcoded values."""

        # Filter by confidence
        filtered_values = [
            hv for hv in hardcoded_values if hv.confidence >= min_confidence
        ]

        # Group by suggested parameter key to deduplicate
        parameter_groups = {}
        for hv in filtered_values:
            key = hv.suggested_parameter_key
            if key not in parameter_groups:
                parameter_groups[key] = []
            parameter_groups[key].append(hv)

        # Create parameter definitions
        parameter_definitions = []
        code_replacements = []

        for param_key, hardcoded_values_group in parameter_groups.items():
            # Use the highest confidence value as the representative
            representative = max(hardcoded_values_group, key=lambda x: x.confidence)

            # Create parameter definition
            param_def = {
                "key": param_key,
                "name": param_key.replace("_", " ").title(),
                "description": f"Parameter migrated from hardcoded value: {representative.value}",
                "parameter_type": self._infer_parameter_type(representative),
                "category": representative.suggested_category,
                "data_type": self._infer_data_type(representative.value),
                "default_value": representative.value,
                "original_locations": [
                    {
                        "file": hv.file_path,
                        "line": hv.line_number,
                        "context": hv.context,
                    }
                    for hv in hardcoded_values_group
                ],
            }

            parameter_definitions.append(param_def)

            # Create code replacement instructions
            for hv in hardcoded_values_group:
                replacement = {
                    "file_path": hv.file_path,
                    "line_number": hv.line_number,
                    "column": hv.column,
                    "old_value": hv.value,
                    "new_code": f"parameter_service.get_parameter_value(session, '{param_key}')",
                    "import_needed": "from parameter_management import parameter_service",
                }

                code_replacements.append(replacement)

        # Estimate effort (rough calculation)
        estimated_hours = (
            len(parameter_definitions) * 0.5 + len(code_replacements) * 0.1
        )

        return MigrationPlan(
            hardcoded_values=filtered_values,
            parameter_definitions=parameter_definitions,
            code_replacements=code_replacements,
            estimated_effort_hours=estimated_hours,
        )

    def execute_migration_plan(
        self,
        session: Session,
        migration_plan: MigrationPlan,
        created_by: str,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Execute the migration plan."""

        results = {
            "parameters_created": 0,
            "parameters_failed": 0,
            "code_files_modified": 0,
            "errors": [],
            "dry_run": dry_run,
        }

        # Create parameters
        for param_def in migration_plan.parameter_definitions:
            try:
                if not dry_run:
                    parameter = self.parameter_service.create_parameter(
                        session=session,
                        key=param_def["key"],
                        name=param_def["name"],
                        description=param_def["description"],
                        parameter_type=ParameterType(param_def["parameter_type"]),
                        category=param_def["category"],
                        data_type=param_def["data_type"],
                        default_value=param_def["default_value"],
                        created_by=created_by,
                    )

                    self.logger.info(f"Created parameter: {param_def['key']}")

                results["parameters_created"] += 1

            except Exception as e:
                error_msg = f"Failed to create parameter {param_def['key']}: {str(e)}"
                results["errors"].append(error_msg)
                results["parameters_failed"] += 1
                self.logger.error(error_msg)

        # Modify code files (group by file)
        files_to_modify = {}
        for replacement in migration_plan.code_replacements:
            file_path = replacement["file_path"]
            if file_path not in files_to_modify:
                files_to_modify[file_path] = []
            files_to_modify[file_path].append(replacement)

        for file_path, replacements in files_to_modify.items():
            try:
                if not dry_run:
                    self._modify_code_file(file_path, replacements)

                results["code_files_modified"] += 1
                self.logger.info(f"Modified file: {file_path}")

            except Exception as e:
                error_msg = f"Failed to modify file {file_path}: {str(e)}"
                results["errors"].append(error_msg)
                self.logger.error(error_msg)

        return results

    def _infer_parameter_type(self, hardcoded_value: HardcodedValue) -> str:
        """Infer parameter type from hardcoded value."""

        context_lower = hardcoded_value.context.lower()

        if any(
            word in context_lower
            for word in ["threshold", "accuracy", "score", "metric"]
        ):
            return ParameterType.THRESHOLD.value
        elif any(
            word in context_lower
            for word in ["prompt", "template", "message", "format"]
        ):
            return ParameterType.PROMPT.value
        elif any(word in context_lower for word in ["rule", "condition", "policy"]):
            return ParameterType.RULE.value
        elif any(word in context_lower for word in ["model", "algorithm", "predictor"]):
            return ParameterType.MODEL_CONFIG.value
        elif any(
            word in context_lower for word in ["enable", "disable", "flag", "feature"]
        ):
            return ParameterType.FEATURE_FLAG.value
        elif any(
            word in context_lower for word in ["batch", "timeout", "retry", "interval"]
        ):
            return ParameterType.PROCESSING_CONFIG.value
        else:
            return ParameterType.THRESHOLD.value  # Default

    def _infer_data_type(self, value: Any) -> str:
        """Infer data type from value."""

        if isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "json"

    def _modify_code_file(
        self, file_path: str, replacements: List[Dict[str, Any]]
    ) -> None:
        """Modify a code file with parameter replacements."""

        # Read file content
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Sort replacements by line number (descending) to avoid offset issues
        replacements.sort(key=lambda x: x["line_number"], reverse=True)

        # Track if we need to add imports
        needs_import = any(r.get("import_needed") for r in replacements)
        import_line = None

        if needs_import:
            import_line = "from parameter_management import parameter_service\n"

            # Find where to insert import (after existing imports)
            insert_pos = 0
            for i, line in enumerate(lines):
                if line.strip().startswith(("import ", "from ")):
                    insert_pos = i + 1
                elif line.strip() == "" and insert_pos > 0:
                    continue
                elif insert_pos > 0:
                    break

            lines.insert(insert_pos, import_line)

        # Apply replacements
        for replacement in replacements:
            line_idx = replacement["line_number"] - 1

            if 0 <= line_idx < len(lines):
                old_line = lines[line_idx]
                old_value_str = repr(replacement["old_value"])

                # Replace the hardcoded value with parameter call
                new_line = old_line.replace(old_value_str, replacement["new_code"])
                lines[line_idx] = new_line

        # Write modified content back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)


def analyze_and_migrate_codebase(
    root_path: Path,
    session: Session,
    created_by: str,
    min_confidence: float = 0.7,
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Complete workflow to analyze and migrate hardcoded values."""

    logger.info(f"Starting codebase analysis and migration (dry_run={dry_run})")

    # Step 1: Analyze codebase
    analyzer = CodeAnalyzer()
    hardcoded_values = analyzer.analyze_codebase(root_path)

    # Step 2: Create migration plan
    migrator = ParameterMigrator(ParameterService())
    migration_plan = migrator.create_migration_plan(hardcoded_values, min_confidence)

    # Step 3: Execute migration
    results = migrator.execute_migration_plan(
        session, migration_plan, created_by, dry_run
    )

    # Step 4: Generate report
    report = {
        "analysis": {
            "total_hardcoded_values": len(hardcoded_values),
            "high_confidence_values": len(
                [hv for hv in hardcoded_values if hv.confidence >= min_confidence]
            ),
            "categories": {},
        },
        "migration_plan": {
            "parameters_to_create": len(migration_plan.parameter_definitions),
            "code_locations_to_modify": len(migration_plan.code_replacements),
            "estimated_effort_hours": migration_plan.estimated_effort_hours,
        },
        "execution_results": results,
        "recommendations": _generate_migration_recommendations(migration_plan, results),
    }

    # Count by category
    for hv in hardcoded_values:
        category = hv.suggested_category
        if category not in report["analysis"]["categories"]:
            report["analysis"]["categories"][category] = 0
        report["analysis"]["categories"][category] += 1

    logger.info(
        f"Migration analysis complete. Found {len(hardcoded_values)} potential parameters."
    )
    return report


def _generate_migration_recommendations(
    migration_plan: MigrationPlan, execution_results: Dict[str, Any]
) -> List[str]:
    """Generate recommendations based on migration results."""

    recommendations = []

    if execution_results["parameters_failed"] > 0:
        recommendations.append(
            "Review failed parameter creations and resolve conflicts before proceeding"
        )

    if execution_results["dry_run"]:
        recommendations.append(
            "Run migration with dry_run=False to apply changes after reviewing the plan"
        )

    if migration_plan.estimated_effort_hours > 8:
        recommendations.append(
            "Consider migrating parameters in phases due to high estimated effort"
        )

    if len(migration_plan.parameter_definitions) > 50:
        recommendations.append(
            "Large number of parameters detected - consider organizing into subcategories"
        )

    recommendations.extend(
        [
            "Test thoroughly after migration to ensure parameter resolution works correctly",
            "Update documentation to reflect new parameter-driven configuration",
            "Consider setting up monitoring for parameter usage and changes",
            "Create brand-specific overrides where different behavior is needed",
        ]
    )

    return recommendations
