#!/usr/bin/env python3
"""
Configuration CLI tool for validating and managing brand configurations.
Single source of truth for configuration operations.
"""

import json
import sys
from pathlib import Path

import click
import structlog
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.config.manager import config_manager
from shared.config.validator import config_validator

logger = structlog.get_logger()


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """Configuration management CLI for Magazine PDF Extractor."""
    if verbose:
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
        )


@cli.command()
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
def validate_all(format):
    """Validate all configuration files."""
    click.echo("🔍 Validating all configurations...")

    results = config_validator.validate_all_configs()

    if format == "json":
        click.echo(json.dumps(results, indent=2, default=str))
    else:
        # YAML output
        _print_validation_results(results)

    if results["overall_status"] != "pass":
        sys.exit(1)


@cli.command()
@click.argument("brand_name")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
def validate_brand(brand_name, format):
    """Validate a specific brand configuration."""
    click.echo(f"🔍 Validating configuration for '{brand_name}'...")

    result = config_validator.validate_brand_config(brand_name)

    if format == "json":
        click.echo(json.dumps(result, indent=2, default=str))
    else:
        _print_brand_validation_result(result)

    if not result["valid"]:
        sys.exit(1)


@cli.command()
def list_brands():
    """List all available brand configurations."""
    brands = config_manager.list_configured_brands()

    if not brands:
        click.echo("❌ No valid brand configurations found.")
        sys.exit(1)

    click.echo("📋 Available brand configurations:")
    for brand in brands:
        try:
            config = config_manager.get_brand_config(brand)
            description = config.description or "No description"
            version = config.version
            click.echo(f"  ✅ {brand} (v{version}) - {description}")
        except Exception as e:
            click.echo(f"  ❌ {brand} - Error: {e}")


@cli.command()
@click.argument("brand_name")
@click.option("--section", "-s", help="Show specific section only")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
def show_config(brand_name, section, format):
    """Show configuration for a specific brand."""
    try:
        config = config_manager.get_brand_config(brand_name)
        config_dict = config.dict()

        if section:
            if section not in config_dict:
                click.echo(f"❌ Section '{section}' not found in configuration.")
                click.echo(f"Available sections: {', '.join(config_dict.keys())}")
                sys.exit(1)
            config_dict = {section: config_dict[section]}

        if format == "json":
            click.echo(json.dumps(config_dict, indent=2, default=str))
        else:
            click.echo(yaml.dump(config_dict, default_flow_style=False))

    except Exception as e:
        click.echo(f"❌ Error loading configuration for '{brand_name}': {e}")
        sys.exit(1)


@cli.command()
@click.argument("brand_name")
@click.option("--output", "-o", help="Output file path")
def generate_template(brand_name, output):
    """Generate a configuration template for a new brand."""
    try:
        template = config_validator.generate_config_template(brand_name)

        if output:
            with open(output, "w") as f:
                f.write(template)
            click.echo(f"✅ Template generated: {output}")
        else:
            click.echo(template)

    except Exception as e:
        click.echo(f"❌ Error generating template: {e}")
        sys.exit(1)


@cli.command()
@click.argument("brand_name")
@click.argument("field")
def get_threshold(brand_name, field):
    """Get confidence threshold for a specific field."""
    try:
        threshold = config_manager.get_confidence_threshold(brand_name, field)
        click.echo(f"{brand_name}.{field}: {threshold}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
@click.argument("brand_name")
@click.argument("feature")
def check_feature(brand_name, feature):
    """Check if a custom feature is enabled for a brand."""
    try:
        enabled = config_manager.is_feature_enabled(brand_name, feature)
        status = "✅ ENABLED" if enabled else "❌ DISABLED"
        click.echo(f"{brand_name}.{feature}: {status}")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        sys.exit(1)


@cli.command()
def schema():
    """Generate JSON schema for brand configuration."""
    from shared.models.brand_config import generate_config_schema

    schema = generate_config_schema()
    click.echo(json.dumps(schema, indent=2))


@cli.command()
def check_consistency():
    """Check configuration consistency across all brands."""
    click.echo("🔍 Checking configuration consistency...")

    brands = config_manager.list_configured_brands()
    if len(brands) < 2:
        click.echo("⚠️  Need at least 2 brands to check consistency.")
        return

    # Compare configurations
    configs = {}
    for brand in brands:
        try:
            configs[brand] = config_manager.get_brand_config(brand)
        except Exception as e:
            click.echo(f"❌ Could not load {brand}: {e}")
            continue

    # Check threshold consistency
    title_thresholds = {}
    body_thresholds = {}

    for brand, config in configs.items():
        if config.confidence_overrides.title:
            title_thresholds[brand] = config.confidence_overrides.title
        if config.confidence_overrides.body:
            body_thresholds[brand] = config.confidence_overrides.body

    if title_thresholds:
        mean_title = sum(title_thresholds.values()) / len(title_thresholds)
        click.echo(f"📊 Title threshold mean: {mean_title:.3f}")

        for brand, threshold in title_thresholds.items():
            if abs(threshold - mean_title) > 0.1:
                click.echo(
                    f"⚠️  {brand} title threshold ({threshold}) deviates significantly from mean"
                )

    # Check version consistency
    versions = set(config.version for config in configs.values())
    if len(versions) > 1:
        click.echo(f"⚠️  Multiple configuration versions found: {versions}")
    else:
        click.echo(f"✅ All configurations use version: {list(versions)[0]}")

    click.echo("✅ Consistency check complete.")


def _print_validation_results(results):
    """Print validation results in human-readable format."""
    status_icon = "✅" if results["overall_status"] == "pass" else "❌"
    click.echo(f"{status_icon} Overall Status: {results['overall_status'].upper()}")

    # XML Schema
    xml_result = results["xml_schema"]
    xml_icon = "✅" if xml_result["valid"] else "❌"
    click.echo(
        f"{xml_icon} XML Schema: {'VALID' if xml_result['valid'] else 'INVALID'}"
    )

    if xml_result.get("errors"):
        for error in xml_result["errors"]:
            click.echo(f"    ❌ {error}")

    # Brand Configurations
    brand_results = results["brand_configs"]
    click.echo(f"\n📋 Brand Configurations ({len(brand_results)} total):")

    for brand, result in brand_results.items():
        icon = "✅" if result["valid"] else "❌"
        status = "VALID" if result["valid"] else "INVALID"
        click.echo(f"  {icon} {brand}: {status}")

        if result.get("errors"):
            for error in result["errors"]:
                click.echo(f"      ❌ {error}")

        if result.get("warnings"):
            for warning in result["warnings"]:
                click.echo(f"      ⚠️  {warning}")

    # Global warnings
    if results.get("warnings"):
        click.echo(f"\n⚠️  Global Warnings:")
        for warning in results["warnings"]:
            click.echo(f"  ⚠️  {warning}")


def _print_brand_validation_result(result):
    """Print brand validation result in human-readable format."""
    icon = "✅" if result["valid"] else "❌"
    status = "VALID" if result["valid"] else "INVALID"

    click.echo(f"{icon} {result['brand']}: {status}")
    click.echo(f"   File: {result['file_path']}")

    if result.get("errors"):
        click.echo("   ❌ Errors:")
        for error in result["errors"]:
            click.echo(f"      • {error}")

    if result.get("warnings"):
        click.echo("   ⚠️  Warnings:")
        for warning in result["warnings"]:
            click.echo(f"      • {warning}")

    if result["valid"] and result.get("config"):
        config = result["config"]
        click.echo(f"   📝 Version: {config.get('version', 'unknown')}")
        click.echo(f"   📄 Description: {config.get('description', 'No description')}")


if __name__ == "__main__":
    cli()
