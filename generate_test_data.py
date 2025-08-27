#!/usr/bin/env python3
"""
Script to generate synthetic test data for magazine extraction testing.

This script provides a command-line interface for generating comprehensive
test suites with 100+ variants per brand for testing the extraction pipeline.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from synthetic_data import (
    GenerationConfig,
    SyntheticDataError,
    SyntheticDataGenerator,
    create_comprehensive_test_suite,
    create_edge_case_test_suite,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("synthetic_data_generation.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for synthetic data generation."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic magazine test data with known ground truth",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate comprehensive test suite with 150 documents per brand
  python generate_test_data.py --output ./test_data --type comprehensive --count 150

  # Generate edge case focused test suite
  python generate_test_data.py --output ./edge_cases --type edge-cases --count 100

  # Generate test data for specific brand only
  python generate_test_data.py --output ./tech_only --brand TechWeekly --count 200

  # Validate setup without generating full suite
  python generate_test_data.py --output ./temp --validate-only
        """,
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output directory for generated test data",
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["comprehensive", "edge-cases", "custom"],
        default="comprehensive",
        help="Type of test suite to generate (default: comprehensive)",
    )

    parser.add_argument(
        "--count",
        "-c",
        type=int,
        default=150,
        help="Number of documents to generate per brand (default: 150)",
    )

    parser.add_argument(
        "--brand", "-b", type=str, help="Generate test data for specific brand only"
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate setup, do not generate full test suite",
    )

    parser.add_argument(
        "--pdf-dpi", type=int, default=300, help="DPI for PDF generation (default: 300)"
    )

    parser.add_argument(
        "--max-pages",
        type=int,
        default=8,
        help="Maximum pages per document (default: 8)",
    )

    parser.add_argument(
        "--edge-case-probability",
        type=float,
        default=0.3,
        help="Probability of edge cases (default: 0.3)",
    )

    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output verbosity"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Ensure output directory exists
        args.output.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {args.output.absolute()}")

        # Create generation configuration
        if args.type == "comprehensive":
            config = GenerationConfig.create_comprehensive_test(args.output)
        elif args.type == "edge-cases":
            config = GenerationConfig.create_edge_case_focused(args.output)
        else:  # custom
            config = GenerationConfig(
                output_directory=args.output,
                documents_per_brand=args.count,
                pages_per_document=(1, args.max_pages),
                edge_case_probability=args.edge_case_probability,
                pdf_dpi=args.pdf_dpi,
            )

        # Override document count if specified
        config.documents_per_brand = args.count

        logger.info(f"Generation configuration:")
        logger.info(f"  - Type: {args.type}")
        logger.info(f"  - Documents per brand: {config.documents_per_brand}")
        logger.info(f"  - Page range: {config.pages_per_document}")
        logger.info(f"  - Edge case probability: {config.edge_case_probability}")
        logger.info(f"  - PDF DPI: {config.pdf_dpi}")

        # Initialize generator
        generator = SyntheticDataGenerator(config)

        # Validate setup
        logger.info("Validating generation setup...")
        validation = generator.validate_generation_setup()

        if not validation["test_generation"]["success"]:
            logger.error("Setup validation failed!")
            logger.error(
                f"Error: {validation['test_generation'].get('error', 'Unknown error')}"
            )
            return 1

        logger.info("Setup validation passed ✓")

        if args.validate_only:
            logger.info(
                "Validation complete. Exiting without generating full test suite."
            )
            return 0

        # Generate test suite
        if args.brand:
            logger.info(f"Generating test suite for brand: {args.brand}")
            suite = generator.generate_brand_focused_suite(args.brand, args.count)
        else:
            if args.type == "comprehensive":
                logger.info("Generating comprehensive test suite...")
                suite = create_comprehensive_test_suite(args.output, args.count)
            elif args.type == "edge-cases":
                logger.info("Generating edge case focused test suite...")
                suite = create_edge_case_test_suite(args.output, args.count)
            else:
                logger.info("Generating custom test suite...")
                suite = generator.generate_complete_test_suite()

        # Print summary
        summary = suite.get_summary()

        print("\n" + "=" * 60)
        print("SYNTHETIC DATA GENERATION SUMMARY")
        print("=" * 60)
        print(f"Suite: {summary['suite_name']}")
        print(f"Total documents: {summary['total_documents']}")
        print(f"Successful generations: {summary['successful_generations']}")
        print(f"Success rate: {summary['success_rate']:.2%}")

        if summary["generation_duration"]:
            print(f"Generation time: {summary['generation_duration']:.2f} seconds")

        print(f"\nBrands generated:")
        for brand, count in summary["documents_per_brand"].items():
            print(f"  - {brand}: {count} documents")

        print(f"\nComplexity distribution:")
        for complexity, count in summary["complexity_distribution"].items():
            print(f"  - {complexity}: {count} documents")

        print(f"\nEdge cases included:")
        for edge_case, count in summary["edge_case_distribution"].items():
            print(f"  - {edge_case}: {count} instances")

        print(f"\nOutput directory: {args.output.absolute()}")
        print("=" * 60)

        # Save detailed summary
        summary_file = args.output / "generation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Generation complete! Detailed summary saved to: {summary_file}")

        return 0

    except SyntheticDataError as e:
        logger.error(f"Synthetic data generation error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Generation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
