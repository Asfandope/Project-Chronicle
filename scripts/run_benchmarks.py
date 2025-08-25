#!/usr/bin/env python3
"""
Benchmark evaluation CLI for Project Chronicle.

Run performance benchmarks against gold standard datasets to measure
system readiness for production deployment.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_management.benchmarks import (
    run_brand_benchmark,
    run_all_brands_benchmark,
    BenchmarkEvaluator,
    AccuracyTargetRegistry
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks against gold standard datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --all                    # Run benchmarks for all brands
  %(prog)s economist                # Run benchmark for economist only
  %(prog)s --targets               # List all accuracy targets
  %(prog)s vogue --verbose         # Run vogue benchmark with detailed output
        """
    )
    
    parser.add_argument(
        "brand",
        nargs="?",
        choices=["economist", "time", "newsweek", "vogue"],
        help="Brand to benchmark (if not specified, use --all)"
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run benchmarks for all brands"
    )
    
    parser.add_argument(
        "--targets",
        action="store_true",
        help="List all accuracy targets and thresholds"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for detailed results (JSON format)"
    )
    
    args = parser.parse_args()
    
    if args.targets:
        print_accuracy_targets()
        return
    
    if args.all:
        print("🎯 Running comprehensive benchmark evaluation...")
        run_all_brands_benchmark()
    elif args.brand:
        print(f"🎯 Running benchmark evaluation for {args.brand}...")
        run_brand_benchmark(args.brand)
    else:
        print("❌ Please specify a brand or use --all flag")
        print("Run with --help for usage information")
        sys.exit(1)


def print_accuracy_targets():
    """Print all accuracy targets in a readable format."""
    registry = AccuracyTargetRegistry()
    targets = registry.list_all_targets()
    
    print("🎯 ACCURACY TARGETS FOR PROJECT CHRONICLE")
    print("=" * 60)
    
    for component_name, component_targets in targets.items():
        print(f"\n📊 {component_name.replace('_', ' ').title()}")
        print("-" * 40)
        
        for metric_name, target in component_targets.items():
            print(f"  • {metric_name}:")
            print(f"    Production:  {target['production_threshold']} {target['unit']}")
            print(f"    Acceptable:  {target['acceptable_threshold']} {target['unit']}")
            print(f"    Needs Work:  {target['improvement_threshold']} {target['unit']}")
            print(f"    Description: {target['description']}")
            print()


if __name__ == "__main__":
    main()