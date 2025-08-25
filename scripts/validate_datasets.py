#!/usr/bin/env python3
"""
Script to validate gold standard datasets.

Usage:
    python scripts/validate_datasets.py [brand_name]
    
If no brand name is provided, validates all brands.
"""

import sys
import argparse
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_management.schema_validator import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate gold standard datasets")
    parser.add_argument("brand", nargs="?", help="Brand to validate (validates all if not specified)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = DatasetValidator()
    
    if args.brand:
        # Validate specific brand
        brands = [args.brand]
    else:
        # Validate all brands
        brands = ['economist', 'time', 'newsweek', 'vogue']
    
    all_passed = True
    
    for brand in brands:
        print(f"\n=== Validating {brand} ===")
        
        try:
            report = validator.validate_brand_dataset(brand)
            
            print(f"Files: {report.total_files}")
            print(f"Valid: {report.valid_files}")
            print(f"Validation Rate: {report.validation_rate:.1f}%")
            
            if report.average_quality_score > 0:
                print(f"Avg Quality Score: {report.average_quality_score:.3f}")
            
            if args.verbose and report.coverage_metrics:
                print(f"Coverage: PDFs={report.coverage_metrics.get('pdf_count', 0)}, "
                      f"XML={report.coverage_metrics.get('xml_count', 0)}")
            
            if report.recommendations:
                print("Top Recommendations:")
                for rec in report.recommendations[:3]:
                    print(f"  - {rec}")
            
            if report.validation_rate < 100:
                all_passed = False
                print("❌ Validation issues found")
            else:
                print("✅ Validation passed")
                
        except Exception as e:
            print(f"❌ Validation failed with error: {e}")
            all_passed = False
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    if len(brands) > 1:
        print(f"\n=== Summary ===")
        if all_passed:
            print("✅ All datasets passed validation")
        else:
            print("❌ Some datasets have validation issues")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)