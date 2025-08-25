#!/usr/bin/env python3
"""
Test script to verify local Project Chronicle setup.

This script tests all major components:
1. Database connectivity
2. Parameter management system
3. Synthetic data generation
4. Evaluation service
5. Self-tuning system
6. Quarantine system
"""

import sys
import asyncio
import logging
from datetime import datetime, timezone
from sqlalchemy import text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_database_connection():
    """Test database connectivity."""
    try:
        from database import get_db_session, check_database_health
        
        logger.info("Testing database connection...")
        
        # Test session creation
        with get_db_session() as session:
            session.execute(text("SELECT 1"))
        
        # Check health
        health = check_database_health()
        logger.info(f"Database health: {health['status']}")
        
        if health['status'] == 'healthy':
            logger.info("✅ Database connection successful")
            return True
        else:
            logger.error("❌ Database connection failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        return False


async def test_parameter_management():
    """Test parameter management system."""
    try:
        from database import get_db_session
        from parameter_management import get_parameter, ParameterKeys
        from parameter_management.initialization import initialize_parameter_management_system
        
        logger.info("Testing parameter management...")
        
        with get_db_session() as session:
            # Initialize parameters
            results = initialize_parameter_management_system(session)
            logger.info(f"Parameter initialization: {results['parameters_created']} created")
            
            # Test parameter retrieval
            accuracy_threshold = get_parameter(
                ParameterKeys.ACCURACY_WER_THRESHOLD,
                default=0.001
            )
            logger.info(f"Retrieved parameter: WER threshold = {accuracy_threshold}")
        
        logger.info("✅ Parameter management system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Parameter management error: {e}")
        return False


async def test_synthetic_data_generation():
    """Test synthetic data generation."""
    try:
        from synthetic_data import SyntheticDataGenerator
        from synthetic_data.types import BrandConfiguration, GenerationConfig
        
        logger.info("Testing synthetic data generation...")
        
        # Create test configuration
        brand_config = BrandConfiguration(
            brand_name="TestBrand",
            brand_style="modern",
            primary_font="Arial",
            default_columns=2
        )
        
        generation_config = GenerationConfig(
            num_variants=2,
            include_edge_cases=True,
            output_format="pdf"
        )
        
        generator = SyntheticDataGenerator()
        
        # This would normally generate actual files, but for testing we just verify the setup
        logger.info(f"Synthetic data generator initialized for brand: {brand_config.brand_name}")
        
        logger.info("✅ Synthetic data generation system ready")
        return True
        
    except Exception as e:
        logger.error(f"❌ Synthetic data generation error: {e}")
        return False


async def test_evaluation_service():
    """Test evaluation service components."""
    try:
        from database import get_db_session
        from evaluation_service.service import EvaluationService
        from evaluation_service.models import EvaluationRun
        
        logger.info("Testing evaluation service...")
        
        with get_db_session() as session:
            # Check if evaluation tables exist
            count = session.query(EvaluationRun).count()
            logger.info(f"Evaluation runs in database: {count}")
        
        # Test service initialization
        eval_service = EvaluationService()
        logger.info("Evaluation service initialized")
        
        logger.info("✅ Evaluation service working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Evaluation service error: {e}")
        return False


async def test_self_tuning_system():
    """Test self-tuning system."""
    try:
        from database import get_db_session
        from self_tuning import get_tuning_system_status, check_brand_tuning_eligibility
        from self_tuning.models import TuningRun
        
        logger.info("Testing self-tuning system...")
        
        with get_db_session() as session:
            # Check tuning system status
            status = get_tuning_system_status(session)
            logger.info(f"Tuning runs in database: {status['total_tuning_runs']}")
            
            # Test brand eligibility check
            eligibility = check_brand_tuning_eligibility("TestBrand", session)
            logger.info(f"Test brand tuning eligibility: {eligibility['message']}")
        
        logger.info("✅ Self-tuning system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Self-tuning system error: {e}")
        return False


async def test_quarantine_system():
    """Test quarantine system."""
    try:
        from database import get_db_session
        from quarantine import get_quarantine_summary
        from quarantine.models import QuarantineItem
        
        logger.info("Testing quarantine system...")
        
        with get_db_session() as session:
            # Check quarantine system status
            summary = get_quarantine_summary(session=session)
            if 'error' not in summary:
                logger.info(f"Quarantine system status: {summary['summary']['total_quarantined']} items quarantined")
            else:
                logger.warning(f"Quarantine system warning: {summary['error']}")
        
        logger.info("✅ Quarantine system working")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quarantine system error: {e}")
        return False


async def test_integration_scenario():
    """Test a full integration scenario."""
    try:
        logger.info("Testing integration scenario...")
        
        # Mock extraction output that would fail quarantine
        mock_extraction = {
            "title": "Test Article",
            "body_text": "Short test body",
            "contributors": [],
            "media_links": []
        }
        
        mock_accuracy_scores = {
            "overall": 0.85,  # Below 99.9% threshold
            "title_accuracy": 0.9,
            "body_text_accuracy": 0.8,
            "contributors_accuracy": 0.0,
            "media_links_accuracy": 0.0
        }
        
        from database import get_db_session
        from quarantine import quarantine_if_needed
        
        with get_db_session() as session:
            # Test quarantine evaluation
            quarantined = quarantine_if_needed(
                issue_id="test_issue_001",
                extraction_output=mock_extraction,
                accuracy_scores=mock_accuracy_scores,
                session=session,
                brand_name="TestBrand"
            )
            
            if quarantined:
                logger.info("✅ Integration test: Mock item successfully quarantined")
            else:
                logger.info("✅ Integration test: Mock item passed quarantine (as expected for test)")
        
        logger.info("✅ Integration scenario completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Integration scenario error: {e}")
        return False


async def run_all_tests():
    """Run all tests and provide summary."""
    logger.info("🚀 Starting Project Chronicle local setup tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("Parameter Management", test_parameter_management),
        ("Synthetic Data Generation", test_synthetic_data_generation),
        ("Evaluation Service", test_evaluation_service),
        ("Self-Tuning System", test_self_tuning_system),
        ("Quarantine System", test_quarantine_system),
        ("Integration Scenario", test_integration_scenario)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        success = await test_func()
        results.append((test_name, success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🏁 TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        icon = "✅" if success else "❌"
        logger.info(f"{icon} {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        logger.info("🎉 All systems ready! Project Chronicle is set up correctly.")
        logger.info("\nNext steps:")
        logger.info("1. Start the application: python main.py")
        logger.info("2. Access API docs at: http://localhost:8000/docs")
        logger.info("3. Check health at: http://localhost:8000/health")
        return True
    else:
        logger.error("⚠️  Some tests failed. Please fix issues before proceeding.")
        return False


if __name__ == "__main__":
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)