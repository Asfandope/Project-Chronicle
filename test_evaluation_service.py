#!/usr/bin/env python3
"""
Test script for the evaluation service.

This script demonstrates and tests the FastAPI evaluation service
with sample evaluation requests and drift detection.
"""

import json
import requests
import time
from datetime import datetime
from pathlib import Path

# Sample ground truth XML
SAMPLE_GROUND_TRUTH = """<?xml version="1.0" encoding="UTF-8"?>
<magazine_ground_truth version="1.0" generator="synthetic_data_generator">
  <document_metadata>
    <document_id>test_doc_001</document_id>
    <brand_name>TechWeekly</brand_name>
    <generation_timestamp>2024-01-15T10:30:00</generation_timestamp>
    <page_count>1</page_count>
    <page_dimensions width="612.0" height="792.0" units="points"/>
    <quality_metrics>
      <expected_accuracy>0.95</expected_accuracy>
      <difficult_elements>2</difficult_elements>
    </quality_metrics>
  </document_metadata>
  
  <articles>
    <article id="test_article_001">
      <title>AI Revolution in Healthcare</title>
      <article_type>feature</article_type>
      <page_range start="1" end="1"/>
      <contributors>
        <contributor role="author" name="Dr. Sarah Johnson" affiliation="Medical AI Institute"/>
      </contributors>
      <text_elements>
        <text_element id="title_001" type="title" page="1" reading_order="1">
          <bbox x0="72" y0="650" x1="540" y1="700"/>
          <content>AI Revolution in Healthcare</content>
          <font family="Helvetica" size="24" style="bold" align="left"/>
          <color r="0.0" g="0.0" b="0.0"/>
          <extraction_metadata confidence="0.95" difficulty="0.1" z_order="1"/>
        </text_element>
        <text_element id="body_001" type="paragraph" page="1" reading_order="2">
          <bbox x0="72" y0="400" x1="540" y1="640"/>
          <content>Artificial intelligence is transforming healthcare delivery through advanced diagnostic tools and personalized treatment recommendations. Machine learning algorithms can now analyze medical imaging with unprecedented accuracy.</content>
          <font family="Times New Roman" size="12" style="normal" align="left"/>
          <color r="0.0" g="0.0" b="0.0"/>
          <extraction_metadata confidence="0.90" difficulty="0.2" z_order="2"/>
        </text_element>
        <text_element id="byline_001" type="byline" page="1" reading_order="3">
          <bbox x0="72" y0="360" x1="300" y1="380"/>
          <content>By Dr. Sarah Johnson, Medical AI Institute</content>
          <font family="Arial" size="10" style="italic" align="left"/>
          <color r="0.0" g="0.0" b="0.0"/>
          <extraction_metadata confidence="0.85" difficulty="0.3" z_order="3"/>
        </text_element>
      </text_elements>
      <image_elements>
        <image_element id="image_001" page="1">
          <bbox x0="320" y0="200" x1="540" y1="350"/>
          <image_properties width="220" height="150" dpi="300" color_space="RGB"/>
          <alt_text>AI diagnostic system interface</alt_text>
          <extraction_metadata confidence="0.80" difficulty="0.4" z_order="4"/>
        </image_element>
      </image_elements>
    </article>
  </articles>
  
  <all_elements>
    <text_elements>
      <text_element id="title_001" type="title" page="1" reading_order="1">
        <bbox x0="72" y0="650" x1="540" y1="700"/>
        <content>AI Revolution in Healthcare</content>
        <font family="Helvetica" size="24" style="bold" align="left"/>
        <color r="0.0" g="0.0" b="0.0"/>
        <extraction_metadata confidence="0.95" difficulty="0.1" z_order="1"/>
      </text_element>
      <text_element id="body_001" type="paragraph" page="1" reading_order="2">
        <bbox x0="72" y0="400" x1="540" y1="640"/>
        <content>Artificial intelligence is transforming healthcare delivery through advanced diagnostic tools and personalized treatment recommendations. Machine learning algorithms can now analyze medical imaging with unprecedented accuracy.</content>
        <font family="Times New Roman" size="12" style="normal" align="left"/>
        <color r="0.0" g="0.0" b="0.0"/>
        <extraction_metadata confidence="0.90" difficulty="0.2" z_order="2"/>
      </text_element>
      <text_element id="byline_001" type="byline" page="1" reading_order="3">
        <bbox x0="72" y0="360" x1="300" y1="380"/>
        <content>By Dr. Sarah Johnson, Medical AI Institute</content>
        <font family="Arial" size="10" style="italic" align="left"/>
        <color r="0.0" g="0.0" b="0.0"/>
        <extraction_metadata confidence="0.85" difficulty="0.3" z_order="3"/>
      </text_element>
    </text_elements>
    <image_elements>
      <image_element id="image_001" page="1">
        <bbox x0="320" y0="200" x1="540" y1="350"/>
        <image_properties width="220" height="150" dpi="300" color_space="RGB"/>
        <alt_text>AI diagnostic system interface</alt_text>
        <extraction_metadata confidence="0.80" difficulty="0.4" z_order="4"/>
      </image_element>
    </image_elements>
  </all_elements>
</magazine_ground_truth>"""

# Perfect extraction (should get 100% accuracy)
PERFECT_EXTRACTION = """<?xml version="1.0" encoding="UTF-8"?>
<extraction_result document_id="test_doc_001">
  <article id="test_article_001">
    <title>AI Revolution in Healthcare</title>
    <contributors>
      <contributor name="Dr. Sarah Johnson" role="author"/>
    </contributors>
    <text_content>AI Revolution in Healthcare. Artificial intelligence is transforming healthcare delivery through advanced diagnostic tools and personalized treatment recommendations. Machine learning algorithms can now analyze medical imaging with unprecedented accuracy.</text_content>
    <media_elements>
      <image>
        <bbox x0="320" y0="200" x1="540" y1="350"/>
        <width>220</width>
        <height>150</height>
        <caption>AI diagnostic system interface</caption>
      </image>
    </media_elements>
  </article>
</extraction_result>"""

# Imperfect extraction (should show accuracy issues)
IMPERFECT_EXTRACTION = """<?xml version="1.0" encoding="UTF-8"?>
<extraction_result document_id="test_doc_001">
  <article id="test_article_001">
    <title>AI in Healthcare</title>
    <contributors>
      <contributor name="Sarah Johnson" role="author"/>
    </contributors>
    <text_content>Artificial intelligence is transforming healthcare through diagnostic tools and treatment recommendations.</text_content>
    <media_elements>
      <image>
        <bbox x0="325" y0="205" x1="535" y1="345"/>
        <width>210</width>
        <height>140</height>
        <caption>AI system interface</caption>
      </image>
    </media_elements>
  </article>
</extraction_result>"""

# Poor extraction (should trigger drift detection)
POOR_EXTRACTION = """<?xml version="1.0" encoding="UTF-8"?>
<extraction_result document_id="test_doc_001">
  <article id="test_article_001">
    <title>Healthcare Technology</title>
    <contributors>
      <contributor name="Johnson" role="writer"/>
    </contributors>
    <text_content>Healthcare is changing with technology.</text_content>
    <media_elements>
      <image>
        <bbox x0="300" y0="180" x1="500" y1="320"/>
        <width>200</width>
        <height>140</height>
        <caption>Technology interface</caption>
      </image>
    </media_elements>
  </article>
</extraction_result>"""


class EvaluationServiceTester:
    """Tests the evaluation service functionality."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.evaluation_base_url = f"{base_url}/api/v1/evaluation"
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        print("Testing health check endpoint...")
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            
            health_data = response.json()
            print(f"✓ Health check passed: {health_data['status']}")
            print(f"  Database connected: {health_data.get('database', 'unknown')}")
            print(f"  Services: {', '.join(health_data.get('services', []))}")
            if 'version' in health_data:
                print(f"  Version: {health_data['version']}")
            return True
            
        except Exception as e:
            print(f"✗ Health check failed: {str(e)}")
            return False
    
    def test_xml_validation(self):
        """Test XML validation endpoint."""
        print("\nTesting XML validation...")
        
        try:
            # Test valid ground truth XML
            response = self.session.post(
                f"{self.evaluation_base_url}/validate/xml",
                json={
                    "xml_content": SAMPLE_GROUND_TRUTH,
                    "xml_type": "ground_truth"
                }
            )
            if response.status_code != 200:
                print(f"  Response status: {response.status_code}")
                print(f"  Response body: {response.text}")
            response.raise_for_status()
            
            validation_result = response.json()
            print(f"✓ Ground truth validation: {validation_result['is_valid']}")
            print(f"  Articles found: {validation_result['article_count']}")
            print(f"  Elements found: {validation_result['element_count']}")
            
            # Test valid extracted XML
            response = self.session.post(
                f"{self.evaluation_base_url}/validate/xml",
                json={
                    "xml_content": PERFECT_EXTRACTION,
                    "xml_type": "extracted"
                }
            )
            response.raise_for_status()
            
            validation_result = response.json()
            print(f"✓ Extracted XML validation: {validation_result['is_valid']}")
            
            return True
            
        except Exception as e:
            print(f"✗ XML validation failed: {str(e)}")
            return False
    
    def test_single_document_evaluation(self):
        """Test single document evaluation."""
        print("\nTesting single document evaluation...")
        
        test_cases = [
            ("Perfect Extraction", PERFECT_EXTRACTION, 1.0),
            ("Imperfect Extraction", IMPERFECT_EXTRACTION, 0.7),
            ("Poor Extraction", POOR_EXTRACTION, 0.3)
        ]
        
        results = []
        
        for case_name, extracted_xml, expected_accuracy in test_cases:
            try:
                print(f"\n  Testing {case_name}...")
                
                request_data = {
                    "document_id": f"test_{case_name.lower().replace(' ', '_')}_{int(time.time())}",
                    "ground_truth_content": SAMPLE_GROUND_TRUTH,
                    "extracted_content": extracted_xml,
                    "brand_name": "TechWeekly",
                    "complexity_level": "simple",
                    "extractor_version": "1.0.0",
                    "model_version": "1.0.0",
                    "notes": f"Test case: {case_name}"
                }
                
                response = self.session.post(
                    f"{self.evaluation_base_url}/evaluate/single",
                    json=request_data
                )
                response.raise_for_status()
                
                evaluation_result = response.json()
                accuracy = evaluation_result['weighted_overall_accuracy']
                
                print(f"    Overall accuracy: {accuracy:.3f}")
                print(f"    Title accuracy: {evaluation_result['title_accuracy']:.3f}")
                print(f"    Body text accuracy: {evaluation_result['body_text_accuracy']:.3f}")
                print(f"    Contributors accuracy: {evaluation_result['contributors_accuracy']:.3f}")
                print(f"    Media links accuracy: {evaluation_result['media_links_accuracy']:.3f}")
                print(f"    Extraction successful: {evaluation_result['extraction_successful']}")
                
                results.append({
                    'case': case_name,
                    'accuracy': accuracy,
                    'document_id': evaluation_result['document_id'],
                    'evaluation_id': evaluation_result['id']
                })
                
                print(f"    ✓ {case_name} completed")
                
            except Exception as e:
                print(f"    ✗ {case_name} failed: {str(e)}")
                continue
        
        print(f"\n✓ Single document evaluation completed: {len(results)} test cases")
        return results
    
    def test_batch_evaluation(self):
        """Test batch evaluation."""
        print("\nTesting batch evaluation...")
        
        try:
            # Create batch request with multiple documents
            documents = []
            
            for i, (case_name, extracted_xml) in enumerate([
                ("Perfect", PERFECT_EXTRACTION),
                ("Imperfect", IMPERFECT_EXTRACTION), 
                ("Poor", POOR_EXTRACTION)
            ]):
                doc_request = {
                    "document_id": f"batch_test_{i+1}_{int(time.time())}",
                    "ground_truth_content": SAMPLE_GROUND_TRUTH,
                    "extracted_content": extracted_xml,
                    "brand_name": "TechWeekly",
                    "complexity_level": "simple",
                    "extractor_version": "1.0.0",
                    "model_version": "1.0.0",
                    "notes": f"Batch test case {i+1}: {case_name}"
                }
                documents.append(doc_request)
            
            batch_request = {
                "evaluation_name": f"test_batch_{int(time.time())}",
                "documents": documents,
                "parallel_processing": True,
                "fail_on_error": False,
                "enable_drift_detection": True,
                "drift_threshold": 0.05
            }
            
            response = self.session.post(
                f"{self.evaluation_base_url}/evaluate/batch",
                json=batch_request
            )
            response.raise_for_status()
            
            batch_result = response.json()
            
            print(f"✓ Batch evaluation completed:")
            print(f"  Evaluation ID: {batch_result['id']}")
            print(f"  Documents processed: {batch_result['document_count']}")
            print(f"  Articles processed: {batch_result['total_articles']}")
            print(f"  Successful extractions: {batch_result['successful_extractions']}")
            print(f"  Failed extractions: {batch_result['failed_extractions']}")
            print(f"  Overall accuracy: {batch_result['overall_weighted_accuracy']:.3f}")
            print(f"  Processing time: {batch_result.get('processing_time_seconds', 0):.2f}s")
            
            return batch_result
            
        except Exception as e:
            print(f"✗ Batch evaluation failed: {str(e)}")
            return None
    
    def test_drift_detection(self, evaluation_run_id: str):
        """Test drift detection functionality."""
        print("\nTesting drift detection...")
        
        try:
            # Trigger drift detection manually
            response = self.session.post(
                f"{self.evaluation_base_url}/drift/detect",
                params={"evaluation_run_id": evaluation_run_id}
            )
            response.raise_for_status()
            
            drift_results = response.json()
            
            print(f"✓ Drift detection completed for {len(drift_results)} metrics:")
            
            for result in drift_results:
                print(f"  {result['metric_type']}:")
                print(f"    Current accuracy: {result['current_accuracy']:.3f}")
                print(f"    Baseline accuracy: {result['baseline_accuracy']:.3f}")
                print(f"    Accuracy drop: {result['accuracy_drop']:.3f}")
                print(f"    Drift detected: {result['drift_detected']}")
                print(f"    Alert triggered: {result['alert_triggered']}")
                print(f"    Auto-tuning triggered: {result['auto_tuning_triggered']}")
            
            return drift_results
            
        except Exception as e:
            print(f"✗ Drift detection failed: {str(e)}")
            return None
    
    def test_drift_status(self):
        """Test drift status endpoints."""
        print("\nTesting drift status...")
        
        try:
            # Get drift status for overall metric
            response = self.session.get(f"{self.evaluation_base_url}/drift/status/overall")
            response.raise_for_status()
            
            status_result = response.json()
            print(f"✓ Drift status for overall metric:")
            print(f"  Status: {status_result['status']}")
            print(f"  Current accuracy: {status_result.get('current_accuracy', 'N/A')}")
            print(f"  Last checked: {status_result.get('last_checked', 'Never')}")
            
            # Get drift summary
            response = self.session.get(f"{self.evaluation_base_url}/drift/summary", params={"days": 7})
            response.raise_for_status()
            
            summary = response.json()
            print(f"✓ Drift summary (last 7 days):")
            print(f"  Total detections: {summary['total_detections']}")
            print(f"  Drift detected count: {summary['drift_detected_count']}")
            print(f"  Alerts triggered: {summary['alerts_triggered_count']}")
            print(f"  Auto-tuning triggered: {summary['auto_tuning_triggered_count']}")
            
            return True
            
        except Exception as e:
            print(f"✗ Drift status check failed: {str(e)}")
            return False
    
    def test_evaluation_history(self):
        """Test evaluation history endpoints."""
        print("\nTesting evaluation history...")
        
        try:
            # Get recent evaluations
            response = self.session.get(
                f"{self.evaluation_base_url}/evaluations",
                params={"page": 1, "page_size": 5}
            )
            response.raise_for_status()
            
            evaluations = response.json()
            print(f"✓ Recent evaluations:")
            print(f"  Total evaluations: {evaluations['total_count']}")
            print(f"  Page {evaluations['page']} of {evaluations['total_pages']}")
            
            for i, evaluation in enumerate(evaluations['items']):
                print(f"  {i+1}. {evaluation['evaluation_type']} - {evaluation['overall_weighted_accuracy']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"✗ Evaluation history failed: {str(e)}")
            return False
    
    def test_system_health(self):
        """Test system health endpoint."""
        print("\nTesting system health...")
        
        try:
            response = self.session.get(
                f"{self.evaluation_base_url}/system/health",
                params={"period_hours": 24}
            )
            response.raise_for_status()
            
            health = response.json()
            print(f"✓ System health (last 24 hours):")
            print(f"  Documents processed: {health['documents_processed']}")
            print(f"  Articles processed: {health['articles_processed']}")
            print(f"  Average overall accuracy: {health.get('average_overall_accuracy', 'N/A')}")
            print(f"  Extraction success rate: {health.get('extraction_success_rate', 'N/A')}")
            print(f"  Drift alerts: {health['drift_alerts_count']}")
            print(f"  Auto-tuning events: {health['auto_tuning_events_count']}")
            
            return True
            
        except Exception as e:
            print(f"✗ System health check failed: {str(e)}")
            return False
    
    def test_accuracy_trends(self):
        """Test accuracy trends endpoint."""
        print("\nTesting accuracy trends...")
        
        try:
            response = self.session.get(
                f"{self.evaluation_base_url}/analytics/trends",
                params={
                    "metric_type": "overall",
                    "days": 7
                }
            )
            response.raise_for_status()
            
            trends = response.json()
            print(f"✓ Accuracy trends (last 7 days):")
            print(f"  Current accuracy: {trends['current_accuracy']:.3f}")
            print(f"  Average accuracy: {trends['average_accuracy']:.3f}")
            print(f"  Trend direction: {trends.get('trend_direction', 'unknown')}")
            print(f"  Data points: {len(trends['data_points'])}")
            
            return True
            
        except Exception as e:
            print(f"✗ Accuracy trends failed: {str(e)}")
            return False
    
    def run_full_test_suite(self):
        """Run the complete test suite."""
        print("Starting Evaluation Service Test Suite")
        print("=" * 50)
        
        # Test health check first
        if not self.test_health_check():
            print("\n❌ Service is not healthy - aborting tests")
            return False
        
        # Run all tests
        test_results = {}
        
        # Basic functionality tests
        test_results['xml_validation'] = self.test_xml_validation()
        test_results['single_evaluation'] = self.test_single_document_evaluation()
        
        # Batch evaluation
        batch_result = self.test_batch_evaluation()
        test_results['batch_evaluation'] = batch_result is not None
        
        # Drift detection (if we have a batch result)
        if batch_result:
            drift_results = self.test_drift_detection(batch_result['id'])
            test_results['drift_detection'] = drift_results is not None
        
        # Status and monitoring
        test_results['drift_status'] = self.test_drift_status()
        test_results['evaluation_history'] = self.test_evaluation_history()
        test_results['system_health'] = self.test_system_health()
        test_results['accuracy_trends'] = self.test_accuracy_trends()
        
        # Summary
        print("\n" + "=" * 50)
        print("TEST SUITE SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in test_results.values() if result)
        total = len(test_results)
        
        for test_name, result in test_results.items():
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
        
        if passed == total:
            print("🎉 All tests passed! The evaluation service is working correctly.")
        else:
            print(f"⚠️  {total-passed} test(s) failed. Check the logs above for details.")
        
        return passed == total


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the evaluation service")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the evaluation service"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for service to be available before testing"
    )
    
    args = parser.parse_args()
    
    tester = EvaluationServiceTester(args.url)
    
    # Wait for service if requested
    if args.wait:
        print("Waiting for service to be available...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{args.url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"Service is available after {attempt + 1} attempts")
                    break
            except:
                pass
            
            if attempt == max_attempts - 1:
                print("Service did not become available within timeout")
                return 1
            
            time.sleep(2)
    
    # Run tests
    success = tester.run_full_test_suite()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())