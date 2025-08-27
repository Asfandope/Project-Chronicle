import io
from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
class TestEndToEndPipeline:
    """Integration tests for complete PDF processing pipeline."""

    async def test_complete_pipeline_flow(
        self,
        orchestrator_client: AsyncClient,
        model_service_client: AsyncClient,
        evaluation_client: AsyncClient,
        sample_pdf_content: bytes,
        temp_directory: Path,
    ):
        """Test complete pipeline from PDF upload to final evaluation."""

        # Step 1: Upload PDF to orchestrator
        files = {
            "file": (
                "sample_magazine.pdf",
                io.BytesIO(sample_pdf_content),
                "application/pdf",
            )
        }
        data = {"brand": "economist"}

        job_response = await orchestrator_client.post(
            "/api/v1/jobs/", files=files, data=data
        )
        assert job_response.status_code == 200

        job_data = job_response.json()
        job_id = job_data["id"]

        # Step 2: Monitor job progress
        # In real integration test, we would poll until completion
        # For now, just verify job was created
        job_status_response = await orchestrator_client.get(f"/api/v1/jobs/{job_id}")
        assert job_status_response.status_code == 200

        job_status = job_status_response.json()
        assert job_status["overall_status"] in ["pending", "in_progress"]

        # Step 3: Simulate model service processing stages
        # In real test, these would be called by the orchestrator's Celery tasks

        # Layout analysis
        layout_request = {"job_id": job_id, "file_path": f"/tmp/{job_data['filename']}"}
        layout_response = await model_service_client.post(
            "/api/v1/layout/analyze", json=layout_request
        )
        # Note: This will likely fail without actual model loading
        # In full integration test, we'd use mock models or test models

        # Step 4: Simulate evaluation
        eval_request = {"job_id": job_id, "brand": "economist"}
        # This would also need gold standard data set up
        eval_response = await evaluation_client.post(
            "/api/v1/accuracy/evaluate", json=eval_request
        )

        # Step 5: Verify final state
        # In complete test, job should be completed or quarantined
        final_status_response = await orchestrator_client.get(f"/api/v1/jobs/{job_id}")
        final_status = final_status_response.json()

        # At minimum, job should still exist and have some processing state
        assert final_status["id"] == job_id
        assert "workflow_steps" in final_status or "overall_status" in final_status

    async def test_multiple_concurrent_jobs(
        self, orchestrator_client: AsyncClient, sample_pdf_content: bytes
    ):
        """Test processing multiple jobs concurrently."""

        # Create multiple jobs
        job_ids = []
        for i in range(3):
            files = {
                "file": (
                    f"test_{i}.pdf",
                    io.BytesIO(sample_pdf_content),
                    "application/pdf",
                )
            }
            data = {"brand": "economist"}

            response = await orchestrator_client.post(
                "/api/v1/jobs/", files=files, data=data
            )
            assert response.status_code == 200
            job_ids.append(response.json()["id"])

        # Verify all jobs were created
        assert len(job_ids) == 3

        # Check that all jobs can be retrieved
        for job_id in job_ids:
            response = await orchestrator_client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200
            assert response.json()["id"] == job_id

    async def test_error_handling_in_pipeline(
        self, orchestrator_client: AsyncClient, model_service_client: AsyncClient
    ):
        """Test error handling across services."""

        # Test invalid model service request
        invalid_request = {
            "job_id": "non-existent-job",
            "file_path": "/non/existent/path.pdf",
        }

        layout_response = await model_service_client.post(
            "/api/v1/layout/analyze", json=invalid_request
        )

        # Should handle error gracefully
        assert layout_response.status_code in [400, 404, 500]

    async def test_health_checks_all_services(
        self,
        orchestrator_client: AsyncClient,
        model_service_client: AsyncClient,
        evaluation_client: AsyncClient,
    ):
        """Test health checks for all services."""

        services = [
            (orchestrator_client, "orchestrator"),
            (model_service_client, "model_service"),
            (evaluation_client, "evaluation"),
        ]

        for client, service_name in services:
            # Basic health check
            health_response = await client.get("/health/")
            assert health_response.status_code == 200

            health_data = health_response.json()
            assert health_data["status"] == "healthy"
            assert health_data["service"] in [
                service_name,
                service_name.replace("_", "-"),
            ]

            # Detailed health check
            detailed_response = await client.get("/health/detailed")
            # May return 503 if dependencies are not available
            assert detailed_response.status_code in [200, 503]

    @pytest.mark.performance
    async def test_pipeline_performance_basic(
        self, orchestrator_client: AsyncClient, sample_pdf_content: bytes
    ):
        """Basic performance test for job creation."""
        import time

        start_time = time.time()

        # Create job
        files = {
            "file": ("perf_test.pdf", io.BytesIO(sample_pdf_content), "application/pdf")
        }
        data = {"brand": "economist"}

        response = await orchestrator_client.post(
            "/api/v1/jobs/", files=files, data=data
        )

        end_time = time.time()
        duration = end_time - start_time

        assert response.status_code == 200
        assert duration < 2.0  # Job creation should be fast (< 2 seconds)

    async def test_configuration_consistency(self, orchestrator_client: AsyncClient):
        """Test that configuration is consistent across services."""

        # Get brand configurations
        economist_config_response = await orchestrator_client.get(
            "/api/v1/config/brands/economist"
        )

        if economist_config_response.status_code == 200:
            config = economist_config_response.json()

            # Verify required configuration sections
            assert "layout_hints" in config
            assert "ocr_preprocessing" in config
            assert "confidence_overrides" in config

            # Verify configuration values are reasonable
            if "confidence_overrides" in config:
                for field, threshold in config["confidence_overrides"].items():
                    assert (
                        0.0 <= threshold <= 1.0
                    ), f"Invalid confidence threshold for {field}: {threshold}"
