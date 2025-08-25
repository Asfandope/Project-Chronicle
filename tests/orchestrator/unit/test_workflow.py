import pytest
from orchestrator.core.workflow import (
    WorkflowEngine, 
    WorkflowStage, 
    WorkflowStatus, 
    WorkflowStep
)

class TestWorkflowEngine:
    """Test workflow engine functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = WorkflowEngine()
    
    def test_get_next_stages_initial(self):
        """Test getting next stages for a new workflow."""
        current_steps = {
            WorkflowStage.INGESTION: WorkflowStep(
                stage=WorkflowStage.INGESTION,
                status=WorkflowStatus.COMPLETED
            )
        }
        
        next_stages = self.engine.get_next_stages(current_steps)
        
        assert WorkflowStage.PREPROCESSING in next_stages
        assert WorkflowStage.LAYOUT_ANALYSIS not in next_stages  # Depends on PREPROCESSING
    
    def test_get_next_stages_multiple_dependencies(self):
        """Test stages with multiple dependencies."""
        current_steps = {
            WorkflowStage.INGESTION: WorkflowStep(
                stage=WorkflowStage.INGESTION,
                status=WorkflowStatus.COMPLETED
            ),
            WorkflowStage.PREPROCESSING: WorkflowStep(
                stage=WorkflowStage.PREPROCESSING,
                status=WorkflowStatus.COMPLETED
            ),
            WorkflowStage.LAYOUT_ANALYSIS: WorkflowStep(
                stage=WorkflowStage.LAYOUT_ANALYSIS,
                status=WorkflowStatus.COMPLETED
            ),
            WorkflowStage.OCR: WorkflowStep(
                stage=WorkflowStage.OCR,
                status=WorkflowStatus.COMPLETED
            )
        }
        
        next_stages = self.engine.get_next_stages(current_steps)
        
        # ARTICLE_RECONSTRUCTION depends on both OCR and LAYOUT_ANALYSIS
        assert WorkflowStage.ARTICLE_RECONSTRUCTION in next_stages
        # IMAGE_EXTRACTION only depends on LAYOUT_ANALYSIS
        assert WorkflowStage.IMAGE_EXTRACTION in next_stages
    
    def test_is_workflow_complete(self):
        """Test workflow completion detection."""
        complete_steps = {
            WorkflowStage.COMPLETED: WorkflowStep(
                stage=WorkflowStage.COMPLETED,
                status=WorkflowStatus.COMPLETED
            )
        }
        
        incomplete_steps = {
            WorkflowStage.EVALUATION: WorkflowStep(
                stage=WorkflowStage.EVALUATION,
                status=WorkflowStatus.COMPLETED
            )
        }
        
        assert self.engine.is_workflow_complete(complete_steps)
        assert not self.engine.is_workflow_complete(incomplete_steps)
    
    def test_is_workflow_failed(self):
        """Test workflow failure detection."""
        failed_steps = {
            WorkflowStage.LAYOUT_ANALYSIS: WorkflowStep(
                stage=WorkflowStage.LAYOUT_ANALYSIS,
                status=WorkflowStatus.FAILED,
                attempts=3,
                max_attempts=3
            )
        }
        
        retry_steps = {
            WorkflowStage.LAYOUT_ANALYSIS: WorkflowStep(
                stage=WorkflowStage.LAYOUT_ANALYSIS,
                status=WorkflowStatus.FAILED,
                attempts=1,
                max_attempts=3
            )
        }
        
        assert self.engine.is_workflow_failed(failed_steps)
        assert not self.engine.is_workflow_failed(retry_steps)
    
    def test_should_quarantine_low_accuracy(self):
        """Test quarantine decision based on accuracy."""
        steps = {}
        
        assert self.engine.should_quarantine(steps, accuracy=0.98)  # Below threshold
        assert not self.engine.should_quarantine(steps, accuracy=0.9995)  # Above threshold
    
    def test_should_quarantine_critical_failure(self):
        """Test quarantine decision based on critical stage failures."""
        failed_layout_steps = {
            WorkflowStage.LAYOUT_ANALYSIS: WorkflowStep(
                stage=WorkflowStage.LAYOUT_ANALYSIS,
                status=WorkflowStatus.FAILED
            )
        }
        
        failed_ocr_steps = {
            WorkflowStage.OCR: WorkflowStep(
                stage=WorkflowStage.OCR,
                status=WorkflowStatus.FAILED
            )
        }
        
        assert self.engine.should_quarantine(failed_layout_steps)
        assert not self.engine.should_quarantine(failed_ocr_steps)  # OCR not critical
    
    def test_stage_dependencies_completeness(self):
        """Test that all stages have valid dependencies."""
        for stage, dependencies in self.engine.STAGE_DEPENDENCIES.items():
            # Check that all dependencies are valid stages
            for dep in dependencies:
                assert isinstance(dep, WorkflowStage)
            
            # Check that there are no circular dependencies
            visited = set()
            self._check_circular_dependency(stage, dependencies, visited)
    
    def _check_circular_dependency(self, stage, dependencies, visited):
        """Helper method to check for circular dependencies."""
        if stage in visited:
            raise ValueError(f"Circular dependency detected involving {stage}")
        
        visited.add(stage)
        for dep in dependencies:
            if dep in self.engine.STAGE_DEPENDENCIES:
                self._check_circular_dependency(
                    dep, 
                    self.engine.STAGE_DEPENDENCIES[dep], 
                    visited.copy()
                )
    
    @pytest.mark.parametrize("stage,expected_deps", [
        (WorkflowStage.PREPROCESSING, [WorkflowStage.INGESTION]),
        (WorkflowStage.LAYOUT_ANALYSIS, [WorkflowStage.PREPROCESSING]),
        (WorkflowStage.OCR, [WorkflowStage.LAYOUT_ANALYSIS]),
        (WorkflowStage.ARTICLE_RECONSTRUCTION, [WorkflowStage.OCR, WorkflowStage.LAYOUT_ANALYSIS]),
    ])
    def test_specific_stage_dependencies(self, stage, expected_deps):
        """Test specific stage dependency requirements."""
        actual_deps = self.engine.STAGE_DEPENDENCIES.get(stage, [])
        assert set(actual_deps) == set(expected_deps)