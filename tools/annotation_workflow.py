#!/usr/bin/env python3
"""
Annotation Workflow System for Gold Standard Dataset Creation

This tool provides a complete workflow for creating high-quality annotated datasets
including automated annotation, manual review, quality control, and batch processing.
"""

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import our custom tools
from dataset_curator import DatasetCurator, PDFAnalysis
from ground_truth_generator import GroundTruthGenerator
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnnotationStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class AnnotationTask:
    """Represents a single annotation task"""

    task_id: str
    brand: str
    pdf_path: str
    issue_id: str
    annotator: str
    status: AnnotationStatus
    created_date: str
    assigned_date: Optional[str] = None
    completed_date: Optional[str] = None
    validation_date: Optional[str] = None
    priority: int = 5  # 1-10, higher is more important
    metadata: Dict[str, Any] = None
    quality_scores: Dict[str, float] = None
    issues: List[str] = None


@dataclass
class AnnotationWorkspace:
    """Represents an annotator's workspace"""

    annotator_id: str
    workspace_path: str
    active_tasks: List[str]
    completed_tasks: List[str]
    statistics: Dict[str, Any]


class AnnotationWorkflow:
    """Main workflow system for annotation management"""

    def __init__(
        self, base_path: str = "data/gold_sets", workspace_path: str = "workspaces"
    ):
        self.base_path = Path(base_path)
        self.workspace_path = Path(workspace_path)
        self.curator = DatasetCurator(str(self.base_path))
        self.generator = GroundTruthGenerator()
        self.validator = ValidationPipeline(str(self.base_path))

        # Initialize workflow directories
        self._setup_directories()

        # Load existing tasks
        self.tasks = self._load_tasks()
        self.workspaces = self._load_workspaces()

        # Quality requirements
        self.quality_requirements = {
            "min_ocr_accuracy": 0.95,
            "min_layout_accuracy": 0.99,
            "max_validation_issues": 3,
            "required_manual_validation": True,
        }

    def _setup_directories(self):
        """Setup required directory structure"""
        directories = [
            self.workspace_path,
            self.workspace_path / "tasks",
            self.workspace_path / "templates",
            self.workspace_path / "completed",
            self.workspace_path / "reports",
            self.base_path / "staging",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        logger.info("Annotation workflow directories initialized")

    def _load_tasks(self) -> Dict[str, AnnotationTask]:
        """Load existing annotation tasks"""
        tasks = {}
        tasks_file = self.workspace_path / "tasks" / "tasks.json"

        if tasks_file.exists():
            try:
                with open(tasks_file, "r") as f:
                    tasks_data = json.load(f)

                for task_data in tasks_data:
                    task = AnnotationTask(**task_data)
                    task.status = AnnotationStatus(task.status)
                    tasks[task.task_id] = task

                logger.info(f"Loaded {len(tasks)} existing annotation tasks")

            except Exception as e:
                logger.error(f"Error loading tasks: {str(e)}")

        return tasks

    def _save_tasks(self):
        """Save annotation tasks to file"""
        tasks_file = self.workspace_path / "tasks" / "tasks.json"

        tasks_data = []
        for task in self.tasks.values():
            task_dict = asdict(task)
            task_dict["status"] = task.status.value
            tasks_data.append(task_dict)

        with open(tasks_file, "w") as f:
            json.dump(tasks_data, f, indent=2)

    def _load_workspaces(self) -> Dict[str, AnnotationWorkspace]:
        """Load annotator workspaces"""
        workspaces = {}
        workspaces_file = self.workspace_path / "workspaces.json"

        if workspaces_file.exists():
            try:
                with open(workspaces_file, "r") as f:
                    workspaces_data = json.load(f)

                for workspace_data in workspaces_data:
                    workspace = AnnotationWorkspace(**workspace_data)
                    workspaces[workspace.annotator_id] = workspace

                logger.info(f"Loaded {len(workspaces)} annotator workspaces")

            except Exception as e:
                logger.error(f"Error loading workspaces: {str(e)}")

        return workspaces

    def _save_workspaces(self):
        """Save annotator workspaces"""
        workspaces_file = self.workspace_path / "workspaces.json"

        workspaces_data = [asdict(workspace) for workspace in self.workspaces.values()]

        with open(workspaces_file, "w") as f:
            json.dump(workspaces_data, f, indent=2)

    def create_annotation_task(
        self, brand: str, pdf_path: str, annotator: str, priority: int = 5
    ) -> str:
        """Create a new annotation task"""

        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Generate task ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_id = f"{brand}_{pdf_file.stem}_{timestamp}"

        # Extract issue ID from filename or generate one
        issue_id = pdf_file.stem

        # Analyze PDF quality
        try:
            analysis = self.curator.analyze_pdf(pdf_path)
            metadata = asdict(analysis)
        except Exception as e:
            logger.error(f"PDF analysis failed: {str(e)}")
            metadata = {"analysis_error": str(e)}

        # Create task
        task = AnnotationTask(
            task_id=task_id,
            brand=brand,
            pdf_path=str(pdf_path),
            issue_id=issue_id,
            annotator=annotator,
            status=AnnotationStatus.PENDING,
            created_date=datetime.now().isoformat(),
            priority=priority,
            metadata=metadata,
            issues=[],
        )

        self.tasks[task_id] = task
        self._save_tasks()

        logger.info(f"Created annotation task: {task_id}")
        return task_id

    def assign_task(self, task_id: str, annotator: str) -> bool:
        """Assign task to an annotator"""

        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return False

        task = self.tasks[task_id]

        if task.status != AnnotationStatus.PENDING:
            logger.error(f"Task {task_id} is not in pending status")
            return False

        # Update task
        task.annotator = annotator
        task.status = AnnotationStatus.IN_PROGRESS
        task.assigned_date = datetime.now().isoformat()

        # Create workspace if needed
        if annotator not in self.workspaces:
            self._create_workspace(annotator)

        # Add to workspace
        workspace = self.workspaces[annotator]
        workspace.active_tasks.append(task_id)

        # Generate annotation template
        self._create_annotation_template(task)

        self._save_tasks()
        self._save_workspaces()

        logger.info(f"Assigned task {task_id} to {annotator}")
        return True

    def _create_workspace(self, annotator: str):
        """Create workspace for annotator"""

        workspace_dir = self.workspace_path / annotator
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (workspace_dir / "active").mkdir(exist_ok=True)
        (workspace_dir / "completed").mkdir(exist_ok=True)
        (workspace_dir / "templates").mkdir(exist_ok=True)

        workspace = AnnotationWorkspace(
            annotator_id=annotator,
            workspace_path=str(workspace_dir),
            active_tasks=[],
            completed_tasks=[],
            statistics={
                "tasks_completed": 0,
                "tasks_rejected": 0,
                "average_quality": 0.0,
                "total_processing_time": 0.0,
            },
        )

        self.workspaces[annotator] = workspace
        logger.info(f"Created workspace for annotator: {annotator}")

    def _create_annotation_template(self, task: AnnotationTask):
        """Create annotation template files for task"""

        annotator = task.annotator
        workspace_dir = self.workspace_path / annotator / "active" / task.task_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDF to workspace
        pdf_path = Path(task.pdf_path)
        workspace_pdf = workspace_dir / pdf_path.name
        shutil.copy2(pdf_path, workspace_pdf)

        # Generate XML template
        xml_template_path = workspace_dir / f"{task.issue_id}.xml"
        xml_root = self.generator.create_ground_truth_template(
            task.brand, task.issue_id, 1  # Assume 1 page for now
        )
        self.generator.save_ground_truth(xml_root, str(xml_template_path))

        # Create annotation guide
        guide_path = workspace_dir / "annotation_guide.md"
        self._create_annotation_guide(guide_path, task)

        # Create task info file
        info_path = workspace_dir / "task_info.json"
        with open(info_path, "w") as f:
            json.dump(asdict(task), f, indent=2)

        logger.info(f"Created annotation template for task {task.task_id}")

    def _create_annotation_guide(self, guide_path: Path, task: AnnotationTask):
        """Create detailed annotation guide for task"""

        guide_content = f"""# Annotation Guide for {task.issue_id}

## Task Information
- **Brand**: {task.brand}
- **Issue ID**: {task.issue_id}
- **Priority**: {task.priority}/10
- **Created**: {task.created_date}

## Quality Requirements
- OCR Accuracy: ≥ {self.quality_requirements['min_ocr_accuracy']*100}%
- Layout Classification: ≥ {self.quality_requirements['min_layout_accuracy']*100}%
- Validation Issues: ≤ {self.quality_requirements['max_validation_issues']}

## PDF Analysis Results
"""

        if task.metadata:
            if "quality_level" in task.metadata:
                guide_content += (
                    f"- **Quality Level**: {task.metadata['quality_level']}\n"
                )
            if "estimated_ocr_quality" in task.metadata:
                guide_content += f"- **Estimated OCR Quality**: {task.metadata['estimated_ocr_quality']:.1%}\n"
            if "layout_complexity" in task.metadata:
                guide_content += (
                    f"- **Layout Complexity**: {task.metadata['layout_complexity']}\n"
                )
            if "issues" in task.metadata:
                guide_content += (
                    f"- **Issues Detected**: {', '.join(task.metadata['issues'])}\n"
                )

        guide_content += """
## Annotation Instructions

### 1. Block Identification
Identify and annotate the following block types:
- **title**: Main article titles and subtitles
- **body**: Article body text paragraphs
- **byline**: Author attribution (e.g., "By John Smith")
- **caption**: Image and figure captions
- **ad**: Advertisement blocks
- **header/footer**: Page headers and footers
- **sidebar**: Sidebar content

### 2. Article Reconstruction
Group related blocks into articles:
- Each article must have at least one title block
- Body blocks should follow logical reading order
- Include all related bylines and captions

### 3. Contributor Extraction
For each contributor:
- Extract full name exactly as written
- Determine role (author, photographer, illustrator, etc.)
- Provide confidence score (0.0-1.0)

### 4. Quality Checklist
Before submitting:
- [ ] All text blocks have accurate bounding boxes
- [ ] Block types are correctly classified
- [ ] Articles are properly structured
- [ ] Contributors are extracted and normalized
- [ ] XML validates without errors

### 5. Validation
Run validation before submission:
```bash
python3 ../../validation_pipeline.py --xml {task.issue_id}.xml
```

## Files in Workspace
- `{Path(task.pdf_path).name}` - Source PDF file
- `{task.issue_id}.xml` - Ground truth XML (edit this)
- `annotation_guide.md` - This guide
- `task_info.json` - Task metadata

## Submission
When annotation is complete:
1. Validate XML file
2. Mark task as completed in workflow system
3. Files will be moved to validation queue
"""

        with open(guide_path, "w") as f:
            f.write(guide_content)

    def complete_task(self, task_id: str, annotator: str) -> bool:
        """Mark task as completed by annotator"""

        if task_id not in self.tasks:
            logger.error(f"Task not found: {task_id}")
            return False

        task = self.tasks[task_id]

        if task.annotator != annotator:
            logger.error(f"Task {task_id} not assigned to {annotator}")
            return False

        if task.status != AnnotationStatus.IN_PROGRESS:
            logger.error(f"Task {task_id} not in progress")
            return False

        # Move files from active to completed workspace
        active_dir = self.workspace_path / annotator / "active" / task_id
        completed_dir = self.workspace_path / annotator / "completed" / task_id

        if active_dir.exists():
            shutil.move(str(active_dir), str(completed_dir))

        # Update task status
        task.status = AnnotationStatus.COMPLETED
        task.completed_date = datetime.now().isoformat()

        # Update workspace
        workspace = self.workspaces[annotator]
        workspace.active_tasks.remove(task_id)
        workspace.completed_tasks.append(task_id)
        workspace.statistics["tasks_completed"] += 1

        self._save_tasks()
        self._save_workspaces()

        # Queue for validation
        self._queue_for_validation(task)

        logger.info(f"Task {task_id} completed by {annotator}")
        return True

    def _queue_for_validation(self, task: AnnotationTask):
        """Queue completed task for validation"""

        # Copy completed XML to staging area for validation
        completed_xml = (
            self.workspace_path
            / task.annotator
            / "completed"
            / task.task_id
            / f"{task.issue_id}.xml"
        )

        if completed_xml.exists():
            staging_dir = self.base_path / "staging" / task.brand
            staging_dir.mkdir(parents=True, exist_ok=True)

            staging_xml = staging_dir / f"{task.issue_id}.xml"
            shutil.copy2(completed_xml, staging_xml)

            logger.info(f"Queued {task.task_id} for validation")

    def validate_completed_tasks(self, batch_size: int = 10) -> Dict[str, Any]:
        """Validate completed tasks in batch"""

        completed_tasks = [
            task
            for task in self.tasks.values()
            if task.status == AnnotationStatus.COMPLETED
        ]

        validation_results = {
            "validation_timestamp": datetime.now().isoformat(),
            "batch_size": batch_size,
            "tasks_validated": 0,
            "tasks_passed": 0,
            "tasks_failed": 0,
            "results": {},
        }

        # Process in batches
        for i in range(0, min(len(completed_tasks), batch_size)):
            task = completed_tasks[i]

            try:
                # Find staging XML file
                staging_xml = (
                    self.base_path / "staging" / task.brand / f"{task.issue_id}.xml"
                )

                if not staging_xml.exists():
                    logger.warning(f"Staging XML not found for task {task.task_id}")
                    continue

                # Run validation
                validation_result = self.validator.validate_dataset(
                    task.brand, task.issue_id
                )

                quality_metrics = validation_result.get("quality_metrics", {})
                overall_score = quality_metrics.get("overall_score", 0.0)

                # Check against requirements
                passed = (
                    overall_score >= 0.90
                    and len(validation_result.get("summary", {}).get("issues", []))
                    <= self.quality_requirements["max_validation_issues"]
                )

                if passed:
                    task.status = AnnotationStatus.VALIDATED
                    task.validation_date = datetime.now().isoformat()
                    task.quality_scores = quality_metrics
                    validation_results["tasks_passed"] += 1

                    # Move to final location
                    self._finalize_validated_task(task)

                else:
                    task.status = AnnotationStatus.FAILED
                    task.issues = validation_result.get("issues", [])
                    validation_results["tasks_failed"] += 1

                validation_results["results"][task.task_id] = {
                    "passed": passed,
                    "score": overall_score,
                    "issues": validation_result.get("issues", []),
                }

                validation_results["tasks_validated"] += 1

            except Exception as e:
                logger.error(f"Validation failed for task {task.task_id}: {str(e)}")
                task.status = AnnotationStatus.FAILED
                task.issues = [f"Validation error: {str(e)}"]
                validation_results["tasks_failed"] += 1

        self._save_tasks()
        return validation_results

    def _finalize_validated_task(self, task: AnnotationTask):
        """Move validated task to final gold standard location"""

        staging_xml = self.base_path / "staging" / task.brand / f"{task.issue_id}.xml"

        final_dir = self.base_path / task.brand / "ground_truth"
        final_dir.mkdir(parents=True, exist_ok=True)

        final_xml = final_dir / f"{task.issue_id}.xml"
        shutil.move(str(staging_xml), str(final_xml))

        # Generate and save metadata
        metadata = self.curator.generate_metadata(
            task.brand,
            task.pdf_path,
            PDFAnalysis(**task.metadata) if task.metadata else None,
            task.annotator,
        )
        metadata.validation_status = "validated"
        metadata.quality_scores.update(task.quality_scores or {})

        metadata_dir = self.base_path / task.brand / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = metadata_dir / f"{task.issue_id}_metadata.json"

        self.curator.save_metadata(metadata, str(metadata_file))

        logger.info(f"Finalized validated task {task.task_id}")

    def generate_workflow_report(
        self, annotator: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate workflow status report"""

        report = {
            "generation_time": datetime.now().isoformat(),
            "annotator_filter": annotator,
            "overall_statistics": {},
            "task_status_distribution": {},
            "annotator_performance": {},
            "quality_trends": {},
        }

        # Filter tasks
        tasks_to_analyze = [
            task
            for task in self.tasks.values()
            if annotator is None or task.annotator == annotator
        ]

        # Overall statistics
        report["overall_statistics"] = {
            "total_tasks": len(tasks_to_analyze),
            "pending_tasks": len(
                [t for t in tasks_to_analyze if t.status == AnnotationStatus.PENDING]
            ),
            "in_progress_tasks": len(
                [
                    t
                    for t in tasks_to_analyze
                    if t.status == AnnotationStatus.IN_PROGRESS
                ]
            ),
            "completed_tasks": len(
                [t for t in tasks_to_analyze if t.status == AnnotationStatus.COMPLETED]
            ),
            "validated_tasks": len(
                [t for t in tasks_to_analyze if t.status == AnnotationStatus.VALIDATED]
            ),
            "failed_tasks": len(
                [t for t in tasks_to_analyze if t.status == AnnotationStatus.FAILED]
            ),
        }

        # Status distribution
        from collections import Counter

        status_counts = Counter(task.status for task in tasks_to_analyze)
        report["task_status_distribution"] = {
            status.value: count for status, count in status_counts.items()
        }

        # Annotator performance
        for annotator_id, workspace in self.workspaces.items():
            if annotator is None or annotator_id == annotator:
                annotator_tasks = [
                    t for t in tasks_to_analyze if t.annotator == annotator_id
                ]

                if annotator_tasks:
                    validated_tasks = [
                        t
                        for t in annotator_tasks
                        if t.status == AnnotationStatus.VALIDATED
                    ]
                    avg_quality = (
                        (
                            sum(
                                t.quality_scores.get("overall_score", 0.0)
                                for t in validated_tasks
                            )
                            / len(validated_tasks)
                        )
                        if validated_tasks
                        else 0.0
                    )

                    report["annotator_performance"][annotator_id] = {
                        "total_assigned": len(annotator_tasks),
                        "completed": len(
                            [
                                t
                                for t in annotator_tasks
                                if t.status
                                in [
                                    AnnotationStatus.COMPLETED,
                                    AnnotationStatus.VALIDATED,
                                ]
                            ]
                        ),
                        "validated": len(validated_tasks),
                        "failed": len(
                            [
                                t
                                for t in annotator_tasks
                                if t.status == AnnotationStatus.FAILED
                            ]
                        ),
                        "average_quality": avg_quality,
                        "workspace_stats": workspace.statistics,
                    }

        return report

    def batch_create_tasks(
        self, pdf_directory: str, brand: str, annotator: str, priority: int = 5
    ) -> List[str]:
        """Create annotation tasks for all PDFs in directory"""

        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_directory}")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        task_ids = []

        logger.info(f"Creating {len(pdf_files)} annotation tasks for {brand}")

        for pdf_file in pdf_files:
            try:
                task_id = self.create_annotation_task(
                    brand, str(pdf_file), annotator, priority
                )
                task_ids.append(task_id)
            except Exception as e:
                logger.error(f"Failed to create task for {pdf_file}: {str(e)}")

        logger.info(f"Created {len(task_ids)} annotation tasks")
        return task_ids


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Annotation Workflow System")
    parser.add_argument(
        "command",
        choices=["create", "assign", "complete", "validate", "report", "batch"],
        help="Command to execute",
    )
    parser.add_argument("--brand", help="Magazine brand")
    parser.add_argument("--pdf", help="PDF file path")
    parser.add_argument(
        "--pdf-dir", help="Directory containing PDFs for batch processing"
    )
    parser.add_argument("--task-id", help="Task ID")
    parser.add_argument("--annotator", help="Annotator ID")
    parser.add_argument("--priority", type=int, default=5, help="Task priority (1-10)")
    parser.add_argument("--output", help="Output path for reports")
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Validation batch size"
    )

    args = parser.parse_args()

    workflow = AnnotationWorkflow()

    if args.command == "create" and args.brand and args.pdf and args.annotator:
        task_id = workflow.create_annotation_task(
            args.brand, args.pdf, args.annotator, args.priority
        )
        print(f"Created task: {task_id}")

    elif args.command == "assign" and args.task_id and args.annotator:
        success = workflow.assign_task(args.task_id, args.annotator)
        print(f"Assignment {'successful' if success else 'failed'}")

    elif args.command == "complete" and args.task_id and args.annotator:
        success = workflow.complete_task(args.task_id, args.annotator)
        print(f"Completion {'successful' if success else 'failed'}")

    elif args.command == "validate":
        results = workflow.validate_completed_tasks(args.batch_size)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        else:
            print(json.dumps(results, indent=2))

    elif args.command == "report":
        report = workflow.generate_workflow_report(args.annotator)
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))

    elif args.command == "batch" and args.brand and args.pdf_dir and args.annotator:
        task_ids = workflow.batch_create_tasks(
            args.pdf_dir, args.brand, args.annotator, args.priority
        )
        print(f"Created {len(task_ids)} tasks: {task_ids}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
