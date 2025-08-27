"""Create comprehensive database schema for job queue, processing history,
brand config tracking, drift detection, and model version management

Revision ID: 0001
Revises:
Create Date: 2024-12-21 12:00:00.000000

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum types
    op.execute(
        "CREATE TYPE workflowstage AS ENUM ("
        "'INGESTION', 'LAYOUT_ANALYSIS', 'OCR', 'ARTICLE_RECONSTRUCTION', "
        "'VALIDATION', 'EXPORT')"
    )
    op.execute(
        "CREATE TYPE workflowstatus AS ENUM ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'QUARANTINED')"
    )

    # Create jobs table
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=512), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("file_hash", sa.String(length=64), nullable=True),
        sa.Column("brand", sa.String(length=50), nullable=True),
        sa.Column("issue_date", sa.String(length=10), nullable=True),
        sa.Column(
            "current_stage",
            sa.Enum(
                "INGESTION",
                "LAYOUT_ANALYSIS",
                "OCR",
                "ARTICLE_RECONSTRUCTION",
                "VALIDATION",
                "EXPORT",
                name="workflowstage",
            ),
            nullable=False,
        ),
        sa.Column(
            "overall_status",
            sa.Enum(
                "PENDING",
                "IN_PROGRESS",
                "COMPLETED",
                "FAILED",
                "QUARANTINED",
                name="workflowstatus",
            ),
            nullable=False,
        ),
        sa.Column(
            "workflow_steps", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("accuracy_score", sa.Float(), nullable=True),
        sa.Column(
            "confidence_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "field_accuracies", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_seconds", sa.Integer(), nullable=True),
        sa.Column("xml_output_path", sa.String(length=512), nullable=True),
        sa.Column("csv_output_path", sa.String(length=512), nullable=True),
        sa.Column("images_output_directory", sa.String(length=512), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("retry_count", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("quarantine_reason", sa.String(length=255), nullable=True),
        sa.Column("quarantined_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("celery_task_id", sa.String(length=36), nullable=True),
        sa.Column("model_version", sa.String(length=20), nullable=True),
        sa.Column("config_version", sa.String(length=20), nullable=True),
        sa.Column("priority", sa.Integer(), nullable=True),
        sa.Column("scheduled_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("file_size > 0", name="check_positive_file_size"),
        sa.CheckConstraint(
            "processing_time_seconds >= 0", name="check_positive_processing_time"
        ),
        sa.CheckConstraint("retry_count >= 0", name="check_non_negative_retry_count"),
        sa.CheckConstraint(
            "accuracy_score >= 0.0 AND accuracy_score <= 1.0",
            name="check_accuracy_range",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR started_at IS NOT NULL",
            name="check_started_before_completed",
        ),
        sa.CheckConstraint(
            "quarantined_at IS NULL OR quarantine_reason IS NOT NULL",
            name="check_quarantine_reason_required",
        ),
    )

    # Create indexes for jobs table
    op.create_index("idx_jobs_accuracy_brand", "jobs", ["accuracy_score", "brand"])
    op.create_index("idx_jobs_brand_created", "jobs", ["brand", "created_at"])
    op.create_index("idx_jobs_brand_status", "jobs", ["brand", "overall_status"])
    op.create_index(
        "idx_jobs_monitoring", "jobs", ["brand", "overall_status", "created_at"]
    )
    op.create_index(
        "idx_jobs_queue_processing",
        "jobs",
        ["overall_status", "priority", "scheduled_at"],
    )
    op.create_index("idx_jobs_status_created", "jobs", ["overall_status", "created_at"])
    op.create_index(op.f("ix_jobs_accuracy_score"), "jobs", ["accuracy_score"])
    op.create_index(op.f("ix_jobs_brand"), "jobs", ["brand"])
    op.create_index(op.f("ix_jobs_celery_task_id"), "jobs", ["celery_task_id"])
    op.create_index(op.f("ix_jobs_completed_at"), "jobs", ["completed_at"])
    op.create_index(op.f("ix_jobs_created_at"), "jobs", ["created_at"])
    op.create_index(op.f("ix_jobs_current_stage"), "jobs", ["current_stage"])
    op.create_index(op.f("ix_jobs_file_hash"), "jobs", ["file_hash"])
    op.create_index(op.f("ix_jobs_filename"), "jobs", ["filename"])
    op.create_index(op.f("ix_jobs_issue_date"), "jobs", ["issue_date"])
    op.create_index(op.f("ix_jobs_model_version"), "jobs", ["model_version"])
    op.create_index(op.f("ix_jobs_overall_status"), "jobs", ["overall_status"])
    op.create_index(op.f("ix_jobs_priority"), "jobs", ["priority"])
    op.create_index(op.f("ix_jobs_scheduled_at"), "jobs", ["scheduled_at"])
    op.create_index(op.f("ix_jobs_started_at"), "jobs", ["started_at"])

    # Create processing_states table
    op.create_table(
        "processing_states",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "stage",
            sa.Enum(
                "INGESTION",
                "LAYOUT_ANALYSIS",
                "OCR",
                "ARTICLE_RECONSTRUCTION",
                "VALIDATION",
                "EXPORT",
                name="workflowstage",
            ),
            nullable=False,
        ),
        sa.Column("stage_version", sa.String(length=20), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processing_time_ms", sa.Integer(), nullable=True),
        sa.Column(
            "stage_output", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "confidence_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "accuracy_metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("overall_confidence", sa.Float(), nullable=True),
        sa.Column(
            "quality_flags", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("memory_usage_mb", sa.Integer(), nullable=True),
        sa.Column("cpu_time_ms", sa.Integer(), nullable=True),
        sa.Column("gpu_time_ms", sa.Integer(), nullable=True),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "error_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("retry_attempt", sa.Integer(), nullable=False),
        sa.Column("celery_task_id", sa.String(length=36), nullable=True),
        sa.Column("worker_id", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "processing_time_ms >= 0", name="check_positive_processing_time"
        ),
        sa.CheckConstraint("memory_usage_mb >= 0", name="check_positive_memory_usage"),
        sa.CheckConstraint("cpu_time_ms >= 0", name="check_positive_cpu_time"),
        sa.CheckConstraint("gpu_time_ms >= 0", name="check_positive_gpu_time"),
        sa.CheckConstraint(
            "overall_confidence >= 0.0 AND overall_confidence <= 1.0",
            name="check_confidence_range",
        ),
        sa.CheckConstraint("retry_attempt >= 0", name="check_non_negative_retry"),
        sa.CheckConstraint(
            "completed_at IS NULL OR started_at IS NOT NULL",
            name="check_started_before_completed",
        ),
        sa.CheckConstraint(
            "success = false OR error_message IS NULL", name="check_success_no_error"
        ),
    )

    # Create indexes for processing_states table
    op.create_index(
        "idx_processing_confidence_stage",
        "processing_states",
        ["overall_confidence", "stage"],
    )
    op.create_index(
        "idx_processing_job_stage", "processing_states", ["job_id", "stage"]
    )
    op.create_index(
        "idx_processing_performance",
        "processing_states",
        ["stage", "processing_time_ms", "memory_usage_mb"],
    )
    op.create_index(
        "idx_processing_stage_time", "processing_states", ["stage", "started_at"]
    )
    op.create_index(
        "idx_processing_success_stage",
        "processing_states",
        ["success", "stage", "started_at"],
    )
    op.create_index(
        "idx_processing_worker_time", "processing_states", ["worker_id", "started_at"]
    )
    op.create_index(
        op.f("ix_processing_states_celery_task_id"),
        "processing_states",
        ["celery_task_id"],
    )
    op.create_index(
        op.f("ix_processing_states_completed_at"), "processing_states", ["completed_at"]
    )
    op.create_index(
        op.f("ix_processing_states_overall_confidence"),
        "processing_states",
        ["overall_confidence"],
    )
    op.create_index(op.f("ix_processing_states_stage"), "processing_states", ["stage"])
    op.create_index(
        op.f("ix_processing_states_stage_version"),
        "processing_states",
        ["stage_version"],
    )
    op.create_index(
        op.f("ix_processing_states_started_at"), "processing_states", ["started_at"]
    )
    op.create_index(
        op.f("ix_processing_states_success"), "processing_states", ["success"]
    )
    op.create_index(
        op.f("ix_processing_states_worker_id"), "processing_states", ["worker_id"]
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("processing_states")
    op.drop_table("jobs")

    # Drop enum types
    op.execute("DROP TYPE workflowstatus")
    op.execute("DROP TYPE workflowstage")
