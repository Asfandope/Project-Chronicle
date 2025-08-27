"""Create brand config, drift detection and model version tables

Revision ID: 0002
Revises: 0001
Create Date: 2024-12-21 12:01:00.000000

"""
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create brand_config_history table
    op.create_table(
        "brand_config_history",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("brand_name", sa.String(length=50), nullable=False),
        sa.Column("config_version", sa.String(length=20), nullable=False),
        sa.Column(
            "config_data", postgresql.JSONB(astext_type=sa.Text()), nullable=False
        ),
        sa.Column("config_schema_version", sa.String(length=10), nullable=False),
        sa.Column("change_type", sa.String(length=20), nullable=False),
        sa.Column("change_summary", sa.Text(), nullable=True),
        sa.Column(
            "changed_fields", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False),
        sa.Column("deployed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deployment_status", sa.String(length=20), nullable=False),
        sa.Column("validation_passed", sa.Boolean(), nullable=False),
        sa.Column(
            "validation_errors", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "validation_warnings",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("created_by", sa.String(length=100), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("rolled_back_from_version", sa.String(length=20), nullable=True),
        sa.Column("rollback_reason", sa.Text(), nullable=True),
        sa.Column(
            "performance_impact", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "accuracy_impact", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "change_type IN ('create', 'update', 'rollback', 'delete', 'activate', 'deactivate')",
            name="check_valid_change_type",
        ),
        sa.CheckConstraint(
            "deployment_status IN ('draft', 'deployed', 'rolled_back', 'archived')",
            name="check_valid_deployment_status",
        ),
        sa.CheckConstraint(
            "deployed_at IS NULL OR deployment_status = 'deployed'",
            name="check_deployed_status_consistency",
        ),
        sa.CheckConstraint(
            "is_active = false OR deployment_status = 'deployed'",
            name="check_active_is_deployed",
        ),
        sa.UniqueConstraint(
            "brand_name",
            "is_active",
            name="unique_active_config_per_brand",
            postgresql_where=sa.text("is_active = true"),
        ),
    )

    # Create indexes for brand_config_history
    op.create_index(
        "idx_brand_config_active",
        "brand_config_history",
        ["brand_name", "is_active", "deployed_at"],
    )
    op.create_index(
        "idx_brand_config_brand_version",
        "brand_config_history",
        ["brand_name", "config_version"],
    )
    op.create_index(
        "idx_brand_config_changes",
        "brand_config_history",
        ["brand_name", "change_type", "created_at"],
    )
    op.create_index(
        "idx_brand_config_deployment",
        "brand_config_history",
        ["deployment_status", "deployed_at"],
    )
    op.create_index(
        "idx_brand_config_timeline",
        "brand_config_history",
        ["brand_name", "created_at"],
    )
    op.create_index(
        "idx_brand_config_validation",
        "brand_config_history",
        ["validation_passed", "deployment_status"],
    )
    op.create_index(
        op.f("ix_brand_config_history_brand_name"),
        "brand_config_history",
        ["brand_name"],
    )
    op.create_index(
        op.f("ix_brand_config_history_change_type"),
        "brand_config_history",
        ["change_type"],
    )
    op.create_index(
        op.f("ix_brand_config_history_config_version"),
        "brand_config_history",
        ["config_version"],
    )
    op.create_index(
        op.f("ix_brand_config_history_created_at"),
        "brand_config_history",
        ["created_at"],
    )
    op.create_index(
        op.f("ix_brand_config_history_deployed_at"),
        "brand_config_history",
        ["deployed_at"],
    )
    op.create_index(
        op.f("ix_brand_config_history_deployment_status"),
        "brand_config_history",
        ["deployment_status"],
    )
    op.create_index(
        op.f("ix_brand_config_history_is_active"), "brand_config_history", ["is_active"]
    )
    op.create_index(
        op.f("ix_brand_config_history_rolled_back_from_version"),
        "brand_config_history",
        ["rolled_back_from_version"],
    )

    # Create brand_config_audit_log table
    op.create_table(
        "brand_config_audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("brand_name", sa.String(length=50), nullable=False),
        sa.Column("config_version", sa.String(length=20), nullable=False),
        sa.Column("access_type", sa.String(length=20), nullable=False),
        sa.Column("accessed_by", sa.String(length=100), nullable=True),
        sa.Column("access_source", sa.String(length=50), nullable=True),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "access_context", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "access_type IN ('read', 'write', 'deploy', 'rollback', 'validate', 'compare')",
            name="check_valid_access_type",
        ),
    )

    # Create indexes for brand_config_audit_log
    op.create_index(
        "idx_audit_access_type",
        "brand_config_audit_log",
        ["access_type", "accessed_at"],
    )
    op.create_index(
        "idx_audit_brand_time", "brand_config_audit_log", ["brand_name", "accessed_at"]
    )
    op.create_index(
        "idx_audit_job_access", "brand_config_audit_log", ["job_id", "accessed_at"]
    )
    op.create_index(
        "idx_audit_source_brand",
        "brand_config_audit_log",
        ["access_source", "brand_name", "accessed_at"],
    )
    op.create_index(
        op.f("ix_brand_config_audit_log_access_type"),
        "brand_config_audit_log",
        ["access_type"],
    )
    op.create_index(
        op.f("ix_brand_config_audit_log_accessed_at"),
        "brand_config_audit_log",
        ["accessed_at"],
    )
    op.create_index(
        op.f("ix_brand_config_audit_log_brand_name"),
        "brand_config_audit_log",
        ["brand_name"],
    )
    op.create_index(
        op.f("ix_brand_config_audit_log_config_version"),
        "brand_config_audit_log",
        ["config_version"],
    )
    op.create_index(
        op.f("ix_brand_config_audit_log_job_id"), "brand_config_audit_log", ["job_id"]
    )

    # Create drift_measurements table
    op.create_table(
        "drift_measurements",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("brand_name", sa.String(length=50), nullable=False),
        sa.Column(
            "measurement_window_start", sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column("measurement_window_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("measurement_type", sa.String(length=30), nullable=False),
        sa.Column("model_version", sa.String(length=20), nullable=False),
        sa.Column("config_version", sa.String(length=20), nullable=False),
        sa.Column("baseline_period_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("baseline_period_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("current_sample_size", sa.Integer(), nullable=False),
        sa.Column("baseline_sample_size", sa.Integer(), nullable=False),
        sa.Column("drift_score", sa.Float(), nullable=False),
        sa.Column("statistical_significance", sa.Float(), nullable=True),
        sa.Column("confidence_interval_lower", sa.Float(), nullable=True),
        sa.Column("confidence_interval_upper", sa.Float(), nullable=True),
        sa.Column(
            "field_drift_scores", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "current_distribution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "baseline_distribution",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "distribution_comparison",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("current_accuracy", sa.Float(), nullable=True),
        sa.Column("baseline_accuracy", sa.Float(), nullable=True),
        sa.Column("accuracy_degradation", sa.Float(), nullable=True),
        sa.Column("current_avg_confidence", sa.Float(), nullable=True),
        sa.Column("baseline_avg_confidence", sa.Float(), nullable=True),
        sa.Column("confidence_drift", sa.Float(), nullable=True),
        sa.Column("confidence_calibration_drift", sa.Float(), nullable=True),
        sa.Column(
            "error_pattern_changes",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "new_error_patterns", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("drift_detected", sa.Boolean(), nullable=False),
        sa.Column("drift_severity", sa.String(length=10), nullable=False),
        sa.Column("requires_intervention", sa.Boolean(), nullable=False),
        sa.Column("alert_sent", sa.Boolean(), nullable=False),
        sa.Column("alert_sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("action_taken", sa.String(length=50), nullable=True),
        sa.Column("action_taken_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("measurement_quality", sa.String(length=10), nullable=False),
        sa.Column(
            "quality_issues", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by", sa.String(length=50), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "current_sample_size > 0", name="check_positive_current_sample"
        ),
        sa.CheckConstraint(
            "baseline_sample_size > 0", name="check_positive_baseline_sample"
        ),
        sa.CheckConstraint(
            "drift_score >= 0.0 AND drift_score <= 1.0", name="check_drift_score_range"
        ),
        sa.CheckConstraint(
            "current_accuracy IS NULL OR (current_accuracy >= 0.0 AND current_accuracy <= 1.0)",
            name="check_current_accuracy_range",
        ),
        sa.CheckConstraint(
            "baseline_accuracy IS NULL OR (baseline_accuracy >= 0.0 AND baseline_accuracy <= 1.0)",
            name="check_baseline_accuracy_range",
        ),
        sa.CheckConstraint(
            "measurement_type IN ('accuracy', 'confidence', 'error_pattern', 'performance', 'distribution')",
            name="check_valid_measurement_type",
        ),
        sa.CheckConstraint(
            "drift_severity IN ('low', 'medium', 'high', 'critical')",
            name="check_valid_drift_severity",
        ),
        sa.CheckConstraint(
            "measurement_quality IN ('high', 'medium', 'low')",
            name="check_valid_measurement_quality",
        ),
        sa.CheckConstraint(
            "measurement_window_end > measurement_window_start",
            name="check_valid_measurement_window",
        ),
        sa.CheckConstraint(
            "baseline_period_end > baseline_period_start",
            name="check_valid_baseline_period",
        ),
    )

    # Create indexes for drift_measurements
    op.create_index(
        "idx_drift_accuracy_degradation",
        "drift_measurements",
        ["brand_name", "accuracy_degradation", "created_at"],
    )
    op.create_index(
        "idx_drift_alert_status",
        "drift_measurements",
        ["alert_sent", "requires_intervention", "created_at"],
    )
    op.create_index(
        "idx_drift_brand_time",
        "drift_measurements",
        ["brand_name", "measurement_window_start"],
    )
    op.create_index(
        "idx_drift_detection_status",
        "drift_measurements",
        ["drift_detected", "drift_severity", "created_at"],
    )
    op.create_index(
        "idx_drift_intervention",
        "drift_measurements",
        ["requires_intervention", "action_taken", "created_at"],
    )
    op.create_index(
        "idx_drift_model_config",
        "drift_measurements",
        ["model_version", "config_version", "brand_name"],
    )
    op.create_index(
        "idx_drift_monitoring",
        "drift_measurements",
        ["brand_name", "drift_detected", "drift_severity", "created_at"],
    )
    op.create_index(
        op.f("ix_drift_measurements_accuracy_degradation"),
        "drift_measurements",
        ["accuracy_degradation"],
    )
    op.create_index(
        op.f("ix_drift_measurements_brand_name"), "drift_measurements", ["brand_name"]
    )
    op.create_index(
        op.f("ix_drift_measurements_config_version"),
        "drift_measurements",
        ["config_version"],
    )
    op.create_index(
        op.f("ix_drift_measurements_created_at"), "drift_measurements", ["created_at"]
    )
    op.create_index(
        op.f("ix_drift_measurements_current_accuracy"),
        "drift_measurements",
        ["current_accuracy"],
    )
    op.create_index(
        op.f("ix_drift_measurements_drift_detected"),
        "drift_measurements",
        ["drift_detected"],
    )
    op.create_index(
        op.f("ix_drift_measurements_drift_score"), "drift_measurements", ["drift_score"]
    )
    op.create_index(
        op.f("ix_drift_measurements_drift_severity"),
        "drift_measurements",
        ["drift_severity"],
    )
    op.create_index(
        op.f("ix_drift_measurements_measurement_type"),
        "drift_measurements",
        ["measurement_type"],
    )
    op.create_index(
        op.f("ix_drift_measurements_measurement_window_start"),
        "drift_measurements",
        ["measurement_window_start"],
    )
    op.create_index(
        op.f("ix_drift_measurements_model_version"),
        "drift_measurements",
        ["model_version"],
    )
    op.create_index(
        op.f("ix_drift_measurements_requires_intervention"),
        "drift_measurements",
        ["requires_intervention"],
    )

    # Create drift_alerts table
    op.create_table(
        "drift_alerts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "drift_measurement_id", postgresql.UUID(as_uuid=True), nullable=False
        ),
        sa.Column("alert_level", sa.String(length=10), nullable=False),
        sa.Column("alert_message", sa.Text(), nullable=False),
        sa.Column(
            "alert_details", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("recipients", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "notification_channels",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
        sa.Column("status", sa.String(length=15), nullable=False),
        sa.Column("acknowledged_by", sa.String(length=100), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolved_by", sa.String(length=100), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution_action", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["drift_measurement_id"],
            ["drift_measurements.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "alert_level IN ('warning', 'critical', 'emergency')",
            name="check_valid_alert_level",
        ),
        sa.CheckConstraint(
            "status IN ('sent', 'acknowledged', 'resolved', 'suppressed', 'failed')",
            name="check_valid_alert_status",
        ),
    )

    # Create indexes for drift_alerts
    op.create_index(
        "idx_alert_drift_measurement", "drift_alerts", ["drift_measurement_id"]
    )
    op.create_index(
        "idx_alert_level_status",
        "drift_alerts",
        ["alert_level", "status", "created_at"],
    )
    op.create_index("idx_alert_resolution", "drift_alerts", ["status", "resolved_at"])
    op.create_index(
        op.f("ix_drift_alerts_alert_level"), "drift_alerts", ["alert_level"]
    )
    op.create_index(op.f("ix_drift_alerts_created_at"), "drift_alerts", ["created_at"])
    op.create_index(op.f("ix_drift_alerts_status"), "drift_alerts", ["status"])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("drift_alerts")
    op.drop_table("drift_measurements")
    op.drop_table("brand_config_audit_log")
    op.drop_table("brand_config_history")
