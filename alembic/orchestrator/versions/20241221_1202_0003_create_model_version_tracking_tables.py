"""Create model version tracking tables

Revision ID: 0003
Revises: 0002
Create Date: 2024-12-21 12:02:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create model_versions table
    op.create_table('model_versions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('version', sa.String(length=20), nullable=False),
        sa.Column('version_hash', sa.String(length=64), nullable=True),
        sa.Column('model_type', sa.String(length=30), nullable=False),
        sa.Column('architecture', sa.String(length=50), nullable=True),
        sa.Column('framework', sa.String(length=20), nullable=False),
        sa.Column('framework_version', sa.String(length=15), nullable=True),
        sa.Column('model_path', sa.String(length=512), nullable=True),
        sa.Column('model_size_mb', sa.Integer(), nullable=True),
        sa.Column('model_checksum', sa.String(length=64), nullable=True),
        sa.Column('config_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('training_dataset', sa.String(length=100), nullable=True),
        sa.Column('training_samples', sa.Integer(), nullable=True),
        sa.Column('training_epochs', sa.Integer(), nullable=True),
        sa.Column('training_time_hours', sa.Float(), nullable=True),
        sa.Column('training_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('benchmark_accuracy', sa.Float(), nullable=True),
        sa.Column('benchmark_f1_score', sa.Float(), nullable=True),
        sa.Column('benchmark_precision', sa.Float(), nullable=True),
        sa.Column('benchmark_recall', sa.Float(), nullable=True),
        sa.Column('benchmark_inference_time_ms', sa.Float(), nullable=True),
        sa.Column('benchmark_memory_usage_mb', sa.Integer(), nullable=True),
        sa.Column('brand_performances', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('supported_brands', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('deployment_status', sa.String(length=15), nullable=False),
        sa.Column('is_production_ready', sa.Boolean(), nullable=False),
        sa.Column('deployed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deprecated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('min_python_version', sa.String(length=10), nullable=True),
        sa.Column('required_packages', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('gpu_required', sa.Boolean(), nullable=False),
        sa.Column('min_gpu_memory_gb', sa.Integer(), nullable=True),
        sa.Column('min_system_memory_gb', sa.Integer(), nullable=True),
        sa.Column('validation_passed', sa.Boolean(), nullable=False),
        sa.Column('validation_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('security_scan_passed', sa.Boolean(), nullable=False),
        sa.Column('security_scan_results', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('total_jobs_processed', sa.Integer(), nullable=False),
        sa.Column('avg_processing_time_ms', sa.Float(), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_by', sa.String(length=100), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('release_notes', sa.Text(), nullable=True),
        sa.Column('breaking_changes', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('migration_guide', sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_name', 'version', name='unique_model_version'),
        sa.CheckConstraint('model_size_mb IS NULL OR model_size_mb > 0', name='check_positive_model_size'),
        sa.CheckConstraint('training_samples IS NULL OR training_samples > 0', name='check_positive_training_samples'),
        sa.CheckConstraint('training_epochs IS NULL OR training_epochs > 0', name='check_positive_training_epochs'),
        sa.CheckConstraint('training_time_hours IS NULL OR training_time_hours > 0', name='check_positive_training_time'),
        sa.CheckConstraint('benchmark_accuracy IS NULL OR (benchmark_accuracy >= 0.0 AND benchmark_accuracy <= 1.0)', name='check_accuracy_range'),
        sa.CheckConstraint('benchmark_f1_score IS NULL OR (benchmark_f1_score >= 0.0 AND benchmark_f1_score <= 1.0)', name='check_f1_range'),
        sa.CheckConstraint('benchmark_precision IS NULL OR (benchmark_precision >= 0.0 AND benchmark_precision <= 1.0)', name='check_precision_range'),
        sa.CheckConstraint('benchmark_recall IS NULL OR (benchmark_recall >= 0.0 AND benchmark_recall <= 1.0)', name='check_recall_range'),
        sa.CheckConstraint('success_rate IS NULL OR (success_rate >= 0.0 AND success_rate <= 1.0)', name='check_success_rate_range'),
        sa.CheckConstraint("deployment_status IN ('development', 'testing', 'staging', 'production', 'deprecated', 'retired')", name='check_valid_deployment_status'),
        sa.CheckConstraint("model_type IN ('transformer', 'cnn', 'traditional_cv', 'ensemble', 'rule_based', 'hybrid')", name='check_valid_model_type'),
        sa.CheckConstraint("framework IN ('pytorch', 'tensorflow', 'onnx', 'scikit_learn', 'opencv', 'custom')", name='check_valid_framework')
    )
    
    # Create indexes for model_versions
    op.create_index('idx_model_brands', 'model_versions', ['model_name', 'deployment_status'])
    op.create_index('idx_model_deployment', 'model_versions', ['deployment_status', 'is_production_ready'])
    op.create_index('idx_model_name_version', 'model_versions', ['model_name', 'version'])
    op.create_index('idx_model_performance', 'model_versions', ['model_name', 'benchmark_accuracy', 'success_rate'])
    op.create_index('idx_model_timeline', 'model_versions', ['model_name', 'created_at'])
    op.create_index('idx_model_usage', 'model_versions', ['model_name', 'last_used_at', 'total_jobs_processed'])
    op.create_index(op.f('ix_model_versions_benchmark_accuracy'), 'model_versions', ['benchmark_accuracy'])
    op.create_index(op.f('ix_model_versions_created_at'), 'model_versions', ['created_at'])
    op.create_index(op.f('ix_model_versions_deployed_at'), 'model_versions', ['deployed_at'])
    op.create_index(op.f('ix_model_versions_deployment_status'), 'model_versions', ['deployment_status'])
    op.create_index(op.f('ix_model_versions_is_production_ready'), 'model_versions', ['is_production_ready'])
    op.create_index(op.f('ix_model_versions_model_name'), 'model_versions', ['model_name'])
    op.create_index(op.f('ix_model_versions_model_type'), 'model_versions', ['model_type'])
    op.create_index(op.f('ix_model_versions_success_rate'), 'model_versions', ['success_rate'])
    op.create_index(op.f('ix_model_versions_version'), 'model_versions', ['version'])
    op.create_index(op.f('ix_model_versions_version_hash'), 'model_versions', ['version_hash'])

    # Create model_deployments table
    op.create_table('model_deployments',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('model_name', sa.String(length=50), nullable=False),
        sa.Column('model_version', sa.String(length=20), nullable=False),
        sa.Column('deployment_type', sa.String(length=20), nullable=False),
        sa.Column('target_environment', sa.String(length=20), nullable=False),
        sa.Column('deployment_status', sa.String(length=15), nullable=False),
        sa.Column('deployment_started_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('deployment_completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('rollback_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('jobs_processed', sa.Integer(), nullable=False),
        sa.Column('avg_processing_time_ms', sa.Float(), nullable=True),
        sa.Column('success_rate', sa.Float(), nullable=True),
        sa.Column('error_rate', sa.Float(), nullable=True),
        sa.Column('health_checks_passed', sa.Integer(), nullable=False),
        sa.Column('health_checks_failed', sa.Integer(), nullable=False),
        sa.Column('last_health_check_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('deployed_by', sa.String(length=100), nullable=True),
        sa.Column('deployment_reason', sa.Text(), nullable=True),
        sa.Column('rollback_reason', sa.Text(), nullable=True),
        sa.Column('deployment_config', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint('jobs_processed >= 0', name='check_non_negative_jobs_processed'),
        sa.CheckConstraint('avg_processing_time_ms IS NULL OR avg_processing_time_ms > 0', name='check_positive_processing_time'),
        sa.CheckConstraint('success_rate IS NULL OR (success_rate >= 0.0 AND success_rate <= 1.0)', name='check_success_rate_range'),
        sa.CheckConstraint('error_rate IS NULL OR (error_rate >= 0.0 AND error_rate <= 1.0)', name='check_error_rate_range'),
        sa.CheckConstraint('health_checks_passed >= 0', name='check_non_negative_health_passed'),
        sa.CheckConstraint('health_checks_failed >= 0', name='check_non_negative_health_failed'),
        sa.CheckConstraint("deployment_type IN ('canary', 'blue_green', 'rolling', 'full', 'hotfix')", name='check_valid_deployment_type'),
        sa.CheckConstraint("target_environment IN ('development', 'testing', 'staging', 'production')", name='check_valid_target_environment'),
        sa.CheckConstraint("deployment_status IN ('deploying', 'deployed', 'failed', 'rolled_back', 'superseded')", name='check_valid_deployment_status'),
        sa.CheckConstraint('deployment_completed_at IS NULL OR deployment_completed_at >= deployment_started_at', name='check_completed_after_started')
    )
    
    # Create indexes for model_deployments
    op.create_index('idx_deployment_health', 'model_deployments', ['target_environment', 'last_health_check_at'])
    op.create_index('idx_deployment_model_env', 'model_deployments', ['model_name', 'target_environment', 'deployment_started_at'])
    op.create_index('idx_deployment_performance', 'model_deployments', ['target_environment', 'success_rate', 'error_rate'])
    op.create_index('idx_deployment_status_time', 'model_deployments', ['deployment_status', 'deployment_started_at'])
    op.create_index(op.f('ix_model_deployments_deployment_started_at'), 'model_deployments', ['deployment_started_at'])
    op.create_index(op.f('ix_model_deployments_deployment_status'), 'model_deployments', ['deployment_status'])
    op.create_index(op.f('ix_model_deployments_deployment_type'), 'model_deployments', ['deployment_type'])
    op.create_index(op.f('ix_model_deployments_model_name'), 'model_deployments', ['model_name'])
    op.create_index(op.f('ix_model_deployments_model_version'), 'model_deployments', ['model_version'])
    op.create_index(op.f('ix_model_deployments_target_environment'), 'model_deployments', ['target_environment'])


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table('model_deployments')
    op.drop_table('model_versions')