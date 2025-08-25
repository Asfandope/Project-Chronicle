-- Initialize databases for development environment

-- Create main database if it doesn't exist
SELECT 'CREATE DATABASE magazine_extractor'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'magazine_extractor')\gexec

-- Create test database if it doesn't exist  
SELECT 'CREATE DATABASE test_magazine_extractor'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'test_magazine_extractor')\gexec

-- Connect to main database and create extensions
\c magazine_extractor;

-- Enable required PostgreSQL extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create database users with appropriate permissions
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'orchestrator_user') THEN
    CREATE USER orchestrator_user WITH PASSWORD 'orchestrator_pass';
  END IF;
  
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = 'evaluation_user') THEN
    CREATE USER evaluation_user WITH PASSWORD 'evaluation_pass';
  END IF;
END
$$;

-- Grant permissions
GRANT CONNECT ON DATABASE magazine_extractor TO orchestrator_user, evaluation_user;
GRANT USAGE ON SCHEMA public TO orchestrator_user, evaluation_user;
GRANT CREATE ON SCHEMA public TO orchestrator_user, evaluation_user;

-- Set up test database
\c test_magazine_extractor;

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

GRANT CONNECT ON DATABASE test_magazine_extractor TO orchestrator_user, evaluation_user;
GRANT USAGE ON SCHEMA public TO orchestrator_user, evaluation_user;
GRANT CREATE ON SCHEMA public TO orchestrator_user, evaluation_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO orchestrator_user, evaluation_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO orchestrator_user, evaluation_user;

-- Switch back to main database
\c magazine_extractor;

-- Create basic monitoring views
CREATE OR REPLACE VIEW job_processing_stats AS
SELECT 
    overall_status,
    brand,
    COUNT(*) as count,
    AVG(processing_time_seconds) as avg_processing_time,
    AVG(accuracy_score) as avg_accuracy,
    MIN(created_at) as earliest_job,
    MAX(created_at) as latest_job
FROM jobs 
GROUP BY overall_status, brand
ORDER BY brand, overall_status;

-- Create indexes for performance
-- (These will be created properly by Alembic migrations, but included here for reference)
-- CREATE INDEX CONCURRENTLY idx_jobs_status_brand ON jobs(overall_status, brand);
-- CREATE INDEX CONCURRENTLY idx_jobs_created_at ON jobs(created_at);
-- CREATE INDEX CONCURRENTLY idx_processing_states_job_id ON processing_states(job_id);

-- Insert sample configuration data for development
-- (This would typically be managed through configuration files)

COMMENT ON DATABASE magazine_extractor IS 'Magazine PDF Extractor - Main database for job management and processing state';
COMMENT ON DATABASE test_magazine_extractor IS 'Magazine PDF Extractor - Test database for automated testing';