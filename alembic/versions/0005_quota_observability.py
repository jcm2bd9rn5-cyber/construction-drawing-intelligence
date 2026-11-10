"""Record quota failures and embedding request efficiency."""
from alembic import op
import sqlalchemy as sa

revision = "0005_quota_observability"
down_revision = "0004_ingestion_job_stats"


def upgrade():
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("error_type", sa.String(64), nullable=True))
        batch.add_column(sa.Column("http_status", sa.Integer(), nullable=True))
        batch.add_column(sa.Column("retry_after_seconds", sa.Float(), nullable=True))
        batch.add_column(sa.Column("quota_metric", sa.String(255), nullable=True))
        batch.add_column(sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("embedding_api_requests", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("embedding_batches", sa.Integer(), nullable=False, server_default="0"))


def downgrade():
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("embedding_batches")
        batch.drop_column("embedding_api_requests")
        batch.drop_column("attempts")
        batch.drop_column("quota_metric")
        batch.drop_column("retry_after_seconds")
        batch.drop_column("http_status")
        batch.drop_column("error_type")
