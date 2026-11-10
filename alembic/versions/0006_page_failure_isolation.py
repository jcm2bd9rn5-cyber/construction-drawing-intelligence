"""Track isolated page failures during ingestion."""
from alembic import op
import sqlalchemy as sa

revision = "0006_page_failure_isolation"
down_revision = "0005_quota_observability"


def upgrade():
    with op.batch_alter_table("pages") as batch:
        batch.add_column(sa.Column("status", sa.String(32), nullable=False, server_default="COMPLETED"))
        batch.add_column(sa.Column("error", sa.Text(), nullable=True))
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("pages_failed", sa.Integer(), nullable=False, server_default="0"))


def downgrade():
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("pages_failed")
    with op.batch_alter_table("pages") as batch:
        batch.drop_column("error")
        batch.drop_column("status")
