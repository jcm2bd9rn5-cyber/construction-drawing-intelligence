"""Track source digest for idempotent ingestion."""
from alembic import op
import sqlalchemy as sa

revision = "0003_version_hash"
down_revision = "0002_chunk_vector"

def upgrade():
    with op.batch_alter_table("document_versions") as batch:
        batch.add_column(sa.Column("content_hash", sa.String(64), nullable=True))
        batch.create_index("ix_document_versions_hash", ["content_hash"])

def downgrade():
    with op.batch_alter_table("document_versions") as batch:
        batch.drop_index("ix_document_versions_hash")
        batch.drop_column("content_hash")
