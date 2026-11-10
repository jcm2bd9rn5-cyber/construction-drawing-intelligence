"""Track ingestion outcomes and enforce source hash idempotency."""
from alembic import op
import sqlalchemy as sa

revision = "0004_ingestion_job_stats"
down_revision = "0003_version_hash"


def upgrade():
    # Earlier ingestion committed document/version rows even when no page chunk
    # survived. They are not searchable and must not become an idempotency hit.
    op.execute(
        """
        UPDATE documents SET current_version_id = NULL
        WHERE current_version_id IN (
          SELECT dv.id FROM document_versions dv
          WHERE NOT EXISTS (
            SELECT 1 FROM pages p JOIN page_chunks pc ON pc.page_id = p.id
            WHERE p.document_version_id = dv.id
          )
        )
        """
    )
    op.execute(
        """
        DELETE FROM document_versions
        WHERE NOT EXISTS (
          SELECT 1 FROM pages p JOIN page_chunks pc ON pc.page_id = p.id
          WHERE p.document_version_id = document_versions.id
        )
        """
    )
    op.execute(
        """
        DELETE FROM documents
        WHERE NOT EXISTS (
          SELECT 1 FROM document_versions dv WHERE dv.document_id = documents.id
        )
        """
    )

    # Keep the most recent copy from any pre-constraint duplicate set.
    op.execute(
        """
        UPDATE documents SET current_version_id = NULL
        WHERE current_version_id IN (
          SELECT id FROM document_versions
          WHERE content_hash IS NOT NULL AND id NOT IN (
            SELECT MAX(id) FROM document_versions
            WHERE content_hash IS NOT NULL GROUP BY file_path, content_hash
          )
        )
        """
    )
    op.execute(
        """
        DELETE FROM document_versions
        WHERE content_hash IS NOT NULL AND id NOT IN (
          SELECT MAX(id) FROM document_versions
          WHERE content_hash IS NOT NULL GROUP BY file_path, content_hash
        )
        """
    )

    with op.batch_alter_table("document_versions") as batch:
        batch.create_unique_constraint(
            "uq_document_versions_source_hash", ["file_path", "content_hash"]
        )
    with op.batch_alter_table("processing_jobs") as batch:
        batch.add_column(sa.Column("pages_created", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("chunks_created", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("embeddings_created", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("vision_pages", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("text_pages", sa.Integer(), nullable=False, server_default="0"))
        batch.add_column(sa.Column("searchable", sa.Boolean(), nullable=False, server_default=sa.false()))
        batch.add_column(sa.Column("warnings", sa.JSON(), nullable=False, server_default="[]"))


def downgrade():
    with op.batch_alter_table("processing_jobs") as batch:
        batch.drop_column("warnings")
        batch.drop_column("searchable")
        batch.drop_column("text_pages")
        batch.drop_column("vision_pages")
        batch.drop_column("embeddings_created")
        batch.drop_column("chunks_created")
        batch.drop_column("pages_created")
    with op.batch_alter_table("document_versions") as batch:
        batch.drop_constraint("uq_document_versions_source_hash", type_="unique")
