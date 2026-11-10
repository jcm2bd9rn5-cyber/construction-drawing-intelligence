"""Index joins used by current-version and evidence retrieval."""
from alembic import op

revision = "0007_search_foreign_key_indexes"
down_revision = "0006_page_failure_isolation"


def upgrade():
    op.create_index("ix_documents_drawing_number", "documents", ["drawing_number"])
    op.create_index("ix_documents_current_version_id", "documents", ["current_version_id"])
    op.create_index("ix_document_versions_document_id", "document_versions", ["document_id"])
    op.create_index("ix_pages_document_version_id", "pages", ["document_version_id"])
    op.create_index("ix_page_chunks_page_id", "page_chunks", ["page_id"])


def downgrade():
    op.drop_index("ix_page_chunks_page_id", table_name="page_chunks")
    op.drop_index("ix_pages_document_version_id", table_name="pages")
    op.drop_index("ix_document_versions_document_id", table_name="document_versions")
    op.drop_index("ix_documents_current_version_id", table_name="documents")
    op.drop_index("ix_documents_drawing_number", table_name="documents")
