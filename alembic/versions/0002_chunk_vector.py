"""Store document chunk embeddings in pgvector on PostgreSQL."""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import VECTOR

revision = "0002_chunk_vector"
down_revision = "0001_initial"

def upgrade():
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    # The initial schema used JSON. Existing embedded rows were never written
    # by that implementation, so the new typed column starts empty.
    op.drop_column("page_chunks", "embedding")
    op.add_column("page_chunks", sa.Column("embedding", VECTOR(768), nullable=True))
    op.create_index("ix_page_chunks_embedding_hnsw", "page_chunks", ["embedding"], postgresql_using="hnsw", postgresql_ops={"embedding": "vector_cosine_ops"})

def downgrade():
    if op.get_bind().dialect.name != "postgresql":
        return
    op.drop_index("ix_page_chunks_embedding_hnsw", table_name="page_chunks")
    op.drop_column("page_chunks", "embedding")
    op.add_column("page_chunks", sa.Column("embedding", sa.JSON(), nullable=True))
