"""${message}\nRevision ID: ${up_revision}\n"""
from alembic import op
import sqlalchemy as sa
${upgrades if upgrades else "def upgrade(): pass"}
${downgrades if downgrades else "def downgrade(): pass"}
