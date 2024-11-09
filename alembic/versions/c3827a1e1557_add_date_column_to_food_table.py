"""Add date column to Food table

Revision ID: c3827a1e1557
Revises: 
Create Date: 2024-11-09 12:43:11.176880

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'c3827a1e1557'
down_revision = None
branch_labels = None
depends_on = None

def upgrade() -> None:
    # Add a new `date` column to the `food` table with a default value
    op.add_column('food', sa.Column('date', sa.DateTime, nullable=True, server_default=sa.func.now()))

def downgrade() -> None:
    # Remove the `date` column from the `food` table in case of a downgrade
    op.drop_column('food', 'date')