from logging.config import fileConfig
import os
from sqlalchemy import create_engine
from sqlalchemy import pool
from alembic import context
from sqlmodel import SQLModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Alembic Config object
config = context.config

# Interpret the config file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata to SQLModel's metadata
target_metadata = SQLModel.metadata

# Define the synchronous database URL for Alembic
sync_postgres_url = os.getenv("POSTGRES_URL").replace("postgresql+asyncpg://", "postgresql://")

# Run migrations offline
def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url") or sync_postgres_url
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()

# Run migrations online
def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Use a synchronous engine for Alembic
    connectable = create_engine(
        sync_postgres_url,
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

# Execute migrations
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()