# scripts/__init__.py
# This file makes the scripts directory a Python package

# Import the migration script
from scripts.migrate_to_postgres import migrate_data

__all__ = ['migrate_data']
