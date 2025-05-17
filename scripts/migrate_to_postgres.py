"""
Migration script to transfer data from JSON files to PostgreSQL database.
"""
import os
import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DATA_DIR
from managers.storage import StorageManager
from managers.postgres_storage import PostgreSQLStorageManager
from managers.conversation import ConversationManager
from managers.emotion import EmotionManager

# Default directories
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
DM_SETTINGS_FILE = DATA_DIR / "dm_enabled_users.json"
USER_PROFILES_DIR = USERS_DIR / "user_profiles"
CONVERSATIONS_DIR = USERS_DIR / "conversations"

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Migrate A2 bot data from files to PostgreSQL')
    parser.add_argument('--db-url', type=str, required=True, 
                        help='PostgreSQL connection URL (postgres://user:password@host:port/dbname)')
    parser.add_argument('--dry-run', action='store_true', 
                        help='Dry run mode - load data but do not write to database')
    parser.add_argument('--data-dir', type=str, default=None,
                        help=f'Data directory (default: {DATA_DIR})')
    return parser.parse_args()

async def migrate_data(db_url, data_dir=None, dry_run=False):
    """Migrate data from files to PostgreSQL database"""
    # Use provided data directory or default
    if data_dir:
        data_dir = Path(data_dir)
    else:
        data_dir = DATA_DIR
    
    print(f"Starting migration from {data_dir} to PostgreSQL")
    
    # Create managers
    emotion_manager = EmotionManager()
    conversation_manager = ConversationManager()
    
    # Set up file storage manager
    file_storage = StorageManager(
        data_dir=data_dir,
        users_dir=data_dir / "users",
        profiles_dir=data_dir / "users" / "profiles",
        dm_settings_file=data_dir / "dm_enabled_users.json",
        user_profiles_dir=data_dir / "users" / "user_profiles",
        conversations_dir=data_dir / "users" / "conversations"
    )
    
    # Verify data directories exist
    if not file_storage.verify_data_directories():
        print("Error: Data directories not available or not readable")
        return False
    
    # Set up PostgreSQL storage manager
    try:
        db_storage = PostgreSQLStorageManager(database_url=db_url, data_dir=data_dir)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return False
    
    # Check database connection
    if not await db_storage.verify_database_connection():
        print("Error: Could not verify database connection")
        return False
    
    # Load data from files
    print("Loading data from files...")
    try:
        await file_storage.load_data(emotion_manager, conversation_manager)
        user_count = len(emotion_manager.user_emotions)
        print(f"Successfully loaded data for {user_count} users from files")
    except Exception as e:
        print(f"Error loading data from files: {e}")
        return False
    
    # If dry run, stop here
    if dry_run:
        print("Dry run mode - not writing to database")
        print(f"Would have migrated data for {len(emotion_manager.user_emotions)} users")
        return True
    
    # Save data to PostgreSQL
    print("Saving data to PostgreSQL...")
    try:
        await db_storage.save_data(emotion_manager, conversation_manager)
        print(f"Successfully migrated data for {len(emotion_manager.user_emotions)} users to PostgreSQL")
    except Exception as e:
        print(f"Error saving data to PostgreSQL: {e}")
        return False
    
    print("Migration completed successfully")
    return True

if __name__ == "__main__":
    args = parse_args()
    
    # Run migration
    success = asyncio.run(migrate_data(
        db_url=args.db_url,
        data_dir=args.data_dir,
        dry_run=args.dry_run
    ))
    
    # Set exit code based on success
    sys.exit(0 if success else 1)
