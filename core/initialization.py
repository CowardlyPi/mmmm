"""
Initialization utilities for the A2 Discord bot.
"""
import os
import sys
from pathlib import Path
from utils.logging_helper import get_logger, setup_logging

def initialize_bot(data_dir=None):
    """
    Initialize bot configuration and components
    
    Args:
        data_dir: Optional custom data directory path
        
    Returns:
        tuple: (config_dict, storage_manager)
    """
    # Import configuration
    from config import DATA_DIR as CONFIG_DATA_DIR
    
    # Use provided data directory or default from config
    data_dir = data_dir or CONFIG_DATA_DIR
    
    # Set up logging
    logger = setup_logging(data_dir)
    logger.info("Initializing A2 Discord bot...")
    
    # Get environment variables
    token = os.getenv("DISCORD_TOKEN")
    if not token:
        logger.error("ERROR: DISCORD_TOKEN environment variable is not set")
        sys.exit(1)
        
    app_id = os.getenv("DISCORD_APP_ID", "")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("ERROR: OPENAI_API_KEY environment variable is not set")
        sys.exit(1)
        
    openai_org_id = os.getenv("OPENAI_ORG_ID", "")
    openai_project_id = os.getenv("OPENAI_PROJECT_ID", "")
    
    # Get batch size for pagination from env or use default
    batch_size = int(os.getenv("DATA_BATCH_SIZE", "50"))
    logger.info(f"Using data batch size: {batch_size}")
    
    # Determine which storage manager to use
    use_postgres = os.getenv("USE_POSTGRES", "0") == "1"
    
    # Initialize appropriate storage manager
    if use_postgres:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            logger.error("ERROR: DATABASE_URL environment variable is not set for PostgreSQL mode")
            sys.exit(1)
            
        logger.info(f"Using PostgreSQL storage mode with database: {database_url.split('@')[-1]}")
        
        from managers.postgres_storage import PostgreSQLStorageManager
        storage_manager = PostgreSQLStorageManager(database_url=database_url, data_dir=data_dir)
    else:
        # Use traditional file-based storage
        logger.info("Using file-based storage mode")
        
        from managers.storage import StorageManager
        storage_manager = StorageManager(
            data_dir=data_dir,
            users_dir=data_dir / "users",
            profiles_dir=data_dir / "users" / "profiles",
            dm_settings_file=data_dir / "dm_enabled_users.json",
            user_profiles_dir=data_dir / "users" / "user_profiles",
            conversations_dir=data_dir / "users" / "conversations"
        )
    
    # Return configuration and storage manager
    config = {
        "token": token,
        "app_id": app_id,
        "openai_api_key": openai_api_key,
        "openai_org_id": openai_org_id,
        "openai_project_id": openai_project_id,
        "batch_size": batch_size,
        "data_dir": data_dir
    }
    
    return config, storage_manager
