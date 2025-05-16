"""
Entry point for the A2 Discord bot.
"""
import os
import sys
from pathlib import Path

# Import the main bot class
from bot import A2Bot

# Import transformer utilities
from utils.transformers_helper import initialize_transformers

# Import logging utilities
from utils.logging_helper import setup_logging
from config import DATA_DIR

if __name__ == "__main__":
    # Set up logging first
    logger = setup_logging(DATA_DIR)
    logger.info("Bot starting from main entry point...")
    
    # Print startup banner for better logs
    logger.info("===== A2 Discord Bot Starting =====")
    logger.info(f"Python version: {sys.version}")
    
    # Initialize transformers only if not disabled
    if os.getenv("DISABLE_TRANSFORMERS", "0") != "1":
        logger.info("Initializing transformers...")
        initialize_transformers()
    else:
        logger.info("Transformers disabled by environment variable")
    
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
    
    # Create and run the bot
    bot = A2Bot(token, app_id, openai_api_key, openai_org_id, openai_project_id)
    logger.info("Starting A2 Discord bot...")
    bot.run()
