"""
Entry point for the A2 Discord bot with modular architecture.
"""
import sys
from core.bot import A2Bot
from core.initialization import initialize_bot
from utils.logging_helper import get_logger

if __name__ == "__main__":
    # Set up logging and initialize configuration
    config, storage_manager = initialize_bot()
    logger = get_logger()
    
    # Print startup banner for better logs
    logger.info("===== A2 Discord Bot Starting =====")
    logger.info(f"Python version: {sys.version}")
    
    # Note: We no longer initialize transformers here - they will be loaded on demand
    transformers_status = "using on-demand loading"
    if os.getenv("DISABLE_TRANSFORMERS", "0") == "1":
        transformers_status = "disabled via environment variable"
    logger.info(f"Transformer models: {transformers_status}")
    
    # Create and run the bot
    bot = A2Bot(
        token=config["token"],
        app_id=config["app_id"],
        openai_api_key=config["openai_api_key"],
        openai_org_id=config["openai_org_id"],
        openai_project_id=config["openai_project_id"],
        storage_manager=storage_manager,
        batch_size=config["batch_size"]
    )
    
    logger.info("Starting A2 Discord bot...")
    bot.run()
