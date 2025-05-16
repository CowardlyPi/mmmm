"""
Logging helper for the A2 Discord bot.
"""
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Global logger
logger = None

def setup_logging(data_dir, log_level=logging.INFO):
    """
    Initialize logging with console and file handlers.
    
    Args:
        data_dir (Path): Directory where log files will be stored
        log_level (int): Logging level (default: INFO)
    
    Returns:
        logger: Configured logger instance
    """
    global logger
    
    if logger is not None:
        return logger  # Already configured
    
    # Ensure logs directory exists
    logs_dir = Path(data_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('a2bot')
    logger.setLevel(log_level)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(console_formatter)
    
    # Create file handler with rotation
    log_file = logs_dir / "a2bot.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,  # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Log startup message
    logger.info("===== A2 Discord Bot Logging Initialized =====")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Python version: {sys.version}")
    
    return logger

def get_logger():
    """
    Get the global logger instance.
    
    Returns:
        logger: The configured logger instance or None if not set up
    """
    global logger
    return logger
