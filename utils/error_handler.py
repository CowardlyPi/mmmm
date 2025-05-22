"""
Enhanced error handling and logging utilities for the A2 Discord bot.
"""
import traceback
import functools
import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from pathlib import Path

class A2Logger:
    """Enhanced logger specifically for A2 bot"""
    
    def __init__(self, name: str = 'a2bot'):
        self.logger = logging.getLogger(name)
        self._setup_custom_levels()
    
    def _setup_custom_levels(self):
        """Add custom logging levels for bot-specific events"""
        # Add custom levels
        logging.addLevelName(25, 'INTERACTION')  # Between INFO and WARNING
        logging.addLevelName(35, 'EMOTION')      # Between WARNING and ERROR
        
        def interaction(self, message, *args, **kwargs):
            if self.isEnabledFor(25):
                self._log(25, message, args, **kwargs)
        
        def emotion(self, message, *args, **kwargs):
            if self.isEnabledFor(35):
                self._log(35, message, args, **kwargs)
        
        # Add methods to logger
        logging.Logger.interaction = interaction
        logging.Logger.emotion = emotion
    
    def log_user_interaction(self, user_id: int, command: str, success: bool = True):
        """Log user interactions with structured data"""
        status = "SUCCESS" if success else "FAILED"
        self.logger.interaction(f"User {user_id} - Command: {command} - Status: {status}")
    
    def log_emotion_change(self, user_id: int, emotion: str, old_value: float, new_value: float):
        """Log emotion changes with structured data"""
        change = new_value - old_value
        self.logger.emotion(f"User {user_id} - Emotion: {emotion} - Change: {old_value:.2f} -> {new_value:.2f} ({change:+.2f})")
    
    def log_error_with_context(self, error: Exception, context: dict = None):
        """Log errors with additional context"""
        context_str = ""
        if context:
            context_items = [f"{k}={v}" for k, v in context.items()]
            context_str = f" | Context: {', '.join(context_items)}"
        
        self.logger.error(f"Error: {type(error).__name__}: {error}{context_str}")

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, exponential_backoff: bool = True):
    """Decorator to retry operations on failure"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger = logging.getLogger('a2bot')
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        await asyncio.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    logger = logging.getLogger('a2bot')
                    
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {current_delay}s...")
                        import time
                        time.sleep(current_delay)
                        
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}: {e}")
            
            raise last_exception
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

def safe_execute(default_return=None, log_errors=True):
    """Decorator to safely execute functions and return default on error"""
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger('a2bot')
                    logger.error(f"Safe execution failed for {func.__name__}: {e}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger = logging.getLogger('a2bot')
                    logger.error(f"Safe execution failed for {func.__name__}: {e}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Traceback: {traceback.format_exc()}")
                return default_return
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class ErrorAggregator:
    """Collect and aggregate errors to prevent spam"""
    
    def __init__(self, max_errors_per_hour: int = 50):
        self.max_errors_per_hour = max_errors_per_hour
        self.error_counts = {}
        self.last_reset = datetime.now(timezone.utc)
    
    def should_log_error(self, error_key: str) -> bool:
        """Check if error should be logged based on rate limiting"""
        now = datetime.now(timezone.utc)
        
        # Reset counts every hour
        if (now - self.last_reset).total_seconds() > 3600:
            self.error_counts.clear()
            self.last_reset = now
        
        # Check error count
        current_count = self.error_counts.get(error_key, 0)
        if current_count >= self.max_errors_per_hour:
            return False
        
        self.error_counts[error_key] = current_count + 1
        return True

# Global error aggregator
error_aggregator = ErrorAggregator()

def log_error_with_aggregation(error: Exception, context: str = "", extra_data: dict = None):
    """Log error with aggregation to prevent spam"""
    error_key = f"{type(error).__name__}:{str(error)[:100]}"
    
    if error_aggregator.should_log_error(error_key):
        logger = logging.getLogger('a2bot')
        
        # Build error message
        error_msg = f"Error in {context}: {type(error).__name__}: {error}"
        if extra_data:
            extra_str = ", ".join(f"{k}={v}" for k, v in extra_data.items())
            error_msg += f" | Extra: {extra_str}"
        
        logger.error(error_msg)
        
        # Log full traceback only for new/rare errors
        if error_aggregator.error_counts.get(error_key, 0) <= 3:
            logger.debug(f"Traceback for {error_key}: {traceback.format_exc()}")

def create_enhanced_command_error_handler():
    """Create an enhanced command error handler"""
    
    async def enhanced_error_handler(ctx, error):
        """Enhanced error handler with better user feedback and logging"""
        
        # Log the error with context
        context_data = {
            "user_id": ctx.author.id,
            "guild_id": ctx.guild.id if ctx.guild else None,
            "channel_id": ctx.channel.id,
            "command": str(ctx.command) if ctx.command else "unknown"
        }
        
        log_error_with_aggregation(error, "command_execution", context_data)
        
        # Handle specific error types with appropriate responses
        if hasattr(error, 'original'):
            error = error.original
        
        # Import discord for error type checking
        import discord
        from discord.ext import commands
        
        # User-friendly responses based on error type
        if isinstance(error, commands.CommandNotFound):
            return  # Ignore unknown commands
        
        elif isinstance(error, commands.MissingRequiredArgument):
            param_name = error.param.name
            await ctx.send(f"A2: Missing required parameter: `{param_name}`. Check the command usage.")
        
        elif isinstance(error, commands.BadArgument):
            await ctx.send("A2: Invalid argument format. Double-check your input.")
        
        elif isinstance(error, commands.MissingPermissions):
            await ctx.send("A2: You lack the necessary permissions for that command.")
        
        elif isinstance(error, commands.BotMissingPermissions):
            missing_perms = ", ".join(error.missing_permissions)
            await ctx.send(f"A2: I need these permissions: {missing_perms}")
        
        elif isinstance(error, commands.CommandOnCooldown):
            await ctx.send(f"A2: Command on cooldown. Try again in {error.retry_after:.1f} seconds.")
        
        elif isinstance(error, discord.Forbidden):
            await ctx.send("A2: Access denied. Check permissions.")
        
        elif isinstance(error, asyncio.TimeoutError):
            await ctx.send("A2: Operation timed out. Try again later.")
        
        else:
            # Generic error for unexpected issues
            error_id = f"{int(datetime.now().timestamp())}"
            await ctx.send(f"A2: System error occurred. Reference ID: {error_id}")
            
            # Log with reference ID for tracking
            logger = logging.getLogger('a2bot')
            logger.error(f"Unhandled error (ID: {error_id}): {type(error).__name__}: {error}")
    
    return enhanced_error_handler

def setup_enhanced_logging(data_dir: Path, debug_mode: bool = False):
    """Set up enhanced logging with better formatting and filtering"""
    
    # Set log level based on debug mode
    log_level = logging.DEBUG if debug_mode else logging.INFO
    
    # Create custom formatter
    class A2Formatter(logging.Formatter):
        """Custom formatter for A2 bot logs"""
        
        COLORS = {
            'DEBUG': '\033[36m',     # Cyan
            'INFO': '\033[32m',      # Green
            'INTERACTION': '\033[35m', # Magenta
            'WARNING': '\033[33m',   # Yellow
            'EMOTION': '\033[34m',   # Blue
            'ERROR': '\033[31m',     # Red
            'CRITICAL': '\033[41m',  # Red background
            'ENDC': '\033[0m'        # End color
        }
        
        def format(self, record):
            # Add color to console output
            if hasattr(record, 'levelname'):
                color = self.COLORS.get(record.levelname, '')
                end_color = self.COLORS['ENDC'] if color else ''
                record.levelname_colored = f"{color}{record.levelname}{end_color}"
            
            return super().format(record)
    
    # Create enhanced logger
    logger = A2Logger()
    
    # Console formatter with colors
    console_formatter = A2Formatter(
        '%(asctime)s - %(levelname_colored)s - %(name)s - %(message)s'
    )
    
    # File formatter without colors
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Update existing handlers with new formatters
    root_logger = logging.getLogger('a2bot')
    for handler in root_logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setFormatter(console_formatter)
        elif isinstance(handler, logging.FileHandler):
            handler.setFormatter(file_formatter)
    
    return logger
