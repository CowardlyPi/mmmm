"""
Input validation and rate limiting utilities for the A2 Discord bot.
"""
import re
import time
import logging
from collections import defaultdict, deque
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, Tuple, Any

logger = logging.getLogger('a2bot')

class RateLimiter:
    """Simple rate limiter for user commands"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 60):
        """
        Initialize rate limiter
        
        Args:
            max_requests: Maximum requests allowed in time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = defaultdict(deque)
    
    def is_allowed(self, user_id: int) -> Tuple[bool, Optional[int]]:
        """
        Check if user is allowed to make a request
        
        Args:
            user_id: Discord user ID
            
        Returns:
            Tuple of (is_allowed, seconds_until_reset)
        """
        now = time.time()
        user_requests = self.requests[user_id]
        
        # Remove old requests outside the time window
        while user_requests and user_requests[0] <= now - self.time_window:
            user_requests.popleft()
        
        # Check if under the limit
        if len(user_requests) < self.max_requests:
            user_requests.append(now)
            return True, None
        
        # Calculate time until oldest request expires
        oldest_request = user_requests[0]
        reset_time = int(oldest_request + self.time_window - now)
        return False, max(1, reset_time)

class InputValidator:
    """Input validation utilities"""
    
    # Constants for validation
    MAX_MESSAGE_LENGTH = 2000
    MAX_NAME_LENGTH = 100
    MAX_INTEREST_LENGTH = 50
    MAX_FACT_LENGTH = 200
    
    # Regex patterns
    SAFE_TEXT_PATTERN = re.compile(r'^[\w\s.,!?\'"-]+$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{1,32}$')
    
    @staticmethod
    def sanitize_text(text: str, max_length: int = MAX_MESSAGE_LENGTH) -> str:
        """
        Sanitize text input by removing dangerous characters and limiting length
        
        Args:
            text: Input text to sanitize
            max_length: Maximum allowed length
            
        Returns:
            Sanitized text
        """
        if not text:
            return ""
        
        # Remove null bytes and control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length].rstrip()
        
        return text.strip()
    
    @staticmethod
    def validate_user_name(name: str) -> Tuple[bool, str]:
        """
        Validate a user name
        
        Args:
            name: Name to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not name:
            return False, "Name cannot be empty"
        
        name = name.strip()
        
        if len(name) > InputValidator.MAX_NAME_LENGTH:
            return False, f"Name too long (max {InputValidator.MAX_NAME_LENGTH} characters)"
        
        if len(name) < 1:
            return False, "Name too short"
        
        # Allow most characters but prevent obvious abuse
        forbidden_patterns = [
            r'<@[!&]?\d+>',  # Discord mentions
            r'https?://',    # URLs
            r'discord\.gg',  # Discord invites
        ]
        
        for pattern in forbidden_patterns:
            if re.search(pattern, name, re.IGNORECASE):
                return False, "Name contains forbidden content"
        
        return True, ""
    
    @staticmethod
    def validate_emotion_value(value: Any, emotion_type: str) -> Tuple[bool, float, str]:
        """
        Validate an emotion value
        
        Args:
            value: Value to validate
            emotion_type: Type of emotion (for bounds checking)
            
        Returns:
            Tuple of (is_valid, sanitized_value, error_message)
        """
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False, 0.0, "Value must be a number"
        
        # Set bounds based on emotion type
        if emotion_type == "affection_points":
            min_val, max_val = -100, 1000
        elif emotion_type == "annoyance":
            min_val, max_val = 0, 100
        else:
            min_val, max_val = 0, 10
        
        if value < min_val or value > max_val:
            return False, 0.0, f"Value must be between {min_val} and {max_val}"
        
        return True, value, ""
    
    @staticmethod
    def validate_list_input(items: str, item_type: str) -> Tuple[bool, list, str]:
        """
        Validate comma-separated list input
        
        Args:
            items: Comma-separated string of items
            item_type: Type of items for validation
            
        Returns:
            Tuple of (is_valid, processed_list, error_message)
        """
        if not items:
            return False, [], "Input cannot be empty"
        
        # Split and clean items
        item_list = [item.strip() for item in items.split(",")]
        item_list = [item for item in item_list if item]  # Remove empty items
        
        if len(item_list) > 10:
            return False, [], "Too many items (max 10)"
        
        # Validate each item based on type
        max_length = {
            "interests": InputValidator.MAX_INTEREST_LENGTH,
            "personality_traits": InputValidator.MAX_INTEREST_LENGTH,
            "notable_facts": InputValidator.MAX_FACT_LENGTH,
            "relationship_context": InputValidator.MAX_FACT_LENGTH,
            "conversation_topics": InputValidator.MAX_INTEREST_LENGTH
        }.get(item_type, InputValidator.MAX_INTEREST_LENGTH)
        
        validated_items = []
        for item in item_list:
            if len(item) > max_length:
                return False, [], f"Item too long: '{item[:20]}...' (max {max_length} characters)"
            
            # Sanitize the item
            sanitized = InputValidator.sanitize_text(item, max_length)
            if sanitized:
                validated_items.append(sanitized)
        
        if not validated_items:
            return False, [], "No valid items found"
        
        return True, validated_items, ""

def add_rate_limiting_to_commands(bot):
    """Add rate limiting decorators to bot commands"""
    
    # Different rate limits for different command types
    general_limiter = RateLimiter(max_requests=20, time_window=60)  # 20 per minute
    resource_limiter = RateLimiter(max_requests=5, time_window=60)   # 5 per minute for heavy commands
    
    def rate_limit(limiter_type="general"):
        """Decorator factory for rate limiting"""
        def decorator(func):
            async def wrapper(ctx, *args, **kwargs):
                user_id = ctx.author.id
                limiter = general_limiter if limiter_type == "general" else resource_limiter
                
                allowed, reset_time = limiter.is_allowed(user_id)
                if not allowed:
                    await ctx.send(f"A2: Slow down. Try again in {reset_time} seconds.")
                    return
                
                return await func(ctx, *args, **kwargs)
            return wrapper
        return decorator
    
    # Apply rate limiting to existing commands
    if hasattr(bot, 'get_command'):
        # Rate limit resource-heavy commands
        heavy_commands = ['stats', 'memories', 'relationship', 'debug_info', 'inspect_user']
        for cmd_name in heavy_commands:
            cmd = bot.get_command(cmd_name)
            if cmd:
                cmd.callback = rate_limit("resource")(cmd.callback)
        
        # Rate limit general commands
        general_commands = ['profile', 'milestones', 'events', 'conversations']
        for cmd_name in general_commands:
            cmd = bot.get_command(cmd_name)
            if cmd:
                cmd.callback = rate_limit("general")(cmd.callback)

def create_safe_error_handler():
    """Create a safe error handler that doesn't expose internal details"""
    
    async def safe_error_handler(ctx, error):
        """Handle errors safely without exposing internal details"""
        
        # User-friendly error messages
        user_friendly_errors = {
            "CommandNotFound": None,  # Ignore command not found
            "MissingRequiredArgument": "A2: Missing required information. Check the command usage.",
            "BadArgument": "A2: Invalid input format. Check your command.",
            "MissingPermissions": "A2: You don't have permission to use that command.",
            "BotMissingPermissions": "A2: I don't have the necessary permissions.",
            "CommandOnCooldown": "A2: Command is on cooldown. Try again later.",
            "CheckFailure": "A2: Command check failed. You may not meet the requirements.",
        }
        
        error_name = type(error).__name__
        user_message = user_friendly_errors.get(error_name)
        
        if user_message:
            await ctx.send(user_message)
        elif user_message is not None:  # Don't send anything for None values
            # Log the actual error for debugging
            logger.error(f"Command error in {ctx.command}: {error}")
            await ctx.send("A2: An error occurred. The issue has been logged.")
    
    return safe_error_handler
