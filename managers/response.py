"""
Response generator for the A2 Discord bot.
"""
import random
import asyncio
import re
from datetime import datetime, timezone
from collections import deque

class ResponseGenerator:
    """Handles conversation management and response generation"""
    
    def __init__(self, openai_client, emotion_manager, conversation_manager):
        self.client = openai_client
        self.emotion_manager = emotion_manager
        self.conversation_manager = conversation_manager
        self.recent_responses = {}
        self.MAX_RECENT_RESPONSES = 10
