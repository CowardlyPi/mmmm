"""Emotion manager for the A2 Discord bot."""
import re
import random
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

class EmotionManager:
    """Manages all emotional and relationship aspects of the bot"""
    
    def __init__(self):
        # ─── State Storage ────────────────────────────────────────────────────
        self.conversation_summaries = {}
        self.conversation_history = defaultdict(list)
        self.user_emotions = {}
        self.recent_responses = {}
        self.user_memories = defaultdict(list)
        self.user_events = defaultdict(list)
        self.user_milestones = defaultdict(list)
        self.interaction_stats = defaultdict(Counter)
        self.relationship_progress = defaultdict(dict)
        self.dm_enabled_users = set()
        self.MAX_RECENT_RESPONSES = 10
        self.pending_messages = set()
