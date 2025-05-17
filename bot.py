"""
Main bot class for the A2 Discord bot.
"""
import os
import sys
import asyncio
import re
import random
import json
from pathlib import Path
from datetime import datetime, timezone

import discord
from discord.ext import commands, tasks
from openai import OpenAI

# Import logging utilities
from utils.logging_helper import get_logger

# Import managers
from managers.conversation import ConversationManager
from managers.emotion import EmotionManager
from managers.storage import StorageManager
from managers.response import ResponseGenerator

# Import enhanced modules
from enhanced_a2 import EnhancedResponseGenerator
from commands.enhanced_commands import setup_enhanced_commands

# Import utilities
from utils.transformers_helper import (
    HAVE_TRANSFORMERS, initialize_transformers,
    get_summarizer, get_toxic, get_sentiment
)

# Import commands
from commands.user_commands import setup_user_commands
from commands.admin_commands import setup_admin_commands

# Import configuration
from config import (
    DATA_DIR, USERS_DIR, PROFILES_DIR, DM_SETTINGS_FILE,
    USER_PROFILES_DIR, CONVERSATIONS_DIR, 
    EMOTION_CONFIG, RELATIONSHIP_LEVELS, PERSONALITY_STATES
)

class A2Bot:
    """Main A2 bot implementation handling commands and event loops"""
    
    def __init__(self, token, app_id, openai_api_key, openai_org_id="", openai_project_id=""):
        # Get logger
        self.logger = get_logger()
        
        # Set up Discord bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.reactions = True
        intents.messages = True
        intents.members = True
        intents.guilds = True
        
        self.prefixes = ["!", "!a2 "]
        self.bot = commands.Bot(
            command_prefix=commands.when_mentioned_or(*self.prefixes), 
            intents=intents, 
            application_id=app_id
        )
        
        self.token = token
        
        # Set up OpenAI client
        self.openai_client = OpenAI(api_key=openai_api_key)
        
        # Store relationship levels and personality states for easy access
        self.bot.RELATIONSHIP_LEVELS = RELATIONSHIP_LEVELS
        self.bot.PERSONALITY_STATES = PERSONALITY_STATES
        
        # Initialize managers
        self.storage_manager = StorageManager(DATA_DIR, USERS_DIR, PROFILES_DIR, DM_SETTINGS_FILE, 
                                             USER_PROFILES_DIR, CONVERSATIONS_DIR)
        self.emotion_manager = EmotionManager()
        self.conversation_manager = ConversationManager()
        self.response_generator = ResponseGenerator(
            self.openai_client, 
            self.emotion_manager, 
            self.conversation_manager
        )
        
        # Initialize enhanced response generator
        self.enhanced_generator = EnhancedResponseGenerator(
            self.openai_client, 
            self.emotion_manager, 
            self.conversation_manager,
            DATA_DIR
        )
        self.use_enhanced = os.getenv("ENABLE_ENHANCED_A2", "1") == "1"
        
        # Store bot start time for uptime tracking
        self.bot.start_time = datetime.now(timezone.utc)
        
        # Set up event handlers and commands
        self._setup_event_handlers()
        self._setup_commands()
    
    def _setup_event_handlers(self):
        """Set up event handlers for the bot"""
        @self.bot.event
        async def on_ready():
            """Handle bot startup"""
            self.logger.info("A2 is online.")
            self.logger.info(f"Connected to {len(self.bot.guilds)} guilds")
            self.logger.info(f"Serving {sum(len(g.members) for g in self.bot.guilds)} users")
            
            # Debug data directories
            self.logger.info(f"Checking data directory: {DATA_DIR}")
            self.logger.info(f"Directory exists: {DATA_DIR.exists()}")
            self.logger.info(f"Profile directory: {PROFILES_DIR}")
            self.logger.info(f"Directory exists: {PROFILES_DIR.exists()}")
            
            # Check for existing profile files
            profile_files = list(PROFILES_DIR.glob("*.json"))
            self.logger.info(f"Found {len(profile_files)} profile files")
            
            # Load all data
            await self.storage_manager.load_data(self.emotion_manager, self.conversation_manager)
            
            # Add first interaction timestamp for users who don't have it
            now = datetime.now(timezone.utc).isoformat()
            for uid in self.emotion_manager.user_emotions:
                if 'first_interaction' not in self.emotion_manager.user_emotions[uid]:
                    self.emotion_manager.user_emotions[uid]['first_interaction'] = (
                        self.emotion_manager.user_emotions[uid].get('last_interaction', now)
                    )
            
            # Initialize enhanced systems if enabled
            if self.use_enhanced:
                await self.enhanced_generator.initialize()
                self.logger.info("Enhanced A2 systems initialized")
            
            # Start background tasks
            self._start_background_tasks()
            
            self.logger.info("All tasks started successfully.")
            self.logger.info("Dynamic stats system enabled")
            
        @self.bot.event
        async def on_message(message):
            """Handle incoming messages"""
            if message.author.bot or message.content.startswith("A2:"):
                return
        
            uid = message.author.id
            content = message.content.strip()
        
            # Clear pending message status if this user had one
            if uid in self.emotion_manager.pending_messages:
                self.emotion_manager.pending_messages.remove(uid)
        
            # Initialize first interaction time if this is a new user
            if uid not in self.emotion_manager.user_emotions:
                now = datetime.now(timezone.utc).isoformat()
                self.emotion_manager.user_emotions[uid] = {
                    "trust": 0, 
                    "resentment": 0, 
                    "attachment": 0, 
                    "protectiveness": 0,
                    "affection_points": 0, 
                    "annoyance": 0,
                    "first_interaction": now,
                    "last_interaction": now,
                    "interaction_count": 0
                }
        
                # Get or create user profile with name from Discord
                self.conversation_manager.get_or_create_profile(uid, message.author.display_name)
                
                self.logger.info(f"New user initialized: {message.author.display_name} ({uid})")
            
                await self.response_generator.handle_first_message_of_day(message, uid)
            
            is_cmd = any(content.startswith(p) for p in self.prefixes)
            is_mention = self.bot.user in getattr(message, 'mentions', [])
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            if not (is_cmd or is_mention or is_dm):
                return
            
            await self.bot.process_commands(message)
            
            if is_cmd:
                return
            
            # Handle regular messages
            self.logger.info(f"Processing message from {message.author.display_name} ({uid}): {content[:30]}...")
            trust = self.emotion_manager.user_emotions.get(uid, {}).get('trust', 0)
            
            # Use enhanced generator if enabled
            if hasattr(self, 'use_enhanced') and self.use_enhanced:
                resp = await self.enhanced_generator.generate_enhanced_response(content, trust, uid, self.storage_manager)
            else:
                resp = await self.response_generator.generate_a2_response(content, trust, uid, self.storage_manager)
            
            # Track user's emotional state in history
            if uid in self.emotion_manager.user_emotions:
                e = self.emotion_manager.user_emotions[uid]
                # Initialize emotion history if it doesn't exist
                if "emotion_history" not in e:
                    e["emotion_history"] = []
                
                # Only record history if enough time has passed since last entry
                if not e["emotion_history"] or (
                    datetime.now(timezone.utc) - 
                    datetime.fromisoformat(e["emotion_history"][-1]["timestamp"])
                ).total_seconds() > 3600:  # One hour between entries
                    e["emotion_history"].append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "trust": e.get("trust", 0),
                        "attachment": e.get("attachment", 0),
                        "resentment": e.get("resentment", 0),
                        "protectiveness": e.get("protectiveness", 0),
                        "affection_points": e.get("affection_points", 0)
                    })
                    
                # Keep history at a reasonable size
                if len(e["emotion_history"]) > 50:
                    e["emotion_history"] = e["emotion_history"][-50:]
            
            # Record interaction data for future analysis
            await self.emotion_manager.record_interaction_data(uid, content, resp, self.storage_manager)
            
            # For longer messages, A2 might sometimes give a thoughtful response
            if len(content) > 100 and random.random() < 0.3 and trust > 5:
                await message.channel.send(f"A2: ...")
                await asyncio.sleep(1.5)
            
            await message.channel.send(f"A2: {resp}")
            
            # Occasionally respond with a follow-up based on relationship
            if random.random() < 0.1 and trust > 7:
                await asyncio.sleep(3)
                followups = [
                    "Something else?",
                    "...",
                    "Still processing that.",
                    "Interesting.",
                ]
                await message.channel.send(f"A2: {random.choice(followups)}")
                
        @self.bot.event
        async def on_command_error(ctx, error):
            """Handle command errors"""
            if isinstance(error, commands.CommandNotFound):
                return
            
            if isinstance(error, commands.MissingRequiredArgument):
                await ctx.send(f"A2: Missing required argument: {error.param.name}")
                return
                
            if isinstance(error, commands.MissingPermissions):
                await ctx.send(f"A2: You don't have the required permissions to use this command.")
                return
                
            # Log the error
            self.logger.error(f"Command error in {ctx.command}: {error}")
            
            # Notify the user
            await ctx.send(f"A2: An error occurred while processing your command.\nError: {error}")
    
    def _setup_commands(self):
        """Set up commands for the bot"""
        # Set up user commands
        setup_user_commands(self.bot, self.emotion_manager, self.conversation_manager, self.storage_manager)
        
        # Set up admin commands
        setup_admin_commands(self.bot, self.emotion_manager, self.conversation_manager, self.storage_manager)
        
        # Set up enhanced commands if enabled
        if self.use_enhanced:
            setup_enhanced_commands(self.bot, self.enhanced_generator.enhanced_system)
    
    def _start_background_tasks(self):
        """Start all background tasks for the bot"""
        # Define task functions with storage manager
        @tasks.loop(minutes=10)
        async def check_inactive_users_task():
            self.logger.debug("Running check_inactive_users_task")
            await self.response_generator.check_inactive_users(self.bot, self.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_affection_task():
            self.logger.debug("Running decay_affection_task")
            await self.emotion_manager.decay_affection(self.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_annoyance_task():
            self.logger.debug("Running decay_annoyance_task")
            await self.emotion_manager.decay_annoyance(self.storage_manager)
            
        @tasks.loop(hours=24)
        async def daily_affection_bonus_task():
            self.logger.debug("Running daily_affection_bonus_task")
            await self.emotion_manager.daily_affection_bonus(self.storage_manager)
            
        @tasks.loop(hours=1)
        async def dynamic_emotional_adjustments_task():
            self.logger.debug("Running dynamic_emotional_adjustments_task")
            await self.emotion_manager.dynamic_emotional_adjustments(self.storage_manager)
            
        @tasks.loop(hours=3)
        async def environmental_mood_effects_task():
            self.logger.debug("Running environmental_mood_effects_task")
            await self.emotion_manager.environmental_mood_effects(self.storage_manager)
            
        @tasks.loop(hours=4)
        async def trigger_random_events_task():
            self.logger.debug("Running trigger_random_events_task")
            await self.response_generator.trigger_random_events(self.bot, self.storage_manager)
            
        @tasks.loop(hours=1)
        async def save_data_task():
            self.logger.debug("Running save_data_task")
            await self.storage_manager.save_data(self.emotion_manager, self.conversation_manager)
        
        # Start all tasks
        check_inactive_users_task.start()
        decay_affection_task.start()
        decay_annoyance_task.start()
        daily_affection_bonus_task.start()
        dynamic_emotional_adjustments_task.start()
        environmental_mood_effects_task.start()
        trigger_random_events_task.start()
        save_data_task.start()
    
    def run(self):
        """Run the bot"""
        self.logger.info("Starting Discord bot...")
        try:
            self.bot.run(self.token)
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")
            raise
