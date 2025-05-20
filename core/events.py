"""
Event handling for the A2 Discord bot.
"""
import asyncio
import random
import discord
from datetime import datetime, timezone
from utils.logging_helper import get_logger

class EventHandler:
    """Handles all Discord event callbacks"""
    
    def __init__(self, bot_instance):
        """
        Initialize the event handler
        
        Args:
            bot_instance: The A2Bot instance
        """
        self.bot = bot_instance
        self.logger = get_logger()
    
    def setup_event_handlers(self):
        """Set up all event handlers for the Discord bot"""
        @self.bot.bot.event
        async def on_ready():
            """Handle bot startup"""
            self.logger.info("A2 is online.")
            self.logger.info(f"Connected to {len(self.bot.bot.guilds)} guilds")
            self.logger.info(f"Serving {sum(len(g.members) for g in self.bot.bot.guilds)} users")
            
            # Debug data directories
            self.logger.info(f"Using storage manager: {self.bot.storage_manager.__class__.__name__}")
            self.logger.info(f"Using batch size: {self.bot.batch_size}")
            
            if hasattr(self.bot.storage_manager, 'verify_database_connection'):
                # PostgreSQL storage
                db_conn_ok = await self.bot.storage_manager.verify_database_connection()
                self.logger.info(f"Database connection verified: {db_conn_ok}")
            elif hasattr(self.bot.storage_manager, 'verify_data_directories'):
                # File-based storage
                self.logger.info(f"Checking data directory: {self.bot.storage_manager.data_dir}")
                self.logger.info(f"Directory exists: {self.bot.storage_manager.data_dir.exists()}")
                self.logger.info(f"Profile directory: {self.bot.storage_manager.profiles_dir}")
                self.logger.info(f"Directory exists: {self.bot.storage_manager.profiles_dir.exists()}")
                
                # Check for existing profile files
                profile_files = list(self.bot.storage_manager.profiles_dir.glob("*.json"))
                self.logger.info(f"Found {len(profile_files)} profile files")
            
            # Load all data
            await self.bot.storage_manager.load_data(self.bot.emotion_manager, self.bot.conversation_manager, self.bot.batch_size)
            
            # Add first interaction timestamp for users who don't have it
            now = datetime.now(timezone.utc).isoformat()
            for uid in self.bot.emotion_manager.user_emotions:
                if 'first_interaction' not in self.bot.emotion_manager.user_emotions[uid]:
                    self.bot.emotion_manager.user_emotions[uid]['first_interaction'] = (
                        self.bot.emotion_manager.user_emotions[uid].get('last_interaction', now)
                    )
            
            # Initialize enhanced systems if enabled
            if self.bot.use_enhanced:
                await self.bot.enhanced_generator.initialize()
                self.logger.info("Enhanced A2 systems initialized")
            
            # Start background tasks
            self.bot.task_manager.start_background_tasks()
            
            self.logger.info("All tasks started successfully.")
            self.logger.info("Dynamic stats system enabled")
            
        @self.bot.bot.event
        async def on_message(message):
            """Handle incoming messages"""
            if message.author.bot or message.content.startswith("A2:"):
                return
        
            uid = message.author.id
            content = message.content.strip()
        
            # Clear pending message status if this user had one
            if uid in self.bot.emotion_manager.pending_messages:
                self.bot.emotion_manager.pending_messages.remove(uid)
        
            # Initialize first interaction time if this is a new user
            if uid not in self.bot.emotion_manager.user_emotions:
                now = datetime.now(timezone.utc).isoformat()
                self.bot.emotion_manager.user_emotions[uid] = {
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
                self.bot.conversation_manager.get_or_create_profile(uid, message.author.display_name)
                
                self.logger.info(f"New user initialized: {message.author.display_name} ({uid})")
            
                await self.bot.response_generator.handle_first_message_of_day(message, uid)
            
            is_cmd = any(content.startswith(p) for p in self.bot.prefixes)
            is_mention = self.bot.bot.user in getattr(message, 'mentions', [])
            is_dm = isinstance(message.channel, discord.DMChannel)
            
            if not (is_cmd or is_mention or is_dm):
                return
            
            await self.bot.bot.process_commands(message)
            
            if is_cmd:
                return
            
            # Handle regular messages
            self.logger.info(f"Processing message from {message.author.display_name} ({uid}): {content[:30]}...")
            trust = self.bot.emotion_manager.user_emotions.get(uid, {}).get('trust', 0)
            
            # Use enhanced generator if enabled
            if hasattr(self.bot, 'use_enhanced') and self.bot.use_enhanced:
                resp = await self.bot.enhanced_generator.generate_enhanced_response(
                    content, trust, uid, self.bot.storage_manager)
            else:
                resp = await self.bot.response_generator.generate_a2_response(
                    content, trust, uid, self.bot.storage_manager)
            
            # Track user's emotional state in history
            if uid in self.bot.emotion_manager.user_emotions:
                e = self.bot.emotion_manager.user_emotions[uid]
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
            await self.bot.emotion_manager.record_interaction_data(uid, content, resp, self.bot.storage_manager)
            
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
                
        @self.bot.bot.event
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
