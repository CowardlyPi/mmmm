"""
Main bot class for the A2 Discord bot with focused responsibilities.
"""
import os
import discord
from discord.ext import commands
from openai import OpenAI

from utils.logging_helper import get_logger
from core.events import EventHandler
from core.background_tasks import BackgroundTaskManager

class A2Bot:
    """Main A2 bot implementation with modular architecture"""
    
    def __init__(self, token, app_id, openai_api_key, openai_org_id="", openai_project_id="", 
                 storage_manager=None, batch_size=50):
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
        
        # Store configuration
        self.batch_size = batch_size
        
        # Set up managers from the imports
        from config import RELATIONSHIP_LEVELS, PERSONALITY_STATES
        from managers.conversation import ConversationManager
        from managers.emotion import EmotionManager
        from managers.response import ResponseGenerator
        from enhanced_a2 import EnhancedResponseGenerator
        
        # Import using configuration
        from config import DATA_DIR, USERS_DIR, PROFILES_DIR, DM_SETTINGS_FILE, USER_PROFILES_DIR, CONVERSATIONS_DIR
        
        # Store relationship levels and personality states for easy access
        self.bot.RELATIONSHIP_LEVELS = RELATIONSHIP_LEVELS
        self.bot.PERSONALITY_STATES = PERSONALITY_STATES
        
        # Initialize managers
        # Use provided storage manager or create default one
        if storage_manager:
            self.storage_manager = storage_manager
        else:
            from managers.storage import StorageManager
            self.storage_manager = StorageManager(DATA_DIR, USERS_DIR, PROFILES_DIR, DM_SETTINGS_FILE, 
                                                USER_PROFILES_DIR, CONVERSATIONS_DIR)
        
        # Initialize core components
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
        
        # Set up modular components
        self.event_handler = EventHandler(self)
        self.task_manager = BackgroundTaskManager(self)
        
        # Set up event handlers and commands
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components of the bot"""
        # Set up event handlers
        self.event_handler.setup_event_handlers()
        
        # Set up commands
        self._setup_commands()
    
    def _setup_commands(self):
        """Set up commands for the bot"""
        # Set up user commands
        from commands.user_commands import setup_user_commands
        setup_user_commands(self.bot, self.emotion_manager, self.conversation_manager, self.storage_manager)
        
        # Set up admin commands
        from commands.admin_commands import setup_admin_commands
        setup_admin_commands(self.bot, self.emotion_manager, self.conversation_manager, self.storage_manager)
        
        # Set up enhanced commands if enabled
        if self.use_enhanced:
            from commands.enhanced_commands import setup_enhanced_commands
            setup_enhanced_commands(self.bot, self.enhanced_generator.enhanced_system)
    
    def run(self):
        """Run the bot"""
        self.logger.info("Starting Discord bot...")
        try:
            self.bot.run(self.token)
        except Exception as e:
            self.logger.error(f"Error running bot: {e}")
            raise
