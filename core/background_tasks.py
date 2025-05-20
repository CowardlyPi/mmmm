"""
Background tasks for the A2 Discord bot.
"""
from discord.ext import tasks
from utils.logging_helper import get_logger

class BackgroundTaskManager:
    """Manages all background tasks for the bot"""
    
    def __init__(self, bot_instance):
        """
        Initialize the task manager
        
        Args:
            bot_instance: The A2Bot instance
        """
        self.bot = bot_instance
        self.logger = get_logger()
        self.tasks = []
    
    def start_background_tasks(self):
        """Start all background tasks for the bot"""
        # Define task functions with storage manager
        @tasks.loop(minutes=10)
        async def check_inactive_users_task():
            self.logger.debug("Running check_inactive_users_task")
            await self.bot.response_generator.check_inactive_users(self.bot.bot, self.bot.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_affection_task():
            self.logger.debug("Running decay_affection_task")
            await self.bot.emotion_manager.decay_affection(self.bot.storage_manager)
            
        @tasks.loop(hours=1)
        async def decay_annoyance_task():
            self.logger.debug("Running decay_annoyance_task")
            await self.bot.emotion_manager.decay_annoyance(self.bot.storage_manager)
            
        @tasks.loop(hours=24)
        async def daily_affection_bonus_task():
            self.logger.debug("Running daily_affection_bonus_task")
            await self.bot.emotion_manager.daily_affection_bonus(self.bot.storage_manager)
            
        @tasks.loop(hours=1)
        async def dynamic_emotional_adjustments_task():
            self.logger.debug("Running dynamic_emotional_adjustments_task")
            await self.bot.emotion_manager.dynamic_emotional_adjustments(self.bot.storage_manager)
            
        @tasks.loop(hours=3)
        async def environmental_mood_effects_task():
            self.logger.debug("Running environmental_mood_effects_task")
            await self.bot.emotion_manager.environmental_mood_effects(self.bot.storage_manager)
            
        @tasks.loop(hours=4)
        async def trigger_random_events_task():
            self.logger.debug("Running trigger_random_events_task")
            await self.bot.response_generator.trigger_random_events(self.bot.bot, self.bot.storage_manager)
            
        @tasks.loop(hours=1)
        async def save_data_task():
            self.logger.debug("Running save_data_task")
            await self.bot.storage_manager.save_data(self.bot.emotion_manager, self.bot.conversation_manager)
        
        # Store tasks for management
        self.tasks = [
            check_inactive_users_task,
            decay_affection_task,
            decay_annoyance_task,
            daily_affection_bonus_task,
            dynamic_emotional_adjustments_task,
            environmental_mood_effects_task,
            trigger_random_events_task,
            save_data_task
        ]
        
        # Start all tasks
        for task in self.tasks:
            task.start()
            self.logger.debug(f"Started task: {task.__name__}")
    
    def stop_all_tasks(self):
        """Stop all running tasks"""
        for task in self.tasks:
            if task.is_running():
                task.cancel()
                self.logger.debug(f"Stopped task: {task.__name__}")
