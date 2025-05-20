"""
Core components for the A2 Discord bot.
"""
from core.bot import A2Bot
from core.events import EventHandler
from core.background_tasks import BackgroundTaskManager
from core.initialization import initialize_bot

__all__ = ['A2Bot', 'EventHandler', 'BackgroundTaskManager', 'initialize_bot']
