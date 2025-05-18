"""
Data models for the A2 Discord bot.
"""
# User profile model
from models.user_profile import UserProfile

# Database models
from models.database import (
    # Base models
    User, UserEmotions, UserMemory, UserEvent, UserMilestone,
    UserProfile as DBUserProfile, Conversation, ConversationSummary, 
    DMSettings, InteractionStats, RelationshipProgress,
    
    # Database initialization
    init_db, get_session_factory,
    
    # SQLAlchemy base
    Base
)

__all__ = [
    # User profile model
    'UserProfile',
    
    # Database models
    'User', 'UserEmotions', 'UserMemory', 'UserEvent', 'UserMilestone',
    'DBUserProfile', 'Conversation', 'ConversationSummary', 
    'DMSettings', 'InteractionStats', 'RelationshipProgress',
    
    # Database utilities
    'init_db', 'get_session_factory', 'Base'
]
