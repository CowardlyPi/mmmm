"""
PostgreSQL storage manager for the A2 Discord bot.
"""
import json
import logging
from datetime import datetime, timezone
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set, Union

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, not_, func

from models.database import (
    User, UserEmotions, UserMemory, UserEvent, UserMilestone,
    UserProfile, Conversation, ConversationSummary, DMSettings,
    InteractionStats, RelationshipProgress,
    init_db, get_session_factory
)

from utils.logging_helper import get_logger


class PostgreSQLStorageManager:
    """Handles all data persistence operations using PostgreSQL"""
    
    def __init__(self, database_url, data_dir=None):
        """Initialize the storage manager with database connection
        
        Args:
            database_url (str): PostgreSQL connection URL
            data_dir (Path, optional): Directory for temp files/fallback storage
        """
        self.logger = get_logger()
        self.data_dir = data_dir  # Keep for temp files if needed
        
        # Initialize database connection
        try:
            self.engine = init_db(database_url)
            self.SessionFactory = get_session_factory(self.engine)
            self.logger.info(f"Database connection established successfully")
        except Exception as e:
            self.logger.error(f"Database connection error: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionFactory()
    
    async def verify_database_connection(self) -> bool:
        """Verify database connection is working"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                self.logger.info("Database connection verified: SUCCESS")
                return True
        except Exception as e:
            self.logger.error(f"Database connection check failed: {e}")
            return False
    
    async def save_user_profile(self, user_id: int, emotion_manager) -> bool:
        """Save user profile data with emotional stats"""
        try:
            with self.get_session() as session:
                # Check if user exists
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    user = User(id=user_id)
                    session.add(user)
                    session.flush()
                
                # Save emotional data
                emotions_data = emotion_manager.user_emotions.get(user_id, {})
                emotions = session.query(UserEmotions).filter(UserEmotions.user_id == user_id).first()
                
                if not emotions:
                    emotions = UserEmotions(user_id=user_id)
                    session.add(emotions)
                
                # Update emotions data
                emotions.trust = emotions_data.get('trust', 0.0)
                emotions.resentment = emotions_data.get('resentment', 0.0)
                emotions.attachment = emotions_data.get('attachment', 0.0)
                emotions.protectiveness = emotions_data.get('protectiveness', 0.0)
                emotions.affection_points = emotions_data.get('affection_points', 0.0)
                emotions.annoyance = emotions_data.get('annoyance', 0.0)
                emotions.interaction_count = emotions_data.get('interaction_count', 0)
                emotions.emotion_history = emotions_data.get('emotion_history', [])
                
                # Parse timestamps if present
                if 'first_interaction' in emotions_data:
                    try:
                        emotions.first_interaction = datetime.fromisoformat(emotions_data['first_interaction'])
                    except (ValueError, TypeError):
                        emotions.first_interaction = datetime.now(timezone.utc)
                
                if 'last_interaction' in emotions_data:
                    try:
                        emotions.last_interaction = datetime.fromisoformat(emotions_data['last_interaction'])
                    except (ValueError, TypeError):
                        emotions.last_interaction = datetime.now(timezone.utc)
                
                # Save relationship progress data
                progress_data = emotion_manager.relationship_progress.get(user_id, {})
                if progress_data:
                    progress = session.query(RelationshipProgress).filter(
                        RelationshipProgress.user_id == user_id).first()
                    
                    if not progress:
                        progress = RelationshipProgress(user_id=user_id)
                        session.add(progress)
                    
                    progress.progress_data = progress_data
                    progress.updated_at = datetime.now(timezone.utc)
                
                # Save interaction stats
                stats_data = dict(emotion_manager.interaction_stats.get(user_id, Counter()))
                if stats_data:
                    stats = session.query(InteractionStats).filter(
                        InteractionStats.user_id == user_id).first()
                    
                    if not stats:
                        stats = InteractionStats(user_id=user_id)
                        session.add(stats)
                    
                    stats.total = stats_data.get('total', 0)
                    stats.positive = stats_data.get('positive', 0)
                    stats.negative = stats_data.get('negative', 0)
                    stats.neutral = stats_data.get('neutral', 0)
                    # Save other stats
                    other_stats = {k: v for k, v in stats_data.items() 
                                  if k not in ['total', 'positive', 'negative', 'neutral']}
                    if other_stats:
                        stats.stats_data = other_stats
                
                # Save memories if they exist
                if user_id in emotion_manager.user_memories and emotion_manager.user_memories[user_id]:
                    # Delete existing memories first (alternative: use upsert logic)
                    session.query(UserMemory).filter(UserMemory.user_id == user_id).delete()
                    
                    # Add new memories
                    for memory_data in emotion_manager.user_memories[user_id]:
                        memory = UserMemory(
                            user_id=user_id,
                            type=memory_data.get('type'),
                            description=memory_data.get('description', ''),
                            importance=memory_data.get('importance', 0.5)
                        )
                        
                        # Parse timestamp if present
                        if 'timestamp' in memory_data:
                            try:
                                memory.timestamp = datetime.fromisoformat(memory_data['timestamp'])
                            except (ValueError, TypeError):
                                memory.timestamp = datetime.now(timezone.utc)
                        
                        # Store additional data as JSON - use memory_metadata instead of metadata
                        metadata = {k: v for k, v in memory_data.items() 
                                   if k not in ['type', 'description', 'timestamp', 'importance']}
                        if metadata:
                            memory.memory_metadata = metadata
                            
                        session.add(memory)
                
                # Save events if they exist
                if user_id in emotion_manager.user_events and emotion_manager.user_events[user_id]:
                    # Delete existing events
                    session.query(UserEvent).filter(UserEvent.user_id == user_id).delete()
                    
                    # Add new events
                    for event_data in emotion_manager.user_events[user_id]:
                        event = UserEvent(
                            user_id=user_id,
                            type=event_data.get('type'),
                            message=event_data.get('message', ''),
                            effects=event_data.get('effects', {})
                        )
                        
                        # Parse timestamp if present
                        if 'timestamp' in event_data:
                            try:
                                event.timestamp = datetime.fromisoformat(event_data['timestamp'])
                            except (ValueError, TypeError):
                                event.timestamp = datetime.now(timezone.utc)
                                
                        session.add(event)
                
                # Save milestones if they exist
                if user_id in emotion_manager.user_milestones and emotion_manager.user_milestones[user_id]:
                    # Delete existing milestones
                    session.query(UserMilestone).filter(UserMilestone.user_id == user_id).delete()
                    
                    # Add new milestones
                    for milestone_data in emotion_manager.user_milestones[user_id]:
                        milestone = UserMilestone(
                            user_id=user_id,
                            name=milestone_data.get('name', ''),
                            description=milestone_data.get('description', ''),
                            score=milestone_data.get('score')
                        )
                        
                        # Parse timestamp if present
                        if 'timestamp' in milestone_data:
                            try:
                                milestone.timestamp = datetime.fromisoformat(milestone_data['timestamp'])
                            except (ValueError, TypeError):
                                milestone.timestamp = datetime.now(timezone.utc)
                                
                        session.add(milestone)
                
                # Commit all changes
                session.commit()
                self.logger.info(f"Successfully saved profile for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving data for user {user_id}: {e}")
            return False
    
    async def load_user_profile(self, user_id: int, emotion_manager):
        """Load user profile data with emotional stats"""
        try:
            with self.get_session() as session:
                # Load emotions data
                emotions = session.query(UserEmotions).filter(UserEmotions.user_id == user_id).first()
                if emotions:
                    emotion_manager.user_emotions[user_id] = emotions.to_dict()
                    self.logger.info(f"Successfully loaded emotions for user {user_id}")
                else:
                    self.logger.info(f"No emotions data found for user {user_id}")
                    return {}
                
                # Load relationship progress
                progress = session.query(RelationshipProgress).filter(
                    RelationshipProgress.user_id == user_id).first()
                if progress:
                    emotion_manager.relationship_progress[user_id] = progress.progress_data
                
                # Load interaction stats
                stats = session.query(InteractionStats).filter(
                    InteractionStats.user_id == user_id).first()
                if stats:
                    counter_data = {
                        'total': stats.total,
                        'positive': stats.positive,
                        'negative': stats.negative,
                        'neutral': stats.neutral
                    }
                    
                    # Add additional stats data if present
                    if stats.stats_data:
                        counter_data.update(stats.stats_data)
                        
                    emotion_manager.interaction_stats[user_id] = Counter(counter_data)
                
                # Load memories
                memories = session.query(UserMemory).filter(UserMemory.user_id == user_id).all()
                if memories:
                    emotion_manager.user_memories[user_id] = [m.to_dict() for m in memories]
                    self.logger.info(f"Loaded {len(memories)} memories for user {user_id}")
                else:
                    emotion_manager.user_memories[user_id] = []
                
                # Load events
                events = session.query(UserEvent).filter(UserEvent.user_id == user_id).all()
                if events:
                    emotion_manager.user_events[user_id] = [e.to_dict() for e in events]
                    self.logger.info(f"Loaded {len(events)} events for user {user_id}")
                else:
                    emotion_manager.user_events[user_id] = []
                
                # Load milestones
                milestones = session.query(UserMilestone).filter(UserMilestone.user_id == user_id).all()
                if milestones:
                    emotion_manager.user_milestones[user_id] = [m.to_dict() for m in milestones]
                    self.logger.info(f"Loaded {len(milestones)} milestones for user {user_id}")
                else:
                    emotion_manager.user_milestones[user_id] = []
                
                return emotion_manager.user_emotions[user_id]
                
        except Exception as e:
            self.logger.error(f"Error loading profile for user {user_id}: {e}")
            return {}
    
    async def save_user_profile_data(self, user_id: int, profile) -> bool:
        """Save user profile data object"""
        try:
            with self.get_session() as session:
                # Check if user exists
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    user = User(id=user_id)
                    session.add(user)
                    session.flush()
                
                # Get or create profile record
                db_profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                
                if not db_profile:
                    db_profile = UserProfile(user_id=user_id)
                    session.add(db_profile)
                
                # Convert profile to dict and update record
                profile_dict = profile.to_dict()
                
                # Update fields
                for key, value in profile_dict.items():
                    if hasattr(db_profile, key) and key != 'user_id':
                        setattr(db_profile, key, value)
                
                # Set updated timestamp
                db_profile.updated_at = datetime.now(timezone.utc)
                
                # Commit changes
                session.commit()
                self.logger.debug(f"Saved user profile for {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving user profile for {user_id}: {e}")
            return False
    
    async def load_user_profile_data(self, user_id: int, conversation_manager) -> bool:
        """Load user profile data object"""
        try:
            with self.get_session() as session:
                # Find profile record
                db_profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                
                if db_profile:
                    # Convert to dict and create profile object
                    profile_dict = db_profile.to_dict()
                    
                    # Create profile object using its class's from_dict method
                    profile = conversation_manager.user_profiles[user_id].__class__.from_dict(profile_dict)
                    conversation_manager.user_profiles[user_id] = profile
                    
                    self.logger.debug(f"Loaded user profile for {user_id}")
                    return True
                else:
                    self.logger.debug(f"No profile found for user {user_id}")
                    return False
                
        except Exception as e:
            self.logger.error(f"Error loading user profile for {user_id}: {e}")
            return False
    
    async def save_conversation(self, user_id: int, conversation_manager) -> bool:
        """Save conversation history and summary"""
        try:
            with self.get_session() as session:
                # Check if user exists
                user = session.query(User).filter(User.id == user_id).first()
                if not user:
                    user = User(id=user_id)
                    session.add(user)
                    session.flush()
                
                # Save conversation history
                if user_id in conversation_manager.conversations:
                    # Delete existing conversation (alternative: keep history)
                    session.query(Conversation).filter(Conversation.user_id == user_id).delete()
                    
                    # Add new messages
                    for msg_data in conversation_manager.conversations[user_id]:
                        message = Conversation(
                            user_id=user_id,
                            content=msg_data.get('content', ''),
                            from_bot=msg_data.get('from_bot', False),
                            topics=msg_data.get('topics', []),
                            referenced_users=msg_data.get('referenced_users', []),
                            sentiment=msg_data.get('sentiment')
                        )
                        
                        # Parse timestamp if present
                        if 'timestamp' in msg_data:
                            try:
                                message.timestamp = datetime.fromisoformat(msg_data['timestamp'])
                            except (ValueError, TypeError):
                                message.timestamp = datetime.now(timezone.utc)
                                
                        session.add(message)
                    
                    self.logger.debug(f"Saved conversation for user {user_id}")
                
                # Save conversation summary
                if user_id in conversation_manager.conversation_summaries:
                    summary = session.query(ConversationSummary).filter(
                        ConversationSummary.user_id == user_id).first()
                    
                    if not summary:
                        summary = ConversationSummary(user_id=user_id)
                        session.add(summary)
                    
                    summary.summary = conversation_manager.conversation_summaries[user_id]
                    summary.updated_at = datetime.now(timezone.utc)
                    
                    self.logger.debug(f"Saved conversation summary for user {user_id}")
                
                # Commit changes
                session.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving conversation for user {user_id}: {e}")
            return False
    
    async def load_conversation(self, user_id: int, conversation_manager) -> bool:
        """Load conversation history and summary"""
        try:
            with self.get_session() as session:
                # Load conversation history
                messages = session.query(Conversation).filter(
                    Conversation.user_id == user_id).order_by(Conversation.timestamp).all()
                
                if messages:
                    conversation_manager.conversations[user_id] = [m.to_dict() for m in messages]
                    self.logger.debug(f"Loaded {len(messages)} messages for user {user_id}")
                else:
                    conversation_manager.conversations[user_id] = []
                
                # Load conversation summary
                summary = session.query(ConversationSummary).filter(
                    ConversationSummary.user_id == user_id).first()
                
                if summary:
                    conversation_manager.conversation_summaries[user_id] = summary.summary
                    self.logger.debug(f"Loaded conversation summary for user {user_id}")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error loading conversation for user {user_id}: {e}")
            return False
    
    async def save_dm_settings(self, dm_enabled_users: Set[int]) -> bool:
        """Save DM permission settings"""
        try:
            with self.get_session() as session:
                # Get all existing settings
                existing_settings = session.query(DMSettings).all()
                existing_user_ids = {s.user_id for s in existing_settings}
                
                # Update existing settings - disable users no longer in set
                for setting in existing_settings:
                    setting.enabled = setting.user_id in dm_enabled_users
                
                # Add new users
                for user_id in dm_enabled_users:
                    if user_id not in existing_user_ids:
                        # Check if user exists
                        user = session.query(User).filter(User.id == user_id).first()
                        if not user:
                            user = User(id=user_id)
                            session.add(user)
                            session.flush()
                        
                        # Create new setting
                        setting = DMSettings(user_id=user_id, enabled=True)
                        session.add(setting)
                
                # Commit changes
                session.commit()
                self.logger.debug(f"Saved DM settings for {len(dm_enabled_users)} users")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving DM settings: {e}")
            return False
    
    async def load_dm_settings(self) -> Set[int]:
        """Load DM permission settings"""
        dm_enabled_users = set()
        
        try:
            with self.get_session() as session:
                # Get all enabled settings
                enabled_settings = session.query(DMSettings).filter(DMSettings.enabled == True).all()
                
                if enabled_settings:
                    dm_enabled_users = {s.user_id for s in enabled_settings}
                    self.logger.info(f"Loaded DM settings for {len(dm_enabled_users)} users")
                else:
                    self.logger.info("No DM settings found")
                    
        except Exception as e:
            self.logger.error(f"Error loading DM settings: {e}")
            
        return dm_enabled_users
    
    async def save_data(self, emotion_manager, conversation_manager=None) -> bool:
        """Save all emotional and conversation data"""
        success = True
        self.logger.info(f"Starting data save for {len(emotion_manager.user_emotions)} users")
        
        # Save emotional data for all users
        for user_id in emotion_manager.user_emotions:
            profile_success = await self.save_user_profile(user_id, emotion_manager)
            success = success and profile_success
            
            # Save conversation data if provided
            if conversation_manager and user_id in conversation_manager.conversations:
                conv_success = await self.save_conversation(user_id, conversation_manager)
                success = success and conv_success
                
                # Save user profile data
                if user_id in conversation_manager.user_profiles:
                    profile_success = await self.save_user_profile_data(
                        user_id, conversation_manager.user_profiles[user_id])
                    success = success and profile_success
        
        # Save DM settings
        dm_success = await self.save_dm_settings(emotion_manager.dm_enabled_users)
        success = success and dm_success
        
        self.logger.info(f"Data save complete for {len(emotion_manager.user_emotions)} users")
        return success
    
    async def load_data(self, emotion_manager, conversation_manager) -> bool:
        """Load all user data"""
        try:
            with self.get_session() as session:
                # Initialize containers
                emotion_manager.user_emotions = {}
                emotion_manager.user_memories = defaultdict(list)
                emotion_manager.user_events = defaultdict(list)
                emotion_manager.user_milestones = defaultdict(list)
                emotion_manager.interaction_stats = defaultdict(Counter)
                emotion_manager.relationship_progress = defaultdict(dict)
                
                # Verify database connection
                if not await self.verify_database_connection():
                    self.logger.error("Database connection unavailable. Data loading aborted.")
                    return False
                
                self.logger.info("Beginning data load process...")
                
                # Load user IDs
                user_ids = [user.id for user in session.query(User.id).all()]
                
                profile_count = 0
                for user_id in user_ids:
                    # Load emotional data
                    emotions = session.query(UserEmotions).filter(UserEmotions.user_id == user_id).first()
                    if emotions:
                        emotion_manager.user_emotions[user_id] = emotions.to_dict()
                        profile_count += 1
                        
                        # Load relationship progress
                        progress = session.query(RelationshipProgress).filter(
                            RelationshipProgress.user_id == user_id).first()
                        if progress:
                            emotion_manager.relationship_progress[user_id] = progress.progress_data
                        
                        # Load interaction stats
                        stats = session.query(InteractionStats).filter(
                            InteractionStats.user_id == user_id).first()
                        if stats:
                            counter_data = {
                                'total': stats.total,
                                'positive': stats.positive,
                                'negative': stats.negative,
                                'neutral': stats.neutral
                            }
                            # Add additional stats data
                            if stats.stats_data:
                                counter_data.update(stats.stats_data)
                                
                            emotion_manager.interaction_stats[user_id] = Counter(counter_data)
                        
                        # Load memories
                        memories = session.query(UserMemory).filter(UserMemory.user_id == user_id).all()
                        if memories:
                            emotion_manager.user_memories[user_id] = [m.to_dict() for m in memories]
                        
                        # Load events
                        events = session.query(UserEvent).filter(UserEvent.user_id == user_id).all()
                        if events:
                            emotion_manager.user_events[user_id] = [e.to_dict() for e in events]
                        
                        # Load milestones
                        milestones = session.query(UserMilestone).filter(UserMilestone.user_id == user_id).all()
                        if milestones:
                            emotion_manager.user_milestones[user_id] = [m.to_dict() for m in milestones]
                
                # Load profiles for conversation manager
                for user_id in user_ids:
                    db_profile = session.query(UserProfile).filter(UserProfile.user_id == user_id).first()
                    if db_profile:
                        # Create profile using from_dict
                        profile_dict = db_profile.to_dict()
                        profile = conversation_manager.get_or_create_profile(user_id)
                        profile = profile.__class__.from_dict(profile_dict)
                        conversation_manager.user_profiles[user_id] = profile
                        
                    # Load conversation data
                    messages = session.query(Conversation).filter(
                        Conversation.user_id == user_id).order_by(Conversation.timestamp).all()
                    if messages:
                        conversation_manager.conversations[user_id] = [m.to_dict() for m in messages]
                        
                    # Load conversation summary
                    summary = session.query(ConversationSummary).filter(
                        ConversationSummary.user_id == user_id).first()
                    if summary:
                        conversation_manager.conversation_summaries[user_id] = summary.summary
                
                # Load DM settings
                enabled_settings = session.query(DMSettings).filter(DMSettings.enabled == True).all()
                emotion_manager.dm_enabled_users = {s.user_id for s in enabled_settings}
                
                self.logger.info(f"Data load complete. Loaded {profile_count} profiles")
                return profile_count > 0
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    async def migrate_from_files(self, file_storage_manager, emotion_manager, conversation_manager) -> bool:
        """Migrate data from file storage to PostgreSQL"""
        try:
            # First, load data from files
            file_load_success = await file_storage_manager.load_data(emotion_manager, conversation_manager)
            
            if not file_load_success:
                self.logger.error("Failed to load data from files. Aborting migration.")
                return False
                
            self.logger.info(f"Loaded data from files for {len(emotion_manager.user_emotions)} users")
            
            # Now save to database
            db_save_success = await self.save_data(emotion_manager, conversation_manager)
            
            if db_save_success:
                self.logger.info("Migration from files to database completed successfully")
                return True
            else:
                self.logger.error("Failed to save data to database during migration")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during migration: {e}")
            return False
