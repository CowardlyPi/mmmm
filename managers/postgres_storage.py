"""
Modified PostgreSQL storage manager with pagination support.
"""
import json
import logging
from datetime import datetime, timezone
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Set, Union, Tuple, Iterator

from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, not_, func

from models.database import (
    User, UserEmotions, UserMemory, UserEvent, UserMilestone,
    UserProfile, Conversation, ConversationSummary, DMSettings,
    InteractionStats, RelationshipProgress,
    init_db, get_session_factory
)

from utils.logging_helper import get_logger
from utils.pagination import Paginator, BatchProcessor

class PostgreSQLStorageManager:
    """Handles all data persistence operations using PostgreSQL with pagination support"""
    
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
        # This method doesn't need pagination as it operates on a single user
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
                
                # Similar code for events and milestones
                # [code omitted for brevity as it's the same pattern]
                
                # Commit all changes
                session.commit()
                self.logger.info(f"Successfully saved profile for user {user_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving data for user {user_id}: {e}")
            return False
    
    async def load_user_profile(self, user_id: int, emotion_manager):
        """Load user profile data with emotional stats"""
        # This method doesn't need pagination as it operates on a single user
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
                
                # Rest of the loading code for a single user (unchanged)
                # [code omitted for brevity as it's unchanged]
                
                return emotion_manager.user_emotions[user_id]
                
        except Exception as e:
            self.logger.error(f"Error loading profile for user {user_id}: {e}")
            return {}
    
    async def save_data(self, emotion_manager, conversation_manager=None) -> bool:
        """Save all emotional and conversation data"""
        # We can still save all data at once since saving is typically done for active users only
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
    
    async def save_dm_settings(self, dm_enabled_users) -> bool:
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
    
    async def load_data(self, emotion_manager, conversation_manager, batch_size=50) -> bool:
        """Load all user data with pagination support
        
        Args:
            emotion_manager: The emotion manager to populate
            conversation_manager: The conversation manager to populate
            batch_size: Number of users to load in each batch
            
        Returns:
            bool: True if data was loaded successfully
        """
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
                
                self.logger.info("Beginning data load process with pagination...")
                
                # Create paginator for users
                user_paginator = Paginator(session, User, page_size=batch_size)
                pagination_info = user_paginator.get_info()
                
                self.logger.info(f"Found {pagination_info['total_records']} users across {pagination_info['total_pages']} pages")
                
                # Process each page of users
                profile_count = 0
                page_number = 1
                total_pages = pagination_info['total_pages']
                
                # Load essential data for all users
                # Option 1: Load each page of users
                for page_number in range(1, total_pages + 1):
                    users_page = user_paginator.get_page(page_number)
                    self.logger.info(f"Loading page {page_number}/{total_pages} ({len(users_page)} users)")
                    
                    user_ids = [user.id for user in users_page]
                    
                    # Batch load emotions for these users
                    emotions = session.query(UserEmotions).filter(UserEmotions.user_id.in_(user_ids)).all()
                    for emotion in emotions:
                        emotion_manager.user_emotions[emotion.user_id] = emotion.to_dict()
                        profile_count += 1
                    
                    # Batch load relationship progress
                    progress_records = session.query(RelationshipProgress).filter(
                        RelationshipProgress.user_id.in_(user_ids)).all()
                    for progress in progress_records:
                        emotion_manager.relationship_progress[progress.user_id] = progress.progress_data
                    
                    # Batch load interaction stats
                    stats_records = session.query(InteractionStats).filter(
                        InteractionStats.user_id.in_(user_ids)).all()
                    for stats in stats_records:
                        counter_data = {
                            'total': stats.total,
                            'positive': stats.positive,
                            'negative': stats.negative,
                            'neutral': stats.neutral
                        }
                        # Add additional stats data
                        if stats.stats_data:
                            counter_data.update(stats.stats_data)
                            
                        emotion_manager.interaction_stats[stats.user_id] = Counter(counter_data)
                    
                    # Report progress
                    self.logger.info(f"Loaded {profile_count} profiles so far")
                
                # Option 2: Process additional data only for active users
                # For active users, load memories, events, etc. which are typically only needed for active users
                
                # Define what "active" means - for example, users with activity in the last 30 days
                cutoff_date = datetime.now(timezone.utc) - timezone.timedelta(days=30)
                active_user_ids = [
                    uid for uid, data in emotion_manager.user_emotions.items()
                    if 'last_interaction' in data and 
                    datetime.fromisoformat(data['last_interaction']) > cutoff_date
                ]
                
                self.logger.info(f"Found {len(active_user_ids)} active users to load detailed data for")
                
                # Process active users in batches
                for i in range(0, len(active_user_ids), batch_size):
                    batch_ids = active_user_ids[i:i+batch_size]
                    self.logger.info(f"Loading detailed data for active users batch {i//batch_size + 1}")
                    
                    # Load memories for active users
                    memories = session.query(UserMemory).filter(UserMemory.user_id.in_(batch_ids)).all()
                    for memory in memories:
                        emotion_manager.user_memories[memory.user_id].append(memory.to_dict())
                    
                    # Load events for active users
                    events = session.query(UserEvent).filter(UserEvent.user_id.in_(batch_ids)).all()
                    for event in events:
                        emotion_manager.user_events[event.user_id].append(event.to_dict())
                    
                    # Load milestones for active users
                    milestones = session.query(UserMilestone).filter(UserMilestone.user_id.in_(batch_ids)).all()
                    for milestone in milestones:
                        emotion_manager.user_milestones[milestone.user_id].append(milestone.to_dict())
                    
                    # Load user profiles for active users
                    profiles = session.query(UserProfile).filter(UserProfile.user_id.in_(batch_ids)).all()
                    for profile in profiles:
                        profile_dict = profile.to_dict()
                        user_profile = conversation_manager.get_or_create_profile(profile.user_id)
                        conversation_manager.user_profiles[profile.user_id] = user_profile.__class__.from_dict(profile_dict)
                    
                    # Load conversations for active users
                    for user_id in batch_ids:
                        # Get the user's recent conversations (last 20)
                        recent_messages = session.query(Conversation).filter(
                            Conversation.user_id == user_id
                        ).order_by(
                            Conversation.timestamp.desc()
                        ).limit(20).all()
                        
                        if recent_messages:
                            # Convert to list and reverse to get chronological order
                            conversation_manager.conversations[user_id] = [
                                m.to_dict() for m in reversed(recent_messages)
                            ]
                        
                        # Load conversation summary
                        summary = session.query(ConversationSummary).filter(
                            ConversationSummary.user_id == user_id
                        ).first()
                        
                        if summary:
                            conversation_manager.conversation_summaries[user_id] = summary.summary
                
                # Load DM settings (not paginated as it's typically a small set)
                enabled_settings = session.query(DMSettings).filter(DMSettings.enabled == True).all()
                emotion_manager.dm_enabled_users = {s.user_id for s in enabled_settings}
                
                self.logger.info(f"Data load complete. Loaded {profile_count} profiles")
                return profile_count > 0
                
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    async def get_active_users(self, days=30) -> List[int]:
        """
        Get list of active user IDs
        
        Args:
            days: Number of days to consider for activity
            
        Returns:
            List of active user IDs
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timezone.timedelta(days=days)
            
            with self.get_session() as session:
                active_ids = session.query(UserEmotions.user_id).filter(
                    UserEmotions.last_interaction >= cutoff_date
                ).all()
                
                return [uid[0] for uid in active_ids]
                
        except Exception as e:
            self.logger.error(f"Error getting active users: {e}")
            return []
    
    async def get_users_iterator(self, batch_size=50) -> Iterator[List[int]]:
        """
        Get an iterator of user IDs in batches
        
        Args:
            batch_size: Number of user IDs per batch
            
        Returns:
            Iterator yielding batches of user IDs
        """
        try:
            with self.get_session() as session:
                # Get total user count
                total_users = session.query(func.count(User.id)).scalar() or 0
                
                # Calculate number of batches
                total_batches = (total_users + batch_size - 1) // batch_size
                
                for batch_num in range(total_batches):
                    offset = batch_num * batch_size
                    user_ids = session.query(User.id).order_by(User.id).offset(offset).limit(batch_size).all()
                    yield [uid[0] for uid in user_ids]
                    
        except Exception as e:
            self.logger.error(f"Error in users iterator: {e}")
            yield []
