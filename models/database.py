"""
Database models for the A2 Discord bot using SQLAlchemy.
"""
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json

from sqlalchemy import (
    Column, Integer, String, Float, Boolean, Text, 
    DateTime, ForeignKey, JSON, Table, MetaData, 
    create_engine, inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.dialects.postgresql import JSONB

Base = declarative_base()

class User(Base):
    """Base user information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)  # Discord user ID
    name = Column(String(255), nullable=True)  # Discord username
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                         onupdate=lambda: datetime.now(timezone.utc))
    
    # Relationships
    emotions = relationship("UserEmotions", back_populates="user", uselist=False, 
                            cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", 
                           cascade="all, delete-orphan")
    events = relationship("UserEvent", back_populates="user", 
                         cascade="all, delete-orphan")
    milestones = relationship("UserMilestone", back_populates="user", 
                             cascade="all, delete-orphan")
    profile = relationship("UserProfile", back_populates="user", uselist=False, 
                          cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", 
                               cascade="all, delete-orphan")
    dm_settings = relationship("DMSettings", back_populates="user", uselist=False, 
                              cascade="all, delete-orphan")


class UserEmotions(Base):
    """Stores emotional state data for a user"""
    __tablename__ = 'user_emotions'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    trust = Column(Float, default=0.0)
    resentment = Column(Float, default=0.0)
    attachment = Column(Float, default=0.0)
    protectiveness = Column(Float, default=0.0)
    affection_points = Column(Float, default=0.0)
    annoyance = Column(Float, default=0.0)
    first_interaction = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_interaction = Column(DateTime, default=lambda: datetime.now(timezone.utc), 
                            onupdate=lambda: datetime.now(timezone.utc))
    interaction_count = Column(Integer, default=0)
    emotion_history = Column(JSONB, default=list)  # Store emotion history as JSON
    
    # Relationships
    user = relationship("User", back_populates="emotions")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "trust": self.trust,
            "resentment": self.resentment,
            "attachment": self.attachment,
            "protectiveness": self.protectiveness,
            "affection_points": self.affection_points,
            "annoyance": self.annoyance,
            "first_interaction": self.first_interaction.isoformat() if self.first_interaction else None,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None,
            "interaction_count": self.interaction_count,
            "emotion_history": self.emotion_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserEmotions':
        """Create from dictionary"""
        emotions = cls()
        emotions.trust = data.get("trust", 0.0)
        emotions.resentment = data.get("resentment", 0.0)
        emotions.attachment = data.get("attachment", 0.0)
        emotions.protectiveness = data.get("protectiveness", 0.0)
        emotions.affection_points = data.get("affection_points", 0.0)
        emotions.annoyance = data.get("annoyance", 0.0)
        emotions.interaction_count = data.get("interaction_count", 0)
        emotions.emotion_history = data.get("emotion_history", [])
        
        # Parse dates if provided
        if data.get("first_interaction"):
            try:
                emotions.first_interaction = datetime.fromisoformat(data["first_interaction"])
            except (ValueError, TypeError):
                emotions.first_interaction = datetime.now(timezone.utc)
                
        if data.get("last_interaction"):
            try:
                emotions.last_interaction = datetime.fromisoformat(data["last_interaction"])
            except (ValueError, TypeError):
                emotions.last_interaction = datetime.now(timezone.utc)
        
        return emotions


class UserMemory(Base):
    """Represents a memory entry for a user"""
    __tablename__ = 'user_memories'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    type = Column(String(50), nullable=True)
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    importance = Column(Float, default=0.5)
    metadata = Column(JSONB, nullable=True)  # Additional memory metadata
    
    # Relationships
    user = relationship("User", back_populates="memories")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "type": self.type,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "importance": self.importance,
            "metadata": self.metadata
        }


class UserEvent(Base):
    """Represents an event record for a user"""
    __tablename__ = 'user_events'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    type = Column(String(50), nullable=True)
    message = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    effects = Column(JSONB, nullable=True)  # Emotional effects of the event
    
    # Relationships
    user = relationship("User", back_populates="events")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "type": self.type,
            "message": self.message,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "effects": self.effects
        }


class UserMilestone(Base):
    """Represents a relationship milestone achievement for a user"""
    __tablename__ = 'user_milestones'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    score = Column(Float, nullable=True)  # Relationship score at milestone
    
    # Relationships
    user = relationship("User", back_populates="milestones")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "score": self.score
        }


class UserProfile(Base):
    """Stores detailed user profile information"""
    __tablename__ = 'user_profiles'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    name = Column(String(255), nullable=True)
    nickname = Column(String(255), nullable=True)
    preferred_name = Column(String(255), nullable=True)
    personality_traits = Column(JSONB, default=list)
    interests = Column(JSONB, default=list)
    notable_facts = Column(JSONB, default=list)
    relationship_context = Column(JSONB, default=list)
    conversation_topics = Column(JSONB, default=list)
    communication_style = Column(JSONB, default=list)
    languages = Column(JSONB, default=list)
    time_zone = Column(String(50), nullable=True)
    preferences = Column(JSONB, default=dict)
    frequent_topics = Column(JSONB, default=dict)
    response_patterns = Column(JSONB, default=dict)
    sentiment_history = Column(JSONB, default=list)
    mentioned_users = Column(JSONB, default=dict)
    shared_interests = Column(JSONB, default=dict)
    relationship_with_others = Column(JSONB, default=dict)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                      onupdate=lambda: datetime.now(timezone.utc))
    interaction_count = Column(Integer, default=0)
    last_interaction = Column(DateTime, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="profile")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "nickname": self.nickname,
            "preferred_name": self.preferred_name,
            "personality_traits": self.personality_traits,
            "interests": self.interests,
            "notable_facts": self.notable_facts,
            "relationship_context": self.relationship_context,
            "conversation_topics": self.conversation_topics,
            "communication_style": self.communication_style,
            "languages": self.languages,
            "time_zone": self.time_zone,
            "preferences": self.preferences,
            "frequent_topics": self.frequent_topics,
            "response_patterns": self.response_patterns,
            "sentiment_history": self.sentiment_history,
            "mentioned_users": self.mentioned_users,
            "shared_interests": self.shared_interests,
            "relationship_with_others": self.relationship_with_others,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "interaction_count": self.interaction_count,
            "last_interaction": self.last_interaction.isoformat() if self.last_interaction else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create profile from dictionary"""
        profile = cls()
        for key, value in data.items():
            if hasattr(profile, key):
                if key in ['created_at', 'updated_at', 'last_interaction'] and value:
                    try:
                        setattr(profile, key, datetime.fromisoformat(value))
                    except (ValueError, TypeError):
                        pass
                else:
                    setattr(profile, key, value)
        return profile


class Conversation(Base):
    """Stores conversation history for a user"""
    __tablename__ = 'conversations'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    from_bot = Column(Boolean, default=False)
    topics = Column(JSONB, default=list)
    referenced_users = Column(JSONB, default=list)
    sentiment = Column(Float, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="conversations")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "from_bot": self.from_bot,
            "topics": self.topics,
            "referenced_users": self.referenced_users,
            "sentiment": self.sentiment
        }


class ConversationSummary(Base):
    """Stores generated conversation summaries"""
    __tablename__ = 'conversation_summaries'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    summary = Column(Text, nullable=False)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "summary": self.summary,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class DMSettings(Base):
    """Stores DM permission settings for users"""
    __tablename__ = 'dm_settings'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    enabled = Column(Boolean, default=False)
    
    # Relationships
    user = relationship("User", back_populates="dm_settings")


class InteractionStats(Base):
    """Stores interaction statistics for users"""
    __tablename__ = 'interaction_stats'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    total = Column(Integer, default=0)
    positive = Column(Integer, default=0)
    negative = Column(Integer, default=0)
    neutral = Column(Integer, default=0)
    stats_data = Column(JSONB, default=dict)  # Additional stats data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "total": self.total,
            "positive": self.positive,
            "negative": self.negative,
            "neutral": self.neutral,
            "stats_data": self.stats_data
        }


class RelationshipProgress(Base):
    """Stores relationship progress information"""
    __tablename__ = 'relationship_progress'
    
    user_id = Column(Integer, ForeignKey('users.id'), primary_key=True)
    progress_data = Column(JSONB, default=dict)
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc),
                      onupdate=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "progress_data": self.progress_data,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


def init_db(database_url):
    """Initialize the database"""
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    return engine


def get_session_factory(engine):
    """Create a session factory"""
    return sessionmaker(bind=engine)
