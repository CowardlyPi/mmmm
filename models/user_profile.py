"""
Enhanced user profile model for the A2 Discord bot.
"""
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

class UserProfile:
    """Stores detailed information about users that A2 interacts with"""
    
    def __init__(self, user_id):
        # Basic identity information
        self.user_id = user_id
        self.name = None
        self.nickname = None
        self.preferred_name = None
        
        # Core profile elements
        self.personality_traits = []
        self.interests = []
        self.notable_facts = []
        self.relationship_context = []
        self.conversation_topics = []
        
        # Enhanced profile information
        self.communication_style = []  # e.g., "formal", "casual", "technical", "direct"
        self.languages = []  # Languages the user speaks
        self.time_zone = None  # User's time zone if revealed
        self.preferences = {}  # Key-value store for user preferences
        
        # Conversation patterns
        self.frequent_topics = {}  # Topic -> frequency count
        self.response_patterns = {}  # User's common responses to certain topics
        self.sentiment_history = []  # Track sentiment over time
        
        # Cross-user relationships
        self.mentioned_users = {}  # User IDs mentioned by this user
        self.shared_interests = {}  # Interests shared with other users
        self.relationship_with_others = {}  # How they talk about other users
        
        # Metadata
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.interaction_count = 0  # Total number of interactions
        self.last_interaction = None  # Timestamp of last interaction
        
    def update_profile(self, field, value):
        """Update a specific field in the profile"""
        if hasattr(self, field):
            setattr(self, field, value)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False
    
    def increment_topic(self, topic, count=1):
        """Increment the frequency count for a conversation topic"""
        self.frequent_topics[topic] = self.frequent_topics.get(topic, 0) + count
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def add_sentiment_entry(self, message_content, sentiment_value):
        """Add a sentiment analysis entry to the user's history"""
        self.sentiment_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "content_preview": message_content[:50] + ("..." if len(message_content) > 50 else ""),
            "sentiment": sentiment_value
        })
        # Keep history at a reasonable size
        if len(self.sentiment_history) > 50:
            self.sentiment_history = self.sentiment_history[-50:]
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def add_mentioned_user(self, mentioned_user_id, context):
        """Record when this user mentions another user"""
        if mentioned_user_id not in self.mentioned_users:
            self.mentioned_users[mentioned_user_id] = []
        
        self.mentioned_users[mentioned_user_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context[:100]  # Store a preview of the mention context
        })
        
        # Keep the list at a reasonable size
        if len(self.mentioned_users[mentioned_user_id]) > 20:
            self.mentioned_users[mentioned_user_id] = self.mentioned_users[mentioned_user_id][-20:]
        
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def update_communication_style(self, style_marker):
        """Update the user's communication style based on observed patterns"""
        if style_marker and style_marker not in self.communication_style:
            self.communication_style.append(style_marker)
            # Keep a reasonable number of style markers
            if len(self.communication_style) > 10:
                self.communication_style = self.communication_style[-10:]
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def record_interaction(self):
        """Record an interaction with this user"""
        self.interaction_count += 1
        self.last_interaction = datetime.now(timezone.utc).isoformat()
        self.updated_at = self.last_interaction
    
    def set_preference(self, key, value):
        """Set a user preference"""
        self.preferences[key] = value
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def get_preference(self, key, default=None):
        """Get a user preference with a default fallback"""
        return self.preferences.get(key, default)
    
    def to_dict(self):
        """Convert profile to dictionary for storage"""
        return {k: v for k, v in self.__dict__.items()}
    
    @classmethod
    def from_dict(cls, data):
        """Create profile from dictionary"""
        profile = cls(data.get('user_id'))
        for k, v in data.items():
            if hasattr(profile, k):
                setattr(profile, k, v)
        return profile
    
    def get_summary(self):
        """Generate a human-readable summary of the profile"""
        summary = []
        
        # Name information
        if self.preferred_name:
            summary.append(f"Name: {self.preferred_name}")
        elif self.nickname:
            summary.append(f"Name: {self.nickname}")
        elif self.name:
            summary.append(f"Name: {self.name}")
        
        # Core personality and interests    
        if self.personality_traits:
            summary.append(f"Personality: {', '.join(self.personality_traits[:3])}")
            
        if self.interests:
            summary.append(f"Interests: {', '.join(self.interests[:3])}")
        
        # Communication patterns
        if self.communication_style:
            summary.append(f"Communication style: {', '.join(self.communication_style[:3])}")
        
        if self.languages:
            summary.append(f"Languages: {', '.join(self.languages)}")
            
        # Important facts and context
        if self.notable_facts:
            summary.append(f"Notable facts: {'; '.join(self.notable_facts[:2])}")
            
        if self.relationship_context:
            summary.append(f"Relationship context: {'; '.join(self.relationship_context[:2])}")
        
        # Most frequent topics if available
        if self.frequent_topics:
            top_topics = sorted(self.frequent_topics.items(), key=lambda x: x[1], reverse=True)[:3]
            topic_strs = [f"{topic}" for topic, count in top_topics]
            if topic_strs:
                summary.append(f"Frequent topics: {', '.join(topic_strs)}")
        
        # Add interaction stats
        if self.interaction_count > 0:
            summary.append(f"Interactions: {self.interaction_count}")
            
        return " | ".join(summary)
    
    def get_detailed_summary(self):
        """Generate a more detailed, multi-line summary for administrative purposes"""
        summary = [f"**User Profile: {self.get_display_name()}**"]
        
        # Basic information
        summary.append("**Basic Information:**")
        if self.name:
            summary.append(f"- Name: {self.name}")
        if self.nickname:
            summary.append(f"- Nickname: {self.nickname}")
        if self.preferred_name:
            summary.append(f"- Preferred name: {self.preferred_name}")
        
        # Personality and interests
        if self.personality_traits:
            summary.append("\n**Personality:**")
            summary.append(f"- Traits: {', '.join(self.personality_traits)}")
        
        if self.interests:
            summary.append("\n**Interests:**")
            summary.append(f"- {', '.join(self.interests)}")
        
        # Communication patterns
        comm_section = []
        if self.communication_style:
            comm_section.append(f"- Style: {', '.join(self.communication_style)}")
        if self.languages:
            comm_section.append(f"- Languages: {', '.join(self.languages)}")
        if self.frequent_topics:
            top_topics = sorted(self.frequent_topics.items(), key=lambda x: x[1], reverse=True)[:5]
            comm_section.append(f"- Top topics: {', '.join([topic for topic, _ in top_topics])}")
        
        if comm_section:
            summary.append("\n**Communication:**")
            summary.extend(comm_section)
        
        # Notable facts
        if self.notable_facts:
            summary.append("\n**Notable Facts:**")
            for fact in self.notable_facts[:5]:
                summary.append(f"- {fact}")
        
        # Relationship context
        if self.relationship_context:
            summary.append("\n**Relationship Context:**")
            for context in self.relationship_context[:3]:
                summary.append(f"- {context}")
        
        # Interaction statistics
        summary.append("\n**Interaction Statistics:**")
        summary.append(f"- Total interactions: {self.interaction_count}")
        if self.last_interaction:
            last_dt = datetime.fromisoformat(self.last_interaction)
            summary.append(f"- Last interaction: {last_dt.strftime('%Y-%m-%d %H:%M')}")
        summary.append(f"- Profile created: {datetime.fromisoformat(self.created_at).strftime('%Y-%m-%d')}")
        
        return '\n'.join(summary)
    
    def get_display_name(self):
        """Get the best display name for this user"""
        if self.preferred_name:
            return self.preferred_name
        if self.nickname:
            return self.nickname
        if self.name:
            return self.name
        return f"User-{self.user_id}"
        
    def analyze_sentiment_trend(self):
        """Analyze the trend in user sentiment over recent interactions"""
        if not self.sentiment_history or len(self.sentiment_history) < 3:
            return "Not enough data to determine sentiment trend."
            
        # Get last 10 sentiment values
        recent = self.sentiment_history[-10:]
        sentiment_values = [entry.get("sentiment", 0) for entry in recent]
        
        # Calculate simple trend (positive slope = improving, negative = declining)
        if len(sentiment_values) >= 5:
            # Simple linear regression would be better, but this is a rough approximation
            first_half = sum(sentiment_values[:len(sentiment_values)//2]) / (len(sentiment_values)//2)
            second_half = sum(sentiment_values[len(sentiment_values)//2:]) / (len(sentiment_values) - len(sentiment_values)//2)
            
            if second_half - first_half > 0.2:
                return "Improving sentiment trend."
            elif first_half - second_half > 0.2:
                return "Declining sentiment trend."
                
        # Calculate average sentiment
        avg_sentiment = sum(sentiment_values) / len(sentiment_values)
        
        if avg_sentiment > 0.7:
            return "Consistently positive sentiment."
        elif avg_sentiment > 0.5:
            return "Generally positive sentiment."
        elif avg_sentiment > 0.3:
            return "Neutral to mildly positive sentiment."
        elif avg_sentiment > 0:
            return "Mixed sentiment, trending neutral."
        else:
            return "Generally negative sentiment."
