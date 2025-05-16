"""
User profile model for the A2 Discord bot.
"""
from datetime import datetime, timezone

class UserProfile:
    """Stores detailed information about users that A2 interacts with"""
    
    def __init__(self, user_id):
        self.user_id = user_id
        self.name = None
        self.nickname = None
        self.preferred_name = None
        self.personality_traits = []
        self.interests = []
        self.notable_facts = []
        self.relationship_context = []
        self.conversation_topics = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def update_profile(self, field, value):
        """Update a specific field in the profile"""
        if hasattr(self, field):
            setattr(self, field, value)
            self.updated_at = datetime.now(timezone.utc).isoformat()
            return True
        return False
    
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
        
        if self.preferred_name:
            summary.append(f"Name: {self.preferred_name}")
        elif self.nickname:
            summary.append(f"Name: {self.nickname}")
        elif self.name:
            summary.append(f"Name: {self.name}")
            
        if self.personality_traits:
            summary.append(f"Personality: {', '.join(self.personality_traits[:3])}")
            
        if self.interests:
            summary.append(f"Interests: {', '.join(self.interests[:3])}")
            
        if self.notable_facts:
            summary.append(f"Notable facts: {'; '.join(self.notable_facts[:2])}")
            
        if self.relationship_context:
            summary.append(f"Relationship context: {'; '.join(self.relationship_context[:2])}")
            
        return " | ".join(summary)
