"""
Conversation manager for the A2 Discord bot.
"""
import re
from datetime import datetime, timezone
from collections import defaultdict

from models.user_profile import UserProfile

class ConversationManager:
    """Manages conversation history and generates summaries"""
    
    def __init__(self):
        self.conversations = defaultdict(list)  # user_id -> list of messages
        self.conversation_summaries = {}  # user_id -> summary string
        self.MAX_HISTORY = 10  # Number of messages to remember
        self.user_profiles = {}  # user_id -> UserProfile
    
    def add_message(self, user_id, content, is_from_bot=False):
        """Add a message to the conversation history"""
        self.conversations[user_id].append({
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_bot": is_from_bot
        })
        
        # Keep only the last MAX_HISTORY messages
        if len(self.conversations[user_id]) > self.MAX_HISTORY:
            self.conversations[user_id] = self.conversations[user_id][-self.MAX_HISTORY:]
    
    def get_conversation_history(self, user_id):
        """Get formatted conversation history"""
        if user_id not in self.conversations:
            return "No prior conversation."
            
        history = []
        for msg in self.conversations[user_id]:
            sender = "A2" if msg["from_bot"] else "User"
            history.append(f"{sender}: {msg['content']}")
            
        return "\n".join(history)
    
    def get_or_create_profile(self, user_id, username=None):
        """Get existing profile or create a new one"""
        if user_id not in self.user_profiles:
            profile = UserProfile(user_id)
            if username:
                profile.name = username
            self.user_profiles[user_id] = profile
        return self.user_profiles[user_id]
    
    def update_name_recognition(self, user_id, name=None, nickname=None, preferred_name=None):
        """Update name recognition for a user"""
        profile = self.get_or_create_profile(user_id)
        
        if name:
            profile.name = name
        if nickname:
            profile.nickname = nickname
        if preferred_name:
            profile.preferred_name = preferred_name
        
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile
    
    def extract_profile_info(self, user_id, message_content):
        """Extract profile information from message content"""
        profile = self.get_or_create_profile(user_id)
        
        # Extract interests
        interest_patterns = [
            r"I (?:like|love|enjoy) (\w+ing)",
            r"I'm (?:interested in|passionate about) ([^.,]+)",
            r"favorite (?:hobby|activity) is ([^.,]+)"
        ]
        
        for pattern in interest_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                interest = match.group(1).strip().lower()
                if interest and interest not in profile.interests:
                    profile.interests.append(interest)
        
        # Extract personality traits
        trait_patterns = [
            r"I am (?:quite |very |extremely |really |)(\w+)",
            r"I'm (?:quite |very |extremely |really |)(\w+)",
            r"I consider myself (?:quite |very |extremely |really |)(\w+)"
        ]
        
        personality_traits = [
            "shy", "outgoing", "confident", "anxious", "creative", "logical", 
            "hardworking", "laid-back", "organized", "spontaneous", "sensitive",
            "resilient", "introverted", "extroverted", "curious", "cautious"
        ]
        
        for pattern in trait_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                trait = match.group(1).strip().lower()
                if trait in personality_traits and trait not in profile.personality_traits:
                    profile.personality_traits.append(trait)
        
        # Extract facts
        fact_patterns = [
            r"I (?:work as|am) an? ([^.,]+)",
            r"I live in ([^.,]+)",
            r"I'm from ([^.,]+)",
            r"I've been ([^.,]+)"
        ]
        
        for pattern in fact_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                fact = match.group(0).strip()
                if fact and fact not in profile.notable_facts:
                    profile.notable_facts.append(fact)
        
        # Extract name references
        name_patterns = [
            r"my name(?:'s| is) ([^.,]+)",
            r"call me ([^.,]+)",
            r"I go by ([^.,]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_content, re.I)
            if match:
                name_value = match.group(1).strip()
                if "nickname" in message_content.lower() or "call me" in message_content.lower():
                    profile.nickname = name_value
                else:
                    profile.name = name_value
        
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile
    
    def generate_summary(self, user_id, transformer_helper=None):
        """Generate a summary of the conversation"""
        if user_id not in self.conversations or len(self.conversations[user_id]) < 3:
            return "Not enough conversation history for a summary."
        
        # Get last few messages
        recent_msgs = self.conversations[user_id][-5:]
        
        # Format for summary generation
        conversation_text = "\n".join([
            f"{'A2' if msg['from_bot'] else 'User'}: {msg['content']}"
            for msg in recent_msgs
        ])
        
        summary = ""
        
        # Try using transformers if available
        if transformer_helper and transformer_helper.HAVE_TRANSFORMERS and transformer_helper.get_summarizer():
            try:
                # Limit to manageable size for the model
                if len(conversation_text) > 1000:
                    conversation_text = conversation_text[-1000:]
                    
                result = transformer_helper.get_summarizer()(conversation_text, max_length=50, min_length=10, do_sample=False)
                if result and len(result) > 0:
                    summary = result[0]['summary_text']
            except Exception as e:
                print(f"Error generating summary: {e}")
        
        # Fallback to a simpler approach
        if not summary:
            # Extract key topics with simple pattern matching
            topics = set()
            for msg in recent_msgs:
                # Find nouns and noun phrases (simple approach)
                content = msg["content"].lower()
                words = content.split()
                for word in words:
                    if len(word) > 4 and word not in ["about", "would", "could", "should", "their", "there", "these", "those", "have", "being"]:
                        topics.add(word)
            
            if topics:
                summary = f"Recent conversation about: {', '.join(list(topics)[:3])}."
            else:
                summary = "Brief conversation with no clear topic."
        
        self.conversation_summaries[user_id] = summary
        return summary
    
    def get_preferred_name(self, user_id):
        """Get the preferred name for a user"""
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if profile.preferred_name:
                return profile.preferred_name
            if profile.nickname:
                return profile.nickname
            if profile.name:
                return profile.name
        return None
