"""
Enhanced conversation manager for the A2 Discord bot.
"""
import re
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import random

class ConversationManager:
    """Manages conversation history and generates summaries"""
    
    def __init__(self):
        self.conversations = defaultdict(list)  # user_id -> list of messages
        self.conversation_summaries = {}  # user_id -> summary string
        self.MAX_HISTORY = 20  # Increased from 10 to 20 as requested
        self.user_profiles = {}  # user_id -> UserProfile
        self.topic_index = defaultdict(list)  # topic -> [(user_id, message_index), ...]
        self.cross_references = defaultdict(set)  # user_id -> set of users they've mentioned
        self.conversation_topics = defaultdict(dict)  # user_id -> {topic: last_discussed_timestamp}
        self.conversation_sentiment = defaultdict(list)  # user_id -> list of sentiment values
    
    def add_message(self, user_id, content, is_from_bot=False):
        """Add a message to the conversation history"""
        message = {
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_bot": is_from_bot,
            "topics": [],  # Will be filled by extract_topics
            "referenced_users": [],  # Will be filled if message references other users
            "sentiment": None  # Will be filled by sentiment analysis if available
        }
        
        self.conversations[user_id].append(message)
        
        # Extract topics from the message for future reference
        topics = self._extract_topics(content)
        message["topics"] = topics
        
        # Update topic index for search capabilities
        msg_index = len(self.conversations[user_id]) - 1
        for topic in topics:
            self.topic_index[topic].append((user_id, msg_index))
            self.conversation_topics[user_id][topic] = message["timestamp"]
        
        # Keep only the last MAX_HISTORY messages
        if len(self.conversations[user_id]) > self.MAX_HISTORY:
            # Remove oldest message from topic index before discarding
            old_message = self.conversations[user_id][0]
            for topic in old_message.get("topics", []):
                self.topic_index[topic] = [(u, i) for u, i in self.topic_index[topic] 
                                         if u != user_id or i != 0]
            
            # Shift all indexes down by 1
            for topic in self.topic_index:
                self.topic_index[topic] = [(u, i-1 if u == user_id and i > 0 else i) 
                                         for u, i in self.topic_index[topic]]
            
            self.conversations[user_id] = self.conversations[user_id][1:]
    
    def _extract_topics(self, content, min_length=4):
        """Extract potential conversation topics from message content"""
        # Naive approach: extract significant words and phrases
        # A more sophisticated approach would use NLP for entity recognition and topic modeling
        
        # Convert to lowercase and remove punctuation
        content = content.lower()
        content = re.sub(r'[^\w\s]', ' ', content)
        
        # Extract words that are likely to be meaningful
        words = content.split()
        topics = []
        
        # Filter out common stop words 
        stop_words = {"the", "and", "or", "a", "an", "in", "on", "at", "to", "for", "with", 
                     "by", "about", "like", "through", "over", "before", "between", "after",
                     "since", "without", "under", "within", "along", "following", "across",
                     "behind", "beyond", "plus", "except", "but", "up", "down", "off", "above",
                     "is", "are", "was", "were", "be", "being", "been", "have", "has", "had",
                     "do", "does", "did", "will", "would", "shall", "should", "may", "might",
                     "must", "can", "could", "i", "you", "he", "she", "it", "we", "they",
                     "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
                     "this", "that", "these", "those", "am", "what", "when", "where", "who", "why",
                     "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", 
                     "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"}
        
        # Extract potential topic words
        for word in words:
            if len(word) >= min_length and word not in stop_words:
                topics.append(word)
        
        # Look for meaningful phrases (simplistic approach)
        potential_phrases = []
        for i in range(len(words) - 1):
            if len(words[i]) >= 3 and len(words[i+1]) >= 3:
                # Avoid phrases with too many stop words
                if words[i] not in stop_words or words[i+1] not in stop_words:
                    phrase = f"{words[i]} {words[i+1]}"
                    potential_phrases.append(phrase)
        
        # Combine single words and phrases
        all_topics = topics + potential_phrases
        
        # Remove duplicates and return
        return list(set(all_topics))[:5]  # Limit to 5 topics per message
    
    def get_conversation_history(self, user_id, limit=None):
        """Get formatted conversation history"""
        if user_id not in self.conversations:
            return "No prior conversation."
            
        history = []
        messages = self.conversations[user_id]
        
        # Apply limit if specified, otherwise use all messages
        if limit and limit < len(messages):
            messages = messages[-limit:]
            
        for msg in messages:
            sender = "A2" if msg["from_bot"] else "User"
            history.append(f"{sender}: {msg['content']}")
            
        return "\n".join(history)
    
    def get_conversations_by_topic(self, topic, max_results=5):
        """Find conversations across users that contain a specific topic"""
        results = []
        if topic not in self.topic_index:
            return results
            
        for user_id, msg_index in self.topic_index[topic]:
            if user_id in self.conversations and msg_index < len(self.conversations[user_id]):
                # Get the message and surrounding context
                message = self.conversations[user_id][msg_index]
                context = self._get_conversation_context(user_id, msg_index, window=1)
                
                # Add to results
                results.append({
                    "user_id": user_id, 
                    "message": message,
                    "context": context,
                    "timestamp": message["timestamp"]
                })
                
                # Break if we have enough results
                if len(results) >= max_results:
                    break
        
        # Sort by recency
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
    
    def _get_conversation_context(self, user_id, message_index, window=1):
        """Get a window of messages around a specific message index"""
        if user_id not in self.conversations:
            return []
            
        messages = self.conversations[user_id]
        start_idx = max(0, message_index - window)
        end_idx = min(len(messages), message_index + window + 1)
        
        return messages[start_idx:end_idx]
    
    def search_conversations(self, query, user_id=None, max_results=5):
        """Search conversations for messages containing the query"""
        results = []
        
        # Normalize query
        query = query.lower()
        
        # If user_id is provided, only search that user's conversations
        if user_id:
            user_ids = [user_id]
        else:
            user_ids = list(self.conversations.keys())
        
        # Search through conversations
        for uid in user_ids:
            for i, message in enumerate(self.conversations[uid]):
                content = message["content"].lower()
                if query in content:
                    results.append({
                        "user_id": uid,
                        "message": message,
                        "context": self._get_conversation_context(uid, i, window=1),
                        "timestamp": message["timestamp"]
                    })
                    
                    # Break if we have enough results
                    if len(results) >= max_results:
                        break
        
        # Sort by recency
        results.sort(key=lambda x: x["timestamp"], reverse=True)
        return results
    
    def get_or_create_profile(self, user_id, username=None):
        """Get existing profile or create a new one"""
        if user_id not in self.user_profiles:
            # Import the UserProfile class if needed
            from models.user_profile import UserProfile
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
        """Extract profile information from message content with improved pattern matching"""
        profile = self.get_or_create_profile(user_id)
        
        # Extract interests with enhanced patterns
        interest_patterns = [
            r"I (?:like|love|enjoy) (\w+ing)",
            r"I'm (?:interested in|passionate about|into) ([^.,]+)",
            r"(?:favorite|preferred|my) (?:hobby|activity|pastime) is ([^.,]+)",
            r"I (?:spend|use) (?:my |a lot of |most )?(?:free )?time ([^.,]+)",
            r"I'm a fan of ([^.,]+)"
        ]
        
        for pattern in interest_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                interest = match.group(1).strip().lower()
                if interest and interest not in profile.interests:
                    profile.interests.append(interest)
                    
                    # If the profile has frequent_topics attribute, update it as well
                    if hasattr(profile, 'increment_topic'):
                        profile.increment_topic(interest, 2)  # Give higher weight to explicit interests
        
        # Extract personality traits with enhanced patterns
        trait_patterns = [
            r"I am (?:quite |very |extremely |really |)(\w+)",
            r"I'm (?:quite |very |extremely |really |)(\w+)",
            r"I consider myself (?:quite |very |extremely |really |)(\w+)",
            r"I tend to be (?:quite |very |extremely |really |)(\w+)",
            r"People say I(?:'m| am) (?:quite |very |extremely |really |)(\w+)"
        ]
        
        # Expanded list of personality traits
        personality_traits = [
            "shy", "outgoing", "confident", "anxious", "creative", "logical", 
            "hardworking", "laid-back", "organized", "spontaneous", "sensitive",
            "resilient", "introverted", "extroverted", "curious", "cautious",
            "analytical", "artistic", "practical", "ambitious", "relaxed", 
            "energetic", "emotional", "rational", "optimistic", "pessimistic",
            "determined", "patient", "impatient", "adaptable", "traditional",
            "unconventional", "reserved", "expressive", "collaborative", 
            "independent", "competitive", "supportive", "meticulous", "flexible"
        ]
        
        for pattern in trait_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                trait = match.group(1).strip().lower()
                if trait in personality_traits and trait not in profile.personality_traits:
                    profile.personality_traits.append(trait)
        
        # Extract languages
        language_patterns = [
            r"I speak (?:fluent |conversational |a little |)([a-zA-Z]+)",
            r"I(?:'m| am) fluent in ([a-zA-Z]+)",
            r"My (?:native|first|second) language is ([a-zA-Z]+)",
            r"I know how to speak ([a-zA-Z]+)"
        ]
        
        # Only update if the profile has the 'languages' attribute (for compatibility)
        if hasattr(profile, 'languages'):
            for pattern in language_patterns:
                matches = re.finditer(pattern, message_content, re.I)
                for match in matches:
                    language = match.group(1).strip().title()
                    if language and language not in profile.languages:
                        profile.languages.append(language)
        
        # Extract facts with enhanced patterns
        fact_patterns = [
            r"I (?:work as|am) an? ([^.,]+)",
            r"I live in ([^.,]+)",
            r"I'm from ([^.,]+)",
            r"I've been ([^.,]+)",
            r"I graduated from ([^.,]+)",
            r"I studied ([^.,]+)",
            r"I(?:'ve| have) (?:recently |just |)([^.,]+)",
            r"My (?:job|career|profession) (?:is|involves) ([^.,]+)"
        ]
        
        for pattern in fact_patterns:
            matches = re.finditer(pattern, message_content, re.I)
            for match in matches:
                fact = match.group(0).strip()
                if fact and fact not in profile.notable_facts:
                    profile.notable_facts.append(fact)
        
        # Extract name references with enhanced patterns
        name_patterns = [
            r"my name(?:'s| is) ([^.,]+)",
            r"call me ([^.,]+)",
            r"I go by ([^.,]+)",
            r"I prefer (?:to be called|the name) ([^.,]+)",
            r"known as ([^.,]+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_content, re.I)
            if match:
                name_value = match.group(1).strip()
                # Check for nickname indicators
                if "nickname" in message_content.lower() or "call me" in message_content.lower():
                    profile.nickname = name_value
                else:
                    profile.name = name_value
        
        # Extract communication style preferences if profile supports it
        if hasattr(profile, 'update_communication_style'):
            style_patterns = {
                "formal": r"I prefer (?:formal|professional|proper) (?:communication|conversation|language)",
                "casual": r"I(?:'m| am) (?:pretty |quite |very |)(?:casual|informal|relaxed)",
                "detailed": r"I like (?:detailed|thorough|comprehensive) (?:explanations|information)",
                "concise": r"I prefer (?:brief|concise|short|direct) (?:explanations|information|messages)",
                "technical": r"I(?:'m| am) (?:pretty |quite |very |)(?:technical|tech-savvy|a techie)",
                "simple": r"I prefer (?:simple|straightforward|easy) (?:explanations|language)"
            }
            
            for style, pattern in style_patterns.items():
                if re.search(pattern, message_content, re.I):
                    profile.update_communication_style(style)
        
        # Extract time zone if profile supports it and message contains it
        if hasattr(profile, 'time_zone'):
            timezone_patterns = [
                r"I(?:'m| am) in (?:the |)([A-Za-z]+(?:\s[A-Za-z]+)?) time ?zone",
                r"My time ?zone is ([A-Za-z]+(?:\s[A-Za-z]+)?)",
                r"It's (\d{1,2}(?::\d{2})? [ap]m) (?:here|for me)"
            ]
            
            for pattern in timezone_patterns:
                match = re.search(pattern, message_content, re.I)
                if match:
                    profile.time_zone = match.group(1)
                    break
        
        # Update interaction count if supported
        if hasattr(profile, 'record_interaction'):
            profile.record_interaction()
        
        profile.updated_at = datetime.now(timezone.utc).isoformat()
        return profile
    
    def extract_cross_references(self, user_id, message_content):
        """Extract references to other users from message content"""
        referenced_users = []
        
        # Check for mentions of other user names
        for other_id, other_profile in self.user_profiles.items():
            # Skip self-references
            if other_id == user_id:
                continue
                
            # Check for references to this user's name variants
            name_variants = []
            if other_profile.preferred_name:
                name_variants.append(other_profile.preferred_name)
            if other_profile.nickname:
                name_variants.append(other_profile.nickname)
            if other_profile.name:
                name_variants.append(other_profile.name)
                
            # Remove duplicates and empty names
            name_variants = [name for name in name_variants if name]
            name_variants = list(set(name_variants))
            
            # Check for each name variant
            for name in name_variants:
                # Use word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(name) + r'\b'
                if re.search(pattern, message_content, re.I):
                    referenced_users.append({
                        "user_id": other_id,
                        "name_used": name,
                        "context": message_content[:100]  # Store a preview of the context
                    })
                    
                    # Update cross-reference tracking
                    self.cross_references[user_id].add(other_id)
                    
                    # If profile supports it, record the mention
                    profile = self.get_or_create_profile(user_id)
                    if hasattr(profile, 'add_mentioned_user'):
                        profile.add_mentioned_user(other_id, message_content)
                    
                    # Only count each user once per message
                    break
        
        return referenced_users
    
    def generate_summary(self, user_id, transformer_helper=None):
        """Generate a summary of the conversation with improved topic extraction"""
        if user_id not in self.conversations or len(self.conversations[user_id]) < 3:
            return "Not enough conversation history for a summary."
        
        # Get last few messages
        recent_msgs = self.conversations[user_id][-8:]  # Increased from 5 to 8 for better coverage
        
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
                    
                result = transformer_helper.get_summarizer()(conversation_text, max_length=75, min_length=20, do_sample=False)
                if result and len(result) > 0:
                    summary = result[0]['summary_text']
            except Exception as e:
                print(f"Error generating summary: {e}")
        
        # Fallback to enhanced topic extraction approach
        if not summary:
            # Extract topics from recent messages
            all_topics = []
            for msg in recent_msgs:
                # Use pre-extracted topics if available
                if "topics" in msg and msg["topics"]:
                    all_topics.extend(msg["topics"])
                else:
                    # Extract topics on-the-fly if not already present
                    topics = self._extract_topics(msg["content"])
                    all_topics.extend(topics)
            
            # Count topic frequencies
            topic_counter = Counter(all_topics)
            
            # Get the most common topics
            common_topics = [topic for topic, count in topic_counter.most_common(5) if count > 1]
            
            if common_topics:
                summary = f"Recent conversation about: {', '.join(common_topics)}."
                
                # Add conversation dynamic if detectable
                bot_msgs = [msg for msg in recent_msgs if msg.get("from_bot")]
                user_msgs = [msg for msg in recent_msgs if not msg.get("from_bot")]
                
                if len(bot_msgs) > 2 and len(user_msgs) > 2:
                    avg_bot_len = sum(len(msg["content"]) for msg in bot_msgs) / len(bot_msgs)
                    avg_user_len = sum(len(msg["content"]) for msg in user_msgs) / len(user_msgs)
                    
                    if avg_bot_len > avg_user_len * 2:
                        summary += " A2 provided detailed explanations."
                    elif avg_user_len > avg_bot_len * 2:
                        summary += " User shared detailed information."
                    elif avg_bot_len > 150 and avg_user_len > 150:
                        summary += " Extended back-and-forth dialogue."
                    elif avg_bot_len < 50 and avg_user_len < 50:
                        summary += " Brief exchanges."
            else:
                summary = "Conversation with varied topics and no clear theme."
        
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
        
    def get_user_by_name(self, name_to_find, current_user_id=None):
        """Find a user by their name, nickname, or preferred name
        
        Args:
            name_to_find (str): The name to search for
            current_user_id (int, optional): ID of the current user to exclude from search
            
        Returns:
            tuple: (user_id, name_type) or (None, None) if not found
        """
        name_to_find = name_to_find.lower().strip()
        
        # Skip very short names or common words
        if len(name_to_find) < 3 or name_to_find.lower() in {"you", "me", "her", "him", "they", "them", "someone", "person"}:
            return None, None
        
        # First check for exact matches in preferred names, nicknames, then regular names
        for user_id, profile in self.user_profiles.items():
            # Skip the current user if specified
            if current_user_id and user_id == current_user_id:
                continue
                
            if profile.preferred_name and profile.preferred_name.lower() == name_to_find:
                return user_id, "preferred_name"
            if profile.nickname and profile.nickname.lower() == name_to_find:
                return user_id, "nickname"
            if profile.name and profile.name.lower() == name_to_find:
                return user_id, "name"
        
        # If no exact match, try partial matches but only for distinctive names
        # (avoid matching common parts of names)
        if len(name_to_find) > 3:
            for user_id, profile in self.user_profiles.items():
                if current_user_id and user_id == current_user_id:
                    continue
                    
                if profile.preferred_name and name_to_find in profile.preferred_name.lower():
                    return user_id, "preferred_name"
                if profile.nickname and name_to_find in profile.nickname.lower():
                    return user_id, "nickname"
                if profile.name and name_to_find in profile.name.lower():
                    return user_id, "name"
        
        return None, None
    
    def get_display_name(self, user_id):
        """Get the best display name for a user
        
        Args:
            user_id (int): The user's ID
            
        Returns:
            str: The user's display name or None if not found
        """
        if user_id not in self.user_profiles:
            return None
            
        profile = self.user_profiles[user_id]
        if profile.preferred_name:
            return profile.preferred_name
        if profile.nickname:
            return profile.nickname
        if profile.name:
            return profile.name
        return None
    
    def get_other_user_conversation(self, user_id, query_terms=None, max_messages=5):
        """
        Retrieve conversation history from another user filtered by query terms
        
        Args:
            user_id (int): The user ID to get conversations from
            query_terms (list, optional): List of terms to filter by
            max_messages (int, optional): Maximum number of messages to return
            
        Returns:
            list: List of relevant conversation messages
        """
        if user_id not in self.conversations:
            return []
            
        # Get the user's conversation history
        history = self.conversations[user_id]
        
        # If no query terms provided, just return the most recent messages
        if not query_terms:
            return history[-max_messages:]
        
        # Filter messages by query terms
        relevant_messages = []
        for msg in history:
            content = msg.get("content", "").lower()
            if any(term.lower() in content for term in query_terms):
                relevant_messages.append(msg)
        
        # Return the most recent relevant messages, up to max_messages
        return relevant_messages[-max_messages:]

    def get_all_users_mentioning(self, name_or_reference, current_user_id=None):
        """
        Find all users who have mentioned a specific term or reference
        
        Args:
            name_or_reference (str): The term to search for
            current_user_id (int, optional): ID of current user to exclude
            
        Returns:
            dict: Dictionary mapping user_ids to relevant messages
        """
        results = {}
        name_lower = name_or_reference.lower()
        
        # Check conversations from all users
        for uid, messages in self.conversations.items():
            # Skip current user if specified
            if current_user_id and uid == current_user_id:
                continue
                
            # Find messages mentioning the reference
            for msg in messages:
                if name_lower in msg.get("content", "").lower():
                    if uid not in results:
                        results[uid] = []
                    results[uid].append(msg)
        
        return results
        
    def find_common_topics(self, user_id1, user_id2, min_count=2):
        """Find topics that two users have discussed
        
        Args:
            user_id1 (int): First user's ID
            user_id2 (int): Second user's ID
            min_count (int): Minimum number of mentions needed
            
        Returns:
            list: List of common topics sorted by relevance
        """
        if user_id1 not in self.conversation_topics or user_id2 not in self.conversation_topics:
            return []
            
        topics1 = set(self.conversation_topics[user_id1].keys())
        topics2 = set(self.conversation_topics[user_id2].keys())
        
        common_topics = topics1.intersection(topics2)
        if not common_topics:
            return []
            
        # Count topic occurrences in both users' conversations
        topic_counts = {}
        for topic in common_topics:
            # Count occurrences in topic index
            count1 = len([1 for u, i in self.topic_index[topic] if u == user_id1])
            count2 = len([1 for u, i in self.topic_index[topic] if u == user_id2])
            
            # Only include topics with sufficient mentions
            if count1 + count2 >= min_count:
                topic_counts[topic] = count1 + count2
        
        # Sort by count and return
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics]
        
    def get_cross_referenced_users(self, user_id, limit=5):
        """Get users that the current user has referenced most frequently
        
        Args:
            user_id (int): The user's ID
            limit (int): Maximum number of users to return
            
        Returns:
            list: List of user IDs sorted by reference frequency
        """
        if user_id not in self.cross_references:
            return []
            
        # Get all referenced users
        referenced_users = self.cross_references[user_id]
        
        # For enhanced profiles, use the detailed mention tracking
        if user_id in self.user_profiles and hasattr(self.user_profiles[user_id], 'mentioned_users'):
            profile = self.user_profiles[user_id]
            # Count mentions for each user
            mention_counts = {}
            for ref_id, mentions in profile.mentioned_users.items():
                mention_counts[ref_id] = len(mentions)
                
            # Sort by mention count
            sorted_users = sorted(mention_counts.items(), key=lambda x: x[1], reverse=True)
            return [uid for uid, count in sorted_users[:limit]]
        
        # Fallback: just return the set of referenced users
        return list(referenced_users)[:limit]
        
    def get_recommended_topics(self, user_id, limit=5):
        """Get recommended conversation topics based on user history and preferences
        
        Args:
            user_id (int): The user's ID
            limit (int): Maximum number of topics to return
            
        Returns:
            list: List of recommended topics
        """
        topics = []
        
        # 1. Get topics from user profile interests
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            if profile.interests:
                topics.extend(profile.interests[:3])
            
            # If enhanced profile is available, use frequent topics
            if hasattr(profile, 'frequent_topics') and profile.frequent_topics:
                # Get the most discussed topics
                frequent = sorted(profile.frequent_topics.items(), key=lambda x: x[1], reverse=True)
                topics.extend([topic for topic, count in frequent[:3]])
        
        # 2. Get topics from conversation history
        if user_id in self.conversation_topics:
            # Sort by recency
            recent_topics = sorted(
                self.conversation_topics[user_id].items(),
                key=lambda x: x[1],  # x[1] is the timestamp
                reverse=True
            )
            topics.extend([topic for topic, timestamp in recent_topics[:3]])
        
        # 3. Get some popular topics across all users (if available)
        popular_topics = Counter()
        for user_topics in self.conversation_topics.values():
            popular_topics.update(user_topics.keys())
            
        if popular_topics:
            # Add some popular topics that the user hasn't discussed yet
            user_topics = set(self.conversation_topics.get(user_id, {}).keys())
            new_topics = [t for t, c in popular_topics.most_common(10) if t not in user_topics]
            topics.extend(new_topics[:2])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_topics = []
        for topic in topics:
            if topic not in seen:
                seen.add(topic)
                unique_topics.append(topic)
        
        # Return limited number of topics
        return unique_topics[:limit]
        
    def get_topic_similarity(self, user_id1, user_id2):
        """Calculate similarity between two users based on conversation topics
        
        Args:
            user_id1 (int): First user's ID
            user_id2 (int): Second user's ID
            
        Returns:
            float: Similarity score (0-1)
        """
        if user_id1 not in self.conversation_topics or user_id2 not in self.conversation_topics:
            return 0.0
            
        topics1 = set(self.conversation_topics[user_id1].keys())
        topics2 = set(self.conversation_topics[user_id2].keys())
        
        # Handle edge case of empty sets
        if not topics1 or not topics2:
            return 0.0
            
        # Calculate Jaccard similarity
        intersection = len(topics1.intersection(topics2))
        union = len(topics1.union(topics2))
        
        return intersection / union if union > 0 else 0.0
