"""
Enhanced response generator for the A2 Discord bot.
"""
import random
import asyncio
import re
import json
from datetime import datetime, timezone, timedelta
from collections import deque

class ResponseGenerator:
    """Handles conversation management and response generation"""
    
    def __init__(self, openai_client, emotion_manager, conversation_manager):
        self.client = openai_client
        self.emotion_manager = emotion_manager
        self.conversation_manager = conversation_manager
        self.recent_responses = {}
        self.MAX_RECENT_RESPONSES = 10
        self.user_references = {}  # Store verified user references
        self.topic_memory = {}  # Remember topics for later reference
    
    def identify_user_references(self, content, current_user_id):
        """Identify references to other users in the message content with enhanced detection
        
        Args:
            content (str): The message content
            current_user_id (int): The ID of the current user
            
        Returns:
            dict: Dictionary mapping referenced names to user info
        """
        referenced_users = {}
        
        # Check for explicit mentions with @username format
        mention_pattern = r"@([a-zA-Z0-9_]+)"
        mentions = re.findall(mention_pattern, content)
        
        # If there's a question about a specific user being "my dude" etc.
        is_question_pattern = r"is\s+(?:my\s+)?(?:dude|friend|buddy|pal)\s+@?([a-zA-Z0-9_]+)"
        is_question_match = re.search(is_question_pattern, content, re.I)
        
        if is_question_match:
            # This is a question about whether someone is "my dude"
            # We should NOT confirm this without evidence
            mentioned_name = is_question_match.group(1)
            # Return the information but mark it as a question
            for member_id, profile in self.conversation_manager.user_profiles.items():
                if profile.name and profile.name.lower() == mentioned_name.lower():
                    return {
                        "question_about_relationship": {
                            "name": mentioned_name,
                            "user_id": member_id,
                            "display_name": self.conversation_manager.get_display_name(member_id)
                        }
                    }
            return {}  # No matching user found
        
        # Use an expanded set of patterns to identify references to other users
        reference_patterns = [
            # Basic questions about others
            r"who (?:is|are) ([a-zA-Z0-9_\s]+)\??",
            r"tell me about ([a-zA-Z0-9_\s]+)",
            r"([a-zA-Z0-9_\s]+)'s profile",
            r"what do you know about ([a-zA-Z0-9_\s]+)",
            r"do you know ([a-zA-Z0-9_\s]+)",
            r"have you met ([a-zA-Z0-9_\s]+)",
            r"remember ([a-zA-Z0-9_\s]+)\?",
            r"what's ([a-zA-Z0-9_\s]+) like",
            r"how's ([a-zA-Z0-9_\s]+)",
            r"what ([a-zA-Z0-9_\s]+) user",
            r"who's ([a-zA-Z0-9_\s]+)",
            
            # Relational references
            r"my (?:friend|buddy|pal|dude) ([a-zA-Z0-9_\s]+)",
            r"my ([a-zA-Z0-9_\s]+)",  # General "my X" pattern
            r"([a-zA-Z0-9_\s]+) is (?:my|a) (?:friend|buddy|pal|dude)",
            
            # Questions about state/status
            r"how is ([a-zA-Z0-9_\s]+)(?:'s| feeling| doing)?",
            r"what did ([a-zA-Z0-9_\s]+) say",
            r"what is ([a-zA-Z0-9_\s]+) (?:talking|saying) about",
            r"did ([a-zA-Z0-9_\s]+) mention",
            r"how was ([a-zA-Z0-9_\s]+)'s day",
            r"is ([a-zA-Z0-9_\s]+) (?:happy|sad|angry|upset|ok|okay)",
            r"can you tell me about ([a-zA-Z0-9_\s]+)'s (?:day|feeling|mood|status)",
            r"status of ([a-zA-Z0-9_\s]+)",
            r"update on ([a-zA-Z0-9_\s]+)",
            
            # NEW: Explicit cross-referencing patterns
            r"what does ([a-zA-Z0-9_\s]+) think about ([a-zA-Z0-9_\s]+)",
            r"has ([a-zA-Z0-9_\s]+) talked about ([a-zA-Z0-9_\s]+)",
            r"when did ([a-zA-Z0-9_\s]+) last mention ([a-zA-Z0-9_\s]+)",
            r"what topics has ([a-zA-Z0-9_\s]+) discussed",
            r"anything in common with ([a-zA-Z0-9_\s]+)",
            r"compare me (?:to|and|with) ([a-zA-Z0-9_\s]+)",
            r"similarities between me and ([a-zA-Z0-9_\s]+)",
            r"do I talk like ([a-zA-Z0-9_\s]+)",
            r"does ([a-zA-Z0-9_\s]+) talk like me",
            r"introduce me to ([a-zA-Z0-9_\s]+)",
            r"find users who talk about ([a-zA-Z0-9_\s]+)"
        ]
        
        # Look for pronoun references if we've recently discussed someone
        pronoun_patterns = [
            r"(?:he|him|his|she|her|they|them|their)(?:\s+is|\s+are|\s+has|\s+have|\s+do|\s+does|\s+did|\s+'s|\s+'re)?"
        ]
        
        # Process direct reference patterns
        for pattern in reference_patterns:
            matches = re.finditer(pattern, content, re.I)
            for match in matches:
                referenced_name = match.group(1).strip()
                
                # Skip common words
                if referenced_name.lower() in {"you", "me", "i", "we", "us", "them", 
                                           "there", "here", "this", "that", "who", 
                                           "what", "where", "when", "why", "how"}:
                    continue
                
                # Check if this name refers to a known user
                user_id, name_type = self.conversation_manager.get_user_by_name(referenced_name, current_user_id)
                if user_id:
                    display_name = self.conversation_manager.get_display_name(user_id)
                    referenced_users[referenced_name] = {
                        "user_id": user_id,
                        "display_name": display_name,
                        "name_type": name_type,
                        # Add relationship info between users if available
                        "relationship": self.get_relationship_info(current_user_id, user_id)
                    }
        
        # Check for topic comparison patterns (new)
        topic_patterns = [
            r"has anyone (?:talked|spoken|discussed|mentioned) about ([a-zA-Z0-9_\s]+)",
            r"who (?:else |)(?:talks|speaks|discusse?s?|mentions) ([a-zA-Z0-9_\s]+)",
            r"find conversations about ([a-zA-Z0-9_\s]+)"
        ]
        
        for pattern in topic_patterns:
            matches = re.finditer(pattern, content, re.I)
            for match in matches:
                topic = match.group(1).strip().lower()
                if len(topic) >= 3 and topic not in {"about", "with", "these", "those", "them"}:
                    referenced_users["topic_search"] = {
                        "topic": topic,
                        "context": content
                    }
        
        # If no direct references found, check for pronouns and use most recently mentioned user
        if not referenced_users:
            for pattern in pronoun_patterns:
                if re.search(pattern, content, re.I):
                    # Find the most recently referenced user from conversation history
                    if current_user_id in self.conversation_manager.conversations:
                        # Go through the last few messages looking for a direct reference
                        history = self.conversation_manager.conversations[current_user_id][-8:]  # Increased from 5
                        for msg in reversed(history):
                            if msg.get("from_bot", False):
                                continue
                            
                            msg_content = msg.get("content", "")
                            # Check this message for references
                            for ref_pattern in reference_patterns:
                                matches = re.finditer(ref_pattern, msg_content, re.I)
                                for match in matches:
                                    ref_name = match.group(1).strip()
                                    user_id, name_type = self.conversation_manager.get_user_by_name(ref_name, current_user_id)
                                    if user_id:
                                        display_name = self.conversation_manager.get_display_name(user_id)
                                        # Found a recent reference, use it for the pronoun
                                        return {
                                            "pronoun_reference": {
                                                "user_id": user_id,
                                                "display_name": display_name,
                                                "name_type": name_type,
                                                "original_reference": ref_name,
                                                "relationship": self.get_relationship_info(current_user_id, user_id)
                                            }
                                        }
        
        return referenced_users
    
    def get_relationship_info(self, user_id1, user_id2):
        """Get information about the relationship between two users
        
        Args:
            user_id1 (int): First user's ID
            user_id2 (int): Second user's ID
            
        Returns:
            dict: Information about the relationship
        """
        relationship = {}
        
        # Check if both users exist
        if user_id1 not in self.emotion_manager.user_emotions or user_id2 not in self.emotion_manager.user_emotions:
            return relationship
        
        # Find common topics
        common_topics = []
        if hasattr(self.conversation_manager, 'find_common_topics'):
            common_topics = self.conversation_manager.find_common_topics(user_id1, user_id2)
            if common_topics:
                relationship["common_topics"] = common_topics[:3]
                
        # Check if they've mentioned each other
        if hasattr(self.conversation_manager, 'cross_references'):
            if user_id2 in self.conversation_manager.cross_references.get(user_id1, set()):
                relationship["user1_has_mentioned_user2"] = True
            if user_id1 in self.conversation_manager.cross_references.get(user_id2, set()):
                relationship["user2_has_mentioned_user1"] = True
        
        # Calculate similarity if available
        if hasattr(self.conversation_manager, 'get_topic_similarity'):
            similarity = self.conversation_manager.get_topic_similarity(user_id1, user_id2)
            if similarity > 0:
                relationship["topic_similarity"] = similarity
        
        return relationship
    
    def extract_conversation_context(self, user_id, current_message):
        """Extract relevant conversation context to provide better responses
        
        Args:
            user_id (int): The user's ID
            current_message (str): Current message content
            
        Returns:
            str: Relevant conversation context
        """
        if user_id not in self.conversation_manager.conversations:
            return ""
            
        # Get last several messages
        history = self.conversation_manager.conversations[user_id][-15:]  # Increased from 10
        
        # Look for relevant topics in current message
        topics = set()
        
        # Try to use the conversation manager's topic extraction if available
        if hasattr(self.conversation_manager, '_extract_topics'):
            extracted_topics = self.conversation_manager._extract_topics(current_message)
            topics.update(extracted_topics)
        else:
            # Fallback: simple word extraction
            important_words = re.findall(r'\b[a-zA-Z]{4,}\b', current_message.lower())
            topics.update(important_words)
        
        # Remove common words
        common_words = {"what", "when", "where", "which", "have", "about", "that", "this", "know", "help"}
        topics = topics - common_words
        
        # Remember these topics for future reference
        self.topic_memory[user_id] = list(topics)
        
        # Find relevant messages
        relevant_messages = []
        
        # Try to use the enhanced search method if available
        if hasattr(self.conversation_manager, 'search_conversations') and topics:
            # Search for each topic
            for topic in topics:
                results = self.conversation_manager.search_conversations(topic, user_id=user_id, max_results=3)
                for result in results:
                    message = result["message"]
                    speaker = "A2" if message.get("from_bot", False) else "User"
                    msg_text = f"{speaker}: {message.get('content', '')}"
                    if msg_text not in relevant_messages:  # Avoid duplicates
                        relevant_messages.append(msg_text)
        else:
            # Fallback: linear search through history
            for msg in history:
                msg_content = msg.get("content", "").lower()
                if any(topic in msg_content for topic in topics):
                    speaker = "A2" if msg.get("from_bot", False) else "User"
                    relevant_messages.append(f"{speaker}: {msg.get('content', '')}")
        
        if not relevant_messages:
            return ""
            
        # Limit to a reasonable amount
        relevant_messages = relevant_messages[-7:]  # Increased from 5
        
        return "\nRelevant conversation history:\n" + "\n".join(relevant_messages)
        
    async def generate_a2_response(self, content, trust, user_id, storage_manager):
        """Generate A2's response to a user message with enhanced context awareness"""
        # Identify any referenced users
        referenced_users = self.identify_user_references(content, user_id)
        
        # Get current user's profile and name
        current_user_profile = self.conversation_manager.get_or_create_profile(user_id)
        current_user_name = self.conversation_manager.get_display_name(user_id) or "User"
        
        # Build prompt for OpenAI
        system_prompt = "You are A2, a combat android from NieR: Automata. You speak in short, clipped sentences, often sarcastic but occasionally showing glimpses of deeper emotion. Keep responses brief and true to character."
        
        # Add user context with enhanced profile information
        user_context = f"You are speaking to {current_user_name}. "
        
        # Add trust level context
        if trust > 7:
            user_context += f"You trust {current_user_name} a lot. "
        elif trust > 4:
            user_context += f"You somewhat trust {current_user_name}. "
        else:
            user_context += f"You are cautious around {current_user_name}. "
        
        # Add enhanced profile information if available
        if hasattr(current_user_profile, 'communication_style') and current_user_profile.communication_style:
            user_context += f"\n\n{current_user_name}'s communication style: {', '.join(current_user_profile.communication_style)}. "
            
        if hasattr(current_user_profile, 'languages') and current_user_profile.languages:
            user_context += f"{current_user_name} speaks: {', '.join(current_user_profile.languages)}. "
            
        if hasattr(current_user_profile, 'frequent_topics') and current_user_profile.frequent_topics:
            # Get top 3 topics
            top_topics = sorted(current_user_profile.frequent_topics.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_topics:
                topics_str = ', '.join(topic for topic, _ in top_topics)
                user_context += f"{current_user_name} frequently discusses: {topics_str}. "
                
        # Add information about recent emotional trends if available
        if hasattr(current_user_profile, 'analyze_sentiment_trend'):
            sentiment_info = current_user_profile.analyze_sentiment_trend()
            if sentiment_info and sentiment_info != "Not enough data to determine sentiment trend.":
                user_context += f"Recently, {current_user_name}'s conversations have shown {sentiment_info} "
        
        # Handle question about user relationship
        if "question_about_relationship" in referenced_users:
            ref_data = referenced_users["question_about_relationship"]
            ref_name = ref_data["name"]
            ref_display_name = ref_data["display_name"]
            
            user_context += f"\n\n{current_user_name} is asking if '{ref_name}' is their 'dude' or friend. "
            user_context += f"You DON'T know if this is true, since you don't track friendships between users. "
            user_context += f"You should respond neutrally without confirming or denying their friendship status."
            
        # Handle topic search
        elif "topic_search" in referenced_users:
            topic_data = referenced_users["topic_search"]
            topic = topic_data["topic"]
            
            # Use the enhanced conversation search if available
            topic_info = ""
            if hasattr(self.conversation_manager, 'get_conversations_by_topic'):
                # Search for conversations about this topic
                conversations = self.conversation_manager.get_conversations_by_topic(topic, max_results=3)
                if conversations:
                    # Format the information
                    for i, conv in enumerate(conversations):
                        other_user_id = conv["user_id"]
                        other_profile = self.conversation_manager.get_or_create_profile(other_user_id)
                        other_name = other_profile.get_display_name() if hasattr(other_profile, 'get_display_name') else "Unknown User"
                        
                        message = conv["message"]
                        preview = message.get("content", "")[:50] + ("..." if len(message.get("content", "")) > 50 else "")
                        
                        topic_info += f"\n- {other_name} mentioned '{topic}': \"{preview}\""
                
                if topic_info:
                    user_context += f"\n\n{current_user_name} is asking about who has discussed the topic '{topic}'. Here are some mentions:{topic_info}"
                else:
                    user_context += f"\n\n{current_user_name} is asking about who has discussed the topic '{topic}', but no one has mentioned this specific topic."
            
        # Handle direct references or pronoun references
        elif referenced_users:
            # Handle pronoun references
            if "pronoun_reference" in referenced_users:
                ref_data = referenced_users["pronoun_reference"]
                ref_user_id = ref_data["user_id"]
                ref_display_name = ref_data["display_name"]
                original_reference = ref_data.get("original_reference", "this person")
                
                if ref_user_id in self.conversation_manager.user_profiles:
                    ref_profile = self.conversation_manager.user_profiles[ref_user_id]
                    
                    # Get a more detailed summary if the enhanced profile is available
                    profile_summary = ""
                    if hasattr(ref_profile, 'get_detailed_summary'):
                        profile_summary = f"Detailed information:\n{ref_profile.get_summary()}"
                    else:
                        profile_summary = ref_profile.get_summary()
                    
                    # Add information about the referenced user
                    user_context += f"\n\n{current_user_name} is referring to {ref_display_name} with a pronoun. "
                    user_context += f"You previously discussed {ref_display_name} who was referenced as '{original_reference}'. "
                    user_context += f"Here's what you know about {ref_display_name}: {profile_summary}"
                    
                    # Add relationship context if available
                    if ref_user_id in self.emotion_manager.user_emotions:
                        ref_trust = self.emotion_manager.user_emotions[ref_user_id].get('trust', 0)
                        if ref_trust > 7:
                            user_context += f" You trust {ref_display_name} a lot."
                        elif ref_trust > 4:
                            user_context += f" You somewhat trust {ref_display_name}."
                        else:
                            user_context += f" You are cautious around {ref_display_name}."
                    
                    # Add relationship info between users if available
                    relationship = ref_data.get("relationship", {})
                    if relationship:
                        if "common_topics" in relationship:
                            user_context += f" {current_user_name} and {ref_display_name} have discussed similar topics: {', '.join(relationship['common_topics'])}."
                        
                        if relationship.get("user1_has_mentioned_user2"):
                            user_context += f" {current_user_name} has mentioned {ref_display_name} before."
                        
                        if relationship.get("user2_has_mentioned_user1"):
                            user_context += f" {ref_display_name} has mentioned {current_user_name} before."
            
            # Handle direct references
            for ref_name, ref_data in referenced_users.items():
                if ref_name in {"pronoun_reference", "topic_search", "question_about_relationship"}:
                    continue
                    
                ref_user_id = ref_data["user_id"]
                ref_display_name = ref_data["display_name"]
                
                if ref_user_id in self.conversation_manager.user_profiles:
                    ref_profile = self.conversation_manager.user_profiles[ref_user_id]
                    
                    # Get more detailed profile information if available
                    profile_summary = ""
                    if hasattr(ref_profile, 'get_detailed_summary'):
                        profile_summary = f"Detailed information:\n{ref_profile.get_summary()}"
                    else:
                        profile_summary = ref_profile.get_summary()
                    
                    # Add information about the referenced user
                    user_context += f"\n\n{current_user_name} is asking about '{ref_name}', which refers to {ref_display_name}. "
                    user_context += f"Here's what you know about {ref_display_name}: {profile_summary}"
                    
                    # Add relationship context if available
                    if ref_user_id in self.emotion_manager.user_emotions:
                        ref_trust = self.emotion_manager.user_emotions[ref_user_id].get('trust', 0)
                        if ref_trust > 7:
                            user_context += f" You trust {ref_display_name} a lot."
                        elif ref_trust > 4:
                            user_context += f" You somewhat trust {ref_display_name}."
                        else:
                            user_context += f" You are cautious around {ref_display_name}."
                    
                    # Add relationship between users if available
                    relationship = ref_data.get("relationship", {})
                    if relationship:
                        if "common_topics" in relationship:
                            user_context += f" {current_user_name} and {ref_display_name} have discussed similar topics: {', '.join(relationship['common_topics'])}."
                        
                        if relationship.get("user1_has_mentioned_user2"):
                            user_context += f" {current_user_name} has mentioned {ref_display_name} before."
                        
                        if relationship.get("user2_has_mentioned_user1"):
                            user_context += f" {ref_display_name} has mentioned {current_user_name} before."
                    
                    # Add recent conversations from the referenced user
                    query_terms = []
                    
                    # Extract potential query terms related to state/feeling/day
                    state_patterns = [
                        r"how (?:is|was) .*? (?:feeling|doing|day)",
                        r"(?:status|update|mood) of",
                        r"is .*? (?:happy|sad|angry|upset|ok|okay)"
                    ]
                    
                    for pattern in state_patterns:
                        if re.search(pattern, content, re.I):
                            query_terms.extend(["day", "feeling", "mood", "status", "bad", "good", "happy", "sad"])
                            break
                    
                    # Get specific query terms from current message
                    content_words = re.findall(r'\b[a-zA-Z]{4,}\b', content.lower())
                    query_terms.extend([w for w in content_words if w not in {"what", "when", "where", "which", "about", "tell", "know"}])
                    
                    # Use conversation search if available
                    recent_convs = []
                    if hasattr(self.conversation_manager, 'search_conversations') and query_terms:
                        # Search for each topic in the other user's conversations
                        for term in query_terms:
                            results = self.conversation_manager.search_conversations(term, user_id=ref_user_id, max_results=2)
                            for result in results:
                                if result["message"] not in recent_convs:  # Avoid duplicates
                                    recent_convs.append(result["message"])
                    else:
                        # Fallback to simpler method
                        recent_convs = self.conversation_manager.get_other_user_conversation(ref_user_id, query_terms)
                    
                    if recent_convs:
                        conv_text = "\n".join([
                            f"{ref_display_name}: {msg['content']}" 
                            for msg in recent_convs if not msg.get('from_bot', False)
                        ])
                        
                        if conv_text:
                            user_context += f"\n\nRecent relevant messages from {ref_display_name}:\n{conv_text}"
        
        # Get conversation history
        conversation_history = self.conversation_manager.get_conversation_history(user_id)
        
        # Extract relevant context based on current message
        relevant_context = self.extract_conversation_context(user_id, content)
        
        # Determine A2's current personality state
        personality_state = self.emotion_manager.select_personality_state(user_id, content)
        
        # Craft the final messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": user_context},
            {"role": "system", "content": f"Previous conversation:\n{conversation_history}"},
            {"role": "system", "content": relevant_context},
            {"role": "user", "content": content}
        ]
        
        # Get parameters based on personality state
        response_params = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.85,
            "top_p": 1,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        # Actually generate the response
        try:
            response = self.client.chat.completions.create(**response_params)
            a2_response = response.choices[0].message.content.strip()
            
            # Save the message in conversation history
            self.conversation_manager.add_message(user_id, content, is_from_bot=False)
            self.conversation_manager.add_message(user_id, a2_response, is_from_bot=True)
            
            # Extract any cross-references from the user's message
            if hasattr(self.conversation_manager, 'extract_cross_references'):
                referenced_users = self.conversation_manager.extract_cross_references(user_id, content)
            
            # Update user profile based on message content
            profile = self.conversation_manager.extract_profile_info(user_id, content)
            
            # Update sentiment history if the profile supports it
            if hasattr(profile, 'add_sentiment_entry'):
                # Simple sentiment analysis based on keywords (replace with a real model in production)
                positive_words = {"like", "love", "good", "great", "awesome", "thanks", "appreciate", "helpful", "nice", "excellent"}
                negative_words = {"hate", "dislike", "bad", "terrible", "awful", "annoying", "useless", "stupid", "broken", "wrong"}
                
                content_words = set(content.lower().split())
                positive_matches = len(content_words.intersection(positive_words))
                negative_matches = len(content_words.intersection(negative_words))
                
                sentiment_value = 0.5  # Neutral default
                if positive_matches > negative_matches:
                    sentiment_value = 0.7 + (0.3 * (positive_matches / (positive_matches + negative_matches + 1)))
                elif negative_matches > 0:
                    sentiment_value = 0.3 - (0.3 * (negative_matches / (positive_matches + negative_matches + 1)))
                
                profile.add_sentiment_entry(content, sentiment_value)
            
            # Update communication style based on message
            if hasattr(profile, 'update_communication_style'):
                # Detect formality level
                if re.search(r'\b(hello|greetings|good [morning|afternoon|evening])\b', content.lower()):
                    profile.update_communication_style("formal")
                if re.search(r'\b(hey|hi|what\'s up|yo|sup)\b', content.lower()):
                    profile.update_communication_style("casual")
                
                # Detect verbosity
                if len(content.split()) > 50:
                    profile.update_communication_style("verbose")
                elif len(content.split()) < 10:
                    profile.update_communication_style("concise")
                
                # Detect technical language
                technical_words = {"code", "programming", "algorithm", "function", "variable", "data", "server", 
                                 "system", "technical", "compile", "execute", "implementation"}
                if len(set(content.lower().split()).intersection(technical_words)) > 2:
                    profile.update_communication_style("technical")
            
            await storage_manager.save_user_profile_data(user_id, profile)
            
            # Track the response
            if user_id not in self.recent_responses:
                self.recent_responses[user_id] = deque(maxlen=self.MAX_RECENT_RESPONSES)
            self.recent_responses[user_id].append(a2_response)
            
            # Update emotional stats based on the interaction
            await self.emotion_manager.update_emotional_stats(user_id, content, a2_response, storage_manager)
            
            return a2_response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "... System error. Connection unstable."
    
    async def handle_first_message_of_day(self, message, user_id):
        """Handle the first message from a user on a new day with improved continuity"""
        # Check if this is indeed the first message of the day
        if user_id in self.emotion_manager.user_emotions:
            emotions = self.emotion_manager.user_emotions[user_id]
            last_interaction = emotions.get("last_interaction")
            
            if last_interaction:
                try:
                    last_time = datetime.fromisoformat(last_interaction)
                    now = datetime.now(timezone.utc)
                    
                    # If last interaction was yesterday or earlier
                    if last_time.date() < now.date():
                        # Tailor greeting based on relationship
                        trust = emotions.get("trust", 0)
                        
                        # Get user name from profile if available
                        user_name = message.author.display_name
                        profile = self.conversation_manager.get_or_create_profile(user_id)
                        if hasattr(profile, 'get_display_name'):
                            display_name = profile.get_display_name()
                            if display_name:
                                user_name = display_name
                                
                        # Check for previous conversation topics to reference
                        topic_reference = ""
                        if hasattr(self, 'topic_memory') and user_id in self.topic_memory:
                            remembered_topics = self.topic_memory[user_id]
                            if remembered_topics:
                                topic = random.choice(remembered_topics)
                                if random.random() < 0.3:  # 30% chance to reference previous topic
                                    topic_reference = f" Still thinking about {topic}?"
                        
                        if trust > 7:
                            greeting = random.choice([
                                f"... Welcome back{', ' + user_name if random.random() < 0.7 else ''}.",
                                f"Good to see you again{', ' + user_name if random.random() < 0.5 else ''}.",
                                f"Hey {user_name}."
                            ])
                        elif trust > 4:
                            greeting = random.choice([
                                "... You're back.",
                                f"Hello, {user_name}.",
                                "Systems online."
                            ])
                        else:
                            greeting = random.choice([
                                "...",
                                "Systems active.",
                                "Sensors online."
                            ])
                            
                        await message.channel.send(f"A2: {greeting}{topic_reference}")
                except Exception as e:
                    print(f"Error in handle_first_message_of_day: {e}")

    async def check_inactive_users(self, bot, storage_manager):
        """Check for inactive users and potentially reach out"""
        self.logger.debug("Checking for inactive users")
        
        # Get current time
        now = datetime.now(timezone.utc)
        
        # Find users who haven't interacted recently but have DMs enabled
        for user_id, emotions in self.emotion_manager.user_emotions.items():
            # Skip if no last interaction
            if 'last_interaction' not in emotions:
                continue
                
            # Parse the last interaction time
            try:
                last_interaction = datetime.fromisoformat(emotions['last_interaction'])
            except (ValueError, TypeError):
                continue
                
            # Calculate days since last interaction
            days_since = (now - last_interaction).days
            
            # Only process users who haven't interacted for 10-30 days
            # and have DMs enabled and decent trust
            if (10 <= days_since <= 30 and 
                user_id in self.emotion_manager.dm_enabled_users and
                emotions.get('trust', 0) >= 4):
                
                # Low chance to reach out (5% daily)
                if random.random() < 0.05:
                    self.logger.info(f"Attempting to reach out to inactive user {user_id}")
                    
                    # Get the user
                    user = bot.get_user(user_id)
                    if not user:
                        continue
                        
                    # Create a message based on trust level
                    trust = emotions.get('trust', 0)
                    
                    if trust > 7:
                        message = random.choice([
                            "... Haven't seen you in a while. Still operational?",
                            "Checking in. Status update?",
                            "... Starting to think you got yourself into trouble."
                        ])
                    else:
                        message = random.choice([
                            "...",
                            "Systems still operational.",
                            "Running routine check."
                        ])
                    
                    # Try to send a DM
                    try:
                        dm = await user.create_dm()
                        await dm.send(f"A2: {message}")
                        
                        # Add this user to pending message list
                        self.emotion_manager.pending_messages.add(user_id)
                        
                        # Record this outreach
                        await self.emotion_manager.create_memory_event(
                            user_id, 
                            "inactive_outreach",
                            f"A2 reached out to {user.display_name} after {days_since} days of inactivity.",
                            {"attachment": 0.1},
                            storage_manager
                        )
                        
                        self.logger.info(f"Successfully reached out to inactive user {user_id}")
                    except Exception as e:
                        self.logger.error(f"Error sending DM to inactive user {user_id}: {e}")
        
        self.logger.debug("Inactive users check completed")

    async def trigger_random_events(self, bot, storage_manager):
        """Trigger random events for active users"""
        self.logger.debug("Checking for random event triggers")
        
        # Get event configuration from constants
        from config import EMOTION_CONFIG
        event_chance = EMOTION_CONFIG.get("RANDOM_EVENT_CHANCE", 0.08)
        event_cooldown = EMOTION_CONFIG.get("EVENT_COOLDOWN_HOURS", 12)
        
        # Get active users
        now = datetime.now(timezone.utc)
        active_users = []
        
        # Find users who:
        # 1. Have interacted recently
        # 2. Have DMs enabled
        # 3. Haven't had an event recently
        for user_id, emotions in self.emotion_manager.user_emotions.items():
            # Skip if no last interaction
            if 'last_interaction' not in emotions:
                continue
                
            # Parse the last interaction time
            try:
                last_interaction = datetime.fromisoformat(emotions['last_interaction'])
            except (ValueError, TypeError):
                continue
                
            # Skip if not active in the last 7 days
            days_since_activity = (now - last_interaction).days
            if days_since_activity > 7:
                continue
                
            # Skip if DMs not enabled
            if user_id not in self.emotion_manager.dm_enabled_users:
                continue
                
            # Check if there was a recent event
            had_recent_event = False
            if user_id in self.emotion_manager.user_events:
                for event in self.emotion_manager.user_events[user_id]:
                    try:
                        event_time = datetime.fromisoformat(event.get('timestamp', ''))
                        hours_since_event = (now - event_time).total_seconds() / 3600
                        if hours_since_event < event_cooldown:
                            had_recent_event = True
                            break
                    except (ValueError, TypeError):
                        pass
                        
            if had_recent_event:
                continue
                
            # Add to active users
            active_users.append((user_id, emotions))
        
        if not active_users:
            self.logger.debug("No eligible users for random events")
            return
        
        self.logger.info(f"Found {len(active_users)} users eligible for random events")
        
        # For each active user, roll a chance
        for user_id, emotions in active_users:
            if random.random() < event_chance:
                # Trigger an event!
                self.logger.info(f"Triggering random event for user {user_id}")
                
                # Build event based on relationship
                trust = emotions.get('trust', 0)
                events = []
                
                # Low trust events
                if trust < 3:
                    events.extend([
                        {
                            "type": "defensive_reaction",
                            "message": "... Stay back. Motion sensors triggered.",
                            "effects": {"resentment": 0.1, "protectiveness": -0.1}
                        },
                        {
                            "type": "system_glitch",
                            "message": "Error... system... recalibrating...",
                            "effects": {"trust": -0.1}
                        }
                    ])
                
                # Medium trust events
                elif trust < 6:
                    events.extend([
                        {
                            "type": "memory_fragment",
                            "message": "I... remember something. The desert. It was... Never mind.",
                            "effects": {"attachment": 0.2, "trust": 0.1}
                        },
                        {
                            "type": "tactical_alert",
                            "message": "Stay alert. Something's not right here.",
                            "effects": {"protectiveness": 0.2, "affection_points": 5}
                        }
                    ])
                
                # High trust events
                else:
                    events.extend([
                        {
                            "type": "protective_instinct",
                            "message": "... I sensed something. Stay close.",
                            "effects": {"protectiveness": 0.3, "trust": 0.1, "affection_points": 10}
                        },
                        {
                            "type": "vulnerability_moment",
                            "message": "Do you ever wonder... what happens when this is all over?",
                            "effects": {"attachment": 0.3, "trust": 0.2, "affection_points": 15}
                        }
                    ])
                
                # Special rare events
                if random.random() < 0.1:  # 10% chance for a rare event
                    events.append({
                        "type": "deep_memory",
                        "message": "I had a dream. I was... human. It felt... real.",
                        "effects": {"attachment": 0.5, "trust": 0.3, "affection_points": 20}
                    })
                
                # Select a random event from the eligible ones
                if not events:
                    continue
                    
                event = random.choice(events)
                
                # Record the event
                timestamp = datetime.now(timezone.utc).isoformat()
                event_record = {
                    "type": event["type"],
                    "message": event["message"],
                    "timestamp": timestamp,
                    "effects": event["effects"]
                }
                
                # Add to user events
                self.emotion_manager.user_events.setdefault(user_id, []).append(event_record)
                
                # Update emotional stats
                for stat, change in event["effects"].items():
                    if stat == "affection_points":
                        emotions[stat] = max(-100, min(1000, emotions.get(stat, 0) + change))
                    else:
                        emotions[stat] = max(0, min(10, emotions.get(stat, 0) + change))
                
                # Create a memory for this event
                await self.emotion_manager.create_memory_event(
                    user_id,
                    event["type"],
                    f"A2 experienced a random {event['type'].replace('_', ' ')} event: {event['message']}",
                    event["effects"],
                    storage_manager
                )
                
                # Try to send a DM to the user
                try:
                    user = bot.get_user(user_id)
                    if user:
                        dm = await user.create_dm()
                        await dm.send(f"A2: {event['message']}")
                        self.logger.info(f"Sent random event DM to user {user_id}")
                except Exception as e:
                    self.logger.error(f"Error sending DM to user {user_id}: {e}")
        
        # Save changes
        await storage_manager.save_data(self.emotion_manager)
        self.logger.debug("Random events check completed")
