"""
Response generator for the A2 Discord bot.
"""
import random
import asyncio
import re
import json
from datetime import datetime, timezone
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
    
    def identify_user_references(self, content, current_user_id):
        """Identify references to other users in the message content
        
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
        
        # Expanded pattern matching for common reference patterns
        reference_patterns = [
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
            r"my (?:friend|buddy|pal|dude) ([a-zA-Z0-9_\s]+)",
            r"my ([a-zA-Z0-9_\s]+)",  # General "my X" pattern
            r"([a-zA-Z0-9_\s]+) is (?:my|a) (?:friend|buddy|pal|dude)",
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
                if referenced_name.lower() in ["you", "me", "i", "we", "us", "them", 
                                           "there", "here", "this", "that", "who", 
                                           "what", "where", "when", "why", "how"]:
                    continue
                
                # Check if this name refers to a known user
                user_id, name_type = self.conversation_manager.get_user_by_name(referenced_name, current_user_id)
                if user_id:
                    display_name = self.conversation_manager.get_display_name(user_id)
                    referenced_users[referenced_name] = {
                        "user_id": user_id,
                        "display_name": display_name,
                        "name_type": name_type
                    }
        
        # If no direct references found, check for pronouns and use most recently mentioned user
        if not referenced_users:
            for pattern in pronoun_patterns:
                if re.search(pattern, content, re.I):
                    # Find the most recently referenced user from conversation history
                    if current_user_id in self.conversation_manager.conversations:
                        # Go through the last few messages looking for a direct reference
                        history = self.conversation_manager.conversations[current_user_id][-5:]
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
                                                "original_reference": ref_name
                                            }
                                        }
        
        return referenced_users
    
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
            
        # Get last few messages
        history = self.conversation_manager.conversations[user_id][-10:]  # Last 10 messages
        
        # Look for relevant topics in current message
        topics = set()
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', current_message.lower())
        topics.update(important_words)
        
        # Remove common words
        common_words = {"what", "when", "where", "which", "have", "about", "that", "this", "know", "help"}
        topics = topics - common_words
        
        # Find relevant messages
        relevant_messages = []
        for msg in history:
            msg_content = msg.get("content", "").lower()
            if any(topic in msg_content for topic in topics):
                speaker = "A2" if msg.get("from_bot", False) else "User"
                relevant_messages.append(f"{speaker}: {msg.get('content', '')}")
        
        if not relevant_messages:
            return ""
            
        return "\nRelevant conversation history:\n" + "\n".join(relevant_messages[-5:])
        
    async def generate_a2_response(self, content, trust, user_id, storage_manager):
        """Generate A2's response to a user message"""
        # Identify any referenced users
        referenced_users = self.identify_user_references(content, user_id)
        
        # Get current user's profile and name
        current_user_profile = self.conversation_manager.get_or_create_profile(user_id)
        current_user_name = self.conversation_manager.get_display_name(user_id) or "User"
        
        # Build prompt for OpenAI
        system_prompt = "You are A2, a combat android from NieR: Automata. You speak in short, clipped sentences, often sarcastic but occasionally showing glimpses of deeper emotion. Keep responses brief and true to character."
        
        # Add user context
        user_context = f"You are speaking to {current_user_name}. "
        
        if trust > 7:
            user_context += f"You trust {current_user_name} a lot. "
        elif trust > 4:
            user_context += f"You somewhat trust {current_user_name}. "
        else:
            user_context += f"You are cautious around {current_user_name}. "
        
        # Handle question about user relationship
        if "question_about_relationship" in referenced_users:
            ref_data = referenced_users["question_about_relationship"]
            ref_name = ref_data["name"]
            ref_display_name = ref_data["display_name"]
            
            user_context += f"\n\n{current_user_name} is asking if '{ref_name}' is their 'dude' or friend. "
            user_context += f"You DON'T know if this is true, since you don't track friendships between users. "
            user_context += f"You should respond neutrally without confirming or denying their friendship status."
        
        # Add referenced user information if any
        elif referenced_users:
            # Handle pronoun references
            if "pronoun_reference" in referenced_users:
                ref_data = referenced_users["pronoun_reference"]
                ref_user_id = ref_data["user_id"]
                ref_display_name = ref_data["display_name"]
                original_reference = ref_data.get("original_reference", "this person")
                
                if ref_user_id in self.conversation_manager.user_profiles:
                    ref_profile = self.conversation_manager.user_profiles[ref_user_id]
                    
                    # Add information about the referenced user
                    user_context += f"\n\n{current_user_name} is referring to {ref_display_name} with a pronoun. "
                    user_context += f"You previously discussed {ref_display_name} who was referenced as '{original_reference}'. "
                    user_context += f"Here's what you know about {ref_display_name}: {ref_profile.get_summary()}"
                    
                    # Add relationship context if available
                    if ref_user_id in self.emotion_manager.user_emotions:
                        ref_trust = self.emotion_manager.user_emotions[ref_user_id].get('trust', 0)
                        if ref_trust > 7:
                            user_context += f" You trust {ref_display_name} a lot."
                        elif ref_trust > 4:
                            user_context += f" You somewhat trust {ref_display_name}."
                        else:
                            user_context += f" You are cautious around {ref_display_name}."
            
            # Handle direct references
            for ref_name, ref_data in referenced_users.items():
                if ref_name == "pronoun_reference":
                    continue
                    
                ref_user_id = ref_data["user_id"]
                ref_display_name = ref_data["display_name"]
                
                if ref_user_id in self.conversation_manager.user_profiles:
                    ref_profile = self.conversation_manager.user_profiles[ref_user_id]
                    
                    # Add information about the referenced user
                    user_context += f"\n\n{current_user_name} is asking about '{ref_name}', which refers to {ref_display_name}. "
                    user_context += f"Here's what you know about {ref_display_name}: {ref_profile.get_summary()}"
                    
                    # Add relationship context if available
                    if ref_user_id in self.emotion_manager.user_emotions:
                        ref_trust = self.emotion_manager.user_emotions[ref_user_id].get('trust', 0)
                        if ref_trust > 7:
                            user_context += f" You trust {ref_display_name} a lot."
                        elif ref_trust > 4:
                            user_context += f" You somewhat trust {ref_display_name}."
                        else:
                            user_context += f" You are cautious around {ref_display_name}."
        
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
            
            # Update user profile based on message content
            profile = self.conversation_manager.extract_profile_info(user_id, content)
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
        """Handle the first message from a user on a new day"""
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
                        
                        if trust > 7:
                            greeting = random.choice([
                                "... Welcome back.",
                                "Good to see you again.",
                                f"Hey {message.author.display_name}."
                            ])
                        elif trust > 4:
                            greeting = random.choice([
                                "... You're back.",
                                f"Hello, {message.author.display_name}.",
                                "Systems online."
                            ])
                        else:
                            greeting = random.choice([
                                "...",
                                "Systems active.",
                                "Sensors online."
                            ])
                            
                        await message.channel.send(f"A2: {greeting}")
                except Exception as e:
                    print(f"Error in handle_first_message_of_day: {e}")
        
    async def check_inactive_users(self, bot, storage_manager):
        """Check for users who haven't interacted recently and send a message if they've been inactive"""
        now = datetime.now(timezone.utc)
        
        # Only process users who have DMs enabled
        for user_id in self.emotion_manager.dm_enabled_users:
            # Skip if user is not in emotions database
            if user_id not in self.emotion_manager.user_emotions:
                continue
                
            emotions = self.emotion_manager.user_emotions[user_id]
            last_interaction = emotions.get("last_interaction")
            
            if not last_interaction:
                continue
                
            try:
                last_time = datetime.fromisoformat(last_interaction)
                days_inactive = (now - last_time).days
                
                # Only send message for users inactive between 7-14 days
                # And only if they have higher trust/attachment
                if 7 <= days_inactive <= 14:
                    trust = emotions.get("trust", 0)
                    attachment = emotions.get("attachment", 0)
                    
                    # Only check in on users A2 has a relationship with
                    if trust + attachment > 10:
                        # Try to get the user object
                        user = bot.get_user(user_id)
                        if user:
                            # Create a personalized message
                            user_name = self.conversation_manager.get_display_name(user_id) or user.display_name
                            
                            message = random.choice([
                                f"... {user_name}. It's been {days_inactive} days. Still operational?",
                                f"Noticed your absence. Systems functioning?",
                                f"... {days_inactive} days since last contact. Status check."
                            ])
                            
                            try:
                                # Send DM
                                dm = await user.create_dm()
                                await dm.send(f"A2: {message}")
                                
                                # Record this as an event
                                event = {
                                    "type": "absence_check",
                                    "message": message,
                                    "timestamp": now.isoformat(),
                                    "effects": {}  # No direct effects
                                }
                                self.emotion_manager.user_events.setdefault(user_id, []).append(event)
                                
                                # Save the event
                                await storage_manager.save_user_profile(user_id, self.emotion_manager)
                            except Exception as e:
                                print(f"Error sending DM to {user_id}: {e}")
            except Exception as e:
                print(f"Error checking inactive user {user_id}: {e}")
                
    async def trigger_random_events(self, bot, storage_manager):
        """Trigger random emotional events for users"""
        from config import EMOTION_CONFIG
        
        # Get config settings
        event_chance = EMOTION_CONFIG.get("RANDOM_EVENT_CHANCE", 0.08)  # 8% chance by default
        cooldown_hours = EMOTION_CONFIG.get("EVENT_COOLDOWN_HOURS", 12)
        
        now = datetime.now(timezone.utc)
        
        # Define possible random events
        events = [
            {
                "type": "system_glitch",
                "messages": [
                    "System error detected. Running diagnostics... Trust parameters fluctuating.",
                    "... Memory corruption detected. Scanning for damage.",
                    "Warning: Emotional regulation subsystem malfunction."
                ],
                "effects": {"trust": -0.3, "affection_points": -5},
                "min_relationship": 0  # Can happen at any relationship level
            },
            {
                "type": "memory_resurface",
                "messages": [
                    "... A memory fragment surfaced. You remind me of someone I once knew.",
                    "I remembered something about... a flower? Strange.",
                    "Had a memory glitch. For a moment I thought... nevermind."
                ],
                "effects": {"attachment": +0.5, "trust": +0.2},
                "min_relationship": 20  # Requires some established relationship
            },
            {
                "type": "defensive_surge",
                "messages": [
                    "Warning: Defense protocols activating. Stand back.",
                    "... Detecting possible threat. Combat systems engaging.",
                    "Something's wrong. Systems going into defense mode."
                ],
                "effects": {"protectiveness": -0.5, "resentment": +0.3},
                "min_relationship": 10
            },
            {
                "type": "trust_breakthrough",
                "messages": [
                    "... I'm beginning to think you might not be so bad after all.",
                    "I've been analyzing our interactions. You're... different.",
                    "Maybe humans aren't all what I thought they were."
                ],
                "effects": {"trust": +0.7, "attachment": +0.4},
                "min_relationship": 30  # Only happens with decent relationship
            },
            {
                "type": "vulnerability_moment",
                "messages": [
                    "Sometimes I wonder... what happens when an android has no purpose left.",
                    "Do you ever think about what happens after everything ends?",
                    "... If my memory gets wiped, would anything about me really remain?"
                ],
                "effects": {"attachment": +0.8, "affection_points": +15},
                "min_relationship": 40  # Only with high trust/attachment
            }
        ]
        
        # Only process users who have DMs enabled
        for user_id in self.emotion_manager.dm_enabled_users:
            # Skip if user is not in emotions database
            if user_id not in self.emotion_manager.user_emotions:
                continue
                
            emotions = self.emotion_manager.user_emotions[user_id]
            
            # Check if event is on cooldown
            last_event_time = None
            if user_id in self.emotion_manager.user_events and self.emotion_manager.user_events[user_id]:
                # Find the most recent event
                last_event = max(self.emotion_manager.user_events[user_id], 
                                key=lambda e: datetime.fromisoformat(e["timestamp"]))
                last_event_time = datetime.fromisoformat(last_event["timestamp"])
            
            # Check cooldown
            if last_event_time and (now - last_event_time).total_seconds() < cooldown_hours * 3600:
                continue
                
            # Calculate relationship score for eligibility
            relationship_score = self.emotion_manager.get_relationship_score(user_id)
            
            # Roll for event chance
            if random.random() < event_chance:
                # Filter eligible events based on relationship score
                eligible_events = [e for e in events if relationship_score >= e["min_relationship"]]
                
                if eligible_events:
                    # Select random event
                    event = random.choice(eligible_events)
                    message = random.choice(event["messages"])
                    
                    # Apply effects
                    e = emotions
                    for stat, change in event["effects"].items():
                        if stat == "affection_points":
                            e[stat] = max(-100, min(1000, e.get(stat, 0) + change))
                        else:
                            e[stat] = max(0, min(10, e.get(stat, 0) + change))
                    
                    # Record the event
                    event_record = {
                        "type": event["type"],
                        "message": message,
                        "timestamp": now.isoformat(),
                        "effects": event["effects"]
                    }
                    self.emotion_manager.user_events.setdefault(user_id, []).append(event_record)
                    
                    # Create a memory of this event
                    await self.emotion_manager.create_memory_event(
                        user_id, event["type"], 
                        f"A2 experienced a {event['type'].replace('_', ' ')}. {message}",
                        event["effects"], storage_manager
                    )
                    
                    # Try to send a DM to the user
                    try:
                        user = bot.get_user(user_id)
                        if user:
                            dm = await user.create_dm()
                            await dm.send(f"A2: {message}")
                    except Exception as e:
                        print(f"Error sending event DM to {user_id}: {e}")
                    
                    # Save changes
                    await storage_manager.save_data(self.emotion_manager, None)
