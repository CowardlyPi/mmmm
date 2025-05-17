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
    
    def identify_user_references(self, content, current_user_id):
        """Identify references to other users in the message content
        
        Args:
            content (str): The message content
            current_user_id (int): The ID of the current user
            
        Returns:
            dict: Dictionary mapping referenced names to user info
        """
        referenced_users = {}
        
        # Simple pattern matching for common reference patterns
        reference_patterns = [
            r"who (?:is|are) ([a-zA-Z0-9_\s]+)\??",
            r"tell me about ([a-zA-Z0-9_\s]+)",
            r"([a-zA-Z0-9_\s]+)'s profile",
            r"what do you know about ([a-zA-Z0-9_\s]+)",
            r"do you know ([a-zA-Z0-9_\s]+)",
            r"have you met ([a-zA-Z0-9_\s]+)",
            r"remember ([a-zA-Z0-9_\s]+)\?"
        ]
        
        for pattern in reference_patterns:
            matches = re.finditer(pattern, content, re.I)
            for match in matches:
                referenced_name = match.group(1).strip()
                
                # Check if this name refers to a known user
                user_id, name_type = self.conversation_manager.get_user_by_name(referenced_name, current_user_id)
                if user_id:
                    display_name = self.conversation_manager.get_display_name(user_id)
                    referenced_users[referenced_name] = {
                        "user_id": user_id,
                        "display_name": display_name,
                        "name_type": name_type
                    }
        
        return referenced_users
        
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
        
        # Add referenced user information if any
        if referenced_users:
            for ref_name, ref_data in referenced_users.items():
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
        
        # Determine A2's current personality state
        personality_state = self.emotion_manager.select_personality_state(user_id, content)
        
        # Craft the final messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": user_context},
            {"role": "system", "content": f"Previous conversation:\n{conversation_history}"},
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
        # Placeholder - this would be implemented with proper logic
        pass
        
    async def check_inactive_users(self, bot, storage_manager):
        """Check for users who haven't interacted recently"""
        # Placeholder - this would be implemented with proper logic
        pass
        
    async def trigger_random_events(self, bot, storage_manager):
        """Trigger random emotional events for users"""
        # Placeholder - this would be implemented with proper logic
        pass
