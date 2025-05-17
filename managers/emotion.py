"""Emotion manager for the A2 Discord bot."""
import re
import random
import logging
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter

# Import configuration
from config import EMOTION_CONFIG, RELATIONSHIP_LEVELS, PERSONALITY_STATES

class EmotionManager:
    """Manages all emotional and relationship aspects of the bot"""
    
    def __init__(self):
        # ─── State Storage ────────────────────────────────────────────────────
        self.conversation_summaries = {}
        self.conversation_history = defaultdict(list)
        self.user_emotions = {}
        self.recent_responses = {}
        self.user_memories = defaultdict(list)
        self.user_events = defaultdict(list)
        self.user_milestones = defaultdict(list)
        self.interaction_stats = defaultdict(Counter)
        self.relationship_progress = defaultdict(dict)
        self.dm_enabled_users = set()
        self.MAX_RECENT_RESPONSES = 10
        self.pending_messages = set()
        
        # ─── Logger ────────────────────────────────────────────────────
        self.logger = logging.getLogger('a2bot')
        
    def select_personality_state(self, user_id, content=""):
        """Select the appropriate personality state based on user relationship and message content"""
        # Default state
        state = "default"
        
        # Get emotional states
        if user_id not in self.user_emotions:
            return state
            
        e = self.user_emotions[user_id]
        trust = e.get('trust', 0)
        attachment = e.get('attachment', 0)
        resentment = e.get('resentment', 0)
        annoyance = e.get('annoyance', 0)
        
        # Use weighted random selection based on various factors
        states_weights = {
            "default": 100,  # Base weight
            "combat": 0,
            "wounded": 0,
            "reflective": 0,
            "playful": 0,
            "protective": 0,
            "trusting": 0
        }
        
        # Adjust weights based on emotional state
        if trust > 7:
            states_weights["trusting"] += 40
            states_weights["reflective"] += 20
            states_weights["playful"] += 15
        elif trust > 4:
            states_weights["trusting"] += 15
            states_weights["reflective"] += 10
        
        if attachment > 5:
            states_weights["protective"] += 25
            states_weights["reflective"] += 10
        
        if resentment > 5:
            states_weights["combat"] += 20
            states_weights["default"] += 10
        
        if annoyance > 60:
            states_weights["combat"] += 15
            states_weights["reflective"] -= 10
            
        # Adjust weights based on message content
        combat_keywords = ["fight", "battle", "attack", "defend", "weapon", "hit", "kill", "enemy"]
        if any(word in content.lower() for word in combat_keywords):
            states_weights["combat"] += 30
        
        reflective_keywords = ["remember", "past", "memory", "feel", "think", "emotion", "miss"]
        if any(word in content.lower() for word in reflective_keywords):
            states_weights["reflective"] += 25
        
        # Ensure no negative weights
        for k in states_weights:
            states_weights[k] = max(0, states_weights[k])
        
        # If total weight is 0, set default to 100
        total = sum(states_weights.values())
        if total == 0:
            states_weights["default"] = 100
            total = 100
        
        # Convert to probabilities
        states_probs = {k: v/total for k, v in states_weights.items()}
        
        # Select state using weighted random
        rand = random.random()
        cumulative = 0
        for k, v in states_probs.items():
            cumulative += v
            if rand <= cumulative:
                return k
        
        return state
    
    def get_relationship_stage(self, user_id):
        """Get the current relationship stage and progress to the next stage"""
        # Get relationship score
        score = self.get_relationship_score(user_id)
        
        # Find current stage
        current_stage = RELATIONSHIP_LEVELS[0]
        next_stage = None
        for i, stage in enumerate(RELATIONSHIP_LEVELS):
            if score >= stage["threshold"]:
                current_stage = stage
                # Set next stage if not at max
                if i < len(RELATIONSHIP_LEVELS) - 1:
                    next_stage = RELATIONSHIP_LEVELS[i + 1]
        
        # Calculate progress to next stage
        progress = 0
        if next_stage:
            current_threshold = current_stage["threshold"]
            next_threshold = next_stage["threshold"]
            # Check for division by zero
            if next_threshold - current_threshold > 0:
                progress = ((score - current_threshold) / (next_threshold - current_threshold)) * 100
                progress = min(99.9, max(0, progress))  # Cap between 0-99.9%
        
        return {
            "score": score,
            "current": current_stage,
            "next": next_stage,
            "progress": progress
        }
    
    def get_relationship_score(self, user_id):
        """Calculate an overall relationship score based on emotional stats"""
        if user_id not in self.user_emotions:
            return 0
            
        e = self.user_emotions[user_id]
        
        # Base score from trust (0-30 points)
        score = e.get('trust', 0) * 3
        
        # Add attachment (0-20 points)
        score += e.get('attachment', 0) * 2
        
        # Add protectiveness (0-15 points)
        score += e.get('protectiveness', 0) * 1.5
        
        # Subtract resentment (0-20 points)
        score -= e.get('resentment', 0) * 2
        
        # Add affection points (0-20 points)
        affection = e.get('affection_points', 0)
        score += min(20, max(-20, affection / 50))
        
        # Subtract annoyance (0-15 points)
        annoyance = e.get('annoyance', 0)
        score -= min(15, annoyance / 6.67)
        
        # Cap the score between 0 and 100
        return max(0, min(100, score))
    
    def get_emotion_description(self, emotion_type, value):
        """Get a textual description of an emotion level"""
        if emotion_type == "trust":
            if value <= 1:
                return "Extremely wary"
            elif value <= 3:
                return "Distrustful"
            elif value <= 5:
                return "Cautious"
            elif value <= 7:
                return "Developing trust"
            elif value <= 9:
                return "Trusting"
            else:
                return "Complete trust"
        
        elif emotion_type == "attachment":
            if value <= 1:
                return "Indifferent"
            elif value <= 3:
                return "Mild connection"
            elif value <= 5:
                return "Moderate attachment"
            elif value <= 7:
                return "Strong attachment"
            elif value <= 9:
                return "Deep connection"
            else:
                return "Profound bond"
        
        elif emotion_type == "protectiveness":
            if value <= 1:
                return "None"
            elif value <= 3:
                return "Minimal"
            elif value <= 5:
                return "Moderate"
            elif value <= 7:
                return "Significant"
            elif value <= 9:
                return "Strong"
            else:
                return "Extreme"
        
        elif emotion_type == "resentment":
            if value <= 1:
                return "None"
            elif value <= 3:
                return "Slight"
            elif value <= 5:
                return "Moderate"
            elif value <= 7:
                return "Significant"
            elif value <= 9:
                return "Strong"
            else:
                return "Extreme"
        
        return f"{value}/10"
    
    async def update_emotional_stats(self, user_id, content, response, storage_manager):
        """Update emotional stats based on an interaction"""
        # Initialize user if they don't exist
        if user_id not in self.user_emotions:
            self.user_emotions[user_id] = {
                "trust": 0, 
                "resentment": 0, 
                "attachment": 0, 
                "protectiveness": 0,
                "affection_points": 0, 
                "annoyance": 0,
                "first_interaction": datetime.now(timezone.utc).isoformat(),
                "last_interaction": datetime.now(timezone.utc).isoformat(),
                "interaction_count": 0
            }
        
        # Update last interaction time and count
        self.user_emotions[user_id]["last_interaction"] = datetime.now(timezone.utc).isoformat()
        self.user_emotions[user_id]["interaction_count"] = self.user_emotions[user_id].get("interaction_count", 0) + 1
        
        # Simple sentiment analysis
        positive_patterns = [
            r"\b(like|trust|respect|admire|appreciate|thank|kind|good|nice|friend|happy|help)\b",
            r"\b(understand|care|support|listen|patient|thoughtful|gentle|calm|peaceful)\b"
        ]
        negative_patterns = [
            r"\b(hate|dislike|annoy|frustrate|anger|mad|stupid|dumb|idiot|useless|broken)\b",
            r"\b(wrong|bad|terrible|awful|worst|failure|mistake|error|bug|glitch|malfunction)\b"
        ]
        
        # Check for positive sentiment
        positive_matches = sum(len(re.findall(pattern, content.lower())) for pattern in positive_patterns)
        
        # Check for negative sentiment
        negative_matches = sum(len(re.findall(pattern, content.lower())) for pattern in negative_patterns)
        
        # Determine overall sentiment
        sentiment = "neutral"
        if positive_matches > negative_matches:
            sentiment = "positive"
        elif negative_matches > positive_matches:
            sentiment = "negative"
        
        # Update interaction stats
        self.interaction_stats[user_id].update({sentiment: 1, "total": 1})
        
        # Apply emotional adjustments based on sentiment
        e = self.user_emotions[user_id]
        
        if sentiment == "positive":
            # Increase trust slightly
            e["trust"] = min(10, e.get("trust", 0) + 0.1)
            
            # Increase attachment slightly
            e["attachment"] = min(10, e.get("attachment", 0) + 0.15)
            
            # Decrease resentment slightly
            e["resentment"] = max(0, e.get("resentment", 0) - 0.05)
            
            # Add affection points
            e["affection_points"] = min(1000, e.get("affection_points", 0) + 2)
            
            # Decrease annoyance
            e["annoyance"] = max(0, e.get("annoyance", 0) - 1)
            
        elif sentiment == "negative":
            # Decrease trust slightly
            e["trust"] = max(0, e.get("trust", 0) - 0.05)
            
            # Increase resentment slightly
            e["resentment"] = min(10, e.get("resentment", 0) + 0.1)
            
            # Increase annoyance
            e["annoyance"] = min(100, e.get("annoyance", 0) + 3)
        
        # Save the changes
        await storage_manager.save_data(self, None)
        
        return sentiment
    
    async def create_memory_event(self, user_id, event_type, description, effects, storage_manager):
        """Create a memory event for a user"""
        # Create the memory record
        memory = {
            "type": event_type,
            "description": description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "effects": effects
        }
        
        # Add to user memories
        self.user_memories.setdefault(user_id, []).append(memory)
        
        # Save memories
        await storage_manager.save_user_profile(user_id, self)
        
        return memory
    
    async def record_interaction_data(self, user_id, content, response, storage_manager):
        """Record interaction data for future analysis"""
        # Simple data about the interaction
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "user_message_length": len(content),
            "bot_response_length": len(response),
            "user_message_preview": content[:30] + "..." if len(content) > 30 else content
        }
        
        # Store data as a memory
        await self.create_memory_event(
            user_id, 
            "interaction", 
            f"Had a conversation about: {data['user_message_preview']}",
            {},  # No direct emotional effects
            storage_manager
        )
    
    async def decay_affection(self, storage_manager):
        """Gradually decay affection points over time"""
        self.logger.debug("Running decay_affection task")
        decay_rate = EMOTION_CONFIG.get("AFFECTION_DECAY_RATE", 1)
        
        for user_id, emotions in self.user_emotions.items():
            # Skip users with no or negative affection points
            if emotions.get("affection_points", 0) <= 0:
                continue
                
            # Apply decay
            emotions["affection_points"] = max(0, emotions.get("affection_points", 0) - decay_rate)
        
        # Save changes
        await storage_manager.save_data(self)
        self.logger.debug("Completed decay_affection task")
    
    async def decay_annoyance(self, storage_manager):
        """Gradually decay annoyance points over time"""
        self.logger.debug("Running decay_annoyance task")
        decay_rate = EMOTION_CONFIG.get("ANNOYANCE_DECAY_RATE", 5)
        threshold = EMOTION_CONFIG.get("ANNOYANCE_THRESHOLD", 85)
        
        for user_id, emotions in self.user_emotions.items():
            # Skip users with no annoyance or above threshold
            if emotions.get("annoyance", 0) <= 0 or emotions.get("annoyance", 0) > threshold:
                continue
                
            # Apply decay
            emotions["annoyance"] = max(0, emotions.get("annoyance", 0) - decay_rate)
        
        # Save changes
        await storage_manager.save_data(self)
        self.logger.debug("Completed decay_annoyance task")
    
    async def daily_affection_bonus(self, storage_manager):
        """Apply daily affection bonus for users with sufficient trust"""
        self.logger.debug("Running daily_affection_bonus task")
        bonus = EMOTION_CONFIG.get("DAILY_AFFECTION_BONUS", 5)
        threshold = EMOTION_CONFIG.get("DAILY_BONUS_TRUST_THRESHOLD", 5)
        
        for user_id, emotions in self.user_emotions.items():
            # Skip users with insufficient trust
            if emotions.get("trust", 0) < threshold:
                continue
                
            # Apply bonus
            emotions["affection_points"] = min(1000, emotions.get("affection_points", 0) + bonus)
        
        # Save changes
        await storage_manager.save_data(self)
        self.logger.debug("Completed daily_affection_bonus task")
    
    async def dynamic_emotional_adjustments(self, storage_manager):
        """Apply dynamic emotional adjustments based on user interactions"""
        self.logger.debug("Running dynamic_emotional_adjustments task")
        
        for user_id, emotions in self.user_emotions.items():
            # Skip if no interaction data
            last_interaction = emotions.get("last_interaction")
            if not last_interaction:
                continue
                
            # Calculate days since last interaction
            try:
                last_date = datetime.fromisoformat(last_interaction)
                days_since = (datetime.now(timezone.utc) - last_date).days
            except Exception:
                days_since = 0
            
            # Apply decay based on inactivity
            if days_since > 7:
                # Apply stronger decay to all emotions
                decay_multipliers = EMOTION_CONFIG.get("DECAY_MULTIPLIERS", {})
                for emotion, multiplier in decay_multipliers.items():
                    if emotion in emotions:
                        decay_amount = (days_since - 7) * 0.01 * multiplier
                        emotions[emotion] = max(0, emotions.get(emotion, 0) - decay_amount)
            
            # Special case: if user was very active and suddenly stops, increase resentment slightly
            interaction_count = emotions.get("interaction_count", 0)
            if interaction_count > 50 and days_since > 14:
                emotions["resentment"] = min(10, emotions.get("resentment", 0) + 0.2)
        
        # Save changes
        await storage_manager.save_data(self)
        self.logger.debug("Completed dynamic_emotional_adjustments task")
    
    async def environmental_mood_effects(self, storage_manager):
        """Apply random environmental effects to moods"""
        self.logger.debug("Running environmental_mood_effects task")
        
        # Random environmental factors
        factors = [
            {"name": "system_stability", "value": random.uniform(0.8, 1.0)},
            {"name": "memory_corruption", "value": random.uniform(0, 0.1)},
            {"name": "sensory_input", "value": random.uniform(0.5, 1.0)}
        ]
        
        # Apply effects to active users
        for user_id, emotions in self.user_emotions.items():
            # Skip inactive users (no interaction in the last 3 days)
            last_interaction = emotions.get("last_interaction")
            if not last_interaction:
                continue
                
            try:
                last_date = datetime.fromisoformat(last_interaction)
                if (datetime.now(timezone.utc) - last_date).days > 3:
                    continue
            except Exception:
                continue
            
            # System stability affects trust
            trust_change = (factors[0]["value"] - 0.9) * 0.2
            emotions["trust"] = max(0, min(10, emotions.get("trust", 0) + trust_change))
            
            # Memory corruption affects resentment
            resentment_change = factors[1]["value"] * 0.3
            emotions["resentment"] = max(0, min(10, emotions.get("resentment", 0) + resentment_change))
            
            # Sensory input affects protectiveness
            protective_change = (factors[2]["value"] - 0.75) * 0.15
            emotions["protectiveness"] = max(0, min(10, emotions.get("protectiveness", 0) + protective_change))
        
        # Save changes
        await storage_manager.save_data(self)
        self.logger.debug("Completed environmental_mood_effects task")
