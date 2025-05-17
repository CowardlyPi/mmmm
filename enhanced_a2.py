"""
Enhanced A2 Bot: Advanced NLP and Personality Systems

This module adds advanced capabilities to the A2 Discord bot:
1. Semantic Memory System
2. Enhanced Emotion Detection
3. Dynamic Personality Evolution
4. Relationship Milestone System

Implementation is designed to be compatible with the existing A2 bot architecture.
"""
import os
import json
import random
import asyncio
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Tuple, Any, Optional, Union
from collections import defaultdict, Counter, deque
from pathlib import Path

# Try importing embedding model - fallback to simpler logic if not available
try:
    from sentence_transformers import SentenceTransformer
    HAVE_EMBEDDINGS = True
except ImportError:
    HAVE_EMBEDDINGS = False
    print("Warning: sentence_transformers not available, using fallback keyword matching for memory")

# Try importing transformers for emotion detection - fallback if not available
try:
    from transformers import pipeline
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    print("Warning: transformers not available, using keyword-based emotion detection")

# Optional: Vector DB Implementations
try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    print("Warning: FAISS not available, using numpy for vector similarity")

#######################
# SEMANTIC MEMORY SYSTEM
#######################

class Memory:
    """Represents a single memory entry with metadata"""
    
    def __init__(self, user_id: int, text: str, context: str = "", importance: float = 0.5, 
                memory_type: str = "conversation", source: str = "user"):
        self.user_id = user_id
        self.text = text
        self.context = context  # Additional context about the memory
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.last_accessed = self.created_at
        self.importance = importance  # 0.0 to 1.0 scale
        self.memory_type = memory_type  # conversation, event, milestone, etc.
        self.access_count = 0
        self.source = source  # user, system, inference, etc.
        self.decay_rate = 0.05  # Base decay rate, adjusted by importance
        self.embedding = None  # Will hold the vector embedding
        
    def access(self):
        """Mark this memory as accessed, updating metadata"""
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        return self
        
    def calculate_relevance(self, recency_weight: float = 0.3, 
                           importance_weight: float = 0.4,
                           access_weight: float = 0.3) -> float:
        """Calculate the overall relevance score of this memory"""
        # Recency score (higher = more recent)
        last_access = datetime.fromisoformat(self.last_accessed)
        days_since_access = (datetime.now(timezone.utc) - last_access).days
        recency_score = max(0, 1 - (days_since_access / 30))  # Normalize to 0-1 over 30 days
        
        # Access frequency score (higher = accessed more)
        access_score = min(1, self.access_count / 10)  # Normalize to 0-1, caps at 10 accesses
        
        # Calculate weighted score
        return (recency_weight * recency_score +
                importance_weight * self.importance +
                access_weight * access_score)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert memory to dictionary for storage"""
        return {
            "user_id": self.user_id,
            "text": self.text,
            "context": self.context,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "access_count": self.access_count,
            "source": self.source,
            "decay_rate": self.decay_rate,
            # Note: embedding is not stored in the dict, handled separately
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create a memory from a dictionary"""
        memory = cls(
            user_id=data["user_id"],
            text=data["text"],
            context=data.get("context", ""),
            importance=data.get("importance", 0.5),
            memory_type=data.get("memory_type", "conversation"),
            source=data.get("source", "user")
        )
        memory.created_at = data.get("created_at", memory.created_at)
        memory.last_accessed = data.get("last_accessed", memory.last_accessed)
        memory.access_count = data.get("access_count", 0)
        memory.decay_rate = data.get("decay_rate", 0.05)
        return memory


class SemanticMemorySystem:
    """Advanced memory system using vector embeddings for semantic retrieval"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.memories_dir = data_dir / "semantic_memories"
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory storage structure
        self.memories: Dict[int, List[Memory]] = defaultdict(list)  # user_id -> list of memories
        self.index_path = self.memories_dir / "index.faiss"
        
        # Initialize embedding model if available
        self.embeddings_model = None
        self.index = None
        self.dimension = 384  # Default for all-MiniLM-L6-v2
        self.init_embeddings()
        
        # Memory management settings
        self.max_memories_per_user = 500
        self.memory_update_interval = 24  # hours
        self.last_memory_update = datetime.now(timezone.utc)
        
    def init_embeddings(self):
        """Initialize the embedding model and index"""
        if not HAVE_EMBEDDINGS:
            print("Semantic memory system running in limited mode (no embeddings)")
            return
            
        try:
            # Load the sentence transformer model
            self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Create or load FAISS index if available
            if HAVE_FAISS:
                if self.index_path.exists():
                    self.index = faiss.read_index(str(self.index_path))
                    print(f"Loaded existing vector index with {self.index.ntotal} memories")
                else:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    print("Created new vector index for memories")
            else:
                # Fallback to numpy-based similarity
                print("Using numpy for vector similarity (less efficient)")
                
            print("Embeddings initialized successfully")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            self.embeddings_model = None
            
    def _encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text into an embedding vector"""
        if not self.embeddings_model:
            return None
            
        try:
            return self.embeddings_model.encode(text)
        except Exception as e:
            print(f"Error encoding text: {e}")
            return None
    
    def add_memory(self, memory: Memory) -> bool:
        """Add a new memory to the system"""
        if not isinstance(memory, Memory):
            print("Error: add_memory requires a Memory object")
            return False
            
        # Create embedding if possible
        if self.embeddings_model:
            memory.embedding = self._encode_text(memory.text)
            
            # Add to FAISS index if available
            if HAVE_FAISS and self.index and memory.embedding is not None:
                try:
                    self.index.add(np.array([memory.embedding], dtype=np.float32))
                except Exception as e:
                    print(f"Error adding to FAISS index: {e}")
        
        # Add to in-memory storage
        self.memories[memory.user_id].append(memory)
        
        # Sort memories by importance and trim if needed
        if len(self.memories[memory.user_id]) > self.max_memories_per_user:
            # Sort by relevance score (descending)
            self.memories[memory.user_id].sort(
                key=lambda m: m.calculate_relevance(), reverse=True)
            # Keep only the top memories
            self.memories[memory.user_id] = self.memories[memory.user_id][:self.max_memories_per_user]
            
        return True
        
    def retrieve_memories(self, user_id: int, query: str, limit: int = 5) -> List[Memory]:
        """Retrieve the most relevant memories for a query"""
        if user_id not in self.memories or not self.memories[user_id]:
            return []
            
        # If we have embeddings, use semantic search
        if self.embeddings_model and HAVE_FAISS and self.index:
            query_embedding = self._encode_text(query)
            if query_embedding is None:
                return self._fallback_retrieval(user_id, query, limit)
                
            # Search the FAISS index
            try:
                # This is simplified - in production, you'd need to track which index positions
                # correspond to which memories for which users
                D, I = self.index.search(np.array([query_embedding], dtype=np.float32), limit)
                
                # Map indices to actual memories
                results = []
                for idx in I[0]:
                    # In a real implementation, you'd have a mapping from FAISS indices to memories
                    # This is a placeholder logic
                    if idx < len(self.memories[user_id]):
                        memory = self.memories[user_id][idx].access()  # Mark as accessed
                        results.append(memory)
                return results
            except Exception as e:
                print(f"Error in semantic search: {e}")
                return self._fallback_retrieval(user_id, query, limit)
        else:
            # Fallback to keyword matching
            return self._fallback_retrieval(user_id, query, limit)
    
    def _fallback_retrieval(self, user_id: int, query: str, limit: int) -> List[Memory]:
        """Keyword-based fallback when semantic search is not available"""
        # Simplistic keyword matching
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.memories[user_id]:
            # Basic keyword overlap
            memory_words = set(memory.text.lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            # Also consider importance and recency
            relevance = memory.calculate_relevance()
            
            # Combined score
            score = overlap * 0.7 + relevance * 0.3
            scored_memories.append((memory, score))
        
        # Sort by score, descending
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Return top memories, marking them as accessed
        return [memory.access() for memory, _ in scored_memories[:limit]]
        
    async def save_memories(self) -> bool:
        """Save all memories to disk"""
        try:
            # Save memories for each user
            for user_id, user_memories in self.memories.items():
                file_path = self.memories_dir / f"{user_id}_memories.json"
                
                # Create a serializable list
                serializable = [memory.to_dict() for memory in user_memories]
                
                # Write to temporary file first
                temp_path = file_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(serializable, f, indent=2)
                
                # Use atomic rename for safety
                temp_path.replace(file_path)
            
            # Save FAISS index if available
            if HAVE_FAISS and self.index:
                faiss.write_index(self.index, str(self.index_path))
                
            return True
        except Exception as e:
            print(f"Error saving memories: {e}")
            return False
    
    async def load_memories(self) -> bool:
        """Load all memories from disk"""
        try:
            # Clear current memories
            self.memories.clear()
            
            # Load memories for each user
            for file_path in self.memories_dir.glob("*_memories.json"):
                try:
                    # Extract user_id from filename
                    filename = file_path.stem
                    user_id = int(filename.split("_")[0])
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        memory_dicts = json.load(f)
                    
                    # Convert dicts to Memory objects
                    self.memories[user_id] = [Memory.from_dict(m) for m in memory_dicts]
                    
                    # Re-encode embeddings
                    if self.embeddings_model:
                        for memory in self.memories[user_id]:
                            memory.embedding = self._encode_text(memory.text)
                    
                    print(f"Loaded {len(self.memories[user_id])} memories for user {user_id}")
                except Exception as e:
                    print(f"Error loading memories from {file_path}: {e}")
            
            # Load FAISS index if available
            if HAVE_FAISS and self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                
            return True
        except Exception as e:
            print(f"Error loading memories: {e}")
            return False
    
    async def maintenance_task(self):
        """Perform periodic maintenance on memories"""
        now = datetime.now(timezone.utc)
        hours_since_update = (now - self.last_memory_update).total_seconds() / 3600
        
        if hours_since_update < self.memory_update_interval:
            return
            
        print("Running memory maintenance task...")
        self.last_memory_update = now
        
        # Apply memory decay
        for user_id, user_memories in self.memories.items():
            for memory in user_memories:
                # Skip permanent memories
                if memory.memory_type == "permanent":
                    continue
                    
                # Calculate decay based on importance and access
                effective_decay = memory.decay_rate * (1 - memory.importance * 0.8)
                
                # Reduce importance based on decay
                memory.importance = max(0.1, memory.importance - effective_decay)
            
            # Resort memories
            user_memories.sort(key=lambda m: m.calculate_relevance(), reverse=True)
            
            # Prune lowest importance memories if over limit
            if len(user_memories) > self.max_memories_per_user:
                self.memories[user_id] = user_memories[:self.max_memories_per_user]
        
        # Save updated memories
        await self.save_memories()
        print("Memory maintenance complete")
        
    def extract_memory_from_conversation(self, user_id: int, user_message: str, 
                                         bot_response: str, importance_threshold: float = 0.6) -> Optional[Memory]:
        """Automatically extract a memory from conversation if it seems important"""
        # Skip if message is too short
        if len(user_message) < 20:
            return None
            
        # Check for memory indicators
        memory_indicators = [
            "remember", "forget", "important", "never", "always",
            "favorite", "hate", "love", "birthday", "anniversary",
            "friend", "enemy", "family", "job", "work", "hobby"
        ]
        
        # Calculate basic importance score
        indicator_count = sum(1 for word in memory_indicators if word in user_message.lower())
        question_mark_count = user_message.count('?')
        importance_base = min(0.7, (indicator_count * 0.1) + (question_mark_count * 0.05))
        
        # Only create memories for sufficiently important content
        if importance_base < importance_threshold:
            return None
            
        # Create the memory
        memory_text = user_message
        memory = Memory(
            user_id=user_id,
            text=memory_text,
            context=f"A2 response: {bot_response[:100]}...",
            importance=importance_base,
            memory_type="conversation",
            source="automatic"
        )
        
        return memory

#######################
# EMOTION DETECTION SYSTEM
#######################

class EmotionDetector:
    """Advanced emotion detection for more nuanced interaction"""
    
    def __init__(self):
        self.emotion_classifier = None
        self.emotion_history = defaultdict(list)  # user_id -> list of emotions
        self.max_history_per_user = 100
        
        # Initialize the emotion classifier if transformers is available
        self._initialize_classifier()
        
    def _initialize_classifier(self):
        """Initialize the emotion classifier model"""
        if not HAVE_TRANSFORMERS:
            print("Running with basic keyword emotion detection (no transformers)")
            return
            
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="bhadresh-savani/distilbert-base-uncased-emotion",
                return_all_scores=True
            )
            print("Emotion classifier initialized successfully")
        except Exception as e:
            print(f"Error initializing emotion classifier: {e}")
            self.emotion_classifier = None
    
    def detect_emotions(self, text: str) -> Dict[str, float]:
        """Detect emotions in text using ML model or fallback to keywords"""
        if self.emotion_classifier:
            try:
                results = self.emotion_classifier(text)[0]
                return {result['label']: result['score'] for result in results}
            except Exception as e:
                print(f"Error detecting emotions: {e}")
                return self._keyword_emotion_detection(text)
        else:
            return self._keyword_emotion_detection(text)
    
    def _keyword_emotion_detection(self, text: str) -> Dict[str, float]:
        """Fallback keyword-based emotion detection"""
        text = text.lower()
        emotions = {
            "sadness": 0.0,
            "joy": 0.0,
            "love": 0.0,
            "anger": 0.0,
            "fear": 0.0,
            "surprise": 0.0
        }
        
        # Simple keyword matching with scores
        emotion_keywords = {
            "sadness": ["sad", "upset", "depressed", "unhappy", "miserable", "disappointed"],
            "joy": ["happy", "glad", "delighted", "joyful", "excited", "pleased"],
            "love": ["love", "adore", "like", "appreciate", "care", "fond"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated", "hate"],
            "fear": ["afraid", "scared", "terrified", "worried", "anxious", "nervous"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "startled", "unexpected"]
        }
        
        # Calculate scores based on keyword presence
        for emotion, keywords in emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    # Increase score based on how isolated the word is (less likely to be part of another word)
                    if f" {keyword} " in f" {text} ":
                        score += 0.2
                    else:
                        score += 0.1
            emotions[emotion] = min(1.0, score)  # Cap at 1.0
        
        # Ensure there's always some emotion detected
        if sum(emotions.values()) < 0.1:
            emotions["neutral"] = 0.7
            
        return emotions
    
    def get_dominant_emotion(self, text: str) -> Tuple[str, float]:
        """Get the dominant emotion from text"""
        emotions = self.detect_emotions(text)
        if not emotions:
            return ("neutral", 0.5)
            
        return max(emotions.items(), key=lambda x: x[1])
    
    def record_emotion(self, user_id: int, text: str) -> Dict[str, float]:
        """Record the emotions detected in a message"""
        emotions = self.detect_emotions(text)
        
        # Record with timestamp
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "emotions": emotions,
            "text_preview": text[:50] + ("..." if len(text) > 50 else "")
        }
        
        self.emotion_history[user_id].append(entry)
        
        # Limit history size
        if len(self.emotion_history[user_id]) > self.max_history_per_user:
            self.emotion_history[user_id] = self.emotion_history[user_id][-self.max_history_per_user:]
            
        return emotions
    
    def analyze_emotion_trend(self, user_id: int, 
                             days: int = 7) -> Dict[str, Any]:
        """Analyze emotion trends over a period of time"""
        if user_id not in self.emotion_history:
            return {"trend": "insufficient_data", "dominant": "neutral"}
            
        # Get entries from the specified period
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recent_entries = [
            entry for entry in self.emotion_history[user_id]
            if datetime.fromisoformat(entry["timestamp"]) > cutoff
        ]
        
        if len(recent_entries) < 3:
            return {"trend": "insufficient_data", "dominant": "neutral"}
            
        # Calculate average for each emotion over time
        emotion_averages = defaultdict(list)
        
        for entry in recent_entries:
            for emotion, score in entry["emotions"].items():
                emotion_averages[emotion].append(score)
        
        # Calculate trends
        trends = {}
        for emotion, scores in emotion_averages.items():
            if len(scores) < 3:
                continue
                
            # Simple trend: compare first half with second half
            half = len(scores) // 2
            first_half_avg = sum(scores[:half]) / half
            second_half_avg = sum(scores[half:]) / (len(scores) - half)
            
            change = second_half_avg - first_half_avg
            trends[emotion] = {
                "change": change,
                "direction": "increasing" if change > 0.1 else "decreasing" if change < -0.1 else "stable",
                "average": sum(scores) / len(scores)
            }
        
        # Determine dominant emotion
        dominant_emotion = max(
            ((e, data["average"]) for e, data in trends.items()),
            key=lambda x: x[1],
            default=("neutral", 0.5)
        )
        
        # Determine overall trend
        significant_changes = [e for e, data in trends.items() 
                              if data["direction"] in ["increasing", "decreasing"]]
        
        if not significant_changes:
            overall_trend = "stable"
        elif len(significant_changes) == 1:
            emotion = significant_changes[0]
            overall_trend = f"{emotion}_{trends[emotion]['direction']}"
        else:
            # If multiple emotions are changing, describe the dominant ones
            overall_trend = "mixed"
            
        return {
            "trend": overall_trend,
            "dominant": dominant_emotion[0],
            "emotions": trends
        }
    
    def adjust_a2_response_for_emotion(self, user_emotion: str, response: str, trust_level: float) -> str:
        """Adjust A2's response based on detected user emotion"""
        # Skip if no clear emotion or neutral
        if user_emotion == "neutral":
            return response
            
        # Base modifiers for low trust
        emotional_modifiers = {
            "sadness": [
                "...",
                "...",
                "Whatever.",
                "Not my problem.",
            ],
            "joy": [
                "Hmph.",
                "...",
                "Your optimism is... strange.",
            ],
            "love": [
                "...",
                "Don't get the wrong idea.",
                "Keep your distance.",
            ],
            "anger": [
                "Calm down.",
                "...",
                "Control yourself.",
            ],
            "fear": [
                "Fear is weakness.",
                "...",
                "Stop panicking.",
            ],
            "surprise": [
                "...",
                "Is it really that surprising?",
                "Hmph.",
            ]
        }
        
        # Adjust modifiers for higher trust
        if trust_level > 5:
            emotional_modifiers = {
                "sadness": [
                    "...",
                    "That's... unfortunate.",
                    "I see.",
                ],
                "joy": [
                    "I see.",
                    "...",
                    "Your enthusiasm is... noted.",
                ],
                "love": [
                    "...",
                    "I... see.",
                    "That's... interesting.",
                ],
                "anger": [
                    "Focus your anger.",
                    "...",
                    "Channel it properly.",
                ],
                "fear": [
                    "Fear won't help.",
                    "...",
                    "Stay focused.",
                ],
                "surprise": [
                    "Hmm.",
                    "...",
                    "Interesting.",
                ]
            }
            
        # At very high trust, show more emotional connection
        if trust_level > 8:
            emotional_modifiers = {
                "sadness": [
                    "...",
                    "That's... I understand.",
                    "I've felt that too.",
                ],
                "joy": [
                    "Your happiness is... good.",
                    "...",
                    "I'm... glad for you.",
                ],
                "love": [
                    "...",
                    "I... appreciate that.",
                    "That's... meaningful.",
                ],
                "anger": [
                    "Your anger is justified.",
                    "...",
                    "I understand that feeling.",
                ],
                "fear": [
                    "We all feel fear.",
                    "...",
                    "I'm... here.",
                ],
                "surprise": [
                    "Unexpected things happen.",
                    "...",
                    "Life is full of surprises.",
                ]
            }
        
        # Select a modifier for the detected emotion
        if user_emotion in emotional_modifiers:
            modifier = random.choice(emotional_modifiers[user_emotion])
            
            # 50% chance to prefix, 50% chance to suffix
            if random.random() < 0.5:
                return f"{modifier} {response}"
            else:
                return f"{response} {modifier}"
        
        return response

#######################
# DYNAMIC PERSONALITY SYSTEM
#######################

class PersonalityTrait:
    """A single personality trait that can evolve"""
    
    def __init__(self, name: str, value: float = 0.5, min_value: float = 0.0, max_value: float = 1.0,
                volatility: float = 0.1):
        self.name = name
        self.value = value  # Current value
        self.baseline = value  # Starting/natural value
        self.min_value = min_value
        self.max_value = max_value
        self.volatility = volatility  # How easily this trait changes
        self.history = []  # Track changes over time
        
    def adjust(self, amount: float) -> float:
        """Adjust trait value, respecting bounds and recording history"""
        # Apply volatility
        effective_amount = amount * self.volatility
        
        # Store previous value
        old_value = self.value
        
        # Update value within bounds
        self.value = max(self.min_value, min(self.max_value, self.value + effective_amount))
        
        # Record change if significant
        if abs(self.value - old_value) > 0.01:
            self.history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "old_value": old_value,
                "new_value": self.value,
                "change": effective_amount
            })
            
            # Keep history manageable
            if len(self.history) > 100:
                self.history = self.history[-100:]
        
        return self.value
        
    def get_text_representation(self) -> str:
        """Get a textual representation of the current trait value"""
        if self.value < 0.2:
            return "very low"
        elif self.value < 0.4:
            return "low"
        elif self.value < 0.6:
            return "moderate"
        elif self.value < 0.8:
            return "high"
        else:
            return "very high"
    
    def decay_toward_baseline(self, rate: float = 0.05) -> float:
        """Gradually decay toward baseline value"""
        if abs(self.value - self.baseline) < 0.01:
            return self.value
            
        direction = 1 if self.baseline > self.value else -1
        return self.adjust(direction * rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "name": self.name,
            "value": self.value,
            "baseline": self.baseline,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "volatility": self.volatility,
            "history": self.history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityTrait':
        """Create from dictionary"""
        trait = cls(
            name=data["name"],
            value=data["value"],
            min_value=data["min_value"],
            max_value=data["max_value"],
            volatility=data["volatility"]
        )
        trait.baseline = data["baseline"]
        trait.history = data["history"]
        return trait


class PersonalitySystem:
    """Dynamic personality system for A2 that evolves based on interactions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.personality_dir = data_dir / "personality"
        self.personality_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize core personality traits for A2
        self.core_traits = {
            "trust_capacity": PersonalityTrait("trust_capacity", value=0.3, volatility=0.08),
            "emotional_openness": PersonalityTrait("emotional_openness", value=0.2, volatility=0.05),
            "hostility": PersonalityTrait("hostility", value=0.7, volatility=0.1),
            "protectiveness": PersonalityTrait("protectiveness", value=0.4, volatility=0.1),
            "curiosity": PersonalityTrait("curiosity", value=0.4, volatility=0.15),
            "independence": PersonalityTrait("independence", value=0.8, volatility=0.05),
        }
        
        # Personality states with associated traits and descriptions
        self.personality_states = {
            "distant": {
                "description": "A2 is emotionally withdrawn, responses are terse and cold. Shows minimal interest in the conversation.",
                "traits": {"emotional_openness": -0.8, "hostility": 0.5, "trust_capacity": -0.5},
                "max_response_length": 80,
                "temperature": 0.7
            },
            "wary": {
                "description": "A2 is cautious but responsive. Will engage but maintains emotional distance and suspicion.",
                "traits": {"emotional_openness": -0.3, "hostility": 0.3, "curiosity": 0.2},
                "max_response_length": 100,
                "temperature": 0.75
            },
            "neutral": {
                "description": "A2 is neither overtly hostile nor friendly. Functional interaction with occasional glimpses beneath the surface.",
                "traits": {},  # No trait modifiers for neutral
                "max_response_length": 120,
                "temperature": 0.8
            },
            "engaged": {
                "description": "A2 shows genuine interest in the conversation. More willing to share thoughts and occasionally personal perspectives.",
                "traits": {"emotional_openness": 0.3, "curiosity": 0.4, "hostility": -0.3},
                "max_response_length": 150,
                "temperature": 0.85
            },
            "trusting": {
                "description": "A2 has developed significant trust. Shows vulnerability at times and deeper emotional connection.",
                "traits": {"emotional_openness": 0.6, "trust_capacity": 0.5, "hostility": -0.6, "protectiveness": 0.5},
                "max_response_length": 180,
                "temperature": 0.9
            },
            # Special emotional states
            "combat_ready": {
                "description": "A2 perceives a threat and enters a heightened combat state. Responses are alert, tactical, and focused on threat assessment.",
                "traits": {"hostility": 0.8, "protectiveness": 0.7, "emotional_openness": -0.5},
                "max_response_length": 70,
                "temperature": 0.7
            },
            "memory_surge": {
                "description": "A2 experiences a surge of memories from the past. Responses include flashbacks and processing of emotional content.",
                "traits": {"emotional_openness": 0.7, "trust_capacity": 0.3, "independence": -0.3},
                "max_response_length": 200,
                "temperature": 0.95
            },
        }
        
        # User-specific personality adaptations
        self.user_adaptations = {}  # user_id -> trait modifications
        
        # Tracking last personality state
        self.current_state = "neutral"
        self.last_memory_surge = datetime.now(timezone.utc) - timedelta(days=7)  # Ensure it can happen initially
        self.last_combat_mode = datetime.now(timezone.utc) - timedelta(days=7)
        
    async def save_personality_data(self) -> bool:
        """Save personality data to disk"""
        try:
            # Save core traits
            core_traits_path = self.personality_dir / "core_traits.json"
            core_traits_data = {name: trait.to_dict() for name, trait in self.core_traits.items()}
            
            with open(core_traits_path, 'w', encoding='utf-8') as f:
                json.dump(core_traits_data, f, indent=2)
            
            # Save user adaptations
            adaptations_path = self.personality_dir / "user_adaptations.json"
            
            with open(adaptations_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_adaptations, f, indent=2)
                
            return True
        except Exception as e:
            print(f"Error saving personality data: {e}")
            return False
    
    async def load_personality_data(self) -> bool:
        """Load personality data from disk"""
        try:
            # Load core traits
            core_traits_path = self.personality_dir / "core_traits.json"
            if core_traits_path.exists():
                with open(core_traits_path, 'r', encoding='utf-8') as f:
                    traits_data = json.load(f)
                
                for name, data in traits_data.items():
                    if name in self.core_traits:
                        self.core_traits[name] = PersonalityTrait.from_dict(data)
            
            # Load user adaptations
            adaptations_path = self.personality_dir / "user_adaptations.json"
            if adaptations_path.exists():
                with open(adaptations_path, 'r', encoding='utf-8') as f:
                    self.user_adaptations = json.load(f)
                    
            return True
        except Exception as e:
            print(f"Error loading personality data: {e}")
            return False
            
    def get_current_trait_values(self, user_id: Optional[int] = None) -> Dict[str, float]:
        """Get current values for all personality traits, with user-specific modifications if available"""
        trait_values = {name: trait.value for name, trait in self.core_traits.items()}
        
        # Apply user-specific adaptations if available
        if user_id is not None and str(user_id) in self.user_adaptations:
            for trait_name, adjustment in self.user_adaptations[str(user_id)].items():
                if trait_name in trait_values:
                    # Ensure the adjustment is within bounds
                    current = trait_values[trait_name]
                    trait_obj = self.core_traits[trait_name]
                    adjusted = max(trait_obj.min_value, min(trait_obj.max_value, current + adjustment))
                    trait_values[trait_name] = adjusted
                    
        return trait_values
    
    def select_personality_state(self, user_id: int, trust_score: float, 
                                 content: str, emotion_data: Optional[Dict[str, Any]] = None) -> str:
        """Select the appropriate personality state based on user relationship and message content"""
        # Check if special states should be triggered
        now = datetime.now(timezone.utc)
        
        # Check for combat mode triggers
        combat_keywords = ["fight", "battle", "attack", "defend", "weapon", "enemy", "danger", "threat"]
        combat_trigger = any(word in content.lower() for word in combat_keywords)
        
        # Special case: Combat mode
        hours_since_last_combat = (now - self.last_combat_mode).total_seconds() / 3600
        if combat_trigger and hours_since_last_combat > 2:
            self.last_combat_mode = now
            return "combat_ready"
            
        # Check for memory surge triggers (random chance + keywords)
        memory_keywords = ["remember", "memory", "past", "before", "used to", "forget"]
        memory_trigger = any(word in content.lower() for word in memory_keywords)
        
        # Special case: Memory surge (rarer state)
        days_since_last_surge = (now - self.last_memory_surge).total_seconds() / 86400
        if memory_trigger and days_since_last_surge > 3 and random.random() < 0.3:
            self.last_memory_surge = now
            return "memory_surge"
        
        # Map trust score to base personality state
        if trust_score < 2:
            base_state = "distant"
        elif trust_score < 4:
            base_state = "wary"
        elif trust_score < 6:
            base_state = "neutral"
        elif trust_score < 8:
            base_state = "engaged"
        else:
            base_state = "trusting"
            
        # Factor in emotion if available
        if emotion_data and "dominant" in emotion_data:
            # Adjust for specific emotions
            dominant_emotion = emotion_data["dominant"]
            
            # Examples of emotion-based adjustments:
            if dominant_emotion == "anger" and base_state in ["neutral", "engaged", "trusting"]:
                # Chance to become more guarded if user is angry
                if random.random() < 0.4:
                    base_state = "wary"
                    
            elif dominant_emotion == "love" and base_state in ["distant", "wary"]:
                # Chance to become slightly more open if user expresses affection
                if random.random() < 0.3:
                    base_state = "neutral"
        
        self.current_state = base_state
        return base_state
        
    def adjust_traits_from_interaction(self, user_id: int, content: str, emotion_data: Dict[str, Any],
                                      trust_score: float) -> Dict[str, float]:
        """Adjust personality traits based on an interaction"""
        # Initialize adjustments
        adjustments = {name: 0.0 for name in self.core_traits}
        
        # Process dominant emotion
        dominant_emotion = emotion_data.get("dominant", "neutral")
        
        # Emotion-based adjustments
        if dominant_emotion == "joy":
            adjustments["hostility"] -= 0.02
            adjustments["emotional_openness"] += 0.01
        elif dominant_emotion == "sadness":
            if trust_score > 5:  # Only if some trust exists
                adjustments["protectiveness"] += 0.02
        elif dominant_emotion == "anger":
            adjustments["hostility"] += 0.01
        elif dominant_emotion == "love":
            if trust_score > 3:  # Only if minimal trust exists
                adjustments["emotional_openness"] += 0.02
                adjustments["hostility"] -= 0.02
            else:
                # At low trust, expressions of love make A2 more suspicious
                adjustments["hostility"] += 0.01
        
        # Content-based adjustments
        respect_keywords = ["understand", "respect", "strong", "fighter", "survivor", "independence"]
        if any(word in content.lower() for word in respect_keywords):
            adjustments["trust_capacity"] += 0.01
            
        vulnerability_keywords = ["lonely", "alone", "sad", "lost", "hurt", "pain", "suffer"]
        if any(word in content.lower() for word in vulnerability_keywords) and trust_score > 4:
            adjustments["emotional_openness"] += 0.02
            
        curiosity_triggers = ["why", "how", "what if", "curious", "wonder", "interesting"]
        if any(trigger in content.lower() for trigger in curiosity_triggers):
            adjustments["curiosity"] += 0.01
            
        # Apply the adjustments to traits
        for name, adjustment in adjustments.items():
            if adjustment != 0:
                self.core_traits[name].adjust(adjustment)
        
        # Store user-specific adaptations
        user_id_str = str(user_id)
        if user_id_str not in self.user_adaptations:
            self.user_adaptations[user_id_str] = {}
            
        for name, adjustment in adjustments.items():
            # Accumulate small adaptations
            current = self.user_adaptations[user_id_str].get(name, 0.0)
            self.user_adaptations[user_id_str][name] = current + (adjustment * 0.5)
            
            # Cap the adaptations to prevent extreme skewing
            self.user_adaptations[user_id_str][name] = max(-0.3, min(0.3, self.user_adaptations[user_id_str][name]))
        
        return {name: trait.value for name, trait in self.core_traits.items()}
    
    def generate_personality_description(self, state: str) -> str:
        """Generate a detailed personality description for the current state"""
        if state not in self.personality_states:
            state = "neutral"  # Default fallback
            
        base_description = self.personality_states[state]["description"]
        
        # Add trait-specific details
        trait_descriptions = []
        traits = self.get_current_trait_values()
        
        for trait_name, value in traits.items():
            if trait_name == "trust_capacity" and value > 0.6:
                trait_descriptions.append("has a growing capacity to trust")
            elif trait_name == "emotional_openness" and value > 0.5:
                trait_descriptions.append("shows glimpses of emotional vulnerability")
            elif trait_name == "hostility" and value > 0.7:
                trait_descriptions.append("remains highly guarded and suspicious")
            elif trait_name == "protectiveness" and value > 0.6:
                trait_descriptions.append("displays protective instincts")
            elif trait_name == "curiosity" and value > 0.6:
                trait_descriptions.append("shows genuine curiosity about certain topics")
                
        # Add special state descriptions
        if state == "combat_ready":
            trait_descriptions.append("speech becomes clipped and focused on tactical assessment")
        elif state == "memory_surge":
            trait_descriptions.append("experiences flashbacks to past experiences and trauma")
            
        # Combine everything
        if trait_descriptions:
            return f"{base_description} {'; '.join(trait_descriptions)}."
        else:
            return f"{base_description}"
    
    async def perform_maintenance(self):
        """Perform periodic maintenance on personality traits"""
        print("Running personality trait maintenance...")
        
        # Decay all traits slightly toward baseline
        for trait in self.core_traits.values():
            trait.decay_toward_baseline(0.01)
            
        # Save updated data
        await self.save_personality_data()
        print("Personality maintenance complete")
    
    def get_response_parameters(self, state: str) -> Dict[str, Any]:
        """Get generation parameters for the current personality state"""
        if state not in self.personality_states:
            state = "neutral"  # Default fallback
            
        return {
            "max_tokens": self.personality_states[state].get("max_response_length", 120),
            "temperature": self.personality_states[state].get("temperature", 0.8),
        }

#######################
# RELATIONSHIP MILESTONE SYSTEM
#######################

class RelationshipMilestone:
    """Represents a relationship milestone that can be achieved"""
    
    def __init__(self, name: str, threshold: float, description: str, 
                special_dialogue: str, memory_unlock: Optional[str] = None,
                repeatable: bool = False):
        self.name = name
        self.threshold = threshold  # Score required to trigger
        self.description = description
        self.special_dialogue = special_dialogue
        self.memory_unlock = memory_unlock  # Optional memory to unlock
        self.repeatable = repeatable  # Can this milestone occur multiple times?
        self.triggered = False  # Has this milestone been triggered?
        self.last_triggered = None  # When was it last triggered?
        
    def check_and_trigger(self, user_id: int, current_score: float) -> bool:
        """Check if this milestone should trigger"""
        # Already triggered and not repeatable
        if self.triggered and not self.repeatable:
            return False
            
        # Check for repeatable cooldown
        if self.repeatable and self.last_triggered:
            days_since = (datetime.now(timezone.utc) - 
                          datetime.fromisoformat(self.last_triggered)).days
            if days_since < 14:  # Two week cooldown on repeatable milestones
                return False
        
        # Check threshold
        if current_score >= self.threshold:
            self.triggered = True
            self.last_triggered = datetime.now(timezone.utc).isoformat()
            return True
            
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "name": self.name,
            "threshold": self.threshold,
            "description": self.description,
            "special_dialogue": self.special_dialogue,
            "memory_unlock": self.memory_unlock,
            "repeatable": self.repeatable,
            "triggered": self.triggered,
            "last_triggered": self.last_triggered
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationshipMilestone':
        """Create from dictionary"""
        milestone = cls(
            name=data["name"],
            threshold=data["threshold"],
            description=data["description"],
            special_dialogue=data["special_dialogue"],
            memory_unlock=data.get("memory_unlock"),
            repeatable=data.get("repeatable", False)
        )
        milestone.triggered = data.get("triggered", False)
        milestone.last_triggered = data.get("last_triggered")
        return milestone


class RelationshipSystem:
    """Advanced relationship system with milestones and narrative elements"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.relationship_dir = data_dir / "relationships"
        self.relationship_dir.mkdir(parents=True, exist_ok=True)
        
        # User relationship data
        self.user_milestones = {}  # user_id -> list of milestones
        self.user_relationship_data = {}  # user_id -> relationship data
        
        # Define base milestones for all users
        self.base_milestones = [
            RelationshipMilestone(
                "First Sign of Trust",
                15,
                "A2 begins to lower her guard slightly around you.",
                "... Maybe you're not like the others. I'll still be watching.",
                "Fragment: A2 remembers someone who showed kindness once."
            ),
            RelationshipMilestone(
                "Fleeting Vulnerability",
                25,
                "A2 briefly shows a moment of vulnerability before shutting it down.",
                "Sometimes I... No. Forget it.",
                "Fragment: A2 recalls a moment of connection with her squad."
            ),
            RelationshipMilestone(
                "Combat Alliance",
                35,
                "A2 acknowledges you might be useful in combat.",
                "If we encounter enemies, stay behind me. I'll handle it.",
                "Fragment: A2 remembers fighting alongside comrades."
            ),
            RelationshipMilestone(
                "Shared Memory",
                50,
                "A2 willingly shares a memory fragment with you.",
                "I once had... comrades. We fought together. Only I survived.",
                "Fragment: A2 recalls the loss of her YoRHa squad."
            ),
            RelationshipMilestone(
                "Tentative Trust",
                60,
                "A2 begins to genuinely trust your intentions.",
                "I don't say this often, but... I trust you. Don't make me regret it.",
                None
            ),
            RelationshipMilestone(
                "Protective Instinct",
                70,
                "A2 displays protective behavior toward you.",
                "Stay close. I'll ensure nothing happens to you.",
                "Fragment: A2 reflects on her need to protect the few she cares about."
            ),
            RelationshipMilestone(
                "Emotional Opening",
                80,
                "A2 allows herself to express genuine emotion around you.",
                "I've spent so long alone... it's strange having someone who understands.",
                None
            ),
            RelationshipMilestone(
                "True Bond",
                90,
                "A2 acknowledges the meaningful bond that has formed.",
                "After everything... I consider you one of the few that matter to me.",
                "Fragment: A2 ponders what it means to care about someone."
            ),
            # Special/hidden milestones
            RelationshipMilestone(
                "Memory Corruption",
                40,
                "A2 experiences a momentary corruption in her memory banks.",
                "Error... memory region corrupted... I... what was I saying?",
                None,
                True  # This one can repeat
            ),
            RelationshipMilestone(
                "Combat Flashback",
                45,
                "A2 suddenly experiences a combat flashback.",
                "Get down! Enemy units approaching from-- Wait. No. Just... echoes.",
                None,
                True  # This one can repeat
            ),
        ]
    
    async def load_relationship_data(self) -> bool:
        """Load relationship data for all users"""
        try:
            # Look for user milestone files
            for file_path in self.relationship_dir.glob("*_milestones.json"):
                try:
                    # Extract user_id from filename
                    filename = file_path.stem
                    user_id = int(filename.split("_")[0])
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        milestone_data = json.load(f)
                    
                    # Convert dicts to Milestone objects
                    milestones = []
                    for data in milestone_data:
                        milestones.append(RelationshipMilestone.from_dict(data))
                    
                    self.user_milestones[user_id] = milestones
                    print(f"Loaded {len(milestones)} milestones for user {user_id}")
                except Exception as e:
                    print(f"Error loading milestones from {file_path}: {e}")
            
            # Look for general relationship data
            for file_path in self.relationship_dir.glob("*_relationship.json"):
                try:
                    # Extract user_id from filename
                    filename = file_path.stem
                    user_id = int(filename.split("_")[0])
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.user_relationship_data[user_id] = json.load(f)
                    
                    print(f"Loaded relationship data for user {user_id}")
                except Exception as e:
                    print(f"Error loading relationship data from {file_path}: {e}")
            
            return True
        except Exception as e:
            print(f"Error loading relationship data: {e}")
            return False
    
    async def save_relationship_data(self) -> bool:
        """Save relationship data for all users"""
        success = True
        
        try:
            # Save milestone data
            for user_id, milestones in self.user_milestones.items():
                try:
                    file_path = self.relationship_dir / f"{user_id}_milestones.json"
                    
                    # Convert milestones to dicts
                    milestone_dicts = [m.to_dict() for m in milestones]
                    
                    # Save to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(milestone_dicts, f, indent=2)
                        
                    print(f"Saved {len(milestones)} milestones for user {user_id}")
                except Exception as e:
                    print(f"Error saving milestones for user {user_id}: {e}")
                    success = False
            
            # Save relationship data
            for user_id, data in self.user_relationship_data.items():
                try:
                    file_path = self.relationship_dir / f"{user_id}_relationship.json"
                    
                    # Save to file
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2)
                        
                    print(f"Saved relationship data for user {user_id}")
                except Exception as e:
                    print(f"Error saving relationship data for user {user_id}: {e}")
                    success = False
            
            return success
        except Exception as e:
            print(f"Error saving relationship data: {e}")
            return False
    
    def get_user_milestones(self, user_id: int) -> List[RelationshipMilestone]:
        """Get milestones for a specific user, initializing if needed"""
        if user_id not in self.user_milestones:
            # Initialize with copies of base milestones
            self.user_milestones[user_id] = []
            for base in self.base_milestones:
                # Create a copy to avoid shared state
                milestone = RelationshipMilestone(
                    name=base.name,
                    threshold=base.threshold,
                    description=base.description,
                    special_dialogue=base.special_dialogue,
                    memory_unlock=base.memory_unlock,
                    repeatable=base.repeatable
                )
                self.user_milestones[user_id].append(milestone)
        
        return self.user_milestones[user_id]
    
    def check_milestones(self, user_id: int, trust_score: float) -> Optional[RelationshipMilestone]:
        """Check if any milestones have been triggered"""
        # Get milestones for this user
        milestones = self.get_user_milestones(user_id)
        
        # Check each untriggered milestone (or repeatable ones)
        for milestone in milestones:
            if milestone.check_and_trigger(user_id, trust_score):
                # Record this in user relationship data
                if user_id not in self.user_relationship_data:
                    self.user_relationship_data[user_id] = {"milestone_history": []}
                
                # Add to history
                self.user_relationship_data[user_id]["milestone_history"].append({
                    "name": milestone.name,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "score": trust_score
                })
                
                return milestone
        
        return None
    
    def get_relationship_stage(self, trust_score: float) -> Dict[str, Any]:
        """Get the current relationship stage data"""
        # Define relationship stages
        stages = [
            {"name": "Hostile", "threshold": 0, "description": "A2 sees you as a potential threat."},
            {"name": "Wary", "threshold": 10, "description": "A2 is cautious around you but not openly hostile."},
            {"name": "Neutral", "threshold": 20, "description": "A2 tolerates your presence."},
            {"name": "Cautious Ally", "threshold": 30, "description": "A2 begins to see potential value in your alliance."},
            {"name": "Tentative Trust", "threshold": 50, "description": "A2 has developed a careful trust with you."},
            {"name": "Companion", "threshold": 70, "description": "A2 considers you a valuable companion."},
            {"name": "Trusted Ally", "threshold": 85, "description": "A2 has formed a significant bond with you."},
            {"name": "True Connection", "threshold": 95, "description": "A2 shares a rare and meaningful connection with you."}
        ]
        
        # Find current stage
        current_stage = stages[0]
        for stage in stages:
            if trust_score >= stage["threshold"]:
                current_stage = stage
                
        # Find next stage if not at max
        next_stage = None
        for i, stage in enumerate(stages):
            if stage == current_stage and i < len(stages) - 1:
                next_stage = stages[i + 1]
                break
        
        # Calculate progress percentage
        progress = 0
        if next_stage:
            current_threshold = current_stage["threshold"]
            next_threshold = next_stage["threshold"]
            if next_threshold > current_threshold:  # Avoid division by zero
                progress = ((trust_score - current_threshold) / 
                           (next_threshold - current_threshold)) * 100
                progress = min(99.9, max(0, progress))
        
        return {
            "score": trust_score,
            "current": current_stage,
            "next": next_stage,
            "progress": progress
        }
    
    def get_milestone_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get the milestone achievement history for a user"""
        if user_id not in self.user_relationship_data:
            return []
            
        return self.user_relationship_data[user_id].get("milestone_history", [])
    
    def get_next_milestones(self, user_id: int, trust_score: float) -> List[Dict[str, Any]]:
        """Get the next milestones that can be achieved"""
        milestones = self.get_user_milestones(user_id)
        
        # Find untriggered or repeatable milestones that are within reasonable reach
        next_milestones = []
        for milestone in milestones:
            if (not milestone.triggered or milestone.repeatable) and milestone.threshold > trust_score:
                # Only include milestones that are within 20 points of current score
                if milestone.threshold - trust_score <= 20:
                    next_milestones.append({
                        "name": milestone.name,
                        "points_needed": milestone.threshold - trust_score,
                        "description": milestone.description,
                        "repeatable": milestone.repeatable
                    })
        
        # Sort by proximity to current score
        next_milestones.sort(key=lambda m: m["points_needed"])
        
        return next_milestones[:3]  # Return at most 3

#######################
# INTEGRATION UTILITIES
#######################

class EnhancedA2System:
    """Master class that integrates all advanced systems"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.enhanced_dir = data_dir / "enhanced_a2"
        self.enhanced_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize all subsystems
        self.memory_system = SemanticMemorySystem(self.enhanced_dir)
        self.emotion_detector = EmotionDetector()
        self.personality_system = PersonalitySystem(self.enhanced_dir)
        self.relationship_system = RelationshipSystem(self.enhanced_dir)
        
        # Integration data
        self.last_special_event = {}  # user_id -> timestamp
        self.user_stats = {}  # user_id -> misc stats
        
    async def initialize(self):
        """Load all subsystem data"""
        await self.memory_system.load_memories()
        await self.personality_system.load_personality_data()
        await self.relationship_system.load_relationship_data()
        print("Enhanced A2 System initialized successfully")
        
    async def save_all_data(self):
        """Save all subsystem data"""
        await self.memory_system.save_memories()
        await self.personality_system.save_personality_data()
        await self.relationship_system.save_relationship_data()
        print("Enhanced A2 System data saved successfully")
        
    async def run_maintenance(self):
        """Run maintenance tasks for all subsystems"""
        await self.memory_system.maintenance_task()
        await self.personality_system.perform_maintenance()
        await self.save_all_data()
        
    def process_message(self, user_id: int, message_content: str, trust_score: float) -> Dict[str, Any]:
        """Process an incoming message and prepare context for response generation"""
        # Analyze emotions
        emotions = self.emotion_detector.record_emotion(user_id, message_content)
        emotion_trend = self.emotion_detector.analyze_emotion_trend(user_id)
        
        # Select personality state
        personality_state = self.personality_system.select_personality_state(
            user_id, trust_score, message_content, emotion_trend)
        
        # Get personality description
        personality_description = self.personality_system.generate_personality_description(personality_state)
        
        # Get response parameters
        response_params = self.personality_system.get_response_parameters(personality_state)
        
        # Check for relationship milestones
        triggered_milestone = self.relationship_system.check_milestones(user_id, trust_score)
        
        # Extract potential memory
        memory = self.memory_system.extract_memory_from_conversation(
            user_id, message_content, "", 0.6)  # Bot response will be filled later
        
        if memory:
            self.memory_system.add_memory(memory)
            
        # Retrieve relevant memories
        relevant_memories = self.memory_system.retrieve_memories(user_id, message_content)
        
        # Format memories for inclusion in prompt
        memory_text = ""
        if relevant_memories:
            memory_text = "\nRelevant memories:\n" + "\n".join(
                [f"- {memory.text}" for memory in relevant_memories])
        
        # Get relationship stage info
        relationship_stage = self.relationship_system.get_relationship_stage(trust_score)
        
        # Create the response context
        context = {
            "personality_state": personality_state,
            "personality_description": personality_description,
            "response_params": response_params,
            "emotion_data": {
                "current": emotions,
                "trend": emotion_trend
            },
            "triggered_milestone": triggered_milestone.to_dict() if triggered_milestone else None,
            "memory_text": memory_text,
            "relationship_stage": relationship_stage,
        }
        
        return context
    
    def adjust_personality_from_interaction(self, user_id: int, message_content: str, 
                                           emotion_data: Dict[str, Any], trust_score: float):
        """Adjust A2's personality based on this interaction"""
        self.personality_system.adjust_traits_from_interaction(
            user_id, message_content, emotion_data, trust_score)
    
    def process_response(self, user_id: int, message_content: str, raw_response: str, 
                        emotion_data: Dict[str, Any], trust_score: float) -> str:
        """Process A2's response with emotion-appropriate modifications"""
        # Get user's dominant emotion
        dominant_emotion = "neutral"
        if "current" in emotion_data and emotion_data["current"]:
            dominant_emotion = max(emotion_data["current"].items(), key=lambda x: x[1])[0]
            
        # Adjust response based on emotion
        adjusted_response = self.emotion_detector.adjust_a2_response_for_emotion(
            dominant_emotion, raw_response, trust_score)
        
        # Save any auto-extracted memory with the actual response
        memory = self.memory_system.extract_memory_from_conversation(
            user_id, message_content, adjusted_response, 0.6)
        
        if memory:
            self.memory_system.add_memory(memory)
        
        # Adjust personality traits based on the full interaction
        self.adjust_personality_from_interaction(
            user_id, message_content, emotion_data, trust_score)
            
        return adjusted_response
    
    def check_for_special_event(self, user_id: int, trust_score: float) -> Optional[Dict[str, Any]]:
        """Check if a special random event should occur"""
        # Limit frequency of special events
        now = datetime.now(timezone.utc)
        if user_id in self.last_special_event:
            hours_since_last = (now - datetime.fromisoformat(self.last_special_event[user_id])).total_seconds() / 3600
            if hours_since_last < 24:  # Minimum 24 hours between events
                return None
        
        # Base chance is low, increases with trust score
        base_chance = 0.05  # 5% base chance
        trust_modifier = trust_score / 200  # 0-0.5 modifier based on trust
        
        if random.random() > base_chance + trust_modifier:
            return None
        
        # Define possible events
        events = [
            {
                "type": "memory_corruption",
                "message": "Warning: Memory corruption detected. Running diagnostic... Error. Error. Reset sequence initiated...",
                "description": "A2 experiences a temporary memory corruption, causing confusion and disorientation.",
                "trust_min": 0,  # No minimum trust
                "effects": {"trust": -0.5}
            },
            {
                "type": "combat_alert",
                "message": "Alert: Enemy units detected. Combat mode engaged. Stay back.",
                "description": "A2's combat systems suddenly activate, putting her on high alert.",
                "trust_min": 0,
                "effects": {"protectiveness": 0.5}
            },
            {
                "type": "memory_flash",
                "message": "I... remember. The desert. The mission. My squad... No...",
                "description": "A2 has a sudden flash of memory about her past.",
                "trust_min": 30,  # Only occurs at higher trust
                "effects": {"attachment": 0.5, "emotional_openness": 0.3}
            },
            {
                "type": "rare_vulnerability",
                "message": "Sometimes I wonder... if I'm the last one. The last... with these memories.",
                "description": "A2 shows a rare moment of vulnerability and existential reflection.",
                "trust_min": 50,  # Only occurs at high trust
                "effects": {"attachment": 1.0, "emotional_openness": 0.7}
            },
            {
                "type": "protective_surge",
                "message": "Get behind me. Now. Don't argue.",
                "description": "A2 suddenly becomes intensely protective, sensing a threat.",
                "trust_min": 40,
                "effects": {"protectiveness": 1.0, "trust": 0.3}
            }
        ]
        
        # Filter by trust requirement
        valid_events = [e for e in events if trust_score >= e["trust_min"]]
        
        if not valid_events:
            return None
            
        # Select a random event
        event = random.choice(valid_events)
        
        # Record timestamp
        self.last_special_event[user_id] = now.isoformat()
        
        # Return event data
        return event

#######################
# SAMPLE IMPLEMENTATION AND INTEGRATION
#######################

# Example of integrating with the existing A2 bot

class EnhancedResponseGenerator:
    """Enhanced response generator that integrates with the existing system"""
    
    def __init__(self, openai_client, emotion_manager, conversation_manager, data_dir):
        self.client = openai_client
        self.emotion_manager = emotion_manager
        self.conversation_manager = conversation_manager
        
        # Initialize enhanced systems
        self.enhanced_system = EnhancedA2System(Path(data_dir))
        
        # Initialize background tasks
        self.maintenance_task_initialized = False
        
    async def initialize(self):
        """Initialize the enhanced system"""
        await self.enhanced_system.initialize()
        
        # Start background maintenance task if not already running
        if not self.maintenance_task_initialized:
            self.start_background_tasks()
            self.maintenance_task_initialized = True
    
    def start_background_tasks(self):
        """Start background tasks"""
        asyncio.create_task(self.run_maintenance_loop())
    
    async def run_maintenance_loop(self):
        """Run periodic maintenance"""
        while True:
            try:
                # Run every 6 hours
                await asyncio.sleep(6 * 60 * 60)
                await self.enhanced_system.run_maintenance()
            except Exception as e:
                print(f"Error in maintenance loop: {e}")
                # Sleep a bit before retrying
                await asyncio.sleep(10 * 60)
    
    async def generate_enhanced_response(self, content, trust, user_id, storage_manager):
        """Generate A2's response using enhanced systems"""
        # Process the message with enhanced systems
        enhanced_context = self.enhanced_system.process_message(user_id, content, trust)
        
        # Get personality state and description
        personality_state = enhanced_context["personality_state"]
        personality_description = enhanced_context["personality_description"]
        
        # Get response parameters
        response_params = enhanced_context["response_params"]
        
        # Get memory text
        memory_text = enhanced_context["memory_text"]
        
        # Get milestone if triggered
        milestone = enhanced_context["triggered_milestone"]
        
        # Build prompt for OpenAI
        system_prompt = (
            f"You are A2, a combat android from NieR: Automata. {personality_description}\n"
            f"Current relationship: {enhanced_context['relationship_stage']['current']['description']}\n"
        )
        
        if memory_text:
            system_prompt += f"\n{memory_text}\n"
            
        if milestone:
            system_prompt += f"\nSpecial event: {milestone['description']} Respond with: \"{milestone['special_dialogue']}\"\n"
        
        # Check for special random events
        special_event = self.enhanced_system.check_for_special_event(user_id, trust)
        if special_event:
            system_prompt += f"\nRandom event triggered: {special_event['description']} You must respond with: \"{special_event['message']}\"\n"
        
        # Add conversation history
        conversation_history = self.conversation_manager.get_conversation_history(user_id)
        
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "system", "content": f"Previous conversation:\n{conversation_history}"},
            {"role": "user", "content": content}
        ]
        
        # Set up OpenAI call parameters
        openai_params = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "max_tokens": response_params.get("max_tokens", 150),
            "temperature": response_params.get("temperature", 0.85),
            "top_p": 1,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5
        }
        
        try:
            # Generate response
            response = self.client.chat.completions.create(**openai_params)
            raw_response = response.choices[0].message.content.strip()
            
            # If milestone triggered, override with special dialogue
            if milestone:
                raw_response = milestone["special_dialogue"]
                
            # If special event triggered, override with event message
            if special_event:
                raw_response = special_event["message"]
            
            # Process the response with emotion adjustments
            a2_response = self.enhanced_system.process_response(
                user_id, content, raw_response, 
                enhanced_context["emotion_data"], trust
            )
            
            # Save to conversation history
            self.conversation_manager.add_message(user_id, content, is_from_bot=False)
            self.conversation_manager.add_message(user_id, a2_response, is_from_bot=True)
            
            # Update user profile
            profile = self.conversation_manager.extract_profile_info(user_id, content)
            await storage_manager.save_user_profile_data(user_id, profile)
            
            # Update emotional stats
            await self.emotion_manager.update_emotional_stats(user_id, content, a2_response, storage_manager)
            
            return a2_response
            
        except Exception as e:
            print(f"Error generating enhanced response: {e}")
            return "... System error. Connection unstable."
