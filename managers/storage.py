python"""
Storage manager for the A2 Discord bot.
"""
import json
import asyncio
from datetime import datetime, timezone
from collections import Counter
from pathlib import Path

class StorageManager:
    """Handles all data persistence operations"""
    
    def __init__(self, data_dir, users_dir, profiles_dir, dm_settings_file, user_profiles_dir, conversations_dir):
        self.data_dir = data_dir
        self.users_dir = users_dir
        self.profiles_dir = profiles_dir
        self.dm_settings_file = dm_settings_file
        self.user_profiles_dir = user_profiles_dir
        self.conversations_dir = conversations_dir
        
        # Ensure directories exist
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self.user_profiles_dir.mkdir(parents=True, exist_ok=True)
        self.conversations_dir.mkdir(parents=True, exist_ok=True)
        
    def verify_data_directories(self):
        """Ensure all required data directories exist and are writable"""
        print(f"Data directory: {self.data_dir}")
        print(f"Directory exists: {self.data_dir.exists()}")
        
        # Check data directory
        if not self.data_dir.exists():
            try:
                self.data_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created data directory: {self.data_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create data directory: {e}")
                return False
        
        # Check users directory
        if not self.users_dir.exists():
            try:
                self.users_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created users directory: {self.users_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create users directory: {e}")
                return False
        
        # Check profiles directory
        if not self.profiles_dir.exists():
            try:
                self.profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created profiles directory: {self.profiles_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create profiles directory: {e}")
                return False
        
        # Check user profiles directory
        if not self.user_profiles_dir.exists():
            try:
                self.user_profiles_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created user profiles directory: {self.user_profiles_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create user profiles directory: {e}")
                return False
        
        # Check conversations directory
        if not self.conversations_dir.exists():
            try:
                self.conversations_dir.mkdir(parents=True, exist_ok=True)
                print(f"Created conversations directory: {self.conversations_dir}")
            except Exception as e:
                print(f"ERROR: Failed to create conversations directory: {e}")
                return False
        
        # Check write access
        try:
            test_file = self.data_dir / "write_test.tmp"
            test_file.write_text("Test write access", encoding="utf-8")
            test_file.unlink()  # Remove test file
            print("Write access verified: SUCCESS")
        except Exception as e:
            print(f"ERROR: Failed to verify write access: {e}")
            return False
        
        return True
        
    async def save_file(self, path, data, temp_suffix='.tmp'):
        """Helper function to safely save a file using atomic write"""
        try:
            # Create a temporary file
            temp_path = path.with_suffix(temp_suffix)
            temp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            
            # Use atomic rename operation
            if temp_path.exists():
                temp_path.replace(path)
                return True
        except Exception as e:
            print(f"Error saving file {path}: {e}")
        return False
        
    async def load_user_profile(self, user_id, emotion_manager):
        """Load user profile data with enhanced stats and error handling"""
        profile_path = self.profiles_dir / f"{user_id}.json"
        
        # Load main profile
        if profile_path.exists():
            try:
                file_content = profile_path.read_text(encoding="utf-8")
                if not file_content.strip():
                    print(f"Warning: Empty profile file for user {user_id}")
                    return {}
                    
                data = json.loads(file_content)
                print(f"Successfully loaded profile for user {user_id}")
                
                # Extract relationship data if present
                if "relationship" in data:
                    emotion_manager.relationship_progress[user_id] = data.pop("relationship")
                
                # Extract interaction stats if present
                if "interaction_stats" in data:
                    emotion_manager.interaction_stats[user_id] = Counter(data.pop("interaction_stats", {}))
                
                return data
            except Exception as e:
                print(f"Error loading profile for user {user_id}: {e}")
        
        return {}
        
    async def save_user_profile(self, user_id, emotion_manager):
        """Save user profile data with enhanced stats and error handling"""
        try:
            # Ensure directory exists
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            
            path = self.profiles_dir / f"{user_id}.json"
            
            # Prepare data to save
            data = emotion_manager.user_emotions.get(user_id, {})
            
            # Add extra data
            data["relationship"] = emotion_manager.relationship_progress.get(user_id, {})
            data["interaction_stats"] = dict(emotion_manager.interaction_stats.get(user_id, Counter()))
            
            # Save main profile
            success = await self.save_file(path, data)
            if success:
                print(f"Successfully saved profile for user {user_id}")
            
            # Save memories if they exist
            if user_id in emotion_manager.user_memories and emotion_manager.user_memories[user_id]:
                memory_path = self.profiles_dir / f"{user_id}_memories.json"
                mem_success = await self.save_file(memory_path, emotion_manager.user_memories[user_id])
                if mem_success:
                    print(f"Saved {len(emotion_manager.user_memories[user_id])} memories for user {user_id}")
            
            # Save events if they exist
            if user_id in emotion_manager.user_events and emotion_manager.user_events[user_id]:
                events_path = self.profiles_dir / f"{user_id}_events.json"
                evt_success = await self.save_file(events_path, emotion_manager.user_events[user_id])
                if evt_success:
                    print(f"Saved {len(emotion_manager.user_events[user_id])} events for user {user_id}")
            
            # Save milestones if they exist
            if user_id in emotion_manager.user_milestones and emotion_manager.user_milestones[user_id]:
                milestones_path = self.profiles_dir / f"{user_id}_milestones.json"
                mile_success = await self.save_file(milestones_path, emotion_manager.user_milestones[user_id])
                if mile_success:
                    print(f"Saved {len(emotion_manager.user_milestones[user_id])} milestones for user {user_id}")
                    
            return True
        except Exception as e:
            print(f"Error saving data for user {user_id}: {e}")
            return False
    
    async def save_conversation(self, user_id, conversation_manager):
        """Save conversation history and summary"""
        try:
            # Save conversation history
            if user_id in conversation_manager.conversations:
                conv_path = self.conversations_dir / f"{user_id}_conversations.json"
                await self.save_file(conv_path, conversation_manager.conversations[user_id])
                
            # Save conversation summary
            if user_id in conversation_manager.conversation_summaries:
                summary_path = self.conversations_dir / f"{user_id}_summary.json"
                await self.save_file(summary_path, {
                    "summary": conversation_manager.conversation_summaries[user_id],
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                
            return True
        except Exception as e:
            print(f"Error saving conversation for user {user_id}: {e}")
            return False
    
    async def load_conversation(self, user_id, conversation_manager):
        """Load conversation history and summary"""
        try:
            # Load conversation history
            conv_path = self.conversations_dir / f"{user_id}_conversations.json"
            if conv_path.exists():
                file_content = conv_path.read_text(encoding="utf-8")
                if file_content.strip():
                    conversation_manager.conversations[user_id] = json.loads(file_content)
            
            # Load conversation summary
            summary_path = self.conversations_dir / f"{user_id}_summary.json"
            if summary_path.exists():
                file_content = summary_path.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    conversation_manager.conversation_summaries[user_id] = data.get("summary", "")
            
            return True
        except Exception as e:
            print(f"Error loading conversation for user {user_id}: {e}")
            return False
    
    async def save_user_profile_data(self, user_id, profile):
        """Save user profile data"""
        try:
            profile_path = self.user_profiles_dir / f"{user_id}_profile.json"
            await self.save_file(profile_path, profile.to_dict())
            return True
        except Exception as e:
            print(f"Error saving user profile for {user_id}: {e}")
            return False
    
    async def load_user_profile_data(self, user_id, conversation_manager):
        """Load user profile data"""
        try:
            profile_path = self.user_profiles_dir / f"{user_id}_profile.json"
            if profile_path.exists():
                file_content = profile_path.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    profile = conversation_manager.user_profiles[user_id].__class__.from_dict(data)
                    conversation_manager.user_profiles[user_id] = profile
                    return True
            return False
        except Exception as e:
            print(f"Error loading user profile for {user_id}: {e}")
            return False
            
    async def load_dm_settings(self):
        """Load DM permission settings"""
        dm_enabled_users = set()
        try:
            if self.dm_settings_file.exists():
                file_content = self.dm_settings_file.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    dm_enabled_users = set(data.get('enabled_users', []))
                    print(f"Loaded DM settings for {len(dm_enabled_users)} users")
                else:
                    print("Warning: Empty DM settings file")
            else:
                print("No DM settings file found")
        except Exception as e:
            print(f"Error loading DM settings: {e}")
        return dm_enabled_users
        
    async def save_dm_settings(self, dm_enabled_users):
        """Save DM permission settings"""
        return await self.save_file(self.dm_settings_file, {"enabled_users": list(dm_enabled_users)})
    
    async def save_data(self, emotion_manager, conversation_manager=None):
        """Save all emotional and conversation data"""
        success = True
        
        # Save emotional data for all users
        for user_id in emotion_manager.user_emotions:
            profile_success = await self.save_user_profile(user_id, emotion_manager)
            success = success and profile_success
            
            # Save conversation data if provided
            if conversation_manager and user_id in conversation_manager.conversations:
                conv_success = await self.save_conversation(user_id, conversation_manager)
                success = success and conv_success
                
                # Save user profile data
                if user_id in conversation_manager.user_profiles:
                    profile_success = await self.save_user_profile_data(
                        user_id, conversation_manager.user_profiles[user_id])
                    success = success and profile_success
        
        # Save DM settings
        dm_success = await self.save_dm_settings(emotion_manager.dm_enabled_users)
        success = success and dm_success
        
        print(f"Data save complete for {len(emotion_manager.user_emotions)} users")
        return success
    
    async def load_data(self, emotion_manager, conversation_manager):
        """Load all user data with improved error handling"""
        # Initialize containers
        emotion_manager.user_emotions = {}
        emotion_manager.user_memories = defaultdict(list)
        emotion_manager.user_events = defaultdict(list)
        emotion_manager.user_milestones = defaultdict(list)
        emotion_manager.interaction_stats = defaultdict(Counter)
        emotion_manager.relationship_progress = defaultdict(dict)
        
        # Ensure directories exist
        if not self.verify_data_directories():
            print("ERROR: Data directories not available. Memory functions disabled.")
            return False
        
        print("Beginning data load process...")
        
        # Load profile data
        profile_count = 0
        error_count = 0
        for file in self.profiles_dir.glob("*.json"):
            if "_" not in file.stem:  # Skip special files like _memories.json
                try:
                    uid = int(file.stem)
                    file_content = file.read_text(encoding="utf-8")
                    if not file_content.strip():
                        print(f"Warning: Empty file {file}")
                        continue
                        
                    data = json.loads(file_content)
                    emotion_manager.user_emotions[uid] = data
                    
                    # Extract relationship data if present
                    if "relationship" in data:
                        emotion_manager.relationship_progress[uid] = data.get("relationship", {})
                    
                    # Extract interaction stats if present
                    if "interaction_stats" in data:
                        emotion_manager.interaction_stats[uid] = Counter(data.get("interaction_stats", {}))
                        
                    profile_count += 1
                except Exception as e:
                    error_count += 1
                    print(f"Error loading profile {file}: {e}")
        
        print(f"Loaded {profile_count} profiles with {error_count} errors")
        
        # Load memories data
        memory_count = 0
        for file in self.profiles_dir.glob("*_memories.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_memories[uid] = json.loads(file_content)
                    memory_count += 1
            except Exception as e:
                print(f"Error loading memories {file}: {e}")
        
        # Load events data
        events_count = 0
        for file in self.profiles_dir.glob("*_events.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_events[uid] = json.loads(file_content)
                    events_count += 1
            except Exception as e:
                print(f"Error loading events {file}: {e}")
        
        # Load milestones data
        milestones_count = 0
        for file in self.profiles_dir.glob("*_milestones.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    emotion_manager.user_milestones[uid] = json.loads(file_content)
                    milestones_count += 1
            except Exception as e:
                print(f"Error loading milestones {file}: {e}")
        
        # Load user profiles
        profile_count = 0
        for file in self.user_profiles_dir.glob("*_profile.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    profile = conversation_manager.user_profiles[uid].__class__.from_dict(data)
                    conversation_manager.user_profiles[uid] = profile
                    profile_count += 1
            except Exception as e:
                print(f"Error loading user profile {file}: {e}")
        print(f"Loaded {profile_count} user profiles")

        # Load conversation data
        conversation_count = 0
        for file in self.conversations_dir.glob("*_conversations.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    conversation_manager.conversations[uid] = json.loads(file_content)
                    conversation_count += 1
            except Exception as e:
                print(f"Error loading conversation {file}: {e}")

        # Load conversation summaries
        summary_count = 0
        for file in self.conversations_dir.glob("*_summary.json"):
            try:
                uid = int(file.stem.split("_")[0])
                file_content = file.read_text(encoding="utf-8")
                if file_content.strip():
                    data = json.loads(file_content)
                    conversation_manager.conversation_summaries[uid] = data.get("summary", "")
                    summary_count += 1
            except Exception as e:
                print(f"Error loading conversation summary {file}: {e}")

        print(f"Loaded {conversation_count} conversations and {summary_count} summaries")

        print(f"Loaded {memory_count} memory files, {events_count} event files, {milestones_count} milestone files")
        
        # Add any missing fields to existing user data
        for uid in emotion_manager.user_emotions:
            if "first_interaction" not in emotion_manager.user_emotions[uid]:
                emotion_manager.user_emotions[uid]["first_interaction"] = emotion_manager.user_emotions[uid].get(
                    "last_interaction", datetime.now(timezone.utc).isoformat())
        
        # Load DM settings
        emotion_manager.dm_enabled_users = await self.load_dm_settings()
        
        print("Data load complete")
        return profile_count > 0
