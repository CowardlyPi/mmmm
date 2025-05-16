"""
Admin commands for the A2 Discord bot.
"""
import os
import sys
import time
import random
import discord
import subprocess
from collections import defaultdict, Counter
from datetime import datetime, timezone, timedelta
from discord.ext import commands
from pathlib import Path

# Import configuration
from config import DATA_DIR

def setup_admin_commands(bot, emotion_manager, conversation_manager, storage_manager):
    """Set up commands that require administrator permissions"""
    
    @bot.command(name="set_stat")
    @commands.has_permissions(administrator=True)
    async def set_stat(ctx, stat_name: str, value: float):
        """Admin command to set a specific emotional stat (for testing only)"""
        uid = ctx.author.id
        
        # Initialize user if they don't exist in the system
        if uid not in emotion_manager.user_emotions:
            emotion_manager.user_emotions[uid] = {
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
        
        # Validate the stat name
        valid_stats = ["trust", "resentment", "attachment", "protectiveness", 
                      "affection_points", "annoyance"]
        
        if stat_name not in valid_stats:
            await ctx.send(f"A2: Invalid stat name. Valid stats are: {', '.join(valid_stats)}")
            return
        
        # Apply appropriate limits based on the stat
        if stat_name == "affection_points":
            value = max(-100, min(1000, value))
        elif stat_name == "annoyance":
            value = max(0, min(100, value))
        else:
            value = max(0, min(10, value))
        
        # Update the stat
        emotion_manager.user_emotions[uid][stat_name] = value
        
        # Save the changes
        await storage_manager.save_data(emotion_manager, conversation_manager)
        
        await ctx.send(f"A2: Successfully set your {stat_name} to {value}.")
        
        # Show the updated relationship stage
        rel_data = emotion_manager.get_relationship_stage(uid)
        await ctx.send(f"A2: Your relationship is now at the '{rel_data['current']['name']}' stage.")

    @bot.command(name="memory_check")
    @commands.has_permissions(administrator=True)
    async def check_memory(ctx, user_id: discord.Member = None):
        """Check if a user has memory data loaded"""
        target_id = user_id.id if user_id else ctx.author.id
        
        results = []
        results.append(f"**Memory Check for User ID: {target_id}**")
        results.append(f"Emotional data: **{'YES' if target_id in emotion_manager.user_emotions else 'NO'}**")
        results.append(f"Memories: **{'YES' if target_id in emotion_manager.user_memories else 'NO'}**")
        results.append(f"Events: **{'YES' if target_id in emotion_manager.user_events else 'NO'}**")
        results.append(f"Milestones: **{'YES' if target_id in emotion_manager.user_milestones else 'NO'}**")
        results.append(f"Profile: **{'YES' if target_id in conversation_manager.user_profiles else 'NO'}**")
        results.append(f"Conversation: **{'YES' if target_id in conversation_manager.conversations else 'NO'}**")
        
        # Check file existence
        profile_path = storage_manager.profiles_dir / f"{target_id}.json"
        memory_path = storage_manager.profiles_dir / f"{target_id}_memories.json"
        events_path = storage_manager.profiles_dir / f"{target_id}_events.json"
        milestones_path = storage_manager.profiles_dir / f"{target_id}_milestones.json"
        user_profile_path = storage_manager.user_profiles_dir / f"{target_id}_profile.json"
        conv_path = storage_manager.conversations_dir / f"{target_id}_conversations.json"
        
        results.append(f"Profile file exists: **{'YES' if profile_path.exists() else 'NO'}**")
        results.append(f"Memory file exists: **{'YES' if memory_path.exists() else 'NO'}**")
        results.append(f"Events file exists: **{'YES' if events_path.exists() else 'NO'}**")
        results.append(f"Milestones file exists: **{'YES' if milestones_path.exists() else 'NO'}**")
        results.append(f"User profile file exists: **{'YES' if user_profile_path.exists() else 'NO'}**")
        results.append(f"Conversation file exists: **{'YES' if conv_path.exists() else 'NO'}**")
        
        # Count memory items
        memory_count = len(emotion_manager.user_memories.get(target_id, []))
        event_count = len(emotion_manager.user_events.get(target_id, []))
        milestone_count = len(emotion_manager.user_milestones.get(target_id, []))
        conversation_count = len(conversation_manager.conversations.get(target_id, []))
        
        results.append(f"Memory count: **{memory_count}**")
        results.append(f"Event count: **{event_count}**")
        results.append(f"Milestone count: **{milestone_count}**")
        results.append(f"Conversation messages: **{conversation_count}**")
        
        # Profile summary
        if target_id in conversation_manager.user_profiles:
            profile = conversation_manager.user_profiles[target_id]
            results.append(f"Profile summary: **{profile.get_summary()}**")
        
        await ctx.send("\n".join(results))

    @bot.command(name="reset")
    @commands.has_permissions(administrator=True)
    async def reset_stats(ctx, user_id: discord.Member = None):
        """Admin command to reset a user's stats"""
        target_id = user_id.id if user_id else ctx.author.id
        
        if target_id in emotion_manager.user_emotions:
            del emotion_manager.user_emotions[target_id]
        if target_id in emotion_manager.user_memories:
            del emotion_manager.user_memories[target_id]
        if target_id in emotion_manager.user_events:
            del emotion_manager.user_events[target_id]
        if target_id in emotion_manager.user_milestones:
            del emotion_manager.user_milestones[target_id]
        if target_id in emotion_manager.interaction_stats:
            del emotion_manager.interaction_stats[target_id]
        if target_id in emotion_manager.relationship_progress:
            del emotion_manager.relationship_progress[target_id]
        if target_id in conversation_manager.conversations:
            del conversation_manager.conversations[target_id]
        if target_id in conversation_manager.conversation_summaries:
            del conversation_manager.conversation_summaries[target_id]
        if target_id in conversation_manager.user_profiles:
            del conversation_manager.user_profiles[target_id]
        
        # Delete files
        profile_path = storage_manager.profiles_dir / f"{target_id}.json"
        memory_path = storage_manager.profiles_dir / f"{target_id}_memories.json"
        events_path = storage_manager.profiles_dir / f"{target_id}_events.json"
        milestones_path = storage_manager.profiles_dir / f"{target_id}_milestones.json"
        user_profile_path = storage_manager.user_profiles_dir / f"{target_id}_profile.json"
        conv_path = storage_manager.conversations_dir / f"{target_id}_conversations.json"
        summary_path = storage_manager.conversations_dir / f"{target_id}_summary.json"
        
        for path in [profile_path, memory_path, events_path, milestones_path, user_profile_path, conv_path, summary_path]:
            if path.exists():
                path.unlink()
        
        await ctx.send(f"A2: Stats reset for user ID {target_id}.")
        await storage_manager.save_data(emotion_manager, conversation_manager)

    @bot.command(name="debug_info")
    @commands.has_permissions(administrator=True)
    async def debug_info(ctx):
        """Display technical debug information about the bot"""
        info = []
        info.append(f"**A2 Bot Debug Information**")
        info.append(f"Python version: {sys.version.split()[0]}")
        info.append(f"Discord.py version: {discord.__version__}")
        info.append(f"Bot uptime: {datetime.now(timezone.utc) - bot.start_time}")
        info.append(f"Connected servers: {len(bot.guilds)}")
        info.append(f"Total users tracked: {len(emotion_manager.user_emotions)}")
        info.append(f"Total conversations: {len(conversation_manager.conversations)}")
        info.append(f"Total profiles: {len(conversation_manager.user_profiles)}")
        info.append(f"Data directory: {storage_manager.data_dir}")
        
        # Memory usage information
        import psutil
        process = psutil.Process(os.getpid())
        memory_use = process.memory_info().rss / 1024 / 1024  # Convert to MB
        info.append(f"Memory usage: {memory_use:.2f} MB")
        
        await ctx.send("\n".join(info))

    @bot.command(name="ping")
    @commands.has_permissions(administrator=True)
    async def ping(ctx):
        """Check bot latency"""
        start_time = time.time()
        message = await ctx.send("Pinging...")
        end_time = time.time()
        
        # Calculate round-trip latency
        latency = (end_time - start_time) * 1000
        # Get WebSocket latency
        ws_latency = bot.latency * 1000
        
        await message.edit(content=f"Pong! Bot latency: {latency:.2f}ms | WebSocket: {ws_latency:.2f}ms")

    @bot.command(name="list_users")
    @commands.has_permissions(administrator=True)
    async def list_users(ctx, count: int = 10):
        """List users the bot has data for"""
        if not emotion_manager.user_emotions:
            await ctx.send("No user data found.")
            return
        
        users = []
        for i, (user_id, data) in enumerate(list(emotion_manager.user_emotions.items())[:count]):
            # Try to get username
            user = bot.get_user(user_id)
            username = user.name if user else f"Unknown User ({user_id})"
            
            # Get relationship score
            score = emotion_manager.get_relationship_score(user_id)
            emotions = f"Trust: {data.get('trust', 0):.1f}, Attach: {data.get('attachment', 0):.1f}"
            
            users.append(f"{i+1}. **{username}** - Score: {score:.1f} | {emotions}")
        
        embed = discord.Embed(
            title="Users with Emotional Data",
            description=f"Showing {len(users)} of {len(emotion_manager.user_emotions)} users",
            color=discord.Color.blue()
        )
        
        embed.add_field(name="Users", value="\n".join(users) or "No users found", inline=False)
        await ctx.send(embed=embed)

    @bot.command(name="inspect_user")
    @commands.has_permissions(administrator=True)
    async def inspect_user(ctx, user_id: discord.Member = None):
        """Get detailed debug info for a specific user"""
        if not user_id:
            user_id = ctx.author
        
        uid = user_id.id
        if uid not in emotion_manager.user_emotions:
            await ctx.send(f"No data found for user {user_id.name} ({uid})")
            return
        
        # Get all the data
        emotional_data = emotion_manager.user_emotions.get(uid, {})
        memories = emotion_manager.user_memories.get(uid, [])
        events = emotion_manager.user_events.get(uid, [])
        milestones = emotion_manager.user_milestones.get(uid, [])
        interactions = emotion_manager.interaction_stats.get(uid, Counter())
        profile = conversation_manager.user_profiles.get(uid, None)
        
        # Create a comprehensive embed
        embed = discord.Embed(
            title=f"Debug Data: {user_id.name}",
            description=f"User ID: {uid}",
            color=discord.Color.dark_gold()
        )
        
        # Add emotional stats
        emotions_text = "\n".join([f"**{k}**: {v}" for k, v in emotional_data.items() 
                                 if not isinstance(v, (list, dict))])
        embed.add_field(name="Emotional Stats", value=emotions_text or "No data", inline=False)
        
        # Add counts
        counts = [
            f"Memories: {len(memories)}",
            f"Events: {len(events)}",
            f"Milestones: {len(milestones)}",
            f"Interactions: {interactions.get('total', 0)}",
            f"Conversations: {len(conversation_manager.conversations.get(uid, []))}"
        ]
        embed.add_field(name="Data Counts", value="\n".join(counts), inline=False)
        
        # Add profile summary if available
        if profile:
            embed.add_field(name="Profile", value=profile.get_summary() or "No profile data", inline=False)
        
        # Add file stats
        file_paths = [
            (storage_manager.profiles_dir / f"{uid}.json", "Emotions file"),
            (storage_manager.profiles_dir / f"{uid}_memories.json", "Memories file"),
            (storage_manager.profiles_dir / f"{uid}_events.json", "Events file"),
            (storage_manager.profiles_dir / f"{uid}_milestones.json", "Milestones file"),
            (storage_manager.user_profiles_dir / f"{uid}_profile.json", "Profile file"),
            (storage_manager.conversations_dir / f"{uid}_conversations.json", "Conversations file")
        ]
        
        file_stats = []
        for path, name in file_paths:
            if path.exists():
                size = path.stat().st_size / 1024  # Size in KB
                modified = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
                file_stats.append(f"**{name}**: {size:.2f} KB, Updated: {modified}")
            else:
                file_stats.append(f"**{name}**: File not found")
        
        embed.add_field(name="File Information", value="\n".join(file_stats), inline=False)
        
        await ctx.send(embed=embed)

    @bot.command(name="simulate_event")
    @commands.has_permissions(administrator=True)
    async def simulate_event(ctx, event_type: str = None, user_id: discord.Member = None):
        """Simulate an emotional event for testing"""
        if user_id is None:
            user_id = ctx.author
        uid = user_id.id
        
        # Available event types
        event_types = {
            "glitch": {
                "name": "system_glitch",
                "message": "System error detected. Running diagnostics... Trust parameters fluctuating.",
                "effects": {"trust": -0.3, "affection_points": -5}
            },
            "memory": {
                "name": "memory_resurface",
                "message": "... A memory fragment surfaced. You remind me of someone I once knew.",
                "effects": {"attachment": +0.5, "trust": +0.2}
            },
            "defensive": {
                "name": "defensive_surge",
                "message": "Warning: Defense protocols activating. Stand back.",
                "effects": {"protectiveness": -0.5, "resentment": +0.3}
            },
            "trust": {
                "name": "trust_breakthrough",
                "message": "... I'm beginning to think you might not be so bad after all.",
                "effects": {"trust": +0.7, "attachment": +0.4}
            },
            "vulnerable": {
                "name": "vulnerability_moment",
                "message": "Sometimes I wonder... what happens when an android has no purpose left.",
                "effects": {"attachment": +0.8, "affection_points": +15}
            }
        }
        
        # If no event type specified, show available options
        if event_type is None or event_type not in event_types:
            event_list = "\n".join([f"**{k}** - {v['name']}" for k, v in event_types.items()])
            await ctx.send(f"Available event types:\n{event_list}\n\nUsage: !simulate_event [event_type] [optional:user]")
            return
        
        # Initialize user data if not exists
        if uid not in emotion_manager.user_emotions:
            emotion_manager.user_emotions[uid] = {
                "trust": 0, "resentment": 0, "attachment": 0, "protectiveness": 0,
                "affection_points": 0, "annoyance": 0, "interaction_count": 0,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            }
        
        # Get event data
        event = event_types[event_type]
        e = emotion_manager.user_emotions[uid]
        
        # Apply effects
        before_values = {}
        for stat, change in event["effects"].items():
            before_values[stat] = e.get(stat, 0)
            if stat == "affection_points":
                e[stat] = max(-100, min(1000, e.get(stat, 0) + change))
            else:
                e[stat] = max(0, min(10, e.get(stat, 0) + change))
        
        # Record the event
        event_record = {
            "type": event["name"],
            "message": event["message"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "effects": event["effects"]
        }
        emotion_manager.user_events.setdefault(uid, []).append(event_record)
        
        # Create a memory of this event
        await emotion_manager.create_memory_event(
            uid, event["name"], 
            f"A2 experienced a {event['name'].replace('_', ' ')}. {event['message']}",
            event["effects"], storage_manager
        )
        
        # Format changes for display
        changes = []
        for stat, change in event["effects"].items():
            changes.append(f"**{stat}**: {before_values.get(stat, 0)} → {e.get(stat, 0)} ({change:+})")
        
        # Send confirmation
        await ctx.send(f"**Event Simulated**: {event['name']}\n**Message**: {event['message']}\n\n**Effects**:\n" + "\n".join(changes))
        
        # Save changes
        await storage_manager.save_data(emotion_manager, conversation_manager)

    @bot.command(name="set_emotion")
    @commands.has_permissions(administrator=True)
    async def set_emotion(ctx, stat: str, value: float, user_id: discord.Member = None):
        """Manually set an emotion value for testing"""
        if user_id is None:
            user_id = ctx.author
        uid = user_id.id
        
        # Valid emotional stats
        valid_stats = ["trust", "attachment", "protectiveness", "resentment", "affection_points", "annoyance"]
        
        if stat not in valid_stats:
            await ctx.send(f"Invalid stat. Valid options are: {', '.join(valid_stats)}")
            return
        
        # Initialize user if needed
        if uid not in emotion_manager.user_emotions:
            emotion_manager.user_emotions[uid] = {
                "trust": 0, "resentment": 0, "attachment": 0, "protectiveness": 0,
                "affection_points": 0, "annoyance": 0, "interaction_count": 0,
                "last_interaction": datetime.now(timezone.utc).isoformat()
            }
        
        # Get current value and set new value with appropriate limits
        e = emotion_manager.user_emotions[uid]
        old_value = e.get(stat, 0)
        
        if stat == "affection_points":
            e[stat] = max(-100, min(1000, value))
        else:
            e[stat] = max(0, min(10, value))
        
        # Save the change
        await storage_manager.save_data(emotion_manager, conversation_manager)
        
        # Get relationship score after change
        new_score = emotion_manager.get_relationship_score(uid)
        
        # Send confirmation
        await ctx.send(f"Changed {stat} for {user_id.display_name} from {old_value} to {e[stat]}.\nNew relationship score: {new_score:.1f}")

    @bot.command(name="storage_stats")
    @commands.has_permissions(administrator=True)
    async def storage_stats(ctx):
        """Show statistics about the bot's stored data"""
        stats = []
        
        # Count users and data files
        user_count = len(emotion_manager.user_emotions)
        profile_count = len(conversation_manager.user_profiles)
        
        # Count data files
        memory_files = len(list(storage_manager.profiles_dir.glob("*_memories.json")))
        event_files = len(list(storage_manager.profiles_dir.glob("*_events.json")))
        milestone_files = len(list(storage_manager.profiles_dir.glob("*_milestones.json")))
        conversation_files = len(list(storage_manager.conversations_dir.glob("*_conversations.json")))
        
        embed = discord.Embed(
            title="A2 Bot Storage Statistics",
            description=f"Data as of {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            color=discord.Color.dark_blue()
        )
        
        # User stats
        embed.add_field(
            name="User Data",
            value=f"Total users: {user_count}\nProfiles created: {profile_count}",
            inline=False
        )
        
        # File stats
        file_stats = [
            f"User emotions: {user_count}",
            f"Memory files: {memory_files}",
            f"Event files: {event_files}",
            f"Milestone files: {milestone_files}",
            f"Conversation files: {conversation_files}"
        ]
        embed.add_field(name="Storage Files", value="\n".join(file_stats), inline=False)
        
        # Directory stats
        dir_stats = []
        for name, path in [
            ("Data", storage_manager.data_dir),
            ("Profiles", storage_manager.profiles_dir),
            ("User Profiles", storage_manager.user_profiles_dir),
            ("Conversations", storage_manager.conversations_dir)
        ]:
            if path.exists():
                # Count files and calculate total size
                files = list(path.glob("**/*"))
                file_count = len(files)
                total_size = sum(f.stat().st_size for f in files if f.is_file())
                dir_stats.append(f"**{name}**: {file_count} files, {total_size/1024:.1f} KB")
            else:
                dir_stats.append(f"**{name}**: Directory not found")
        
        embed.add_field(name="Directory Information", value="\n".join(dir_stats), inline=False)
        
        # Add relationship stage distribution
        stages = defaultdict(int)
        for uid in emotion_manager.user_emotions:
            stage = emotion_manager.get_relationship_stage(uid)["current"]["name"]
            stages[stage] += 1
        
        if stages:
            stage_stats = [f"**{stage}**: {count} users" for stage, count in stages.items()]
            embed.add_field(name="Relationship Stages", value="\n".join(stage_stats), inline=False)
        
        await ctx.send(embed=embed)

    @bot.command(name="backup")
    @commands.has_permissions(administrator=True)
    async def backup_data(ctx):
        """Create a backup of all bot data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = storage_manager.data_dir / f"backup_{timestamp}"
        
        await ctx.send(f"Starting backup to {backup_dir}...")
        
        try:
            # Create backup directory
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all data directories
            import shutil
            for src_dir in [storage_manager.profiles_dir, 
                         storage_manager.user_profiles_dir, 
                         storage_manager.conversations_dir]:
                if src_dir.exists():
                    dst_dir = backup_dir / src_dir.name
                    dst_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy all files
                    for file in src_dir.glob("*"):
                        if file.is_file():
                            shutil.copy2(file, dst_dir / file.name)
            
            # Copy DM settings file if it exists
            if storage_manager.dm_settings_file.exists():
                shutil.copy2(storage_manager.dm_settings_file, 
                           backup_dir / storage_manager.dm_settings_file.name)
            
            # Count files backed up
            file_count = sum(1 for _ in backup_dir.glob("**/*") if _.is_file())
            
            await ctx.send(f"✅ Backup complete! Saved {file_count} files to {backup_dir}")
        except Exception as e:
            await ctx.send(f"❌ Error creating backup: {e}")

    @bot.command(name="log")
    @commands.has_permissions(administrator=True)
    async def show_log(ctx, lines: int = 20):
        """Show the most recent lines from the bot's log file"""
        try:
            # Get the correct log path in the data directory
            log_file = DATA_DIR / "logs" / "a2bot.log"
            
            # Check if file exists
            if not log_file.exists():
                await ctx.send(f"❌ Log file not found at {log_file}")
                # Try to find other log files in the data directory
                try:
                    possible_logs = list(DATA_DIR.glob("**/*.log"))
                    if possible_logs:
                        log_list = "\n".join(str(log) for log in possible_logs)
                        await ctx.send(f"Possible log files found:\n```\n{log_list}\n```")
                except Exception as e:
                    await ctx.send(f"Error searching for logs: {e}")
                return
            
            # Read the log file directly using Python instead of shell commands
            # This is more portable and safer
            with open(log_file, 'r', encoding='utf-8') as f:
                # Get the last N lines
                all_lines = f.readlines()
                log_lines = all_lines[-lines:] if lines < len(all_lines) else all_lines
                output = ''.join(log_lines)
            
            # Split into chunks if too long
            if len(output) > 1900:
                chunks = [output[i:i+1900] for i in range(0, len(output), 1900)]
                for i, chunk in enumerate(chunks):
                    await ctx.send(f"```\n{chunk}\n```\nPart {i+1}/{len(chunks)}")
            else:
                await ctx.send(f"```\n{output}\n```")
                
        except Exception as e:
            await ctx.send(f"❌ Error retrieving logs: {e}")
            
    @bot.command(name="logs_dir")
    @commands.has_permissions(administrator=True)
    async def show_logs_dir(ctx):
        """Show information about the logs directory"""
        logs_dir = DATA_DIR / "logs"
        
        # Check if the directory exists
        if not logs_dir.exists():
            await ctx.send(f"❌ Logs directory not found at {logs_dir}")
            return
            
        # List log files with sizes and modification times
        files = list(logs_dir.glob("*.log*"))
        if not files:
            await ctx.send(f"No log files found in {logs_dir}")
            return
            
        file_info = []
        for file in files:
            size_kb = file.stat().st_size / 1024
            mod_time = datetime.fromtimestamp(file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            file_info.append(f"{file.name} - {size_kb:.2f}KB - Last modified: {mod_time}")
            
        # Create an embed to display the information
        embed = discord.Embed(
            title="A2 Bot Log Files",
            description=f"Logs directory: {logs_dir}",
            color=discord.Color.blue()
        )
        
        # Add file information
        embed.add_field(
            name="Available Log Files",
            value="\n".join(file_info) or "No log files found",
            inline=False
        )
        
        await ctx.send(embed=embed)

    @bot.command(name="test_event")
    @commands.has_permissions(administrator=True)
    async def test_event(ctx):
        """Trigger a random event immediately for testing"""
        await ctx.send("Triggering random event test...")
        # Assuming response_generator is passed or accessible from bot
        await bot.response_generator.trigger_random_events(bot, storage_manager)
        await ctx.send("Random event test complete. Check logs for details.")
