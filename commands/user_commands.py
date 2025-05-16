"""
User-facing commands for the A2 Discord bot.
"""
import random
import discord
from discord.ext import commands
from datetime import datetime, timezone

def setup_user_commands(bot, emotion_manager, conversation_manager, storage_manager):
    """Set up commands accessible to normal users"""
    
    @bot.command(name="stats")
    async def stats(ctx):
        """Display enhanced, dynamic relationship stats"""
        uid = ctx.author.id
        e = emotion_manager.user_emotions.get(uid, {})
        
        # Calculate relationship score
        rel_data = emotion_manager.get_relationship_stage(uid)
        
        # Create a more visual and dynamic embed
        embed = discord.Embed(
            title=f"A2's Perception of {ctx.author.display_name}", 
            description=f"Relationship Stage: **{rel_data['current']['name']}**",
            color=discord.Color.from_hsv(min(0.99, max(0, rel_data['score']/100)), 0.8, 0.8)  # Color changes with score
        )
        
        # Add description of current relationship
        embed.add_field(
            name="Status", 
            value=rel_data['current']['description'],
            inline=False
        )
        
        # Show progress to next stage if not at max
        if rel_data['next']:
            progress_bar = "█" * int(rel_data['progress']/10) + "░" * (10 - int(rel_data['progress']/10))
            embed.add_field(
                name=f"Progress to {rel_data['next']['name']}", 
                value=f"`{progress_bar}` {rel_data['progress']:.1f}%",
                inline=False
            )
        
        # Visual bars for stats using Discord emoji blocks
        for stat_name, value, max_val, emoji in [
            ("Trust", e.get('trust', 0), 10, "🔒"),
            ("Attachment", e.get('attachment', 0), 10, "🔗"),
            ("Protectiveness", e.get('protectiveness', 0), 10, "🛡️"),
            ("Resentment", e.get('resentment', 0), 10, "⚔️"),
            ("Affection", e.get('affection_points', 0), 1000, "💠"),
            ("Annoyance", e.get('annoyance', 0), 100, "🔥")
        ]:
            # Normalize to 0-10 range for emoji bars
            norm_val = value / max_val * 10 if max_val > 10 else value
            bar = "█" * int(norm_val) + "░" * (10 - int(norm_val))
            
            if stat_name.lower() in ["trust", "attachment", "protectiveness", "resentment"]:
                desc = f"{emotion_manager.get_emotion_description(stat_name.lower(), value)}"
            else:
                desc = f"{value}/{max_val}"
                
            embed.add_field(name=f"{emoji} {stat_name}", value=f"`{bar}` {desc}", inline=False)
        
        # Add dynamic mood and state info
        current_state = emotion_manager.select_personality_state(uid, "")
        embed.add_field(name="Current Mood", value=f"{current_state.capitalize()}", inline=True)
        
        # Add interaction stats
        embed.add_field(name="Total Interactions", value=str(e.get('interaction_count', 0)), inline=True)
        
        # Add profile info if available
        if uid in conversation_manager.user_profiles:
            profile = conversation_manager.user_profiles[uid]
            if profile.interests:
                embed.add_field(name="Recognized Interests", value=', '.join(profile.interests[:3]), inline=True)
            if profile.personality_traits:
                embed.add_field(name="Recognized Traits", value=', '.join(profile.personality_traits[:3]), inline=True)
        
        # Add a contextual response
        responses = [
            "...",
            "Don't read too much into this.",
            "Numbers don't matter.",
            "Still functioning.",
            "Is this what you wanted to see?",
            "Analyzing you too, human."
        ]
        
        if rel_data['score'] > 60:
            responses.extend([
                "Your presence is... acceptable.",
                "We've come a long way.",
                "Trust doesn't come easily for me."
            ])
        
        if e.get('annoyance', 0) > 60:
            responses.extend([
                "Don't push it.",
                "You're testing my patience."
            ])
        
        embed.set_footer(text=random.choice(responses))
        
        await ctx.send(embed=embed)

    @bot.command(name="memories")
    async def memories(ctx):
        """Show memories A2 has formed with this user"""
        uid = ctx.author.id
        if uid not in emotion_manager.user_memories or not emotion_manager.user_memories[uid]:
            await ctx.send("A2: ... No significant memories stored.")
            return
        
        embed = discord.Embed(title="A2's Memory Logs", color=discord.Color.purple())
        
        # Sort memories by timestamp (newest first)
        sorted_memories = sorted(emotion_manager.user_memories[uid], 
                                key=lambda m: datetime.fromisoformat(m["timestamp"]), 
                                reverse=True)
        
        # Display the 5 most recent memories
        for i, memory in enumerate(sorted_memories[:5]):
            timestamp = datetime.fromisoformat(memory["timestamp"])
            embed.add_field(
                name=f"Memory Log #{len(sorted_memories)-i}",
                value=f"*{timestamp.strftime('%Y-%m-%d %H:%M')}*\n{memory['description']}",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @bot.command(name="milestones")
    async def show_milestones(ctx):
        """Show relationship milestones achieved with this user"""
        uid = ctx.author.id
        if uid not in emotion_manager.user_milestones or not emotion_manager.user_milestones[uid]:
            await ctx.send("A2: No notable milestones recorded yet.")
            return
        
        embed = discord.Embed(title="Relationship Milestones", color=discord.Color.gold())
        
        # Sort milestones by timestamp
        sorted_milestones = sorted(emotion_manager.user_milestones[uid], 
                                  key=lambda m: datetime.fromisoformat(m["timestamp"]))
        
        for i, milestone in enumerate(sorted_milestones):
            timestamp = datetime.fromisoformat(milestone["timestamp"])
            embed.add_field(
                name=f"Milestone #{i+1}",
                value=f"*{timestamp.strftime('%Y-%m-%d')}*\n{milestone['description']}",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @bot.command(name="relationship")
    async def relationship(ctx):
        """Show detailed relationship progression info"""
        uid = ctx.author.id
        rel_data = emotion_manager.get_relationship_stage(uid)
        e = emotion_manager.user_emotions.get(uid, {})
        
        # Create graphical representation
        embed = discord.Embed(
            title=f"Relationship with {ctx.author.display_name}",
            description=f"Overall Score: {rel_data['score']:.1f}/100",
            color=discord.Color.dark_purple()
        )
        
        # Create relationship progression bar
        stages_bar = ""
        for i, stage in enumerate(bot.RELATIONSHIP_LEVELS):
            if rel_data["current"] == stage:
                stages_bar += "**[" + stage["name"] + "]** → "
            elif i < bot.RELATIONSHIP_LEVELS.index(rel_data["current"]):
                stages_bar += stage["name"] + " → "
            elif i == bot.RELATIONSHIP_LEVELS.index(rel_data["current"]) + 1:
                stages_bar += stage["name"] + " → ..."
                break
            else:
                continue
        
        embed.add_field(name="Progression", value=stages_bar, inline=False)
        
        # Show current relationship details
        embed.add_field(
            name="Current Stage", 
            value=f"**{rel_data['current']['name']}**\n{rel_data['current']['description']}",
            inline=False
        )
        
        # Add interaction stats
        stats = emotion_manager.interaction_stats.get(uid, {})
        total = stats.get("total", 0)
        if total > 0:
            positive = stats.get("positive", 0)
            negative = stats.get("negative", 0)
            neutral = stats.get("neutral", 0)
            
            stats_txt = f"Total interactions: {total}\n"
            stats_txt += f"Positive: {positive} ({positive/total*100:.1f}%)\n"
            stats_txt += f"Negative: {negative} ({negative/total*100:.1f}%)\n"
            stats_txt += f"Neutral: {neutral} ({neutral/total*100:.1f}%)"
            
            embed.add_field(name="Interaction Analysis", value=stats_txt, inline=False)
        
        # Add key contributing factors
        factors = []
        if e.get('trust', 0) > 5:
            factors.append(f"High trust (+{e.get('trust', 0):.1f})")
        if e.get('attachment', 0) > 5:
            factors.append(f"Strong attachment (+{e.get('attachment', 0):.1f})")
        if e.get('resentment', 0) > 3:
            factors.append(f"Lingering resentment (-{e.get('resentment', 0):.1f})")
        if e.get('protectiveness', 0) > 5:
            factors.append(f"Protective instincts (+{e.get('protectiveness', 0):.1f})")
        if e.get('affection_points', 0) > 50:
            factors.append(f"Positive affection (+{e.get('affection_points', 0)/100:.1f})")
        elif e.get('affection_points', 0) < -20:
            factors.append(f"Negative affection ({e.get('affection_points', 0)/100:.1f})")
        
        if factors:
            embed.add_field(name="Key Factors", value="\n".join(factors), inline=False)
        
        # Add a personalized note based on relationship
        if rel_data['score'] < 10:
            note = "Systems registering high caution levels. Threat assessment ongoing."
        elif rel_data['score'] < 25:
            note = "Your presence is tolerable. For now."
        elif rel_data['score'] < 50:
            note = "You're... different from the others. Still evaluating."
        elif rel_data['score'] < 75:
            note = "I've grown somewhat accustomed to your presence."
        else:
            note = "There are few I've trusted this much. Don't make me regret it."
        
        embed.set_footer(text=note)
        
        await ctx.send(embed=embed)

    @bot.command(name="events")
    async def show_events(ctx):
        """Show recent random events"""
        uid = ctx.author.id
        if uid not in emotion_manager.user_events or not emotion_manager.user_events[uid]:
            await ctx.send("A2: No notable events recorded.")
            return
        
        embed = discord.Embed(title="Recent Events", color=discord.Color.dark_red())
        
        # Sort events by timestamp (newest first)
        sorted_events = sorted(emotion_manager.user_events[uid], 
                              key=lambda e: datetime.fromisoformat(e["timestamp"]), 
                              reverse=True)
        
        for i, event in enumerate(sorted_events[:5]):
            timestamp = datetime.fromisoformat(event["timestamp"])
            
            # Format the effects for display
            effects_txt = ""
            for stat, value in event.get("effects", {}).items():
                if value >= 0:
                    effects_txt += f"{stat}: +{value}\n"
                else:
                    effects_txt += f"{stat}: {value}\n"
            
            embed.add_field(
                name=f"Event {i+1}: {event['type'].replace('_', ' ').title()}",
                value=f"*{timestamp.strftime('%Y-%m-%d %H:%M')}*\n"
                      f"\"{event['message']}\"\n\n"
                      f"{effects_txt if effects_txt else 'No measurable effects.'}",
                inline=False
            )
        
        await ctx.send(embed=embed)

    @bot.command(name="profile")
    async def show_profile(ctx, user_id: discord.Member = None):
        """Show your profile as A2 knows it"""
        target_id = user_id.id if user_id else ctx.author.id
        target_name = user_id.display_name if user_id else ctx.author.display_name
        
        if target_id not in conversation_manager.user_profiles:
            if target_id == ctx.author.id:
                await ctx.send("A2: I don't have a profile for you yet. Keep talking to me so I can learn more.")
            else:
                await ctx.send(f"A2: I don't have a profile for {target_name} yet.")
            return
        
        profile = conversation_manager.user_profiles[target_id]
        
        embed = discord.Embed(
            title=f"Profile: {target_name}",
            description="Information A2 has learned about you",
            color=discord.Color.blue()
        )
        
        # Add name information
        names = []
        if profile.name:
            names.append(f"Name: {profile.name}")
        if profile.nickname:
            names.append(f"Nickname: {profile.nickname}")
        if profile.preferred_name:
            names.append(f"Preferred name: {profile.preferred_name}")
        
        if names:
            embed.add_field(name="Identity", value="\n".join(names), inline=False)
        
        # Add personality traits
        if profile.personality_traits:
            embed.add_field(
                name="Personality Traits", 
                value=", ".join(profile.personality_traits),
                inline=False
            )
        
        # Add interests
        if profile.interests:
            embed.add_field(
                name="Interests", 
                value=", ".join(profile.interests),
                inline=False
            )
        
        # Add notable facts
        if profile.notable_facts:
            embed.add_field(
                name="Notable Information", 
                value="\n".join(profile.notable_facts),
                inline=False
            )
        
        # Add relationship context
        if profile.relationship_context:
            embed.add_field(
                name="Relationship Context", 
                value="\n".join(profile.relationship_context),
                inline=False
            )
        
        # Add conversation topics
        if profile.conversation_topics:
            embed.add_field(
                name="Common Conversation Topics", 
                value=", ".join(profile.conversation_topics),
                inline=False
            )
        
        # Add last updated info
        embed.set_footer(text=f"Last updated: {datetime.fromisoformat(profile.updated_at).strftime('%Y-%m-%d %H:%M:%S')}")
        
        await ctx.send(embed=embed)

    @bot.command(name="set_name")
    async def set_name(ctx, *, name):
        """Set your preferred name for A2 to use"""
        uid = ctx.author.id
        profile = conversation_manager.update_name_recognition(uid, preferred_name=name)
        
        # Save the updated profile
        await storage_manager.save_user_profile_data(uid, profile)
        
        await ctx.send(f"A2: I'll remember your name as {name}.")

    @bot.command(name="set_nickname")
    async def set_nickname(ctx, *, nickname):
        """Set your nickname for A2 to use"""
        uid = ctx.author.id
        profile = conversation_manager.update_name_recognition(uid, nickname=nickname)
        
        # Save the updated profile
        await storage_manager.save_user_profile_data(uid, profile)
        
        await ctx.send(f"A2: I'll remember your nickname as {nickname}.")

    @bot.command(name="conversations")
    async def show_conversations(ctx):
        """Show recent conversation history"""
        uid = ctx.author.id
        
        if uid not in conversation_manager.conversations or not conversation_manager.conversations[uid]:
            await ctx.send("A2: No conversation history found.")
            return
        
        embed = discord.Embed(
            title="Recent Conversation History",
            description="Last few messages exchanged with A2",
            color=discord.Color.green()
        )
        
        # Get and format conversation history
        history = conversation_manager.conversations[uid][-5:]  # Last 5 messages
        
        formatted_history = ""
        for i, msg in enumerate(history):
            speaker = "A2" if msg.get("from_bot", False) else "You"
            formatted_history += f"**{speaker}**: {msg.get('content', '')}\n\n"
        
        embed.add_field(name="Messages", value=formatted_history or "No messages found.", inline=False)
        
        # Add conversation summary if available
        if uid in conversation_manager.conversation_summaries:
            summary = conversation_manager.conversation_summaries[uid]
            if summary and summary != "Not enough conversation history for a summary.":
                embed.add_field(name="Summary", value=summary, inline=False)
        
        await ctx.send(embed=embed)

    @bot.command(name="update_profile")
    async def update_profile(ctx, field, *, value):
        """Update a field in your profile"""
        uid = ctx.author.id
        
        if uid not in conversation_manager.user_profiles:
            conversation_manager.get_or_create_profile(uid, ctx.author.display_name)
        
        profile = conversation_manager.user_profiles[uid]
        
        valid_fields = ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]
        
        if field not in valid_fields:
            await ctx.send(f"A2: Invalid field. Valid fields are: {', '.join(valid_fields)}")
            return
        
        # Handle list fields
        if field in ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]:
            # Add to list
            items = [item.strip() for item in value.split(",")]
            current_list = getattr(profile, field, [])
            for item in items:
                if item and item not in current_list:
                    current_list.append(item)
            profile.update_profile(field, current_list)
            
            await storage_manager.save_user_profile_data(uid, profile)
            await ctx.send(f"A2: Updated your {field.replace('_', ' ')} with: {value}")
        else:
            # Handle scalar fields
            profile.update_profile(field, value)
            await storage_manager.save_user_profile_data(uid, profile)
            await ctx.send(f"A2: Updated your {field.replace('_', ' ')} to: {value}")

    @bot.command(name="clear_profile_field")
    async def clear_profile_field(ctx, field):
        """Clear a field in your profile"""
        uid = ctx.author.id
        
        if uid not in conversation_manager.user_profiles:
            await ctx.send("A2: You don't have a profile yet.")
            return
        
        profile = conversation_manager.user_profiles[uid]
        
        valid_fields = ["name", "nickname", "preferred_name", "interests", "personality_traits", 
                       "notable_facts", "relationship_context", "conversation_topics"]
        
        if field not in valid_fields:
            await ctx.send(f"A2: Invalid field. Valid fields are: {', '.join(valid_fields)}")
            return
        
        # Handle list fields
        if field in ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]:
            profile.update_profile(field, [])
        else:
            # Handle scalar fields
            profile.update_profile(field, None)
        
        await storage_manager.save_user_profile_data(uid, profile)
        await ctx.send(f"A2: Cleared your {field.replace('_', ' ')}.")

    @bot.command(name="dm_toggle")
    async def toggle_dm(ctx):
        """Toggle whether A2 can send you DMs for events"""
        uid = ctx.author.id
        
        if uid in emotion_manager.dm_enabled_users:
            emotion_manager.dm_enabled_users.discard(uid)
            await ctx.send("A2: DM notifications disabled.")
        else:
            emotion_manager.dm_enabled_users.add(uid)
            
            # Test DM permissions
            try:
                dm = await ctx.author.create_dm()
                await dm.send("A2: DM access confirmed. Notifications enabled.")
                await ctx.send("A2: DM notifications enabled. Test message sent.")
            except discord.errors.Forbidden:
                await ctx.send("A2: Cannot send DMs. Check your privacy settings.")
                emotion_manager.dm_enabled_users.discard(uid)
        
        await storage_manager.save_dm_settings(emotion_manager.dm_enabled_users)
