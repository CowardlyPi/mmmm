"""
Enhanced user commands with better UX and error handling.
"""
import discord
from discord.ext import commands
from datetime import datetime, timezone
from utils.validation_utils import InputValidator, RateLimiter
from utils.error_handler import safe_execute, log_error_with_aggregation

def setup_enhanced_user_commands(bot, emotion_manager, conversation_manager, storage_manager):
    """Set up enhanced user commands with better UX"""
    
    # Rate limiters for different command types
    general_limiter = RateLimiter(max_requests=15, time_window=60)
    heavy_limiter = RateLimiter(max_requests=5, time_window=60)
    
    def rate_limited(limiter_type="general"):
        """Rate limiting decorator"""
        def decorator(func):
            async def wrapper(ctx, *args, **kwargs):
                limiter = general_limiter if limiter_type == "general" else heavy_limiter
                allowed, reset_time = limiter.is_allowed(ctx.author.id)
                
                if not allowed:
                    await ctx.send(f"A2: Slow down. Try again in {reset_time} seconds.")
                    return
                
                return await func(ctx, *args, **kwargs)
            return wrapper
        return decorator
    
    @bot.command(name="stats", help="Display your relationship stats with A2")
    @rate_limited("heavy")
    @safe_execute(default_return=None)
    async def enhanced_stats(ctx):
        """Enhanced stats command with better formatting and error handling"""
        uid = ctx.author.id
        e = emotion_manager.user_emotions.get(uid, {})
        
        if not e:
            await ctx.send("A2: No interaction data found. Talk to me first.")
            return
        
        # Calculate relationship data safely
        try:
            rel_data = emotion_manager.get_relationship_stage(uid)
        except Exception as error:
            log_error_with_aggregation(error, "calculate_relationship_stage", {"user_id": uid})
            await ctx.send("A2: Error calculating relationship data.")
            return
        
        # Create enhanced embed
        embed = discord.Embed(
            title=f"A2's Assessment: {ctx.author.display_name}",
            description=f"**Relationship Level:** {rel_data['current']['name']}",
            color=discord.Color.from_hsv(min(0.99, max(0, rel_data['score']/100)), 0.7, 0.9)
        )
        
        # Add relationship description
        embed.add_field(
            name="Current Status",
            value=rel_data['current']['description'],
            inline=False
        )
        
        # Show progress to next level
        if rel_data['next']:
            progress = rel_data['progress']
            progress_bar = "█" * int(progress/10) + "░" * (10 - int(progress/10))
            embed.add_field(
                name=f"Progress to {rel_data['next']['name']}",
                value=f"`{progress_bar}` {progress:.1f}%",
                inline=False
            )
        
        # Emotional stats with visual bars and descriptions
        emotion_fields = [
            ("Trust", e.get('trust', 0), 10, "🔒", emotion_manager.get_emotion_description("trust", e.get('trust', 0))),
            ("Attachment", e.get('attachment', 0), 10, "🔗", emotion_manager.get_emotion_description("attachment", e.get('attachment', 0))),
            ("Protectiveness", e.get('protectiveness', 0), 10, "🛡️", emotion_manager.get_emotion_description("protectiveness", e.get('protectiveness', 0))),
            ("Resentment", e.get('resentment', 0), 10, "⚔️", emotion_manager.get_emotion_description("resentment", e.get('resentment', 0))),
        ]
        
        for name, value, max_val, emoji, description in emotion_fields:
            bar_length = int((value / max_val) * 10) if max_val > 0 else 0
            bar = "█" * bar_length + "░" * (10 - bar_length)
            embed.add_field(
                name=f"{emoji} {name}",
                value=f"`{bar}` {description}",
                inline=True
            )
        
        # Special stats
        affection = e.get('affection_points', 0)
        annoyance = e.get('annoyance', 0)
        
        embed.add_field(
            name="💠 Affection",
            value=f"{affection}/1000 points",
            inline=True
        )
        
        if annoyance > 10:  # Only show if notable
            embed.add_field(
                name="🔥 Annoyance",
                value=f"{annoyance}/100",
                inline=True
            )
        
        # Interaction statistics
        interaction_count = e.get('interaction_count', 0)
        embed.add_field(
            name="📊 Interactions",
            value=f"Total: {interaction_count}",
            inline=True
        )
        
        # Add contextual footer
        footers = [
            "Systems operational.",
            "Data analysis complete.",
            "Monitoring continues.",
            "...",
        ]
        
        if rel_data['score'] > 70:
            footers.extend([
                "Your presence is... acceptable.",
                "We've built something here.",
            ])
        elif annoyance > 60:
            footers.extend([
                "Don't push it.",
                "Testing my patience.",
            ])
        
        embed.set_footer(text=random.choice(footers))
        
        await ctx.send(embed=embed)
    
    @bot.command(name="help_me", aliases=["help", "commands"], help="Show available commands")
    @rate_limited("general")
    async def enhanced_help(ctx, command_name: str = None):
        """Enhanced help command with categorized commands"""
        
        if command_name:
            # Show help for specific command
            cmd = bot.get_command(command_name)
            if not cmd:
                await ctx.send(f"A2: Unknown command '{command_name}'.")
                return
            
            embed = discord.Embed(
                title=f"Command: {cmd.name}",
                description=cmd.help or "No description available.",
                color=discord.Color.blue()
            )
            
            if cmd.aliases:
                embed.add_field(name="Aliases", value=", ".join(cmd.aliases), inline=False)
            
            if cmd.signature:
                embed.add_field(name="Usage", value=f"`!{cmd.name} {cmd.signature}`", inline=False)
            
            await ctx.send(embed=embed)
            return
        
        # Show all commands categorized
        embed = discord.Embed(
            title="A2 Command Interface",
            description="Available commands organized by category",
            color=discord.Color.dark_blue()
        )
        
        # User commands
        user_commands = [
            ("stats", "View your relationship statistics"),
            ("profile", "View your profile as A2 knows it"),
            ("memories", "View memories A2 has of you"),
            ("milestones", "View relationship milestones achieved"),
            ("relationship", "Detailed relationship progression"),
            ("conversations", "View recent conversation history"),
        ]
        
        user_cmd_text = "\n".join([f"`!{name}` - {desc}" for name, desc in user_commands])
        embed.add_field(name="📊 Personal Data", value=user_cmd_text, inline=False)
        
        # Profile management commands
        profile_commands = [
            ("set_name <name>", "Set your preferred name"),
            ("set_nickname <nickname>", "Set your nickname"),
            ("update_profile <field> <value>", "Update profile information"),
            ("clear_profile_field <field>", "Clear a profile field"),
        ]
        
        profile_cmd_text = "\n".join([f"`!{name}` - {desc}" for name, desc in profile_commands])
        embed.add_field(name="👤 Profile Management", value=profile_cmd_text, inline=False)
        
        # Settings commands
        settings_commands = [
            ("dm_toggle", "Toggle DM notifications"),
        ]
        
        settings_cmd_text = "\n".join([f"`!{name}` - {desc}" for name, desc in settings_commands])
        embed.add_field(name="⚙️ Settings", value=settings_cmd_text, inline=False)
        
        # Add usage tips
        embed.add_field(
            name="💡 Tips",
            value="• Use `!help_me <command>` for detailed help\n• Commands are rate-limited\n• Talk naturally - A2 learns from conversation",
            inline=False
        )
        
        embed.set_footer(text="A2: These are the functions I've made available to you.")
        
        await ctx.send(embed=embed)
    
    @bot.command(name="set_name", help="Set your preferred name for A2 to use")
    @rate_limited("general")
    async def enhanced_set_name(ctx, *, name: str):
        """Enhanced name setting with validation"""
        uid = ctx.author.id
        
        # Validate the name
        is_valid, error_message = InputValidator.validate_user_name(name)
        if not is_valid:
            await ctx.send(f"A2: Invalid name. {error_message}")
            return
        
        # Sanitize the name
        clean_name = InputValidator.sanitize_text(name, InputValidator.MAX_NAME_LENGTH)
        
        try:
            profile = conversation_manager.update_name_recognition(uid, preferred_name=clean_name)
            await storage_manager.save_user_profile_data(uid, profile)
            
            await ctx.send(f"A2: I'll remember your name as '{clean_name}'.")
        except Exception as error:
            log_error_with_aggregation(error, "set_name", {"user_id": uid, "name": clean_name})
            await ctx.send("A2: Error updating name. Try again later.")
    
    @bot.command(name="update_profile", help="Update your profile information")
    @rate_limited("general")
    async def enhanced_update_profile(ctx, field: str, *, value: str):
        """Enhanced profile update with better validation"""
        uid = ctx.author.id
        
        # Validate field
        valid_fields = ["interests", "personality_traits", "notable_facts", "relationship_context", "conversation_topics"]
        
        if field not in valid_fields:
            fields_list = ", ".join(valid_fields)
            await ctx.send(f"A2: Invalid field. Valid fields are: {fields_list}")
            return
        
        # Validate the input as a list
        is_valid, processed_items, error_message = InputValidator.validate_list_input(value, field)
        if not is_valid:
            await ctx.send(f"A2: {error_message}")
            return
        
        try:
            # Get or create profile
            if uid not in conversation_manager.user_profiles:
                conversation_manager.get_or_create_profile(uid, ctx.author.display_name)
            
            profile = conversation_manager.user_profiles[uid]
            
            # Update the field
            current_list = getattr(profile, field, [])
            
            # Add new items to existing list
            for item in processed_items:
                if item not in current_list:
                    current_list.append(item)
            
            # Limit list size
            if len(current_list) > 20:
                current_list = current_list[-20:]  # Keep most recent 20
            
            profile.update_profile(field, current_list)
            await storage_manager.save_user_profile_data(uid, profile)
            
            field_display = field.replace('_', ' ')
            items_text = ", ".join(processed_items)
            await ctx.send(f"A2: Updated your {field_display}. Added: {items_text}")
            
        except Exception as error:
            log_error_with_aggregation(error, "update_profile", {"user_id": uid, "field": field})
            await ctx.send("A2: Error updating profile. Try again later.")
    
    @bot.command(name="quick_stats", aliases=["qs"], help="Quick stats summary")
    @rate_limited("general")
    async def quick_stats(ctx):
        """Quick, lightweight stats display"""
        uid = ctx.author.id
        e = emotion_manager.user_emotions.get(uid, {})
        
        if not e:
            await ctx.send("A2: No data found.")
            return
        
        try:
            score = emotion_manager.get_relationship_score(uid)
            trust = e.get('trust', 0)
            attachment = e.get('attachment', 0)
            
            # Quick summary
            status = "Hostile"
            if score >= 50:
                status = "Bonded"
            elif score >= 25:
                status = "Trusted"
            elif score >= 10:
                status = "Neutral"
            elif score >= 5:
                status = "Wary"
            
            await ctx.send(f"A2: Status: {status} | Trust: {trust:.1f}/10 | Bond: {attachment:.1f}/10 | Score: {score:.1f}")
            
        except Exception as error:
            log_error_with_aggregation(error, "quick_stats", {"user_id": uid})
            await ctx.send("A2: Error retrieving data.")
    
    @bot.command(name="my_data", help="Export your data")
    @rate_limited("heavy")
    async def export_data(ctx):
        """Allow users to export their data"""
        uid = ctx.author.id
        
        if uid not in emotion_manager.user_emotions:
            await ctx.send("A2: No data found for export.")
            return
        
        try:
            # Collect user data
            data = {
                "emotions": emotion_manager.user_emotions.get(uid, {}),
                "memories": emotion_manager.user_memories.get(uid, []),
                "events": emotion_manager.user_events.get(uid, []),
                "milestones": emotion_manager.user_milestones.get(uid, []),
                "interactions": dict(emotion_manager.interaction_stats.get(uid, {})),
            }
            
            if uid in conversation_manager.user_profiles:
                data["profile"] = conversation_manager.user_profiles[uid].to_dict()
            
            # Create a summary instead of full data (privacy)
            summary = f"""
**A2 Data Summary for {ctx.author.display_name}**

**Emotional Stats:**
- Trust: {data['emotions'].get('trust', 0):.2f}/10
- Attachment: {data['emotions'].get('attachment', 0):.2f}/10
- Total Interactions: {data['emotions'].get('interaction_count', 0)}

**Data Counts:**
- Memories: {len(data['memories'])}
- Events: {len(data['events'])}
- Milestones: {len(data['milestones'])}

**Profile Elements:**
- Interests: {len(data.get('profile', {}).get('interests', []))}
- Personality traits: {len(data.get('profile', {}).get('personality_traits', []))}

*Full data export can be requested from administrators.*
            """
            
            await ctx.send(f"```{summary}```")
            
        except Exception as error:
            log_error_with_aggregation(error, "export_data", {"user_id": uid})
            await ctx.send("A2: Error compiling data export.")
    
    # Import random for footers
    import random
