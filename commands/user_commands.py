"""
User-facing commands for the A2 Discord bot.
"""
import discord
from discord.ext import commands

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
    
    # Include all the other user commands...
    # [All other user-facing commands from original code]
