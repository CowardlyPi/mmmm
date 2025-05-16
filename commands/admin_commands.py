"""
Admin commands for the A2 Discord bot.
"""
import os
import sys
import time
import random
import discord
from datetime import datetime, timezone
from discord.ext import commands

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
    
    # Include all the other admin commands...
    # [All other admin commands from original code]
