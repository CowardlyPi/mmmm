"""
Example implementation of commands that utilize the enhanced A2 module.
"""
import discord
from discord.ext import commands
from datetime import datetime, timezone, timedelta

def setup_enhanced_commands(bot, enhanced_system):
    """Set up commands for the enhanced A2 features"""
    
    @bot.command(name="semantic_memories")
    async def semantic_memories(ctx):
        """Show semantic memories A2 has about you"""
        uid = ctx.author.id
        
        # Get memories for this user
        memories = enhanced_system.memory_system.retrieve_memories(uid, "important memories", limit=5)
        
        if not memories:
            await ctx.send("A2: No significant semantic memories found.")
            return
        
        embed = discord.Embed(title="A2's Semantic Memory Logs", color=discord.Color.purple())
        
        # Sort by importance (descending)
        sorted_memories = sorted(memories, key=lambda m: m.importance, reverse=True)
        
        for i, memory in enumerate(sorted_memories):
            # Format timestamp
            timestamp = datetime.fromisoformat(memory.created_at)
            
            embed.add_field(
                name=f"Memory {i+1} (Importance: {memory.importance:.2f})",
                value=f"*{timestamp.strftime('%Y-%m-%d')}*\n{memory.text}",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @bot.command(name="add_memory")
    async def add_memory(ctx, *, memory_text):
        """Add a custom memory for A2 to remember about you"""
        uid = ctx.author.id
        
        # Create a new memory
        from enhanced_a2 import Memory
        memory = Memory(
            user_id=uid,
            text=memory_text,
            importance=0.8,  # User-created memories are important
            memory_type="user_created",
            source="user_command"
        )
        
        # Add to memory system
        result = enhanced_system.memory_system.add_memory(memory)
        
        if result:
            await ctx.send("A2: ... I'll remember that.")
        else:
            await ctx.send("A2: Error storing memory.")
    
    @bot.command(name="personality")
    async def personality(ctx):
        """Show A2's current personality traits"""
        uid = ctx.author.id
        
        # Get personality system
        personality_system = enhanced_system.personality_system
        
        # Get current traits
        traits = personality_system.get_current_trait_values(uid)
        
        # Get current state
        state = personality_system.current_state
        state_desc = personality_system.personality_states.get(state, {}).get("description", "Unknown state")
        
        embed = discord.Embed(
            title="A2's Current Personality",
            description=f"Current State: **{state.replace('_', ' ').title()}**\n*{state_desc}*",
            color=discord.Color.dark_teal()
        )
        
        # Add traits
        for trait_name, value in traits.items():
            # Format trait name for display
            display_name = trait_name.replace("_", " ").title()
            
            # Create a visual bar
            bar_length = 10
            filled_bars = int(value * bar_length)
            bar = "█" * filled_bars + "░" * (bar_length - filled_bars)
            
            embed.add_field(
                name=display_name,
                value=f"`{bar}` {value:.2f}",
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @bot.command(name="emotion_analysis")
    async def emotion_analysis(ctx):
        """Show A2's analysis of your emotions"""
        uid = ctx.author.id
        
        # Get emotion detector
        emotion_detector = enhanced_system.emotion_detector
        
        # Get emotion trend
        trend_data = emotion_detector.analyze_emotion_trend(uid)
        
        if trend_data["trend"] == "insufficient_data":
            await ctx.send("A2: Not enough interaction data for emotion analysis.")
            return
        
        embed = discord.Embed(
            title="A2's Emotional Analysis",
            description=f"Overall trend: **{trend_data['trend'].replace('_', ' ')}**\nDominant emotion: **{trend_data['dominant']}**",
            color=discord.Color.gold()
        )
        
        # Add detailed emotion data
        if "emotions" in trend_data:
            for emotion, data in trend_data["emotions"].items():
                direction = data["direction"]
                arrow = "↑" if direction == "increasing" else "↓" if direction == "decreasing" else "→"
                embed.add_field(
                    name=f"{emotion.title()} {arrow}",
                    value=f"Average: {data['average']:.2f}\nChange: {data['change']:.2f}",
                    inline=True
                )
        
        await ctx.send(embed=embed)
    
    @bot.command(name="relationship_progress")
    async def relationship_progress(ctx):
        """Show detailed relationship progress and next milestones"""
        uid = ctx.author.id
        
        # Get trust score from emotion manager
        trust_score = bot.emotion_manager.get_relationship_score(uid)
        
        # Get relationship system
        relationship_system = enhanced_system.relationship_system
        
        # Get current stage
        stage_data = relationship_system.get_relationship_stage(trust_score)
        
        # Get next milestones
        next_milestones = relationship_system.get_next_milestones(uid, trust_score)
        
        # Get milestone history
        milestone_history = relationship_system.get_milestone_history(uid)
        
        embed = discord.Embed(
            title="Relationship with A2",
            description=f"Current Stage: **{stage_data['current']['name']}**\n*{stage_data['current']['description']}*",
            color=discord.Color.dark_purple()
        )
        
        # Add progress to next stage
        if stage_data['next']:
            progress_bar = "█" * int(stage_data['progress'] / 10) + "░" * (10 - int(stage_data['progress'] / 10))
            embed.add_field(
                name=f"Progress to {stage_data['next']['name']}",
                value=f"`{progress_bar}` {stage_data['progress']:.1f}%",
                inline=False
            )
        
        # Add next milestones
        if next_milestones:
            milestones_text = ""
            for milestone in next_milestones:
                milestones_text += f"**{milestone['name']}** - {milestone['points_needed']:.1f} points needed\n"
                milestones_text += f"*{milestone['description']}*\n\n"
                
            embed.add_field(
                name="Upcoming Milestones",
                value=milestones_text,
                inline=False
            )
        
        # Add recent milestone history
        if milestone_history:
            recent_milestones = sorted(milestone_history, key=lambda m: m['timestamp'], reverse=True)[:3]
            history_text = ""
            
            for milestone in recent_milestones:
                timestamp = datetime.fromisoformat(milestone['timestamp'])
                history_text += f"**{milestone['name']}** - {timestamp.strftime('%Y-%m-%d')}\n"
                
            embed.add_field(
                name="Recent Milestones Achieved",
                value=history_text,
                inline=False
            )
        
        await ctx.send(embed=embed)
    
    @bot.command(name="trigger_memory_event")
    @commands.has_permissions(administrator=True)
    async def trigger_memory_event(ctx):
        """Admin command to trigger a memory event for testing"""
        uid = ctx.author.id
        
        # Get personality system
        personality_system = enhanced_system.personality_system
        
        # Force memory surge state
        personality_system.last_memory_surge = datetime.now(timezone.utc) - timedelta(days=7)
        personality_system.current_state = "memory_surge"
        
        # Generate memory event dialogue
        memory_event_text = "I remember... the desert. The mission. No signal. Just silence. My squad... they... [Memory corruption detected]"
        
        await ctx.send(f"A2: {memory_event_text}")
        
        # Add a memory about this event
        from enhanced_a2 import Memory
        memory = Memory(
            user_id=uid,
            text=f"A2 experienced a memory surge about her squad in the desert",
            importance=0.9,
            memory_type="event",
            source="admin_command"
        )
        
        enhanced_system.memory_system.add_memory(memory)
        
        await ctx.send("*Memory event triggered and recorded*")
