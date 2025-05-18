"""
Memory usage monitoring tools for A2 Discord bot.
"""
import os
import sys
import psutil
import gc
from datetime import datetime

def get_memory_usage():
    """
    Get current memory usage information
    
    Returns:
        dict: Memory usage information
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
        "timestamp": datetime.now().isoformat()
    }

def log_memory_usage(logger, label=""):
    """
    Log current memory usage
    
    Args:
        logger: Logger instance
        label: Optional label for the log entry
    """
    memory = get_memory_usage()
    prefix = f"{label}: " if label else ""
    logger.info(f"{prefix}Memory usage: {memory['rss_mb']:.2f} MB (RSS), {memory['percent']:.1f}% of system RAM")

def force_garbage_collection():
    """
    Force a full garbage collection
    
    Returns:
        int: Number of objects collected
    """
    # Collect statistics before
    gc.collect(0)  # Collect generation 0 objects
    gc.collect(1)  # Collect generation 1 objects
    collected = gc.collect(2)  # Collect generation 2 objects (full collection)
    
    return collected

def print_memory_report(logger):
    """
    Print a comprehensive memory usage report
    
    Args:
        logger: Logger instance
    """
    mem_before = get_memory_usage()
    logger.info(f"Current memory usage: {mem_before['rss_mb']:.2f} MB")
    
    # Force garbage collection
    collected = force_garbage_collection()
    logger.info(f"Garbage collection freed {collected} objects")
    
    mem_after = get_memory_usage()
    logger.info(f"Memory usage after GC: {mem_after['rss_mb']:.2f} MB")
    logger.info(f"Memory saved by GC: {mem_before['rss_mb'] - mem_after['rss_mb']:.2f} MB")
    
    # Get system memory info
    system_mem = psutil.virtual_memory()
    logger.info(f"System memory: {system_mem.total / (1024*1024*1024):.1f} GB total, {system_mem.available / (1024*1024*1024):.1f} GB available ({system_mem.percent}% used)")
    
    # Get top 10 objects by memory usage
    try:
        import objgraph
        logger.info("Top memory-consuming object types:")
        for obj_type, count in objgraph.most_common_types(10):
            logger.info(f"  {obj_type}: {count} instances")
    except ImportError:
        logger.info("Install 'objgraph' for detailed object memory usage")

def monitor_memory_threshold(logger, threshold_mb=500, check_interval=60, callback=None):
    """
    Start background monitoring of memory usage with threshold alerts
    
    Args:
        logger: Logger instance
        threshold_mb: Memory threshold in MB
        check_interval: Check interval in seconds
        callback: Function to call when threshold is exceeded
    
    Returns:
        The monitoring task
    """
    import asyncio
    
    async def monitor_task():
        while True:
            memory = get_memory_usage()
            if memory["rss_mb"] > threshold_mb:
                logger.warning(f"Memory threshold exceeded: {memory['rss_mb']:.2f} MB / {threshold_mb} MB")
                if callback:
                    await callback()
            await asyncio.sleep(check_interval)
    
    # Create and return the task (caller needs to add it to the event loop)
    return asyncio.create_task(monitor_task())

# Memory reduction actions for high memory situations
async def reduce_memory_usage(bot):
    """
    Perform actions to reduce memory usage in high-memory situations
    
    Args:
        bot: A2Bot instance
    """
    # 1. Unload transformer models if they're loaded
    try:
        from utils.transformers_helper import unload_models
        unload_models()
    except ImportError:
        pass
    
    # 2. Clear any caches
    bot.conversation_manager.conversations.clear()
    
    # 3. Force garbage collection
    force_garbage_collection()
