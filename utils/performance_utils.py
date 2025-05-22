"""
Performance optimization utilities for the A2 Discord bot.
"""
import asyncio
import time
import functools
import weakref
import gc
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Optional, Dict, List
from datetime import datetime, timezone, timedelta

class LRUCache:
    """Simple LRU cache implementation with TTL support"""
    
    def __init__(self, max_size: int = 128, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.timestamps = {}
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        # Check TTL
        if time.time() - self.timestamps[key] > self.ttl_seconds:
            del self.cache[key]
            del self.timestamps[key]
            return None
        
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: Any, value: Any):
        """Put value in cache"""
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                del self.timestamps[oldest_key]
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def clear(self):
        """Clear the cache"""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)

def cached_method(ttl_seconds: int = 300, max_size: int = 128):
    """Decorator to cache method results with TTL"""
    def decorator(func: Callable) -> Callable:
        cache = LRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
        
        @functools.wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            # Create cache key
            cache_key = (id(self), args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = await func(self, *args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        @functools.wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            # Create cache key
            cache_key = (id(self), args, tuple(sorted(kwargs.items())))
            
            # Try to get from cache
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(self, *args, **kwargs)
            cache.put(cache_key, result)
            return result
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._cache = cache  # Store cache reference for clearing if needed
        return wrapper
    
    return decorator

class AsyncSemaphore:
    """Enhanced semaphore with priority support"""
    
    def __init__(self, value: int = 1):
        self._semaphore = asyncio.Semaphore(value)
        self._priority_queue = asyncio.PriorityQueue()
        self._processing = False
    
    async def acquire(self, priority: int = 5):
        """Acquire semaphore with priority (lower number = higher priority)"""
        if not self._processing and self._semaphore._value > 0:
            # Fast path - semaphore is available
            await self._semaphore.acquire()
            return
        
        # Queue the request with priority
        future = asyncio.Future()
        await self._priority_queue.put((priority, time.time(), future))
        
        if not self._processing:
            asyncio.create_task(self._process_queue())
        
        await future
    
    async def _process_queue(self):
        """Process the priority queue"""
        self._processing = True
        
        while not self._priority_queue.empty():
            try:
                priority, timestamp, future = await asyncio.wait_for(
                    self._priority_queue.get(), timeout=0.1
                )
                
                # Acquire semaphore
                await self._semaphore.acquire()
                
                # Signal the waiting coroutine
                if not future.cancelled():
                    future.set_result(None)
                    
            except asyncio.TimeoutError:
                break
            except Exception:
                continue
        
        self._processing = False
    
    def release(self):
        """Release the semaphore"""
        self._semaphore.release()

class PerformanceMonitor:
    """Monitor performance metrics for optimization"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str):
        """End timing an operation and record the duration"""
        if operation in self.start_times:
            duration = time.time() - self.start_times[operation]
            self.metrics[operation].append(duration)
            del self.start_times[operation]
            
            # Keep only last 100 measurements
            if len(self.metrics[operation]) > 100:
                self.metrics[operation] = self.metrics[operation][-100:]
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation"""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = self.metrics[operation]
        return {
            "count": len(durations),
            "avg": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "recent_avg": sum(durations[-10:]) / min(10, len(durations))
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations"""
        return {op: self.get_stats(op) for op in self.metrics.keys()}

def performance_timer(operation_name: str = None):
    """Decorator to time function execution"""
    monitor = PerformanceMonitor()
    
    def decorator(func: Callable) -> Callable:
        op_name = operation_name or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor.start_timer(op_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                monitor.end_timer(op_name)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor.start_timer(op_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                monitor.end_timer(op_name)
        
        wrapper = async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        wrapper._monitor = monitor
        return wrapper
    
    return decorator

class BatchProcessor:
    """Process items in batches to improve performance"""
    
    def __init__(self, batch_size: int = 50, max_wait_time: float = 1.0):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_items = []
        self.pending_futures = []
        self.last_batch_time = time.time()
        self._processing = False
    
    async def add_item(self, item: Any) -> Any:
        """Add item to batch and return future for result"""
        future = asyncio.Future()
        self.pending_items.append(item)
        self.pending_futures.append(future)
        
        # Process batch if full or timeout reached
        if (len(self.pending_items) >= self.batch_size or 
            time.time() - self.last_batch_time >= self.max_wait_time):
            if not self._processing:
                asyncio.create_task(self._process_batch())
        
        return await future
    
    async def _process_batch(self):
        """Process the current batch"""
        if self._processing or not self.pending_items:
            return
        
        self._processing = True
        
        # Get current batch
        items = self.pending_items[:]
        futures = self.pending_futures[:]
        
        # Clear pending
        self.pending_items.clear()
        self.pending_futures.clear()
        self.last_batch_time = time.time()
        
        try:
            # Process items (override this method in subclasses)
            results = await self.process_batch(items)
            
            # Set results for futures
            for future, result in zip(futures, results):
                if not future.cancelled():
                    future.set_result(result)
                    
        except Exception as e:
            # Set exception for all futures
            for future in futures:
                if not future.cancelled():
                    future.set_exception(e)
        finally:
            self._processing = False
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Override this method to implement batch processing logic"""
        # Default implementation - just return items as-is
        return items

class MemoryOptimizer:
    """Utilities for memory optimization"""
    
    @staticmethod
    def cleanup_cache_by_age(cache_dict: dict, max_age_seconds: int = 3600):
        """Clean up cache entries older than max_age"""
        if not hasattr(cache_dict, '_timestamps'):
            return
        
        current_time = time.time()
        keys_to_remove = []
        
        for key, timestamp in cache_dict._timestamps.items():
            if current_time - timestamp > max_age_seconds:
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            cache_dict.pop(key, None)
            cache_dict._timestamps.pop(key, None)
    
    @staticmethod
    def limit_collection_size(collection: list, max_size: int):
        """Limit collection size by removing oldest items"""
        if len(collection) > max_size:
            # Remove oldest items (assuming they're at the beginning)
            del collection[:-max_size]
    
    @staticmethod
    def force_garbage_collection():
        """Force garbage collection and return memory freed"""
        import psutil
        import os
        
        # Get memory before
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss
        
        # Force collection
        collected = gc.collect()
        
        # Get memory after
        memory_after = process.memory_info().rss
        memory_freed = memory_before - memory_after
        
        return {
            "objects_collected": collected,
            "memory_freed_mb": memory_freed / (1024 * 1024),
            "memory_before_mb": memory_before / (1024 * 1024),
            "memory_after_mb": memory_after / (1024 * 1024)
        }

# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()

# Global response rate limiter
response_semaphore = AsyncSemaphore(value=5)  # Max 5 concurrent responses

def optimize_for_memory_usage(func):
    """Decorator to optimize function for memory usage"""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after memory-intensive operations
            if hasattr(func, '_memory_intensive') and func._memory_intensive:
                gc.collect()
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after memory-intensive operations
            if hasattr(func, '_memory_intensive') and func._memory_intensive:
                gc.collect()
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
