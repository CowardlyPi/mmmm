"""
Utility modules for the A2 Discord bot.
"""
# Existing imports...
from utils.transformers_helper import (
    HAVE_TRANSFORMERS, initialize_transformers,
    get_summarizer, get_toxic, get_sentiment,
    get_model_status, unload_models
)
from utils.logging_helper import setup_logging, get_logger
from utils.pagination import Paginator, BatchProcessor
from utils.memory_monitor import (
    get_memory_usage, log_memory_usage, force_garbage_collection,
    print_memory_report, monitor_memory_threshold, reduce_memory_usage
)

# NEW imports for the enhancements
from utils.validation_utils import InputValidator, RateLimiter, add_rate_limiting_to_commands
from utils.error_handler import (
    A2Logger, retry_on_failure, safe_execute, log_error_with_aggregation,
    create_enhanced_command_error_handler, setup_enhanced_logging
)
from utils.performance_utils import (
    LRUCache, cached_method, AsyncSemaphore, PerformanceMonitor,
    performance_timer, BatchProcessor as PerfBatchProcessor, MemoryOptimizer,
    global_performance_monitor, response_semaphore, optimize_for_memory_usage
)

__all__ = [
    # Existing exports...
    'HAVE_TRANSFORMERS', 'initialize_transformers',
    'get_summarizer', 'get_toxic', 'get_sentiment',
    'get_model_status', 'unload_models',
    'setup_logging', 'get_logger',
    'Paginator', 'BatchProcessor',
    'get_memory_usage', 'log_memory_usage', 'force_garbage_collection',
    'print_memory_report', 'monitor_memory_threshold', 'reduce_memory_usage',
    
    # NEW exports
    'InputValidator', 'RateLimiter', 'add_rate_limiting_to_commands',
    'A2Logger', 'retry_on_failure', 'safe_execute', 'log_error_with_aggregation',
    'create_enhanced_command_error_handler', 'setup_enhanced_logging',
    'LRUCache', 'cached_method', 'AsyncSemaphore', 'PerformanceMonitor',
    'performance_timer', 'PerfBatchProcessor', 'MemoryOptimizer',
    'global_performance_monitor', 'response_semaphore', 'optimize_for_memory_usage'
]
