"""
Helper functions for transformer models with proper error handling and lazy loading.
"""
import os
import sys
import time
import logging
from pathlib import Path
from functools import lru_cache

# Configure PyTorch early to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NO_CUDA"] = "1"

# Set flag to track transformers availability
HAVE_TRANSFORMERS = False
transformers_disabled = os.getenv("DISABLE_TRANSFORMERS", "0") == "1"

# Track initialization state of each model
_models_initialized = {
    "summarizer": False,
    "toxic": False,
    "sentiment": False
}

# Model cache
_model_cache = {}

# Get logger
logger = logging.getLogger('a2bot')

def _can_load_transformers():
    """Check if transformers can be loaded"""
    global HAVE_TRANSFORMERS
    
    if transformers_disabled:
        logger.info("Transformer models disabled via environment variable")
        return False
        
    if HAVE_TRANSFORMERS:
        return True
        
    try:
        from transformers import pipeline
        import torch
        
        # Configure torch for efficiency
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        
        # Print torch configuration for debugging
        logger.info(f"PyTorch configuration:")
        logger.info(f"  Version: {torch.__version__}")
        logger.info(f"  Threads: {torch.get_num_threads()}")
        logger.info(f"  Available devices: CPU only (forced)")
        
        HAVE_TRANSFORMERS = True
        return True
        
    except ImportError as e:
        logger.warning(f"Transformers library not available: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking transformers: {e}")
        return False

def initialize_transformers():
    """
    Initialize the transformers library.
    
    Note: This does NOT load any models, just checks if transformers is available.
    Models will be loaded on-demand when first requested.
    """
    return _can_load_transformers()

def _load_model(model_type):
    """
    Load a specific model if not already loaded
    
    Args:
        model_type: Type of model to load ('summarizer', 'toxic', or 'sentiment')
        
    Returns:
        The loaded model or None if loading failed
    """
    global _models_initialized, _model_cache
    
    # Return from cache if already loaded
    if _models_initialized.get(model_type, False) and model_type in _model_cache:
        return _model_cache[model_type]
        
    # Check if transformers can be loaded
    if not _can_load_transformers():
        return None
        
    try:
        from transformers import pipeline
        
        logger.info(f"Lazy loading {model_type} model...")
        start_time = time.time()
        
        # Load the requested model
        if model_type == "summarizer":
            model = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        elif model_type == "toxic":
            model = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
        elif model_type == "sentiment":
            model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
            
        # Record successful loading
        _models_initialized[model_type] = True
        _model_cache[model_type] = model
        
        elapsed_time = time.time() - start_time
        logger.info(f"✓ {model_type.capitalize()} model loaded in {elapsed_time:.2f} seconds")
        
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_type} model: {e}")
        _models_initialized[model_type] = False
        return None

def get_summarizer():
    """Get the summarization model, loading it if needed."""
    return _load_model("summarizer")

def get_toxic():
    """Get the toxicity model, loading it if needed."""
    return _load_model("toxic")

def get_sentiment():
    """Get the sentiment model, loading it if needed."""
    return _load_model("sentiment")

# Helper to check initialization status
def get_model_status():
    """Get the initialization status of all models"""
    return {
        "transformers_available": HAVE_TRANSFORMERS,
        "models_initialized": _models_initialized.copy(),
        "transformers_disabled": transformers_disabled
    }

# Memory management function
def unload_models():
    """Unload all models to free memory"""
    global _models_initialized, _model_cache
    
    # Clear cache
    _model_cache.clear()
    
    # Reset initialization flags
    for model_type in _models_initialized:
        _models_initialized[model_type] = False
        
    # Force garbage collection
    import gc
    gc.collect()
    
    logger.info("All transformer models unloaded to free memory")
