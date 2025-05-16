"""
Helper functions for transformer models with proper error handling.
Extracted from bot_helper.py
"""
import os
import sys
import time
from pathlib import Path

# Configure PyTorch early to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NO_CUDA"] = "1"

# Set flag to track transformers availability
HAVE_TRANSFORMERS = False
transformers_disabled = os.getenv("DISABLE_TRANSFORMERS", "0") == "1"

# Initialize model placeholders
local_summarizer = None
local_toxic = None
local_sentiment = None

def initialize_transformers():
    """Initialize transformer models with proper error handling."""
    global HAVE_TRANSFORMERS, local_summarizer, local_toxic, local_sentiment
    
    if transformers_disabled:
        print("Transformer models disabled via environment variable")
        return False
        
    print("Initializing transformer models...")
    start_time = time.time()
    
    try:
        from transformers import pipeline
        import torch
        
        # Configure torch for efficiency
        torch.set_num_threads(int(os.getenv("OMP_NUM_THREADS", "2")))
        
        # Print torch configuration for debugging
        print(f"PyTorch configuration:")
        print(f"  Version: {torch.__version__}")
        print(f"  Threads: {torch.get_num_threads()}")
        print(f"  Available devices: CPU only (forced)")
        
        # Initialize models one by one with error handling
        try:
            print("Loading summarization model...")
            local_summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            print("✓ Summarization model loaded")
        except Exception as e:
            print(f"ERROR loading summarization model: {e}")
            local_summarizer = None
            
        try:
            print("Loading toxicity model...")
            local_toxic = pipeline("text-classification", model="unitary/toxic-bert", top_k=None)
            print("✓ Toxicity model loaded")
        except Exception as e:
            print(f"ERROR loading toxicity model: {e}")
            local_toxic = None
            
        try:
            print("Loading sentiment model...")
            local_sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            print("✓ Sentiment model loaded")
        except Exception as e:
            print(f"ERROR loading sentiment model: {e}")
            local_sentiment = None
        
        elapsed_time = time.time() - start_time
        print(f"Model initialization completed in {elapsed_time:.2f} seconds")
        
        # Set flag only if at least one model was loaded successfully
        HAVE_TRANSFORMERS = any([local_summarizer, local_toxic, local_sentiment])
        return HAVE_TRANSFORMERS
        
    except ImportError as e:
        print(f"Transformers library not available: {e}")
        HAVE_TRANSFORMERS = False
        return False
    except Exception as e:
        print(f"Unexpected error initializing transformers: {e}")
        HAVE_TRANSFORMERS = False
        return False

def get_summarizer():
    """Get the summarization model, initializing if needed."""
    if not HAVE_TRANSFORMERS:
        return None
    return local_summarizer

def get_toxic():
    """Get the toxicity model, initializing if needed."""
    if not HAVE_TRANSFORMERS:
        return None
    return local_toxic

def get_sentiment():
    """Get the sentiment model, initializing if needed."""
    if not HAVE_TRANSFORMERS:
        return None
    return local_sentiment
