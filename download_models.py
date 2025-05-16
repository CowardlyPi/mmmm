import os
import time
import sys

print("Starting model download script...")
print(f"Python version: {sys.version}")

# Try to import transformers
try:
    from transformers import pipeline
    print("Successfully imported transformers")
except ImportError as e:
    print(f"Error importing transformers: {e}")
    print("Continuing without downloading models")
    sys.exit(0)

# Create directory for models
os.makedirs("/root/.cache/huggingface", exist_ok=True)
print("Created huggingface cache directory")

# Set smaller models for faster building
summarizer_model = "sshleifer/distilbart-cnn-12-6"
toxic_model = "unitary/toxic-bert"
sentiment_model = "distilbert-base-uncased-finetuned-sst-2-english"

total_start = time.time()

try:
    print(f"Downloading summarization model: {summarizer_model}")
    start_time = time.time()
    summarizer = pipeline("summarization", model=summarizer_model)
    print(f"✓ Summarization model downloaded in {time.time() - start_time:.2f} seconds")
    
    print(f"Downloading toxicity detection model: {toxic_model}")
    start_time = time.time()
    toxic = pipeline("text-classification", model=toxic_model, top_k=None)
    print(f"✓ Toxicity model downloaded in {time.time() - start_time:.2f} seconds")
    
    print(f"Downloading sentiment analysis model: {sentiment_model}")
    start_time = time.time()
    sentiment = pipeline("sentiment-analysis", model=sentiment_model)
    print(f"✓ Sentiment model downloaded in {time.time() - start_time:.2f} seconds")
    
    print(f"All models downloaded successfully in {time.time() - total_start:.2f} seconds total!")
except Exception as e:
    print(f"Error downloading models: {e}")
    print("Continuing with build despite model download issues...")

print("Model download script completed.")
