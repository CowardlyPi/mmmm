#!/bin/bash

# Print environment information
echo "===== Starting A2 Discord Bot ====="
echo "Date: $(date)"
echo "Python version: $(python --version)"
echo "Available memory:"
free -h

# Check for required environment variables
if [ -z "$DISCORD_TOKEN" ]; then
    echo "ERROR: DISCORD_TOKEN environment variable is not set"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY environment variable is not set"
    exit 1
fi

# Check data directory
echo "Checking data directory: $DATA_DIR"
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory..."
    mkdir -p "$DATA_DIR"
fi

# Create logs directory
echo "Checking logs directory..."
if [ ! -d "$DATA_DIR/logs" ]; then
    echo "Creating logs directory..."
    mkdir -p "$DATA_DIR/logs"
fi

# Set proper permissions
echo "Setting directory permissions..."
chmod -R 777 "$DATA_DIR"

# Test write permissions
echo "Testing write permissions to data directory..."
if touch "$DATA_DIR/write_test.tmp" && rm "$DATA_DIR/write_test.tmp"; then
    echo "Data directory is writable."
else
    echo "WARNING: Data directory is not writable. Memory features may not work."
fi

# Run the bot with timeout protection
echo "Starting bot..."
python /app/main.py

# If we got here, the bot exited
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
    echo "Bot exited with code $EXIT_CODE"
    echo "Memory status:"
    free -h
    echo "Last 20 lines of log:"
    if [ -f "$DATA_DIR/logs/a2bot.log" ]; then
        tail -n 20 "$DATA_DIR/logs/a2bot.log"
    else
        echo "Log file not found"
    fi
fi

exit $EXIT_CODE
