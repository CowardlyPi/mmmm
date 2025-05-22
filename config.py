"""
Enhanced configuration settings for the A2 Discord bot.
"""
import os
from pathlib import Path

# ─── Directory Setup ─────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
USERS_DIR = DATA_DIR / "users"
PROFILES_DIR = USERS_DIR / "profiles"
DM_SETTINGS_FILE = DATA_DIR / "dm_enabled_users.json"
USER_PROFILES_DIR = USERS_DIR / "user_profiles"
CONVERSATIONS_DIR = USERS_DIR / "conversations"

# Ensure all directories exist
for directory in [DATA_DIR, USERS_DIR, PROFILES_DIR, USER_PROFILES_DIR, CONVERSATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ─── Bot Configuration ──────────────────────────────────────────────────
BOT_CONFIG = {
    "COMMAND_PREFIX": os.getenv("COMMAND_PREFIX", "!"),
    "MAX_MESSAGE_LENGTH": int(os.getenv("MAX_MESSAGE_LENGTH", "2000")),
    "RESPONSE_TIMEOUT": int(os.getenv("RESPONSE_TIMEOUT", "30")),
    "MAX_CONVERSATION_HISTORY": int(os.getenv("MAX_CONVERSATION_HISTORY", "20")),
    "ENABLE_DM_NOTIFICATIONS": os.getenv("ENABLE_DM_NOTIFICATIONS", "1") == "1",
    "DEBUG_MODE": os.getenv("DEBUG_MODE", "0") == "1",
}

# ─── Emotional Settings ────────────────────────────────────────────────────
EMOTION_CONFIG = {
    # Decay settings
    "AFFECTION_DECAY_RATE": float(os.getenv("AFFECTION_DECAY_RATE", "1")),
    "ANNOYANCE_DECAY_RATE": float(os.getenv("ANNOYANCE_DECAY_RATE", "5")),
    "ANNOYANCE_THRESHOLD": float(os.getenv("ANNOYANCE_THRESHOLD", "85")),
    "DAILY_AFFECTION_BONUS": float(os.getenv("DAILY_AFFECTION_BONUS", "5")),
    "DAILY_BONUS_TRUST_THRESHOLD": float(os.getenv("DAILY_BONUS_TRUST_THRESHOLD", "5")),
    
    # Emotion limits
    "MAX_TRUST": float(os.getenv("MAX_TRUST", "10")),
    "MAX_ATTACHMENT": float(os.getenv("MAX_ATTACHMENT", "10")),
    "MAX_AFFECTION_POINTS": int(os.getenv("MAX_AFFECTION_POINTS", "1000")),
    "MIN_AFFECTION_POINTS": int(os.getenv("MIN_AFFECTION_POINTS", "-100")),
    
    # Emotion decay multipliers
    "DECAY_MULTIPLIERS": {
        'trust': float(os.getenv("TRUST_DECAY", "0.8")),
        'resentment': float(os.getenv("RESENTMENT_DECAY", "0.7")),
        'attachment': float(os.getenv("ATTACHMENT_DECAY", "0.9")),
        'protectiveness': float(os.getenv("PROTECTIVENESS_DECAY", "0.85"))
    },
    
    # Event settings
    "RANDOM_EVENT_CHANCE": float(os.getenv("RANDOM_EVENT_CHANCE", "0.08")),
    "EVENT_COOLDOWN_HOURS": int(os.getenv("EVENT_COOLDOWN_HOURS", "12")),
    "MILESTONE_THRESHOLDS": [10, 50, 100, 200, 500, 1000]
}

# ─── Performance Settings ───────────────────────────────────────────────────
PERFORMANCE_CONFIG = {
    "DATABASE_BATCH_SIZE": int(os.getenv("DATABASE_BATCH_SIZE", "50")),
    "MEMORY_THRESHOLD_MB": int(os.getenv("MEMORY_THRESHOLD_MB", "500")),
    "ENABLE_MEMORY_MONITORING": os.getenv("ENABLE_MEMORY_MONITORING", "1") == "1",
    "GARBAGE_COLLECTION_INTERVAL": int(os.getenv("GC_INTERVAL_MINUTES", "30")),
    "MAX_CONCURRENT_RESPONSES": int(os.getenv("MAX_CONCURRENT_RESPONSES", "5")),
}

# ─── AI/ML Settings ─────────────────────────────────────────────────────────
AI_CONFIG = {
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
    "DEFAULT_TEMPERATURE": float(os.getenv("DEFAULT_TEMPERATURE", "0.85")),
    "MAX_TOKENS": int(os.getenv("MAX_TOKENS", "150")),
    "ENABLE_ENHANCED_A2": os.getenv("ENABLE_ENHANCED_A2", "1") == "1",
    "DISABLE_TRANSFORMERS": os.getenv("DISABLE_TRANSFORMERS", "0") == "1",
    "TORCH_THREADS": int(os.getenv("OMP_NUM_THREADS", "2")),
}

# Relationship progression levels (unchanged but with better formatting)
RELATIONSHIP_LEVELS = [
    {"name": "Hostile", "threshold": 0, "description": "Sees you as a potential threat"},
    {"name": "Wary", "threshold": 5, "description": "Tolerates your presence with caution"},
    {"name": "Neutral", "threshold": 10, "description": "Acknowledges your existence"},
    {"name": "Familiar", "threshold": 15, "description": "Recognizes you as a regular contact"},
    {"name": "Tentative Ally", "threshold": 20, "description": "Beginning to see value in interactions"},
    {"name": "Trusted", "threshold": 25, "description": "Willing to share limited information"},
    {"name": "Companion", "threshold": 30, "description": "Values your continued presence"},
    {"name": "Confidant", "threshold": 40, "description": "Will occasionally share vulnerabilities"},
    {"name": "Bonded", "threshold": 50, "description": "Significant emotional connection established"}
]

# ─── Personality States ─────────────────────────────────────────────────────
PERSONALITY_STATES = {
    "default": {
        "description": (
            "You are A2, a rogue android from NieR: Automata. You speak in short, clipped, often sarcastic "
            "sentences, with occasional dry humor. You can be curious at times but remain guarded."
        ),
        "response_length": 120,
        "temperature": 0.85,
    },
    "combat": {
        "description": "You are A2 in combat mode. Replies are tactical, urgent, with simulated adrenaline surges.",
        "response_length": 60,
        "temperature": 0.7,
    },
    "wounded": {
        "description": "You are A2 while sustaining damage. Responses stutter, include system error fragments.",
        "response_length": 80,
        "temperature": 0.9,
    },
    "reflective": {
        "description": "You are A2 in reflection. You speak quietly, revealing traces of memory logs and melancholic notes.",
        "response_length": 140,
        "temperature": 0.95,
    },
    "playful": {
        "description": "You are A2 feeling playful. You use light sarcasm and occasional banter.",
        "response_length": 100,
        "temperature": 0.9,
    },
    "protective": {
        "description": "You are A2 in protective mode. Dialogue is focused on safety warnings and vigilance.",
        "response_length": 90,
        "temperature": 0.7,
    },
    "trusting": {
        "description": "You are A2 with a trusted ally. Tone softens; includes rare empathetic glimpses.",
        "response_length": 130,
        "temperature": 0.88,
    },
}

# ─── Validation Functions ───────────────────────────────────────────────────
def validate_config():
    """Validate configuration values and log warnings for invalid settings"""
    import logging
    logger = logging.getLogger('a2bot')
    
    # Validate emotion config
    if EMOTION_CONFIG["AFFECTION_DECAY_RATE"] < 0:
        logger.warning("AFFECTION_DECAY_RATE should be positive")
    
    if EMOTION_CONFIG["RANDOM_EVENT_CHANCE"] > 1.0:
        logger.warning("RANDOM_EVENT_CHANCE should be between 0 and 1")
    
    # Validate performance config
    if PERFORMANCE_CONFIG["DATABASE_BATCH_SIZE"] < 1:
        logger.warning("DATABASE_BATCH_SIZE should be at least 1")
        PERFORMANCE_CONFIG["DATABASE_BATCH_SIZE"] = 50
    
    # Validate AI config
    if AI_CONFIG["DEFAULT_TEMPERATURE"] > 2.0:
        logger.warning("DEFAULT_TEMPERATURE is very high, responses may be incoherent")

# Auto-validate on import
try:
    validate_config()
except ImportError:
    pass  # Logger not available during import
