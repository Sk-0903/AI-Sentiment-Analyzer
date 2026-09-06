"""
config.py - Configuration management for AI Sentiment Analyzer.
Loads settings from environment variables with safe defaults.
"""

import os
from dotenv import load_dotenv

# Load variables from .env if present
load_dotenv()


class Config:
    """Central configuration class."""
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() in ("true", "1", "t")
    SECRET_KEY = os.getenv("SECRET_KEY") or os.urandom(24).hex()

    # NLP Model Configuration
    # Configurable through MODEL_NAME environment variable
    MODEL_NAME = os.getenv("MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest")

    # Input validation
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 2000))

    # MongoDB Configuration
    MONGODB_URI = os.getenv("MONGODB_URI", "")
    MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "sentiment_analyzer_db")
    MONGODB_COLLECTION_NAME = os.getenv("MONGODB_COLLECTION_NAME", "analysis_history")
    # Strict 2-second timeout to prevent blocking UI when MongoDB is unavailable
    MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", 2000))

    # Dashboard & History Settings
    HISTORY_DEFAULT_LIMIT = int(os.getenv("HISTORY_DEFAULT_LIMIT", 50))
