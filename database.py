"""
database.py - MongoDB Persistence Layer with Resilient In-Memory Fallback.

Provides:
- MongoDB connection using MONGODB_URI.
- Strict 2-second server selection timeout to prevent UI hanging.
- Resilient In-Memory store if MongoDB is offline or unconfigured.
- Distinct status reporting: "MongoDB Connected" vs "In-Memory History".
- Full CRUD operations: save_analysis, get_history, clear_history, get_statistics.
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from config import Config

logger = logging.getLogger(__name__)

# Global in-memory fallback list (capped to prevent memory leaks)
_IN_MEMORY_HISTORY: List[Dict[str, Any]] = []
_MAX_IN_MEMORY_ITEMS = 200


class DatabaseManager:
    """Manages MongoDB connection with transparent in-memory fallback."""

    def __init__(self):
        self.uri = Config.MONGODB_URI.strip()
        self.db_name = Config.MONGODB_DB_NAME
        self.collection_name = Config.MONGODB_COLLECTION_NAME
        self.timeout_ms = Config.MONGODB_TIMEOUT_MS

        self.client = None
        self.db = None
        self.collection = None
        self.is_connected = False
        self.connection_error: Optional[str] = None

        self._init_connection()

    def _init_connection(self) -> None:
        """Attempts to connect to MongoDB with a short timeout."""
        if not self.uri:
            logger.info("No MONGODB_URI provided. Running in In-Memory History mode.")
            self.is_connected = False
            self.connection_error = "No MONGODB_URI configured."
            return

        try:
            from pymongo import MongoClient
            from pymongo.errors import PyMongoError

            self.client = MongoClient(
                self.uri,
                serverSelectionTimeoutMS=self.timeout_ms,
                connectTimeoutMS=self.timeout_ms,
                socketTimeoutMS=self.timeout_ms
            )
            # Ping database to verify connection immediately
            self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            self.is_connected = True
            self.connection_error = None
            logger.info("Successfully connected to MongoDB (%s)", self.db_name)
        except Exception as e:
            self.is_connected = False
            self.connection_error = str(e)
            self.client = None
            self.db = None
            self.collection = None
            # Log technical details server-side only; do not crash
            logger.warning("MongoDB unavailable (%s). Falling back to In-Memory History.", e)

    def get_status(self) -> Dict[str, Any]:
        """Returns database status for UI indicators."""
        if self.is_connected:
            return {
                "connected": True,
                "status_label": "MongoDB Connected",
                "mode": "mongodb",
                "database": self.db_name
            }
        return {
            "connected": False,
            "status_label": "In-Memory History",
            "mode": "in_memory",
            "warning": "MongoDB offline or not configured. Data is stored in-memory."
        }

    def save_analysis(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Saves an analysis record to MongoDB or in-memory fallback.
        Ensures record structure with id, timestamps, and formatted scores.
        """
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record = {
            "id": record_id,
            "text": analysis_data.get("text", ""),
            "sentiment": analysis_data.get("sentiment", "Neutral"),
            "confidence": float(analysis_data.get("confidence", 0.0)),
            "positive_score": float(analysis_data.get("scores", {}).get("positive", 0.0)),
            "neutral_score": float(analysis_data.get("scores", {}).get("neutral", 0.0)),
            "negative_score": float(analysis_data.get("scores", {}).get("negative", 0.0)),
            "explanation": analysis_data.get("explanation", ""),
            "engine": analysis_data.get("engine", "transformer"),
            "model_name": analysis_data.get("model_name", ""),
            "timestamp": now.isoformat(),
            "formatted_time": now.strftime("%b %d, %Y - %H:%M:%S UTC")
        }

        # Try saving to MongoDB if connected
        if self.is_connected and self.collection is not None:
            try:
                # Include MongoDB _id matching our record id
                mongo_doc = dict(record)
                mongo_doc["_id"] = record_id
                self.collection.insert_one(mongo_doc)
                return record
            except Exception as e:
                logger.warning("Failed writing to MongoDB (%s). Writing to in-memory store.", e)
                # Fall through to in-memory fallback
                self.is_connected = False

        # In-memory storage
        global _IN_MEMORY_HISTORY
        _IN_MEMORY_HISTORY.insert(0, record)
        if len(_IN_MEMORY_HISTORY) > _MAX_IN_MEMORY_ITEMS:
            _IN_MEMORY_HISTORY.pop()

        return record

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent analysis history, newest first."""
        if self.is_connected and self.collection is not None:
            try:
                cursor = self.collection.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                return list(cursor)
            except Exception as e:
                logger.warning("MongoDB read error (%s). Using in-memory history.", e)
                self.is_connected = False

        return _IN_MEMORY_HISTORY[:limit]

    def clear_history(self) -> int:
        """Clears all stored history."""
        deleted_count = 0
        if self.is_connected and self.collection is not None:
            try:
                res = self.collection.delete_many({})
                deleted_count += res.deleted_count
            except Exception as e:
                logger.warning("MongoDB clear error (%s).", e)
                self.is_connected = False

        global _IN_MEMORY_HISTORY
        deleted_count += len(_IN_MEMORY_HISTORY)
        _IN_MEMORY_HISTORY.clear()
        return deleted_count

    def get_statistics(self) -> Dict[str, Any]:
        """Calculates total analyses and sentiment distribution."""
        history = self.get_history(limit=500)
        total = len(history)
        pos = sum(1 for item in history if item.get("sentiment") == "Positive")
        neu = sum(1 for item in history if item.get("sentiment") == "Neutral")
        neg = sum(1 for item in history if item.get("sentiment") == "Negative")

        return {
            "total": total,
            "positive": pos,
            "neutral": neu,
            "negative": neg,
            "positive_percent": round((pos / total * 100), 1) if total > 0 else 0,
            "neutral_percent": round((neu / total * 100), 1) if total > 0 else 0,
            "negative_percent": round((neg / total * 100), 1) if total > 0 else 0
        }


# Singleton database instance
db_manager = DatabaseManager()
