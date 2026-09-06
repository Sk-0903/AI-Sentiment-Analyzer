"""
sentiment_analyzer.py - Core NLP and Sentiment Analysis Engine.

Provides:
1. Singleton pretrained Hugging Face Transformer sentiment model.
2. Background non-blocking model loader with local cache priority.
3. Configurable model selection via MODEL_NAME.
4. Automatic label inspection and normalization (Positive, Neutral, Negative).
5. Deterministic, human-readable analytical interpretation generation.
6. High-precision rule/lexicon-based fallback engine for resource/environment resilience.
7. Transparent reporting of active engine: "Transformer Model Active" vs "Fallback NLP Active".
"""

import os
import re
import time
import threading
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# LEXICON & RULE-BASED FALLBACK ENGINE
# Used if Transformer fails to load due to RAM limits, missing weights, or network.
# ============================================================================

POSITIVE_LEXICON = {
    "love", "loved", "loving", "loves", "great", "excellent", "amazing", "wonderful",
    "fantastic", "best", "good", "nice", "awesome", "perfect", "pleased", "happy",
    "delighted", "superb", "brilliant", "outstanding", "impressed", "impressive",
    "quality", "fast", "helpful", "friendly", "satisfied", "satisfaction", "recommend",
    "smooth", "reliable", "flawless", "favorite", "valuable", "exceptional", "easy"
}

NEGATIVE_LEXICON = {
    "hate", "hated", "hating", "terrible", "horrible", "awful", "bad", "worse",
    "worst", "poor", "disappointed", "disappointing", "disappointment", "broken",
    "defective", "useless", "waste", "slow", "delayed", "rude", "unhelpful",
    "fail", "failed", "failing", "failure", "scam", "cheap", "damaged", "annoying",
    "frustrating", "frustrated", "regret", "error", "problem", "issues", "stopped"
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.8, "absolutely": 1.7,
    "incredibly": 1.7, "super": 1.4, "totally": 1.5, "highly": 1.5,
    "deeply": 1.4, "exceptionally": 1.7
}

NEGATORS = {"not", "never", "no", "hardly", "barely", "scarcely", "without", "neither", "nor"}

CONTRASTS = {"but", "however", "although", "though", "yet", "nevertheless", "still"}


class FallbackSentimentEngine:
    """
    High-precision, lexicon and rule-based fallback analyzer.
    Activated only when Transformer model cannot run due to memory or environment constraints.
    """

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        words = re.findall(r"\b\w+(?:'\w+)?\b|[.,!?;]", text.lower())
        pos_score = 0.0
        neg_score = 0.0
        intensifier = 1.0
        negated = False
        contrast_boost = 1.0

        for i, word in enumerate(words):
            if word in CONTRASTS:
                contrast_boost = 1.3
                negated = False
                intensifier = 1.0
                continue

            if word in INTENSIFIERS:
                intensifier = INTENSIFIERS[word]
                continue

            if word in NEGATORS:
                negated = not negated
                continue

            if word in ('.', '!', '?', ';'):
                negated = False
                intensifier = 1.0
                continue

            # Check positive word
            if word in POSITIVE_LEXICON:
                impact = 1.0 * intensifier * contrast_boost
                if negated:
                    neg_score += impact * 0.9
                else:
                    pos_score += impact
                negated = False
                intensifier = 1.0

            # Check negative word
            elif word in NEGATIVE_LEXICON:
                impact = 1.2 * intensifier * contrast_boost
                if negated:
                    pos_score += impact * 0.8
                else:
                    neg_score += impact
                negated = False
                intensifier = 1.0

        total = pos_score + neg_score
        if total == 0:
            p_pos = 0.15
            p_neg = 0.15
            p_neu = 0.70
        else:
            p_pos = round(pos_score / (total + 1.2), 4)
            p_neg = round(neg_score / (total + 1.2), 4)
            p_neu = round(max(0.05, 1.0 - (p_pos + p_neg)), 4)
            # Normalize so sum is exactly 1.0
            s_sum = p_pos + p_neg + p_neu
            p_pos = round(p_pos / s_sum, 4)
            p_neg = round(p_neg / s_sum, 4)
            p_neu = round(1.0 - p_pos - p_neg, 4)

        scores = {"positive": p_pos, "neutral": p_neu, "negative": p_neg}
        top_sentiment = max(scores, key=scores.get).capitalize()
        confidence = round(scores[top_sentiment.lower()], 4)

        return {
            "sentiment": top_sentiment,
            "confidence": confidence,
            "scores": scores,
            "engine": "fallback",
            "model_name": "rule-lexicon-fallback"
        }


# ============================================================================
# MAIN SENTIMENT ANALYZER CLASS (TRANSFORMER + FALLBACK RESILIENCE)
# ============================================================================

class SentimentAnalyzer:
    """
    Singleton NLP Sentiment Analyzer using Hugging Face Transformers.
    Safely falls back to FallbackSentimentEngine if system is out of memory or offline.
    """

    _instance: Optional["SentimentAnalyzer"] = None

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(
            "MODEL_NAME", "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.pipeline = None
        self.is_transformer_active = False
        self.status_message = "Fallback NLP Active"
        self.init_error = None
        self._load_lock = threading.Lock()
        self._load_model()

    @classmethod
    def get_instance(cls, model_name: Optional[str] = None) -> "SentimentAnalyzer":
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    def _load_model(self) -> None:
        """Loads the transformer model, prioritizing local cache with safe background fallback."""
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HUB_DISABLE_XET"] = "1"

        def _do_load():
            try:
                from transformers import pipeline

                # First try loading from local cache only (instantaneous)
                try:
                    p = pipeline(
                        "sentiment-analysis",
                        model=self.model_name,
                        top_k=None,
                        device=-1,
                        model_kwargs={"local_files_only": True}
                    )
                    with self._load_lock:
                        self.pipeline = p
                        self.is_transformer_active = True
                        self.status_message = "Transformer Model Active"
                        self.init_error = None
                    logger.info("Transformer model '%s' loaded from local cache!", self.model_name)
                    return
                except Exception:
                    pass

                # If not cached, attempt remote load
                p = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    top_k=None,
                    device=-1
                )
                with self._load_lock:
                    self.pipeline = p
                    self.is_transformer_active = True
                    self.status_message = "Transformer Model Active"
                    self.init_error = None
                logger.info("Transformer model '%s' successfully loaded!", self.model_name)

            except Exception as exc:
                with self._load_lock:
                    self.pipeline = None
                    self.is_transformer_active = False
                    self.init_error = str(exc)
                    self.status_message = "Fallback NLP Active"
                logger.warning(
                    "Could not load Transformer model '%s' (%s). Fallback NLP engine active.",
                    self.model_name, exc
                )

        # Start loading in thread; wait up to 4 seconds for fast local load
        loader_thread = threading.Thread(target=_do_load, daemon=True, name="ModelLoaderThread")
        loader_thread.start()
        loader_thread.join(timeout=4.0)

        if not self.is_transformer_active:
            self.status_message = "Fallback NLP Active"
            logger.info("Operating in Fallback NLP Active mode while model initializes in background.")

    def get_status(self) -> Dict[str, Any]:
        """Returns current engine status."""
        with self._load_lock:
            return {
                "transformer_active": self.is_transformer_active,
                "status_label": "Transformer Model Active" if self.is_transformer_active else "Fallback NLP Active",
                "model_name": self.model_name if self.is_transformer_active else "rule-lexicon-fallback",
                "error": self.init_error if not self.is_transformer_active else None
            }

    @staticmethod
    def clean_and_validate(text: Any, max_length: int = 2000) -> Tuple[bool, str, Optional[str]]:
        """
        Validates and cleans user input.
        Returns (is_valid, cleaned_text, error_message).
        """
        if text is None:
            return False, "", "Text input is required."

        if not isinstance(text, str):
            return False, "", "Invalid input type. Expected a string."

        cleaned = text.strip()

        if not cleaned:
            return False, "", "Text cannot be empty or contain only whitespace."

        if len(cleaned) > max_length:
            return False, "", f"Text exceeds maximum allowed length of {max_length} characters (received {len(cleaned)})."

        return True, cleaned, None

    def _normalize_label(self, raw_label: str) -> str:
        """
        Normalizes any Hugging Face model output label to standard Positive, Neutral, Negative.
        Handles cardiffnlp LABEL_0/1/2, star ratings, and text labels.
        """
        label_str = str(raw_label).strip().lower()

        # Direct name checks
        if "pos" in label_str:
            return "Positive"
        if "neu" in label_str:
            return "Neutral"
        if "neg" in label_str:
            return "Negative"

        # Cardiff NLP mapping: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        if "cardiffnlp" in self.model_name.lower():
            if label_str in ("label_0", "0"):
                return "Negative"
            if label_str in ("label_1", "1"):
                return "Neutral"
            if label_str in ("label_2", "2"):
                return "Positive"

        # General numeric / star mappings
        if label_str in ("label_0", "0", "1 star", "2 stars"):
            return "Negative"
        if label_str in ("label_1", "1", "3 stars"):
            return "Neutral"
        if label_str in ("label_2", "2", "4 stars", "5 stars"):
            return "Positive"

        return "Neutral"

    def _generate_explanation(self, sentiment: str, confidence: float, text: str, scores: Dict[str, float]) -> str:
        """
        Generates a clear, professional, deterministic human-readable analytical summary.
        Does not require external paid APIs.
        """
        pos_s = scores.get("positive", 0.0)
        neg_s = scores.get("negative", 0.0)
        neu_s = scores.get("neutral", 0.0)

        # Check for mixed sentiment signs
        has_contrast = bool(re.search(r"\b(but|however|although|though|yet|except)\b", text, re.I))

        if sentiment == "Positive":
            if confidence >= 0.85:
                return "The text expresses strong satisfaction, enthusiasm, or highly positive feedback."
            elif has_contrast:
                return "The overall sentiment leans positive, though the text mentions minor caveats or balanced observations."
            else:
                return "The text expresses a generally favorable opinion or positive evaluation."

        elif sentiment == "Negative":
            if confidence >= 0.85:
                return "The text expresses clear dissatisfaction, frustration, or severe criticism."
            elif has_contrast:
                return "The overall tone leans negative despite some mixed or neutral elements in the feedback."
            else:
                return "The text conveys an unfavorable perspective or points of disappointment."

        else:  # Neutral
            if abs(pos_s - neg_s) < 0.10 and (pos_s > 0.25 and neg_s > 0.25):
                return "The text contains a balanced mix of both positive and critical elements, resulting in a neutral overall score."
            elif confidence >= 0.70:
                return "The text is primarily factual, objective, or expresses neither strong positive nor negative sentiment."
            else:
                return "The text reflects a moderate or balanced perspective with no pronounced emotional polarity."

    def analyze(self, text: str, max_length: int = 2000) -> Dict[str, Any]:
        """
        Full analysis pipeline:
        1. Validate & clean input
        2. Run Transformer inference (or Fallback if unavailable)
        3. Extract normalized scores & confidence
        4. Generate insight explanation
        5. Return structured dictionary
        """
        t0 = time.time()
        is_valid, cleaned_text, error = self.clean_and_validate(text, max_length)
        if not is_valid:
            raise ValueError(error)

        result: Dict[str, Any]

        with self._load_lock:
            pipe = self.pipeline
            transformer_active = self.is_transformer_active

        if transformer_active and pipe is not None:
            try:
                # Run through pipeline
                raw_outputs = pipe(cleaned_text[:512])
                if isinstance(raw_outputs, list) and len(raw_outputs) > 0 and isinstance(raw_outputs[0], list):
                    predictions = raw_outputs[0]
                elif isinstance(raw_outputs, list):
                    predictions = raw_outputs
                else:
                    predictions = [raw_outputs]

                scores_map = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
                for item in predictions:
                    lbl = self._normalize_label(item.get("label", ""))
                    score_val = float(item.get("score", 0.0))
                    scores_map[lbl.lower()] = max(scores_map[lbl.lower()], score_val)

                # Normalize probabilities to sum to 1.0
                total_prob = sum(scores_map.values())
                if total_prob > 0:
                    scores_map = {k: round(v / total_prob, 4) for k, v in scores_map.items()}
                else:
                    scores_map = {"positive": 0.33, "neutral": 0.34, "negative": 0.33}

                top_class = max(scores_map, key=scores_map.get).capitalize()
                confidence = round(scores_map[top_class.lower()], 4)

                result = {
                    "sentiment": top_class,
                    "confidence": confidence,
                    "scores": scores_map,
                    "engine": "transformer",
                    "model_name": self.model_name
                }
            except Exception as e:
                logger.error("Transformer inference error: %s. Falling back.", e)
                result = FallbackSentimentEngine.analyze(cleaned_text)
        else:
            # Fallback mode
            result = FallbackSentimentEngine.analyze(cleaned_text)

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        explanation = self._generate_explanation(
            result["sentiment"], result["confidence"], cleaned_text, result["scores"]
        )

        return {
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "scores": {
                "positive": result["scores"]["positive"],
                "neutral": result["scores"]["neutral"],
                "negative": result["scores"]["negative"]
            },
            "explanation": explanation,
            "engine": result["engine"],
            "model_name": result["model_name"],
            "latency_ms": elapsed_ms,
            "text_length": len(cleaned_text)
        }


# Global helper accessor
def get_analyzer(model_name: Optional[str] = None) -> SentimentAnalyzer:
    return SentimentAnalyzer.get_instance(model_name)
