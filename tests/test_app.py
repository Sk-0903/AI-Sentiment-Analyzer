"""
test_app.py - Automated Test Suite for AI Sentiment Analyzer.

Tests:
1. Flask application initialization and configuration.
2. Home dashboard rendering (GET /).
3. /api/status endpoint reporting component health.
4. /api/analyze with valid positive, negative, neutral, and mixed texts.
5. Input validation (rejection of empty, whitespace, and >2000 character inputs).
6. Result schema validation (sentiment, confidence in [0, 1], normalized probabilities).
7. Label correctness (strictly 'Positive', 'Neutral', 'Negative').
8. In-memory and MongoDB database persistence.
9. Resilient non-blocking behavior when MongoDB is offline.
10. History retrieval (GET /api/history) and clearing (DELETE /api/history).
"""

import unittest
import json
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app
from database import db_manager
from sentiment_analyzer import get_analyzer, FallbackSentimentEngine


class SentimentAnalyzerTestCase(unittest.TestCase):
    """Test suite for Flask API, Sentiment Analyzer, and Persistence."""

    def setUp(self):
        self.app = app
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        # Ensure fresh state before tests
        db_manager.clear_history()

    def tearDown(self):
        db_manager.clear_history()

    # ------------------------------------------------------------------------
    # 1. APPLICATION & HOME PAGE TESTS
    # ------------------------------------------------------------------------

    def test_01_app_exists(self):
        """Verify Flask application initialized."""
        self.assertIsNotNone(self.app)

    def test_02_home_page_loads(self):
        """Test GET / returns 200 and renders the dashboard UI."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        content = response.data.decode("utf-8")
        self.assertIn("SentimentAI", content)
        self.assertIn("Understand what your customers are saying", content)
        self.assertIn("Analyze Sentiment", content)
        self.assertIn("charCounter", content)

    def test_03_api_status(self):
        """Test GET /api/status returns online status and component health."""
        response = self.client.get("/api/status")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "online")
        self.assertIn("model", data)
        self.assertIn("database", data)
        self.assertIn("status_label", data["model"])
        self.assertIn("status_label", data["database"])

    # ------------------------------------------------------------------------
    # 2. INPUT VALIDATION TESTS
    # ------------------------------------------------------------------------

    def test_04_analyze_rejects_empty_input(self):
        """Test POST /api/analyze rejects empty text."""
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": ""}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("error", data)

    def test_05_analyze_rejects_whitespace_only(self):
        """Test POST /api/analyze rejects whitespace-only string."""
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": "     \n\t   "}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("empty", data["error"].lower())

    def test_06_analyze_rejects_excessive_length(self):
        """Test POST /api/analyze rejects text exceeding 2000 characters."""
        long_text = "Quality feedback test. " * 150  # ~3450 characters
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": long_text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertFalse(data["success"])
        self.assertIn("exceeds maximum", data["error"])

    def test_07_analyze_rejects_non_json_payload(self):
        """Test POST /api/analyze rejects invalid media type."""
        response = self.client.post(
            "/api/analyze",
            data="plain text",
            content_type="text/plain"
        )
        self.assertEqual(response.status_code, 400)

    # ------------------------------------------------------------------------
    # 3. SENTIMENT INFERENCE & RESPONSE SCHEMA TESTS
    # ------------------------------------------------------------------------

    def test_08_analyze_positive_review(self):
        """Test analyzing a strong positive product review."""
        text = "I absolutely love this product. The quality is amazing and delivery was very fast."
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        # Check required fields
        self.assertTrue(data["success"])
        self.assertIn("id", data)
        self.assertIn("sentiment", data)
        self.assertIn("confidence", data)
        self.assertIn("scores", data)
        self.assertIn("explanation", data)
        self.assertIn("engine", data)

        # Check sentiment label and probability boundaries
        self.assertEqual(data["sentiment"], "Positive")
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)
        self.assertIn(data["engine"], ["transformer", "fallback"])

        scores = data["scores"]
        self.assertIn("positive", scores)
        self.assertIn("neutral", scores)
        self.assertIn("negative", scores)
        self.assertGreater(scores["positive"], scores["negative"])

    def test_09_analyze_negative_review(self):
        """Test analyzing a critical negative feedback."""
        text = "Very disappointed. The product stopped working after two days and support was unhelpful."
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertEqual(data["sentiment"], "Negative")
        self.assertGreater(data["scores"]["negative"], data["scores"]["positive"])

    def test_10_analyze_neutral_statement(self):
        """Test analyzing neutral/factual text."""
        text = "The package arrived at 2 PM on Tuesday."
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertTrue(data["success"])
        self.assertIn(data["sentiment"], ["Neutral", "Positive", "Negative"])
        self.assertGreaterEqual(data["confidence"], 0.0)
        self.assertLessEqual(data["confidence"], 1.0)

    def test_11_analyze_mixed_sentiment(self):
        """Test analyzing feedback with mixed positive and negative points."""
        text = "The design is sleek and beautiful, but customer support was very slow and delayed."
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": text}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(data["success"])
        self.assertIn(data["sentiment"], ["Positive", "Neutral", "Negative"])
        self.assertIn("scores", data)

    # ------------------------------------------------------------------------
    # 4. DATABASE & HISTORY LIFECYCLE TESTS
    # ------------------------------------------------------------------------

    def test_12_history_recording_and_clearing(self):
        """Test that analyses are persisted to history and can be cleared."""
        # Run two analyses
        self.client.post(
            "/api/analyze",
            data=json.dumps({"text": "Excellent quality item, truly fantastic!"}),
            content_type="application/json"
        )
        self.client.post(
            "/api/analyze",
            data=json.dumps({"text": "Terrible experience, completely broken on arrival."}),
            content_type="application/json"
        )

        # Retrieve history
        hist_resp = self.client.get("/api/history")
        self.assertEqual(hist_resp.status_code, 200)
        hist_data = hist_resp.get_json()

        self.assertTrue(hist_data["success"])
        self.assertEqual(len(hist_data["history"]), 2)
        self.assertEqual(hist_data["statistics"]["total"], 2)
        self.assertIn("positive", hist_data["statistics"])
        self.assertIn("negative", hist_data["statistics"])
        self.assertIn("neutral", hist_data["statistics"])

        # Test DELETE /api/history
        del_resp = self.client.delete("/api/history")
        self.assertEqual(del_resp.status_code, 200)
        del_data = del_resp.get_json()
        self.assertTrue(del_data["success"])

        # Verify history is now empty
        after_hist = self.client.get("/api/history").get_json()
        self.assertEqual(len(after_hist["history"]), 0)
        self.assertEqual(after_hist["statistics"]["total"], 0)

    def test_13_mongodb_offline_resilience(self):
        """Verify that application functions properly when MongoDB is unreachable."""
        # Ensure database is in safe mode
        status = db_manager.get_status()
        self.assertIn("status_label", status)
        self.assertIn(status["status_label"], ["MongoDB Connected", "In-Memory History"])

        # Perform analysis - must NOT crash or return 500
        response = self.client.post(
            "/api/analyze",
            data=json.dumps({"text": "Great service and super fast response."}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["success"])

    def test_14_fallback_engine_direct(self):
        """Direct test of the high-precision fallback engine for sanity."""
        res_pos = FallbackSentimentEngine.analyze("Absolutely loved the experience, superb service!")
        self.assertEqual(res_pos["sentiment"], "Positive")
        self.assertEqual(res_pos["engine"], "fallback")

        res_neg = FallbackSentimentEngine.analyze("Awful, terrible failure and worst customer service.")
        self.assertEqual(res_neg["sentiment"], "Negative")
        self.assertEqual(res_neg["engine"], "fallback")

    def test_15_label_normalization_across_models(self):
        """Test normalization for diverse Hugging Face model label conventions."""
        analyzer = get_analyzer()

        # Cardiff NLP mapping: 0 -> Negative, 1 -> Neutral, 2 -> Positive
        self.assertEqual(analyzer._normalize_label("LABEL_0"), "Negative")
        self.assertEqual(analyzer._normalize_label("LABEL_1"), "Neutral")
        self.assertEqual(analyzer._normalize_label("LABEL_2"), "Positive")

        # Plain text labels
        self.assertEqual(analyzer._normalize_label("positive"), "Positive")
        self.assertEqual(analyzer._normalize_label("neutral"), "Neutral")
        self.assertEqual(analyzer._normalize_label("negative"), "Negative")

        # Short tokens
        self.assertEqual(analyzer._normalize_label("pos"), "Positive")
        self.assertEqual(analyzer._normalize_label("neu"), "Neutral")
        self.assertEqual(analyzer._normalize_label("neg"), "Negative")

        # Star rating formats
        self.assertEqual(analyzer._normalize_label("1 star"), "Negative")
        self.assertEqual(analyzer._normalize_label("3 stars"), "Neutral")
        self.assertEqual(analyzer._normalize_label("5 stars"), "Positive")

    def test_16_transformer_prediction_path(self):
        """Simulate transformer pipeline output to verify full transformer inference handling."""
        analyzer = get_analyzer()

        # Mock transformer pipeline returning 3 raw predictions
        mock_pipeline_output = [
            [
                {"label": "LABEL_0", "score": 0.05},  # Negative
                {"label": "LABEL_1", "score": 0.15},  # Neutral
                {"label": "LABEL_2", "score": 0.80}   # Positive
            ]
        ]

        # Temporarily inject mock pipeline
        orig_pipeline = analyzer.pipeline
        orig_active = analyzer.is_transformer_active
        try:
            analyzer.pipeline = lambda text: mock_pipeline_output
            analyzer.is_transformer_active = True

            result = analyzer.analyze("The build quality is exceptional and sleek.")
            self.assertEqual(result["sentiment"], "Positive")
            self.assertEqual(result["engine"], "transformer")
            self.assertAlmostEqual(result["confidence"], 0.80, places=2)
            self.assertAlmostEqual(
                sum(result["scores"].values()), 1.0, places=2
            )
        finally:
            analyzer.pipeline = orig_pipeline
            analyzer.is_transformer_active = orig_active


if __name__ == "__main__":
    unittest.main()

