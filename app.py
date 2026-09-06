"""
app.py - Main Flask Application for AI Sentiment Analyzer.

Exposes:
- GET  /             : Main SaaS Analytics Dashboard
- POST /api/analyze  : Analyze text sentiment
- GET  /api/history  : Retrieve analysis history and aggregate statistics
- DELETE /api/history: Clear history
- GET  /api/status   : Live health check and component status
"""

import logging
from flask import Flask, render_template, request, jsonify, make_response
from config import Config
from sentiment_analyzer import get_analyzer
from database import db_manager

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
app.config.from_object(Config)

# Initialize NLP model once on startup
logger.info("Initializing NLP Sentiment Engine on application startup...")
analyzer = get_analyzer(Config.MODEL_NAME)
logger.info("NLP Engine status: %s", analyzer.get_status()["status_label"])


# ============================================================================
# SECURITY & CORS HEADERS
# ============================================================================

@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response


# ============================================================================
# WEB & DASHBOARD ROUTES
# ============================================================================

@app.route("/", methods=["GET"])
def index():
    """Renders the main analytics dashboard."""
    model_status = analyzer.get_status()
    db_status = db_manager.get_status()
    stats = db_manager.get_statistics()
    return render_template(
        "index.html",
        model_status=model_status,
        db_status=db_status,
        initial_stats=stats,
        max_length=Config.MAX_TEXT_LENGTH
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns current health and system statuses."""
    return jsonify({
        "status": "online",
        "model": analyzer.get_status(),
        "database": db_manager.get_status(),
        "max_length": Config.MAX_TEXT_LENGTH
    }), 200


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Analyzes submitted text for sentiment.
    Expected JSON payload: { "text": "..." }
    """
    if not request.is_json:
        return jsonify({
            "success": False,
            "error": "Request body must be valid JSON with Content-Type: application/json."
        }), 400

    data = request.get_json(silent=True)
    if data is None:
        return jsonify({
            "success": False,
            "error": "Malformed or unparseable JSON payload."
        }), 400

    raw_text = data.get("text")

    # Validate and clean input
    is_valid, cleaned_text, error_msg = analyzer.clean_and_validate(
        raw_text, max_length=Config.MAX_TEXT_LENGTH
    )
    if not is_valid:
        return jsonify({
            "success": False,
            "error": error_msg
        }), 400

    try:
        # Run inference
        analysis_result = analyzer.analyze(cleaned_text, max_length=Config.MAX_TEXT_LENGTH)

        # Persist record in MongoDB or in-memory fallback
        payload_to_save = {
            "text": cleaned_text,
            "sentiment": analysis_result["sentiment"],
            "confidence": analysis_result["confidence"],
            "scores": analysis_result["scores"],
            "explanation": analysis_result["explanation"],
            "engine": analysis_result["engine"],
            "model_name": analysis_result["model_name"]
        }
        saved_record = db_manager.save_analysis(payload_to_save)

        # Merge results for response
        response_payload = {
            "success": True,
            "id": saved_record["id"],
            "sentiment": analysis_result["sentiment"],
            "confidence": analysis_result["confidence"],
            "scores": analysis_result["scores"],
            "explanation": analysis_result["explanation"],
            "engine": analysis_result["engine"],
            "model_name": analysis_result["model_name"],
            "latency_ms": analysis_result["latency_ms"],
            "timestamp": saved_record["timestamp"],
            "formatted_time": saved_record["formatted_time"],
            "database_mode": db_manager.get_status()["mode"]
        }
        return jsonify(response_payload), 200

    except ValueError as val_err:
        return jsonify({"success": False, "error": str(val_err)}), 400
    except Exception as exc:
        logger.error("Unexpected error during analysis: %s", exc, exc_info=True)
        return jsonify({
            "success": False,
            "error": "Unable to analyze the text right now. Please try again."
        }), 500


@app.route("/api/history", methods=["GET"])
def api_history():
    """
    Returns recent analysis history from MongoDB / In-Memory store,
    along with aggregate metrics and current database status.
    """
    try:
        limit = request.args.get("limit", default=Config.HISTORY_DEFAULT_LIMIT, type=int)
        if limit < 1 or limit > 200:
            limit = Config.HISTORY_DEFAULT_LIMIT

        history_items = db_manager.get_history(limit=limit)
        stats = db_manager.get_statistics()
        db_status = db_manager.get_status()

        return jsonify({
            "success": True,
            "history": history_items,
            "statistics": stats,
            "database_status": db_status
        }), 200
    except Exception as exc:
        logger.error("Failed to retrieve history: %s", exc)
        return jsonify({
            "success": False,
            "error": "Unable to retrieve analysis history at this time."
        }), 500


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    """Clears analysis history."""
    try:
        deleted = db_manager.clear_history()
        stats = db_manager.get_statistics()
        return jsonify({
            "success": True,
            "message": "Analysis history cleared successfully.",
            "deleted_count": deleted,
            "statistics": stats
        }), 200
    except Exception as exc:
        logger.error("Failed to clear history: %s", exc)
        return jsonify({
            "success": False,
            "error": "Failed to clear analysis history."
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found_handler(e):
    if request.path.startswith("/api/"):
        return jsonify({"success": False, "error": "Endpoint not found."}), 404
    return render_template("index.html", error="Page not found"), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"success": False, "error": "Method not allowed for this endpoint."}), 405


@app.errorhandler(500)
def server_error_handler(e):
    logger.error("Internal Server Error: %s", e)
    return jsonify({
        "success": False,
        "error": "Internal server error. Please try again later."
    }), 500


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=Config.PORT,
        debug=Config.DEBUG
    )
