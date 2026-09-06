# AI Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.1-black.svg)](https://flask.palletsprojects.com/)
[![Hugging Face](https://img.shields.io/badge/Transformers-Hugging%20Face-yellow.svg)](https://huggingface.co/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Supported-green.svg)](https://www.mongodb.com/)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An AI-powered sentiment analytics web application built with **Python**, **Flask**, **Hugging Face Transformers**, and **MongoDB**. The platform evaluates customer feedback, product reviews, and social media posts, returning granular sentiment classification (**Positive**, **Neutral**, **Negative**), calibrated confidence metrics, probability distributions, deterministic analytical insights, and persistent historical tracking.

Built as **Project 7 of an AI Engineering Internship Portfolio**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture & How It Works](#architecture--how-it-works)
- [NLP Model Architecture](#nlp-model-architecture)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Environment Variables](#environment-variables)
- [Local Setup & Installation (Windows)](#local-setup--installation-windows)
- [MongoDB Setup](#mongodb-setup)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)
- [Example Inputs](#example-inputs)
- [Render Cloud Deployment](#render-cloud-deployment)
- [Automated Testing](#automated-testing)
- [Future Improvements](#future-improvements)

---

## Overview

In modern customer intelligence and e-commerce, understanding user sentiment at scale is critical. **AI Sentiment Analyzer** delivers an enterprise-grade, lightweight analytics dashboard that processes user text inputs and provides:

1. **3-Class Sentiment Classification**: Distinct detection of `Positive`, `Neutral`, and `Negative` sentiments.
2. **Probability Distribution & Confidence**: Real-time softmax probability calculations across all three classes.
3. **Automated Analytical Interpretation**: Deterministic, context-aware natural language explanations highlighting customer satisfaction, mixed opinions, or severe dissatisfaction without paid third-party API dependencies.
4. **Historical Analytics & Aggregate Metrics**: Real-time MongoDB persistence with an automatic in-memory fallback for offline resilience.

---

## Key Features

- **Actual Pretrained Transformer**: Powered by Hugging Face sequence classification (`cardiffnlp/twitter-roberta-base-sentiment-latest` by default, or configurable via `MODEL_NAME`).
- **Startup Singleton Pattern**: The NLP model loads once on server startup, guaranteeing fast sub-second inferences on subsequent user requests.
- **Dynamic Label Normalization**: Automatically identifies and maps raw model labels (`LABEL_0`/`LABEL_1`/`LABEL_2`, star ratings, or string labels) to standard classes.
- **Fail-Safe Fallback Engine**: If memory constraints or network issues prevent loading heavy model weights, the application seamlessly switches to a built-in high-precision lexicon engine while transparently reporting its active state.
- **Non-Blocking MongoDB Integration**: Equipped with a 2-second connection timeout and in-memory persistence fallback. The analyzer never crashes or hangs if the database is temporarily unreachable.
- **Executive SaaS Dashboard**: Designed with clean typography, charcoal text, subtle borders, interactive Chart.js doughnut visualizations, and a responsive grid layout.

---

## Architecture & How It Works

```
┌────────────────────────────────────────────────────────┐
│                   User Text Input                      │
│     (Product Review / Customer Feedback / Tweet)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Text Validation & Sanitization             │
│        (Empty check, strip, 0-2000 length limit)       │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│        Singleton Pretrained Transformer Model          │
│   (cardiffnlp/twitter-roberta-base-sentiment-latest)   │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────────┐ ┌───────────────────────┐
│     Class Probabilities      │ │  Analytical Insights  │
│  Pos: 96.4% | Neu: 2.5% ...  │ │ (Deterministic NLP)   │
└──────────────┬───────────────┘ └───────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│        MongoDB Persistence (with In-Memory Fallback)   │
│         Stores: Text, Sentiment, Confidence, Scores    │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             Responsive SaaS UI Dashboard               │
│        (Chart.js Visualization + History Table)        │
└────────────────────────────────────────────────────────┘
```

---

## NLP Model Architecture

### Primary Model
- **Model ID**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Architecture**: RoBERTa-base fine-tuned on ~124M tweets for 3-way sentiment classification.
- **Output Classes**: `Negative` (0), `Neutral` (1), `Positive` (2).
- **Inference**: PyTorch CPU sequence classification pipeline.

### Model Config & Render Compatibility
The model is fully configurable via the `MODEL_NAME` environment variable:
- **Local / Dedicated Instances**: `cardiffnlp/twitter-roberta-base-sentiment-latest` (~499MB weights, requires ~500-650MB RAM).
- **Low-Resource Environments (e.g. Render Free Tier 512MB RAM)**: You can set `MODEL_NAME=lxyuan/distilbert-base-multilingual-cased-sentiments-student` or rely on the built-in Fallback Engine.
- **Transparent Status Indicators**: The UI clearly displays whether the Transformer or Fallback engine is active (`Transformer Model Active` vs `Fallback NLP Active`).

---

## Technologies Used

- **Backend**: Python 3.10+, Flask 3.1
- **Machine Learning**: Hugging Face Transformers, PyTorch
- **Database**: MongoDB (via `pymongo`) with In-Memory fallback
- **Server**: Gunicorn WSGI
- **Frontend**: HTML5, Modern CSS3 (Grid & Flexbox), Vanilla JavaScript
- **Data Visualization**: Chart.js 4.4 (CDN)

---

## Project Structure

```
AI-Sentiment-Analyzer/
├── app.py                  # Flask server and API route handlers
├── sentiment_analyzer.py   # Transformer singleton and fallback engine
├── database.py             # MongoDB persistence layer and in-memory store
├── config.py               # Environment configuration and defaults
├── requirements.txt        # Python dependency manifest
├── Procfile                # Render / cloud deployment process definition
├── README.md               # Project documentation
├── .gitignore              # Git ignore configuration
├── .env.example            # Environment variables template
│
├── templates/
│   └── index.html          # Responsive SaaS analytics dashboard template
│
├── static/
│   ├── style.css           # Modern SaaS styling and responsive breakpoints
│   └── script.js           # Client interactions, chart updates, and API calls
│
└── tests/
    └── test_app.py         # Automated test suite (unit and integration)
```

---

## Environment Variables

Create a `.env` file in the root directory (or copy from `.env.example`):

```bash
# MongoDB Connection String (Atlas or Local)
# If left empty, application operates in resilient In-Memory mode
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.example.mongodb.net/?retryWrites=true&w=majority
MONGODB_DB_NAME=sentiment_analyzer_db
MONGODB_COLLECTION_NAME=analysis_history

# Model Selection (Defaults to Cardiff NLP RoBERTa)
MODEL_NAME=cardiffnlp/twitter-roberta-base-sentiment-latest

# Application Server
PORT=5000
FLASK_DEBUG=False
SECRET_KEY=sentiment-ai-secure-key-2026
```

---

## Local Setup & Installation (Windows)

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.14 installed with `pip`
- Git

### 1. Clone or Navigate to the Project
```powershell
cd c:\Users\Keshav.S\OneDrive\Documents\AI-Sentiment-Analyzer
```

### 2. Create and Activate Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## MongoDB Setup

### Option A: MongoDB Atlas (Cloud - Free Tier)
1. Register for a free cluster at [mongodb.com/atlas](https://www.mongodb.com/atlas).
2. Under **Database Access**, create a database user and password.
3. Under **Network Access**, whitelist your IP or allow access from anywhere (`0.0.0.0/0`).
4. Copy the connection string and paste it into `.env`:
   ```env
   MONGODB_URI=mongodb+srv://user:password@cluster0.mongodb.net/?retryWrites=true&w=majority
   ```

### Option B: Local MongoDB
1. Start your local MongoDB Community Server on port 27017:
   ```env
   MONGODB_URI=mongodb://localhost:27017/
   ```

### Option C: Standalone In-Memory Mode (Zero Config)
If `MONGODB_URI` is omitted or MongoDB is unreachable, SentimentAI automatically enables **In-Memory History**. All features, history tables, and statistics will function without error.

---

## Running the Application

Start the Flask development server:
```powershell
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## API Documentation

### 1. Analyze Sentiment
**Endpoint**: `POST /api/analyze`  
**Content-Type**: `application/json`

#### Request Body:
```json
{
  "text": "I absolutely love this product. The quality is amazing and delivery was very fast."
}
```

#### Response (`200 OK`):
```json
{
  "success": true,
  "id": "e8d7f6b4-4b53-4811-9a99-4c126588b39e",
  "sentiment": "Positive",
  "confidence": 0.9721,
  "scores": {
    "positive": 0.9721,
    "neutral": 0.0215,
    "negative": 0.0064
  },
  "explanation": "The text expresses strong satisfaction, enthusiasm, or highly positive feedback.",
  "engine": "transformer",
  "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
  "latency_ms": 38.5,
  "timestamp": "2026-09-06T09:30:00.000Z",
  "formatted_time": "Sep 06, 2026 - 09:30:00 UTC",
  "database_mode": "mongodb"
}
```

---

### 2. Retrieve History & Stats
**Endpoint**: `GET /api/history?limit=50`

#### Response (`200 OK`):
```json
{
  "success": true,
  "history": [
    {
      "id": "e8d7f6b4-...",
      "text": "I absolutely love this product...",
      "sentiment": "Positive",
      "confidence": 0.9721,
      "positive_score": 0.9721,
      "neutral_score": 0.0215,
      "negative_score": 0.0064,
      "timestamp": "2026-09-06T09:30:00.000Z"
    }
  ],
  "statistics": {
    "total": 24,
    "positive": 13,
    "neutral": 6,
    "negative": 5,
    "positive_percent": 54.2,
    "neutral_percent": 25.0,
    "negative_percent": 20.8
  },
  "database_status": {
    "connected": true,
    "status_label": "MongoDB Connected",
    "mode": "mongodb"
  }
}
```

---

### 3. Clear History
**Endpoint**: `DELETE /api/history`

#### Response (`200 OK`):
```json
{
  "success": true,
  "message": "Analysis history cleared successfully.",
  "deleted_count": 24,
  "statistics": {
    "total": 0,
    "positive": 0,
    "neutral": 0,
    "negative": 0
  }
}
```

---

### 4. Health Check
**Endpoint**: `GET /api/status`

#### Response (`200 OK`):
```json
{
  "status": "online",
  "model": {
    "transformer_active": true,
    "status_label": "Transformer Model Active",
    "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest"
  },
  "database": {
    "connected": true,
    "status_label": "MongoDB Connected",
    "mode": "mongodb"
  },
  "max_length": 2000
}
```

---

## Example Inputs

| Category | Example Text | Expected Sentiment |
| :--- | :--- | :--- |
| **Positive** | *"I absolutely love this product. The quality is amazing and delivery was very fast."* | **Positive** (>90%) |
| **Neutral** | *"The package arrived on Tuesday afternoon as scheduled."* | **Neutral** (>70%) |
| **Negative** | *"Very disappointed. The product stopped working after two days and support was unhelpful."* | **Negative** (>90%) |
| **Mixed** | *"The design is sleek and features are great, but the battery life is quite disappointing."* | **Neutral / Leaning Negative** |

---

## Render Cloud Deployment

The repository is pre-configured for one-click deployment on [Render](https://render.com/).

### 1. Build & Start Commands
- **Environment**: Python 3
- **Build Command**:
  ```bash
  pip install -r requirements.txt
  ```
- **Start Command**:
  ```bash
  gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120
  ```

> **Note on Render Free Tier (512MB RAM)**:  
> Running `cardiffnlp/twitter-roberta-base-sentiment-latest` requires ~500-600MB RSS memory. On 512MB free tier instances, use `--workers 1` (configured in `Procfile`). If RAM is constrained, set `MODEL_NAME=lxyuan/distilbert-base-multilingual-cased-sentiments-student` in Render Environment Variables.

### 2. Render Environment Variables
Add these under **Environment**:
- `MONGODB_URI`: Your MongoDB Atlas URI.
- `SECRET_KEY`: Secure random string.
- `MODEL_NAME`: `cardiffnlp/twitter-roberta-base-sentiment-latest` (or lighter model).

---

## Automated Testing

Run the comprehensive test suite with `unittest`:

```powershell
python -m unittest discover tests -v
```

The test suite validates:
- Flask server and dashboard view rendering
- Input rejection on empty, whitespace, or >2000 character inputs
- Sentiment inference and schema conformity
- Label boundary constraints and score normalization
- Database persistence and graceful MongoDB offline recovery
- History retrieval, statistics calculation, and deletion

---

## Screenshots

*(Add application screenshots here for GitHub / portfolio display)*

- **Dashboard Hero & Input Form**: `[Placeholder: /screenshots/dashboard_hero.png]`
- **Real-time Probability Visualization**: `[Placeholder: /screenshots/results_chart.png]`
- **Historical Analysis & Aggregates**: `[Placeholder: /screenshots/history_stats.png]`

---

## Future Improvements

- **Aspect-Based Sentiment Analysis (ABSA)**: Deconstruct multi-feature feedback into distinct sentiment scores per product attribute (e.g. shipping, durability, price).
- **Batch CSV Analysis**: Allow e-commerce merchants to upload customer feedback spreadsheets for bulk sentiment processing.
- **Multilingual Support**: Integrate multilingual transformer checkpoints for cross-lingual sentiment intelligence.
- **Time-Series Sentiment Trends**: Chart sentiment trends over time to detect shifts in brand reputation.

---

## License

Distributed under the MIT License. See `LICENSE` for details.
