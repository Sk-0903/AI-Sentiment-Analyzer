/**
 * script.js - Vanilla JavaScript client application for SentimentAI.
 * Manages form submissions, validation, Chart.js updates, and MongoDB history.
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const form = document.getElementById("sentimentForm");
    const textInput = document.getElementById("textInput");
    const charCounter = document.getElementById("charCounter");
    const validationMsg = document.getElementById("validationMsg");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const btnSpinner = document.getElementById("btnSpinner");
    const btnLabel = document.getElementById("btnLabel");
    const clearInputBtn = document.getElementById("clearInputBtn");
    const exampleBtns = document.querySelectorAll(".example-btn");

    // Result Elements
    const resultsSection = document.getElementById("resultsSection");
    const sentimentBadge = document.getElementById("sentimentBadge");
    const sentimentValue = document.getElementById("sentimentValue");
    const sentimentIcon = document.getElementById("sentimentIcon");
    const confidenceValue = document.getElementById("confidenceValue");
    const posPercent = document.getElementById("posPercent");
    const neuPercent = document.getElementById("neuPercent");
    const negPercent = document.getElementById("negPercent");
    const posBar = document.getElementById("posBar");
    const neuBar = document.getElementById("neuBar");
    const negBar = document.getElementById("negBar");
    const insightText = document.getElementById("insightText");
    const engineBadge = document.getElementById("engineBadge");
    const latencyBadge = document.getElementById("latencyBadge");

    // Status Elements
    const modelStatusPill = document.getElementById("modelStatusPill");
    const modelStatusText = document.getElementById("modelStatusText");
    const dbStatusPill = document.getElementById("dbStatusPill");
    const dbStatusText = document.getElementById("dbStatusText");

    // Stats Elements
    const statTotal = document.getElementById("statTotal");
    const statPos = document.getElementById("statPos");
    const statPosPct = document.getElementById("statPosPct");
    const statNeu = document.getElementById("statNeu");
    const statNeuPct = document.getElementById("statNeuPct");
    const statNeg = document.getElementById("statNeg");
    const statNegPct = document.getElementById("statNegPct");

    // History Elements
    const historyList = document.getElementById("historyList");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const historySourceNote = document.getElementById("historySourceNote");
    const toast = document.getElementById("toast");

    // Chart.js instance reference
    let sentimentChart = null;
    const MAX_LENGTH = parseInt(textInput.getAttribute("maxlength") || "2000", 10);

    // ========================================================================
    // TOAST NOTIFICATIONS
    // ========================================================================
    function showToast(message, duration = 3200) {
        if (!toast) return;
        toast.textContent = message;
        toast.classList.remove("hidden");
        setTimeout(() => {
            toast.classList.add("hidden");
        }, duration);
    }

    // ========================================================================
    // CHARACTER COUNTER & VALIDATION
    // ========================================================================
    function updateCharCount() {
        const count = textInput.value.length;
        charCounter.textContent = `${count} / ${MAX_LENGTH}`;

        if (count > MAX_LENGTH) {
            validationMsg.textContent = `Exceeds max length of ${MAX_LENGTH} characters.`;
            analyzeBtn.disabled = true;
        } else {
            validationMsg.textContent = "";
            analyzeBtn.disabled = false;
        }
    }

    textInput.addEventListener("input", updateCharCount);

    // Clear input button
    clearInputBtn.addEventListener("click", () => {
        textInput.value = "";
        updateCharCount();
        validationMsg.textContent = "";
        textInput.focus();
    });

    // Quick examples handler
    exampleBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const exampleText = btn.getAttribute("data-text");
            if (exampleText) {
                textInput.value = exampleText;
                updateCharCount();
                validationMsg.textContent = "";
                textInput.focus();
            }
        });
    });

    // ========================================================================
    // CHART.JS INITIALIZATION & UPDATE
    // ========================================================================
    function updateChart(posScore, neuScore, negScore) {
        const ctx = document.getElementById("sentimentChart");
        if (!ctx) return;

        const pPos = Math.round(posScore * 100);
        const pNeu = Math.round(neuScore * 100);
        const pNeg = Math.round(negScore * 100);

        const data = {
            labels: ["Positive", "Neutral", "Negative"],
            datasets: [{
                data: [pPos, pNeu, pNeg],
                backgroundColor: [
                    "#10b981", // Emerald
                    "#64748b", // Slate
                    "#ef4444"  // Red
                ],
                borderWidth: 2,
                borderColor: "#ffffff",
                hoverOffset: 4
            }]
        };

        const options = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return ` ${context.label}: ${context.raw}%`;
                        }
                    }
                }
            },
            cutout: "68%"
        };

        if (sentimentChart) {
            sentimentChart.data = data;
            sentimentChart.update();
        } else if (typeof Chart !== "undefined") {
            sentimentChart = new Chart(ctx, {
                type: "doughnut",
                data: data,
                options: options
            });
        }
    }

    // ========================================================================
    // FORM SUBMISSION & INFERENCE
    // ========================================================================
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const rawText = textInput.value;
        const text = rawText.trim();

        if (!text) {
            validationMsg.textContent = "Please enter text to analyze.";
            textInput.focus();
            return;
        }

        if (text.length > MAX_LENGTH) {
            validationMsg.textContent = `Text length exceeds maximum of ${MAX_LENGTH} characters.`;
            return;
        }

        // Set Loading State
        analyzeBtn.disabled = true;
        btnSpinner.style.display = "inline-block";
        btnLabel.textContent = "Analyzing...";
        validationMsg.textContent = "";

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                const errorMsg = data.error || "Analysis failed. Please try again.";
                validationMsg.textContent = errorMsg;
                showToast(`Error: ${errorMsg}`);
                return;
            }

            // Display Results
            renderResults(data);

            // Refresh History and Aggregate Stats
            loadHistory();

            // Auto-scroll gently to result card
            resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

        } catch (err) {
            console.error("Network or parsing error:", err);
            validationMsg.textContent = "Unable to reach server. Please ensure backend is running.";
            showToast("Network error. Please try again.");
        } finally {
            // Restore button state
            analyzeBtn.disabled = false;
            btnSpinner.style.display = "none";
            btnLabel.textContent = "Analyze Sentiment";
        }
    });

    // ========================================================================
    // RENDER ANALYSIS RESULTS
    // ========================================================================
    function renderResults(data) {
        resultsSection.classList.remove("hidden");

        const sentiment = data.sentiment || "Neutral";
        const confidence = data.confidence || 0.0;
        const scores = data.scores || { positive: 0, neutral: 0, negative: 0 };
        const engine = data.engine || "transformer";
        const modelName = data.model_name || "NLP Model";
        const latency = data.latency_ms || 0;
        const explanation = data.explanation || "";

        // Update Sentiment Badge
        sentimentValue.textContent = sentiment;
        sentimentBadge.className = "sentiment-badge";

        if (sentiment === "Positive") {
            sentimentBadge.classList.add("badge-positive");
            sentimentIcon.textContent = "✓";
        } else if (sentiment === "Negative") {
            sentimentBadge.classList.add("badge-negative");
            sentimentIcon.textContent = "✕";
        } else {
            sentimentBadge.classList.add("badge-neutral");
            sentimentIcon.textContent = "—";
        }

        // Confidence
        confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;

        // Probability Score Bars
        const pPos = (scores.positive * 100).toFixed(1);
        const pNeu = (scores.neutral * 100).toFixed(1);
        const pNeg = (scores.negative * 100).toFixed(1);

        posPercent.textContent = `${pPos}%`;
        neuPercent.textContent = `${pNeu}%`;
        negPercent.textContent = `${pNeg}%`;

        posBar.style.width = `${pPos}%`;
        neuBar.style.width = `${pNeu}%`;
        negBar.style.width = `${pNeg}%`;

        // Insight Text
        insightText.textContent = explanation;

        // Meta tags & Engine disclosure
        if (engine === "transformer") {
            engineBadge.textContent = `AI Model: Transformer (${modelName})`;
            engineBadge.style.color = "var(--sentiment-pos-text)";
        } else {
            engineBadge.textContent = "AI Model: Fallback Mode (Rule-Lexicon)";
            engineBadge.style.color = "var(--text-secondary)";
        }

        latencyBadge.textContent = `Latency: ${latency} ms`;

        // Update Chart
        updateChart(scores.positive, scores.neutral, scores.negative);
    }

    // ========================================================================
    // HISTORY & AGGREGATE STATS
    // ========================================================================
    async function loadHistory() {
        try {
            const response = await fetch("/api/history?limit=30");
            if (!response.ok) return;

            const data = await response.json();
            if (!data.success) return;

            // Render Aggregate Statistics
            if (data.statistics) {
                const s = data.statistics;
                statTotal.textContent = s.total;
                statPos.textContent = s.positive;
                statPosPct.textContent = `(${s.positive_percent}%)`;
                statNeu.textContent = s.neutral;
                statNeuPct.textContent = `(${s.neutral_percent}%)`;
                statNeg.textContent = s.negative;
                statNegPct.textContent = `(${s.negative_percent}%)`;
            }

            // Update Database Status pill
            if (data.database_status) {
                const db = data.database_status;
                dbStatusText.textContent = db.status_label;
                if (db.connected) {
                    dbStatusPill.className = "status-pill status-active";
                    historySourceNote.textContent = "Live records stored in MongoDB";
                } else {
                    dbStatusPill.className = "status-pill status-memory";
                    historySourceNote.textContent = "In-memory session records";
                }
            }

            // Render History List
            renderHistoryList(data.history || []);

        } catch (err) {
            console.error("Failed to fetch history:", err);
        }
    }

    function renderHistoryList(items) {
        historyList.innerHTML = "";

        if (!items || items.length === 0) {
            historyList.innerHTML = `
                <div class="history-empty">
                    <p>No analysis history yet. Enter text above or click an example to begin.</p>
                </div>
            `;
            return;
        }

        items.forEach(item => {
            const row = document.createElement("div");
            row.className = "history-item";

            let badgeClass = "badge-neutral";
            if (item.sentiment === "Positive") badgeClass = "badge-positive";
            if (item.sentiment === "Negative") badgeClass = "badge-negative";

            const confFormatted = `${Math.round((item.confidence || 0) * 100)}%`;
            const timeFormatted = item.formatted_time || (new Date(item.timestamp)).toLocaleTimeString();

            row.innerHTML = `
                <div class="history-item-left">
                    <span class="history-badge-small ${badgeClass}">${escapeHtml(item.sentiment)}</span>
                    <span class="history-text" title="${escapeHtml(item.text)}">${escapeHtml(item.text)}</span>
                </div>
                <div class="history-item-right">
                    <span class="history-confidence">${confFormatted} conf</span>
                    <span class="history-time">${escapeHtml(timeFormatted)}</span>
                </div>
            `;

            // Click history item to populate textarea and re-view
            row.addEventListener("click", () => {
                textInput.value = item.text;
                updateCharCount();
                renderResults({
                    sentiment: item.sentiment,
                    confidence: item.confidence,
                    scores: {
                        positive: item.positive_score,
                        neutral: item.neutral_score,
                        negative: item.negative_score
                    },
                    explanation: item.explanation,
                    engine: item.engine,
                    model_name: item.model_name,
                    latency_ms: 0
                });
                resultsSection.scrollIntoView({ behavior: "smooth", block: "nearest" });
            });

            historyList.appendChild(row);
        });
    }

    // ========================================================================
    // CLEAR HISTORY
    // ========================================================================
    clearHistoryBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all analysis history?")) {
            return;
        }

        try {
            const res = await fetch("/api/history", { method: "DELETE" });
            const data = await res.json();
            if (data.success) {
                showToast("Analysis history cleared.");
                loadHistory();
            } else {
                showToast("Failed to clear history.");
            }
        } catch (err) {
            console.error("Error clearing history:", err);
            showToast("Network error while clearing history.");
        }
    });

    // Helper: Escape HTML to avoid XSS
    function escapeHtml(str) {
        if (!str) return "";
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // Initial Load
    updateCharCount();
    loadHistory();
});
