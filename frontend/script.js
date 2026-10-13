/**
 * Frontend logic for the AI Question Generator UI.
 */

// ============================================================
// Config
// ============================================================
const API_BASE = "";

// ============================================================
// Small helpers
// ============================================================
function $(id) {
    return document.getElementById(id);
}

function escapeHtml(str) {
    if (str === null || str === undefined) return "";
    return String(str)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function labelClass(label) {
    if (label === "EXACT_DUPLICATE") return "exact";
    if (label === "SEMANTIC_DUPLICATE") return "semantic";
    return "different";
}

function scoreClass(value, kind) {
    if (kind === "exact") {
        if (value >= 90) return "high";
        if (value >= 60) return "mid";
        return "low";
    }
    if (value >= 0.8) return "high";
    if (value >= 0.6) return "mid";
    return "low";
}

async function apiCall(path, options = {}) {
    const res = await fetch(API_BASE + path, {
        headers: { "Content-Type": "application/json" },
        ...options,
    });
    if (!res.ok) {
        const err = await res.text();
        throw new Error(err || `Request failed with status ${res.status}`);
    }
    return res.json();
}

function setLoading(btn, isLoading) {
    const label = btn.querySelector(".btn-label");
    const spinner = btn.querySelector(".spinner");
    btn.disabled = isLoading;
    if (spinner) spinner.hidden = !isLoading;
    if (label) label.style.opacity = isLoading ? "0.7" : "1";
}

// ============================================================
// Tab switching
// ============================================================
function initTabs() {
    const buttons = document.querySelectorAll(".nav-btn");
    const panels = document.querySelectorAll(".tab-panel");

    buttons.forEach((btn) => {
        btn.addEventListener("click", () => {
            const target = btn.dataset.tab;
            buttons.forEach((b) => b.classList.toggle("active", b === btn));
            panels.forEach((p) => {
                p.classList.toggle("active", p.id === `tab-${target}`);
            });
        });
    });
}

// ============================================================
// API health check
// ============================================================
async function checkHealth() {
    const badge = $("apiStatus");
    const text = $("apiStatusText");
    try {
        await apiCall("/api/health");
        badge.classList.add("online");
        badge.classList.remove("offline");
        text.textContent = "API online";
    } catch (e) {
        badge.classList.add("offline");
        badge.classList.remove("online");
        text.textContent = "API offline";
    }
}

// ============================================================
// Tab 1 — Generate
// ============================================================
function renderComparisons(rows) {
    const tbody = $("comparisonTable").querySelector("tbody");
    tbody.innerHTML = rows
        .map((r) => {
            const cls = labelClass(r.label);
            const exactCls = scoreClass(r.exact_score, "exact");
            const semCls = scoreClass(r.semantic_score, "semantic");
            return `
                <tr>
                    <td><strong>${escapeHtml(r.existing_id)}</strong></td>
                    <td>${escapeHtml(r.existing_question)}</td>
                    <td><span class="label ${cls}">${escapeHtml(r.label)}</span></td>
                    <td><span class="score ${exactCls}">${r.exact_score.toFixed(2)}</span></td>
                    <td><span class="score ${semCls}">${Math.round(r.semantic_score * 100)}%</span></td>
                </tr>
            `;
        })
        .join("");
}

function renderContext(questions) {
    const list = $("contextList");
    list.innerHTML = questions
        .map((q) => `<li>${escapeHtml(q)}</li>`)
        .join("");
}

async function onGenerate() {
    const btn = $("generateBtn");
    const topic = $("topicInput").value.trim();
    const contextSize = parseInt($("contextSizeInput").value, 10);

    if (!topic) {
        alert("Please enter a topic.");
        return;
    }

    setLoading(btn, true);
    try {
        const data = await apiCall("/api/generate", {
            method: "POST",
            body: JSON.stringify({ topic, context_size: contextSize }),
        });

        $("generatedQuestion").textContent = `"${data.generated_question}"`;

        const verdict = $("generateVerdict");
        if (data.is_duplicate) {
            verdict.className = "verdict danger";
            verdict.textContent = "⚠ Duplicate detected — avoidance failed";
        } else {
            verdict.className = "verdict success";
            verdict.textContent = "✓ New question — avoidance worked";
        }

        renderContext(data.context_used);
        renderComparisons(data.comparisons);

        $("generateResult").classList.remove("result-hidden");
    } catch (e) {
        alert("Generation failed: " + e.message);
    } finally {
        setLoading(btn, false);
    }
}

// ============================================================
// Tab 2 — Evaluate
// ============================================================
function renderMetrics(m) {
    const grid = $("metricGrid");
    const items = [
        { label: "True Positives", value: m.true_positives },
        { label: "False Positives", value: m.false_positives },
        { label: "False Negatives", value: m.false_negatives },
        { label: "True Negatives", value: m.true_negatives },
        { label: "Precision", value: m.precision.toFixed(3) },
        { label: "Recall", value: m.recall.toFixed(3) },
        { label: "F1 Score", value: m.f1.toFixed(3) },
    ];
    grid.innerHTML = items
        .map(
            (it) => `
        <div class="metric">
            <div class="metric-label">${it.label}</div>
            <div class="metric-value">${it.value}</div>
        </div>
    `
        )
        .join("");
}

function renderEvaluationRows(rows) {
    const tbody = $("evalTable").querySelector("tbody");
    tbody.innerHTML = rows
        .map((r) => {
            const cls = labelClass(r.detector_says);
            const semCls = scoreClass(r.semantic_score, "semantic");
            return `
                <tr>
                    <td><strong>${escapeHtml(r.comparison)}</strong></td>
                    <td>${escapeHtml(r.question)}</td>
                    <td>${escapeHtml(r.expected)}</td>
                    <td><span class="label ${cls}">${escapeHtml(r.detector_says)}</span></td>
                    <td><span class="score ${semCls}">${Math.round(r.semantic_score * 100)}%</span></td>
                </tr>
            `;
        })
        .join("");
}

async function onEvaluate() {
    const btn = $("evaluateBtn");
    setLoading(btn, true);
    try {
        const data = await apiCall("/api/evaluate", { method: "POST" });
        renderMetrics(data.metrics);
        renderEvaluationRows(data.rows);
        $("evaluateResult").classList.remove("result-hidden");
    } catch (e) {
        alert("Evaluation failed: " + e.message);
    } finally {
        setLoading(btn, false);
    }
}

// ============================================================
// Tab 3 — Compare
// ============================================================
async function onCompare() {
    const btn = $("compareBtn");
    const qA = $("qA").value.trim();
    const qB = $("qB").value.trim();

    if (!qA || !qB) {
        alert("Please enter both questions.");
        return;
    }

    setLoading(btn, true);
    try {
        const data = await apiCall("/api/detect", {
            method: "POST",
            body: JSON.stringify({ question_a: qA, question_b: qB }),
        });

        // Big verdict badge
        const labelEl = $("compareLabel");
        labelEl.className = `verdict-big ${data.label}`;
        labelEl.textContent = data.label.replaceAll("_", " ");

        // Interpretation text
        const interp = $("compareInterpretation");
        if (data.label === "EXACT_DUPLICATE") {
            interp.textContent =
                "These two questions have essentially identical wording after normalizing " +
                "punctuation and casing. They are duplicates.";
        } else if (data.label === "SEMANTIC_DUPLICATE") {
            interp.textContent =
                "These two questions ask for the same concept but use different wording. " +
                "They are duplicates at the meaning level.";
        } else {
            interp.textContent =
                "These two questions cover different concepts. They are not duplicates.";
        }

        // Questions shown side by side
        $("compareQA").textContent = qA;
        $("compareQB").textContent = qB;

        // Scores — exact as-is (0-100), semantic as percentage
        $("compareExact").textContent = data.exact_score.toFixed(2);
        $("compareSemantic").textContent = Math.round(data.semantic_score * 100) + "%";

        // Progress bars
        const exactPct = Math.min(data.exact_score, 100);
        const semPct = Math.min(data.semantic_score * 100, 100);

        requestAnimationFrame(() => {
            $("compareExactBar").style.width = exactPct + "%";
            $("compareSemanticBar").style.width = semPct + "%";
        });

        $("compareResult").classList.remove("result-hidden");
    } catch (e) {
        alert("Comparison failed: " + e.message);
    } finally {
        setLoading(btn, false);
    }
}

// ============================================================
// Boot
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
    initTabs();
    checkHealth();

    $("generateBtn").addEventListener("click", onGenerate);
    $("evaluateBtn").addEventListener("click", onEvaluate);
    $("compareBtn").addEventListener("click", onCompare);
});