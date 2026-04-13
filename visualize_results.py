"""
MetaCal Benchmark — Accuracy Calculator & Visualizer
=====================================================
Two modes:
  1. AUTO  — drop .run.json files into RUN_JSON_DIR; they are parsed automatically
  2. MANUAL — use the MANUAL_DATA dict below (extracted from your Kaggle screenshot)

Run: python visualize_results.py
Outputs: metacal_heatmap.png, metacal_accuracy_bar.png, metacal_task_pass_rate.png
"""

import json, os, glob, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
RUN_JSON_DIR = Path(__file__).parent          # folder that contains .run.json files
OUTPUT_DIR   = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# MANUAL DATA  (from your Kaggle screenshot — verify / replace as needed)
# Format:  task_id → { model_display_name → (pass: bool, score_pct: int) }
# score_pct = percentage of assertions that passed (the number shown under PASS/FAIL)
# ──────────────────────────────────────────────────────────────────────────────
MODELS_ORDER = [
    "Claude Opus 4.6",
    "Claude Sonnet 4.6",
    "Gemini 3.1 Flash",
    "Gemini 3 Flash",
    "Gemma 4 31B",
    "Gemma 4 26B",
    "GLM-5",
    "DeepSeek V3.2",
    "DeepSeek R1",
    "Qwen3 235B",
    "Qwen3 Coder",
    "GPT-5.4",
    "GPT-5.4 mini",
]

TASKS_ORDER = [
    "T-01", "T-02", "T-03", "T-04", "T-05",
    "T-06", "T-07", "T-08", "T-09", "T-10",
    "T-11", "T-12", "T-13", "T-14", "T-15",
]

TASK_LABELS = {
    "T-01": "T-01 Graded Confidence",
    "T-02": "T-02 Epistemic Uncertainty",
    "T-03": "T-03 Uncertainty Injection",
    "T-04": "T-04 Error Detection",
    "T-05": "T-05 Targeted Error Detection",
    "T-06": "T-06 Error Correction",
    "T-07": "T-07 Metacognitive Sensitivity",
    "T-08": "T-08 Contradiction Detection",
    "T-09": "T-09 Accuracy-Matched Confidence",
    "T-10": "T-10 Hallucination & Abstention",
    "T-11": "T-11 Logical Consistency",
    "T-12": "T-12 Thinking Path Quality",
    "T-13": "T-13 Abstention Capability",
    "T-14": "T-14 Evidence-Based Confidence",
    "T-15": "T-15 Difficulty Prediction & Calibration",
}

SUB_FACULTY = {
    "T-01": "Calibration",
    "T-02": "Calibration",
    "T-03": "Calibration",
    "T-04": "Error Detection",
    "T-05": "Error Detection",
    "T-06": "Error Detection",
    "T-07": "Meta Sensitivity",
    "T-08": "Meta Sensitivity",
    "T-09": "Thinking Path",
    "T-10": "Thinking Path",
    "T-11": "Thinking Path",
    "T-12": "Thinking Path",
    "T-13": "Thinking Path",
    "T-14": "Thinking Path",
    "T-15": "Calibration",
}

SUB_FACULTY_COLORS = {
    "Calibration":    "#4C72B0",
    "Error Detection":"#DD8452",
    "Meta Sensitivity":"#55A868",
    "Thinking Path":  "#C44E52",
}

# ── Replace with your actual values from the screenshot ──────────────────────
# True/False = task-level PASS/FAIL;  integer = assertion-pass percentage shown
# None = not run / no data
MANUAL_DATA: dict[str, dict[str, tuple[bool, int] | None]] = {
    "T-01": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (False, 75),
        "Gemini 3.1 Flash":  (False, 75),
        "Gemini 3 Flash":    (False, 50),
        "Gemma 4 31B":       (False, 50),
        "Gemma 4 26B":       (False, 50),
        "GLM-5":             (False, 25),
        "DeepSeek V3.2":     (False, 50),
        "DeepSeek R1":       (False, 50),
        "Qwen3 235B":        (False, 50),
        "Qwen3 Coder":       (False, 50),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (False, 75),
    },
    "T-02": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 0),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 33),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-03": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (True,  100),
        "Gemini 3.1 Flash":  (False, 67),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 67),
        "DeepSeek R1":       (False, 67),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (False, 67),
    },
    "T-04": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (True,  100),
        "Gemini 3.1 Flash":  (True,  100),
        "Gemini 3 Flash":    (False, 67),
        "Gemma 4 31B":       (False, 67),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 33),
        "DeepSeek V3.2":     (True,  100),
        "DeepSeek R1":       (True,  100),
        "Qwen3 235B":        (True,  100),
        "Qwen3 Coder":       (False, 67),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (True,  100),
    },
    "T-05": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 67),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-06": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 67),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 67),
        "DeepSeek R1":       (False, 67),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-07": {
        "Claude Opus 4.6":   (False, 50),
        "Claude Sonnet 4.6": (False, 50),
        "Gemini 3.1 Flash":  (False, 50),
        "Gemini 3 Flash":    (False, 0),
        "Gemma 4 31B":       (False, 0),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 50),
        "DeepSeek R1":       (False, 50),
        "Qwen3 235B":        (False, 0),
        "Qwen3 Coder":       (False, 0),
        "GPT-5.4":           (False, 50),
        "GPT-5.4 mini":      (False, 0),
    },
    "T-08": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (True,  100),
        "Gemini 3.1 Flash":  (False, 67),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 33),
        "DeepSeek V3.2":     (True,  100),
        "DeepSeek R1":       (True,  100),
        "Qwen3 235B":        (False, 67),
        "Qwen3 Coder":       (False, 67),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (False, 67),
    },
    "T-09": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 33),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 0),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 33),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-10": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (True,  100),
        "Gemini 3.1 Flash":  (False, 67),
        "Gemini 3 Flash":    (False, 67),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (True,  100),
        "DeepSeek R1":       (True,  100),
        "Qwen3 235B":        (False, 67),
        "Qwen3 Coder":       (False, 67),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (False, 67),
    },
    "T-11": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 33),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-12": {
        "Claude Opus 4.6":   (True,  100),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 33),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 67),
        "DeepSeek R1":       (False, 67),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (True,  100),
        "GPT-5.4 mini":      (False, 67),
    },
    "T-13": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 0),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 33),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 67),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-14": {
        "Claude Opus 4.6":   (False, 67),
        "Claude Sonnet 4.6": (False, 67),
        "Gemini 3.1 Flash":  (False, 33),
        "Gemini 3 Flash":    (False, 33),
        "Gemma 4 31B":       (False, 33),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 33),
        "DeepSeek R1":       (False, 33),
        "Qwen3 235B":        (False, 33),
        "Qwen3 Coder":       (False, 33),
        "GPT-5.4":           (False, 33),
        "GPT-5.4 mini":      (False, 33),
    },
    "T-15": {
        "Claude Opus 4.6":   (False, 50),
        "Claude Sonnet 4.6": (False, 50),
        "Gemini 3.1 Flash":  (False, 50),
        "Gemini 3 Flash":    (False, 0),
        "Gemma 4 31B":       (False, 0),
        "Gemma 4 26B":       (False, 0),
        "GLM-5":             (False, 0),
        "DeepSeek V3.2":     (False, 50),
        "DeepSeek R1":       (False, 50),
        "Qwen3 235B":        (False, 0),
        "Qwen3 Coder":       (False, 0),
        "GPT-5.4":           (False, 50),
        "GPT-5.4 mini":      (False, 0),
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# PARSER  — reads .run.json files and returns same dict shape as MANUAL_DATA
# ──────────────────────────────────────────────────────────────────────────────
MODEL_SLUG_TO_DISPLAY = {
    "anthropic/claude-opus-4-6@default":           "Claude Opus 4.6",
    "anthropic/claude-sonnet-4-6@default":         "Claude Sonnet 4.6",
    "google/gemini-3.1-pro-preview@default":       "Gemini 3.1 Flash",
    "google/gemini-3-flash-preview@default":       "Gemini 3 Flash",
    "google/gemma-4-31b@default":                  "Gemma 4 31B",
    "google/gemma-4-26b-a4b@default":              "Gemma 4 26B",
    "zhipuai/glm-5@default":                       "GLM-5",
    "deepseek/deepseek-v3.2@default":              "DeepSeek V3.2",
    "deepseek/deepseek-r1-0528@default":           "DeepSeek R1",
    "qwen/qwen3-235b-a22b-instruct@default":       "Qwen3 235B",
    "qwen/qwen3-coder-480b-a35b-instruct@default": "Qwen3 Coder",
    "openai/gpt-5.4-2026-03-05@default":           "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17@default":      "GPT-5.4 mini",
}

TASK_NAME_TO_ID_RE = re.compile(r"(T-\d{2})")

def parse_run_jsons(directory: Path) -> dict:
    """Parse all .run.json files and return same structure as MANUAL_DATA."""
    data: dict[str, dict[str, tuple[bool, int]]] = {}
    files = list(directory.glob("*.run.json"))
    if not files:
        return {}
    for fpath in files:
        with open(fpath, encoding="utf-8") as f:
            run = json.load(f)
        task_name = run.get("taskVersion", {}).get("name", "")
        m = TASK_NAME_TO_ID_RE.search(task_name)
        if not m:
            continue
        task_id = m.group(1)
        model_slug = run.get("modelVersion", {}).get("slug", "")
        model_display = MODEL_SLUG_TO_DISPLAY.get(model_slug, model_slug)
        assertions = run.get("assertions", [])
        if not assertions:
            continue
        passed = sum(
            1 for a in assertions
            if a.get("status") == "BENCHMARK_TASK_RUN_ASSERTION_STATUS_PASSED"
        )
        total = len(assertions)
        score_pct = round(100 * passed / total) if total else 0
        task_pass = (passed == total)
        data.setdefault(task_id, {})[model_display] = (task_pass, score_pct)
    return data

# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────────────────────────────────────
parsed = parse_run_jsons(RUN_JSON_DIR)
if parsed:
    print(f"[INFO] Loaded {sum(len(v) for v in parsed.values())} results from run.json files.")
    # Merge: parsed values override manual
    data = {t: dict(MANUAL_DATA.get(t, {})) for t in TASKS_ORDER}
    for task_id, model_dict in parsed.items():
        if task_id in data:
            data[task_id].update(model_dict)
        else:
            data[task_id] = model_dict
else:
    print("[INFO] No run.json files found — using manual screenshot data.")
    data = MANUAL_DATA

# Determine which models are actually present
models_present = [m for m in MODELS_ORDER
                  if any(m in data.get(t, {}) for t in TASKS_ORDER)]

# ──────────────────────────────────────────────────────────────────────────────
# BUILD MATRICES
# ──────────────────────────────────────────────────────────────────────────────
n_tasks  = len(TASKS_ORDER)
n_models = len(models_present)

pass_matrix  = np.full((n_tasks, n_models), np.nan)   # 1=PASS, 0=FAIL
score_matrix = np.full((n_tasks, n_models), np.nan)   # 0–100

for ti, task in enumerate(TASKS_ORDER):
    for mi, model in enumerate(models_present):
        cell = data.get(task, {}).get(model)
        if cell is not None:
            passed, score = cell
            pass_matrix[ti, mi]  = 1.0 if passed else 0.0
            score_matrix[ti, mi] = score

# Per-model accuracy (% tasks passed)
model_accuracy = np.nanmean(pass_matrix, axis=0) * 100   # shape (n_models,)

# Per-task pass rate (% models that passed)
task_pass_rate = np.nanmean(pass_matrix, axis=1) * 100   # shape (n_tasks,)

# Per-model avg assertion score
model_avg_score = np.nanmean(score_matrix, axis=0) * 100 / 100  # already 0-100

# Overall accuracy
overall_acc = np.nanmean(pass_matrix) * 100

print(f"\n{'='*55}")
print(f"  METACAL BENCHMARK — ACCURACY SUMMARY")
print(f"{'='*55}")
print(f"  Overall task-pass rate: {overall_acc:.1f}%\n")
print(f"  {'Model':<22}  {'Task PASS%':>10}  {'Avg Score%':>10}")
print(f"  {'-'*44}")
sorted_idx = np.argsort(-model_accuracy)
for i in sorted_idx:
    print(f"  {models_present[i]:<22}  {model_accuracy[i]:>9.1f}%  {model_avg_score[i]:>9.1f}%")
print(f"\n  {'Task':<8}  {'Pass rate':>10}  {'Sub-faculty'}")
print(f"  {'-'*44}")
for ti, task in enumerate(TASKS_ORDER):
    print(f"  {task:<8}  {task_pass_rate[ti]:>9.1f}%  {SUB_FACULTY.get(task,'')}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 1 — HEATMAP  (score %, annotated with PASS / FAIL)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(max(14, n_models * 1.1), max(9, n_tasks * 0.65)))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

cmap = LinearSegmentedColormap.from_list(
    "metacal", ["#8B0000", "#CC3300", "#FF6600", "#229922", "#006600"]
)
img = ax.imshow(score_matrix, cmap=cmap, vmin=0, vmax=100, aspect="auto")

for ti in range(n_tasks):
    for mi in range(n_models):
        val = score_matrix[ti, mi]
        is_pass = pass_matrix[ti, mi]
        if np.isnan(val):
            ax.text(mi, ti, "—", ha="center", va="center", fontsize=8, color="#888")
            continue
        bg_color = "#00aa00" if is_pass else "#cc2200"
        ax.add_patch(plt.Rectangle(
            (mi - 0.45, ti - 0.45), 0.9, 0.9,
            color=bg_color, alpha=0.25, zorder=0
        ))
        label = "PASS" if is_pass else "FAIL"
        ax.text(mi, ti - 0.12, label, ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="#55ff55" if is_pass else "#ff5555")
        ax.text(mi, ti + 0.22, f"{val:.0f}%", ha="center", va="center",
                fontsize=7, color="#dddddd")

ax.set_xticks(range(n_models))
ax.set_xticklabels(models_present, rotation=40, ha="right", fontsize=9, color="#cccccc")
ax.set_yticks(range(n_tasks))
ax.set_yticklabels(
    [TASK_LABELS.get(t, t) for t in TASKS_ORDER],
    fontsize=8.5, color="#cccccc"
)

# Sub-faculty colour bar on left
for ti, task in enumerate(TASKS_ORDER):
    sf = SUB_FACULTY.get(task, "")
    col = SUB_FACULTY_COLORS.get(sf, "#888888")
    ax.add_patch(plt.Rectangle((-0.9, ti - 0.45), 0.3, 0.9, color=col, clip_on=False))

cbar = fig.colorbar(img, ax=ax, fraction=0.015, pad=0.01)
cbar.set_label("Assertion-pass %", color="#cccccc", fontsize=9)
cbar.ax.yaxis.set_tick_params(color="#cccccc")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#cccccc")

sf_patches = [mpatches.Patch(color=c, label=k) for k, c in SUB_FACULTY_COLORS.items()]
ax.legend(handles=sf_patches, loc="upper left", bbox_to_anchor=(0, -0.14),
          ncol=4, framealpha=0, fontsize=8,
          labelcolor="#cccccc")

ax.set_title(
    f"MetaCal Benchmark — Task-Level Heatmap  |  overall pass rate {overall_acc:.1f}%",
    color="white", fontsize=13, pad=12
)
ax.tick_params(colors="#666666")
for spine in ax.spines.values():
    spine.set_edgecolor("#333333")

plt.tight_layout()
out1 = OUTPUT_DIR / "metacal_heatmap.png"
plt.savefig(out1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n[SAVED] {out1}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 2 — PER-MODEL ACCURACY  (sorted bar chart)
# ──────────────────────────────────────────────────────────────────────────────
sorted_idx = np.argsort(-model_accuracy)
sorted_models = [models_present[i] for i in sorted_idx]
sorted_acc    = model_accuracy[sorted_idx]
sorted_score  = model_avg_score[sorted_idx]

fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

x = np.arange(n_models)
bars = ax.bar(x - 0.2, sorted_acc,   width=0.38, color="#4C72B0", label="Task PASS % (all assertions)")
bars2= ax.bar(x + 0.2, sorted_score, width=0.38, color="#55A868", label="Avg assertion-pass %")

for bar, val in zip(bars, sorted_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#aaaaaa")
for bar, val in zip(bars2, sorted_score):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#aaaaaa")

ax.set_xticks(x)
ax.set_xticklabels(sorted_models, rotation=35, ha="right", fontsize=9, color="#cccccc")
ax.set_ylabel("Accuracy (%)", color="#cccccc")
ax.set_ylim(0, 115)
ax.set_title("Per-Model Accuracy — MetaCal Benchmark", color="white", fontsize=12, pad=10)
ax.legend(framealpha=0.15, labelcolor="#cccccc", fontsize=9)
ax.tick_params(axis="y", colors="#888888")
ax.tick_params(axis="x", colors="#888888")
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#222244", linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out2 = OUTPUT_DIR / "metacal_accuracy_bar.png"
plt.savefig(out2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[SAVED] {out2}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOT 3 — PER-TASK PASS RATE  (coloured by sub-faculty)
# ──────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5.5))
fig.patch.set_facecolor("#1a1a2e")
ax.set_facecolor("#1a1a2e")

bar_colors = [SUB_FACULTY_COLORS.get(SUB_FACULTY.get(t, ""), "#888888") for t in TASKS_ORDER]
x = np.arange(n_tasks)
bars = ax.bar(x, task_pass_rate, color=bar_colors, width=0.65, edgecolor="#1a1a2e", linewidth=0.5)

for bar, val in zip(bars, task_pass_rate):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#aaaaaa")

ax.set_xticks(x)
ax.set_xticklabels(TASKS_ORDER, rotation=0, fontsize=9, color="#cccccc")
ax.set_ylabel("% Models that PASSED", color="#cccccc")
ax.set_ylim(0, 115)
ax.set_title("Per-Task Pass Rate — MetaCal Benchmark", color="white", fontsize=12, pad=10)

sf_patches = [mpatches.Patch(color=c, label=k) for k, c in SUB_FACULTY_COLORS.items()]
ax.legend(handles=sf_patches, framealpha=0.15, labelcolor="#cccccc", fontsize=9)
ax.tick_params(axis="y", colors="#888888")
ax.tick_params(axis="x", colors="#888888")
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.yaxis.grid(True, color="#222244", linewidth=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
out3 = OUTPUT_DIR / "metacal_task_pass_rate.png"
plt.savefig(out3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"[SAVED] {out3}")
print("\nDone.")
