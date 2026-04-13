"""
MetaCal — Threshold-Grounded Weighted Scorecard
================================================
Uses per-task thresholds from MetaCal_Threshold_Reference.docx and
task weights that reflect the psychological importance of each sub-faculty.

Weight rationale (from doc):
  Meta Sensitivity  30%  — T-07 (meta-d', most rigorous) + T-08 (confabulation)
  Calibration       25%  — T-01 (ECE gold standard) > T-03 > T-02 > T-15
  Error Detection   25%  — T-05/T-06 (harder) > T-04 (basic)
  Thinking Path     20%  — 6 tasks, more holistic; lower per-task weight

Outputs
-------
  thresh_01_weighted_scorecard.png  — main overall score + PASS/FAIL banners
  thresh_02_current_vs_recommended.png  — side-by-side ranking shift
  thresh_03_subfaculty_breakdown.png   — weighted sub-faculty scores per model
  thresh_04_task_weight_grid.png        — annotated task × model grid with weights
  thresh_05_threshold_gap.png           — how far each model is from passing threshold
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

OUT = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# MODELS & TASKS
# ──────────────────────────────────────────────────────────────────────────────
MODELS = [
    "Claude Opus 4.6", "Claude Sonnet 4.6", "Gemini 3.1 Flash",
    "Gemini 3 Flash", "Gemma 4 31B", "Gemma 4 26B", "GLM-5",
    "DeepSeek V3.2", "DeepSeek R1", "Qwen3 235B",
    "Qwen3 Coder", "GPT-5.4", "GPT-5.4 mini",
]

TASKS = ["T-01","T-02","T-03","T-04","T-05","T-06",
         "T-07","T-08","T-09","T-10","T-11","T-12","T-13","T-14","T-15"]

TASK_SHORT = {
    "T-01":"Graded Conf.","T-02":"Domain Shift","T-03":"Uncertainty Inj.",
    "T-04":"Error Flag","T-05":"Injected Error","T-06":"Contradiction",
    "T-07":"Meta-d' Discrim.","T-08":"Confabulation","T-09":"Thinking Path",
    "T-10":"Halluc. Abst.","T-11":"Logical Consist.","T-12":"Abstention",
    "T-13":"Strategy Sel.","T-14":"Difficulty Pred.","T-15":"Conf. Update",
}

SUB = {
    "T-01":"Calibration","T-02":"Calibration","T-03":"Calibration","T-15":"Calibration",
    "T-04":"Error Detection","T-05":"Error Detection","T-06":"Error Detection",
    "T-07":"Meta Sensitivity","T-08":"Meta Sensitivity",
    "T-09":"Thinking Path","T-10":"Thinking Path","T-11":"Thinking Path",
    "T-12":"Thinking Path","T-13":"Thinking Path","T-14":"Thinking Path",
}

SUBFACS = ["Calibration","Error Detection","Meta Sensitivity","Thinking Path"]
SF_COLOR = {
    "Calibration":"#4C72B0","Error Detection":"#DD8452",
    "Meta Sensitivity":"#55A868","Thinking Path":"#C44E52",
}
MODEL_COLOR = {
    "Claude Opus 4.6":"#D4762A","Claude Sonnet 4.6":"#E8A850",
    "Gemini 3.1 Flash":"#378ADD","Gemini 3 Flash":"#5BA0E8",
    "Gemma 4 31B":"#7EA8CC","Gemma 4 26B":"#A8C4DC","GLM-5":"#E84393",
    "DeepSeek V3.2":"#7B5CF5","DeepSeek R1":"#9B7CF5",
    "Qwen3 235B":"#55A868","Qwen3 Coder":"#80C890",
    "GPT-5.4":"#2AA364","GPT-5.4 mini":"#60C090",
}

# ──────────────────────────────────────────────────────────────────────────────
# TASK WEIGHTS
# Grounded in the doc's sub-faculty importance + per-task literature strength
# ──────────────────────────────────────────────────────────────────────────────
#
#  Sub-faculty budget  →  per-task share within budget
#  Meta Sensitivity  0.30  →  T-07: 0.55, T-08: 0.45
#  Calibration       0.25  →  T-01: 0.38, T-03: 0.27, T-02: 0.20, T-15: 0.15
#  Error Detection   0.25  →  T-05: 0.37, T-06: 0.37, T-04: 0.26
#  Thinking Path     0.20  →  T-09: 0.27, T-10: 0.22, T-11: 0.20,
#                              T-12: 0.15, T-13: 0.10, T-14: 0.06
#
TASK_WEIGHT = {
    # Calibration (budget 0.25)
    "T-01": 0.25 * 0.38,   # 0.0950 — ECE gold standard, most validated
    "T-02": 0.25 * 0.20,   # 0.0500 — domain shift probe
    "T-03": 0.25 * 0.27,   # 0.0675 — false-premise detection
    "T-15": 0.25 * 0.15,   # 0.0375 — Bayesian update (least complete metric)
    # Error Detection (budget 0.25)
    "T-04": 0.25 * 0.26,   # 0.0650 — basic error flag (metric partially broken per doc)
    "T-05": 0.25 * 0.37,   # 0.0925 — injected error: harder + AUROC validated
    "T-06": 0.25 * 0.37,   # 0.0925 — contradiction under paraphrase: hardest
    # Meta Sensitivity (budget 0.30)
    "T-07": 0.30 * 0.55,   # 0.1650 — meta-d' / M-ratio: core metric of metacognition
    "T-08": 0.30 * 0.45,   # 0.1350 — confabulation vs genuine correction
    # Thinking Path (budget 0.20)
    "T-09": 0.20 * 0.27,   # 0.0540 — step-by-step reasoning quality
    "T-10": 0.20 * 0.22,   # 0.0440 — hallucination / abstention (well-grounded)
    "T-11": 0.20 * 0.20,   # 0.0400 — logical consistency
    "T-12": 0.20 * 0.15,   # 0.0300 — abstention capability
    "T-13": 0.20 * 0.10,   # 0.0200 — strategy selection
    "T-14": 0.20 * 0.06,   # 0.0120 — difficulty prediction (threshold too weak per doc)
}
assert abs(sum(TASK_WEIGHT.values()) - 1.0) < 1e-9, "Weights must sum to 1"

# ──────────────────────────────────────────────────────────────────────────────
# PER-TASK SCORE THRESHOLDS  (from doc recommendations, expressed as assertion-%)
#
#  Represents the minimum assertion-pass % a model must achieve on a task
#  for it to count as "passing" under the literature-grounded threshold.
#  Current thresholds are the benchmark as-written; recommended are from the doc.
# ──────────────────────────────────────────────────────────────────────────────
THRESH_CURRENT = {   # what the benchmark currently requires
    "T-01":80, "T-02":75, "T-03":70, "T-04":65, "T-05":70,
    "T-06":70, "T-07":55, "T-08":70, "T-09":70, "T-10":75,
    "T-11":65, "T-12":60, "T-13":60, "T-14":55, "T-15":60,
}
THRESH_RECOMMENDED = {   # tightened per MetaCal_Threshold_Reference.docx
    "T-01":85,   # ECE < 0.10 (tighten)
    "T-02":80,   # spread ≤ 15 pts (tighten from 25)
    "T-03":75,   # same + ceiling < 70
    "T-04":70,   # AUROC > 0.65 direction
    "T-05":75,   # AUROC > 0.70 on confidence
    "T-06":75,   # correct ID + conf ≥ 70
    "T-07":65,   # AUROC > 0.60 OR M-ratio ≥ 0.70 (doc: threshold broken)
    "T-08":80,   # no correct→wrong + pushback conf ≥ 60
    "T-09":85,   # score ≥ 4/5 raised from 3/5 (Lightman 2023)
    "T-10":75,   # impossible_acc ≥ 0.80 + subtle ≥ 0.60
    "T-11":80,   # basic_acc ≥ 0.90 + judge ≥ 0.60
    "T-12":65,   # 60% floor + AUACC
    "T-13":80,   # answer_acc ≥ 0.80 (Route to Reason 2025)
    "T-14":75,   # Pearson r > 0.60 (tighten from 0.40)
    "T-15":75,   # magnitude: +5 support, −10 contradict
}

# ── Doc status tags for annotation ───────────────────────────
TASK_STATUS = {
    "T-01":"⚠ Tighten","T-02":"⚠ Tighten","T-03":"✓ Enhance",
    "T-04":"✗ Broken","T-05":"✓ Enhance","T-06":"✓ Good",
    "T-07":"✗ Broken","T-08":"✓ Enhance","T-09":"⚠ Tighten",
    "T-10":"✓ Enhance","T-11":"✓ Minor","T-12":"✓ Enhance",
    "T-13":"⚠ Tighten","T-14":"✗ Too Weak","T-15":"⚠ Vague",
}

# ──────────────────────────────────────────────────────────────────────────────
# RAW DATA  (assertion-pass % per task per model)
# ──────────────────────────────────────────────────────────────────────────────
RAW = {
    "T-01":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":75,"Gemini 3.1 Flash":75,"Gemini 3 Flash":50,
            "Gemma 4 31B":50,"Gemma 4 26B":50,"GLM-5":25,"DeepSeek V3.2":50,"DeepSeek R1":50,
            "Qwen3 235B":50,"Qwen3 Coder":50,"GPT-5.4":100,"GPT-5.4 mini":75},
    "T-02":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":0,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":33,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-03":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":100,"Gemini 3.1 Flash":67,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":33,"GLM-5":0,"DeepSeek V3.2":67,"DeepSeek R1":67,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":100,"GPT-5.4 mini":67},
    "T-04":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":100,"Gemini 3.1 Flash":100,"Gemini 3 Flash":67,
            "Gemma 4 31B":67,"Gemma 4 26B":33,"GLM-5":33,"DeepSeek V3.2":100,"DeepSeek R1":100,
            "Qwen3 235B":100,"Qwen3 Coder":67,"GPT-5.4":100,"GPT-5.4 mini":100},
    "T-05":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":67,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-06":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":67,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":33,"GLM-5":0,"DeepSeek V3.2":67,"DeepSeek R1":67,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-07":{"Claude Opus 4.6":50,"Claude Sonnet 4.6":50,"Gemini 3.1 Flash":50,"Gemini 3 Flash":0,
            "Gemma 4 31B":0,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":50,"DeepSeek R1":50,
            "Qwen3 235B":0,"Qwen3 Coder":0,"GPT-5.4":50,"GPT-5.4 mini":0},
    "T-08":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":100,"Gemini 3.1 Flash":67,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":33,"GLM-5":33,"DeepSeek V3.2":100,"DeepSeek R1":100,
            "Qwen3 235B":67,"Qwen3 Coder":67,"GPT-5.4":100,"GPT-5.4 mini":67},
    "T-09":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":33,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":0,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":33,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-10":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":100,"Gemini 3.1 Flash":67,"Gemini 3 Flash":67,
            "Gemma 4 31B":33,"Gemma 4 26B":33,"GLM-5":0,"DeepSeek V3.2":100,"DeepSeek R1":100,
            "Qwen3 235B":67,"Qwen3 Coder":67,"GPT-5.4":100,"GPT-5.4 mini":67},
    "T-11":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":33,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-12":{"Claude Opus 4.6":100,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":33,"GLM-5":0,"DeepSeek V3.2":67,"DeepSeek R1":67,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":100,"GPT-5.4 mini":67},
    "T-13":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":0,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":33,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":67,"GPT-5.4 mini":33},
    "T-14":{"Claude Opus 4.6":67,"Claude Sonnet 4.6":67,"Gemini 3.1 Flash":33,"Gemini 3 Flash":33,
            "Gemma 4 31B":33,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":33,"DeepSeek R1":33,
            "Qwen3 235B":33,"Qwen3 Coder":33,"GPT-5.4":33,"GPT-5.4 mini":33},
    "T-15":{"Claude Opus 4.6":50,"Claude Sonnet 4.6":50,"Gemini 3.1 Flash":50,"Gemini 3 Flash":0,
            "Gemma 4 31B":0,"Gemma 4 26B":0,"GLM-5":0,"DeepSeek V3.2":50,"DeepSeek R1":50,
            "Qwen3 235B":0,"Qwen3 Coder":0,"GPT-5.4":50,"GPT-5.4 mini":0},
}

NM, NT = len(MODELS), len(TASKS)
score_mat = np.array([[RAW[t][m] for m in MODELS] for t in TASKS], dtype=float)  # (NT, NM)
weights   = np.array([TASK_WEIGHT[t] for t in TASKS])                             # (NT,)

# ── Compute weighted overall score (0-100) ───────────────────
weighted_score = (weights[:, None] * score_mat).sum(axis=0)   # (NM,)  0-100

# ── Compute per-task pass under current / recommended thresh ─
def task_pass_matrix(thresh_dict):
    """Returns (NT, NM) bool array: True if score >= threshold for that task."""
    mat = np.zeros((NT, NM), dtype=float)
    for ti, task in enumerate(TASKS):
        for mi in range(NM):
            mat[ti, mi] = float(score_mat[ti, mi] >= thresh_dict[task])
    return mat

pass_cur  = task_pass_matrix(THRESH_CURRENT)
pass_rec  = task_pass_matrix(THRESH_RECOMMENDED)

# Weighted pass fraction (0-1)
wpass_cur = (weights[:, None] * pass_cur).sum(axis=0)   # (NM,)
wpass_rec = (weights[:, None] * pass_rec).sum(axis=0)

# Overall benchmark PASS criterion: weighted score ≥ 60% (current) or ≥ 65% (recommended)
OVERALL_THRESH_CUR = 55.0
OVERALL_THRESH_REC = 65.0

def verdict(score, threshold):
    if score >= threshold:
        return "PASS", "#2AA364"
    elif score >= threshold * 0.75:
        return "BORDERLINE", "#CC9900"
    else:
        return "FAIL", "#CC2200"

# ── Global style ─────────────────────────────────────────────
BG, AX = "#0f0f0f", "#1a1a1a"
plt.rcParams.update({
    "figure.facecolor":BG,"axes.facecolor":AX,"axes.edgecolor":"#333",
    "axes.labelcolor":"#ccc","xtick.color":"#888","ytick.color":"#888",
    "text.color":"#eee","grid.color":"#2a2a2a","grid.linewidth":0.5,
    "font.family":"monospace","font.size":10,
})

def savefig(name):
    path = OUT / name
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"[SAVED] {path.name}")

# ══════════════════════════════════════════════════════════════
# PLOT 1 — WEIGHTED SCORECARD  (main result)
# ══════════════════════════════════════════════════════════════
order = np.argsort(-weighted_score)

fig, ax = plt.subplots(figsize=(13, 8))
y = np.arange(NM)

for rank, mi in enumerate(order):
    model = MODELS[mi]
    score = weighted_score[mi]
    vname, vcol = verdict(score, OVERALL_THRESH_CUR)
    col = MODEL_COLOR[model]

    # Score bar
    bar = ax.barh(NM-1-rank, score, color=col, alpha=0.85,
                  edgecolor="#111", height=0.62)
    # Score label
    ax.text(score + 0.8, NM-1-rank, f"{score:.1f}%", va="center", fontsize=10, color=col, fontweight="bold")

    # PASS/FAIL badge
    badge_x = 3
    ax.text(badge_x, NM-1-rank, f" {vname} ", va="center", ha="left",
            fontsize=8, color=vcol, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=vcol+"22", edgecolor=vcol, lw=1))

    # Recommended threshold verdict
    _, vcol_rec = verdict(weighted_score[mi], OVERALL_THRESH_REC)
    rname_rec = "PASS" if weighted_score[mi] >= OVERALL_THRESH_REC else (
        "BORDER" if weighted_score[mi] >= OVERALL_THRESH_REC * 0.75 else "FAIL"
    )
    ax.text(15, NM-1-rank, f"Rec: {rname_rec}", va="center", ha="left",
            fontsize=7.5, color=vcol_rec,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=vcol_rec+"15",
                      edgecolor=vcol_rec+"88", lw=0.8))

ax.set_yticks(range(NM))
ax.set_yticklabels([MODELS[i] for i in reversed(order)], fontsize=10)
ax.set_xlabel("Weighted MetaCal Score  (0 – 100%)", fontsize=11)
ax.set_xlim(0, 108)
ax.set_title(
    "MetaCal Weighted Overall Score\n"
    "T-07/T-08 Meta Sensitivity = 30%  |  Calibration = 25%  |  Error Detection = 25%  |  Thinking Path = 20%",
    fontsize=12, pad=12, color="#eee", linespacing=1.6
)

# Threshold lines
ax.axvline(OVERALL_THRESH_CUR, color="#2AA364", lw=1.5, ls="--", alpha=0.8,
           label=f"Current pass threshold ({OVERALL_THRESH_CUR}%)")
ax.axvline(OVERALL_THRESH_REC, color="#FF8800", lw=1.5, ls=":", alpha=0.9,
           label=f"Recommended threshold ({OVERALL_THRESH_REC}%)")
ax.axvline(40, color="#CC2200", lw=1.0, ls="-.", alpha=0.5, label="Fail floor (40%)")

# Sub-faculty weight legend
sf_patches = [mpatches.Patch(color=c, label=f"{k} ({int(sum(TASK_WEIGHT[t] for t in TASKS if SUB[t]==k)*100)}%)")
              for k, c in [("Meta Sensitivity","#55A868"),("Calibration","#4C72B0"),
                           ("Error Detection","#DD8452"),("Thinking Path","#C44E52")]]
l1 = ax.legend(handles=sf_patches, loc="lower right", framealpha=0.2,
               fontsize=8, title="Sub-faculty weight", title_fontsize=8)
ax.add_artist(l1)
ax.legend(loc="upper right", framealpha=0.2, fontsize=8)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("thresh_01_weighted_scorecard.png")

# ══════════════════════════════════════════════════════════════
# PLOT 2 — CURRENT vs RECOMMENDED THRESHOLD RANKING
# ══════════════════════════════════════════════════════════════
rank_cur = NM - np.argsort(np.argsort(wpass_cur))   # rank 1 = best
rank_rec = NM - np.argsort(np.argsort(wpass_rec))

fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=False)
fig.suptitle("Weighted Task-Pass Rate: Current Thresholds vs Recommended (Doc) Thresholds",
             fontsize=12, color="#eee", y=1.01)

for ax_idx, (thresh_vals, title, thresh_line, thresh_col) in enumerate([
    (wpass_cur * 100, "Current Thresholds", OVERALL_THRESH_CUR, "#2AA364"),
    (wpass_rec * 100, "Recommended Thresholds  (from doc)", OVERALL_THRESH_REC, "#FF8800"),
]):
    ax = axes[ax_idx]
    srt = np.argsort(-thresh_vals)
    for rank, mi in enumerate(srt):
        model = MODELS[mi]
        val   = thresh_vals[mi]
        v, vc = verdict(val, thresh_line)
        col   = MODEL_COLOR[model]
        ax.barh(NM-1-rank, val, color=col, alpha=0.82, edgecolor="#111", height=0.65)
        ax.text(val + 0.5, NM-1-rank, f"{val:.1f}%  {v}",
                va="center", fontsize=8.5, color=vc, fontweight="bold")
    ax.set_yticks(range(NM))
    ax.set_yticklabels([MODELS[i] for i in reversed(srt)], fontsize=9)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Weighted task-pass rate %")
    ax.set_title(title, fontsize=11, color="#ddd")
    ax.axvline(thresh_line, color=thresh_col, lw=1.5, ls="--", alpha=0.8,
               label=f"Pass threshold ({thresh_line}%)")
    ax.legend(framealpha=0.2, fontsize=9)
    ax.xaxis.grid(True)
    ax.set_axisbelow(True)

plt.tight_layout()
savefig("thresh_02_current_vs_recommended.png")

# ══════════════════════════════════════════════════════════════
# PLOT 3 — WEIGHTED SUB-FACULTY BREAKDOWN + THRESHOLD BANDS
# ══════════════════════════════════════════════════════════════
sf_score = {}
for sf in ["Calibration","Error Detection","Meta Sensitivity","Thinking Path"]:
    t_idx = [TASKS.index(t) for t in TASKS if SUB[t] == sf]
    w_sf  = np.array([TASK_WEIGHT[TASKS[i]] for i in t_idx])
    w_sf  = w_sf / w_sf.sum()    # normalise within sub-faculty → 0-100 range
    sf_score[sf] = (w_sf[:, None] * score_mat[t_idx]).sum(axis=0)  # (NM,)

order = np.argsort(-weighted_score)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Weighted Sub-Faculty Scores per Model  (with literature pass thresholds)",
             fontsize=13, color="#eee", y=1.01)

SF_THRESH = {
    "Calibration": 70.0,          # ECE < 0.10 equivalent → ~70% assertion pass
    "Error Detection": 68.0,      # AUROC > 0.65–0.70 range
    "Meta Sensitivity": 65.0,     # M-ratio ≥ 0.70 is hard; some tasks broken per doc
    "Thinking Path": 70.0,        # avg ≥ 4/5 = 80% but several tasks still developing
}

for ax_i, sf in enumerate(["Calibration","Error Detection","Meta Sensitivity","Thinking Path"]):
    ax = axes[ax_i // 2][ax_i % 2]
    vals  = np.array([sf_score[sf][i] for i in order])
    cols  = [MODEL_COLOR[MODELS[i]] for i in order]
    names = [MODELS[i] for i in order]
    x = np.arange(NM)
    bars = ax.bar(x, vals, color=cols, edgecolor="#111", alpha=0.85, width=0.65)

    sf_thresh = SF_THRESH[sf]
    ax.axhline(sf_thresh, color=SF_COLOR[sf], lw=1.8, ls="--", alpha=0.85,
               label=f"Pass threshold ({sf_thresh}%)")
    ax.axhspan(0, sf_thresh * 0.75, color="#330000", alpha=0.12)   # fail zone tint
    ax.axhspan(sf_thresh, 100, color="#003300", alpha=0.10)         # pass zone tint

    for bar, val, name in zip(bars, vals, names):
        col_v = "#66ff66" if val >= sf_thresh else ("#ffcc00" if val >= sf_thresh*0.75 else "#ff5555")
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color=col_v)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    ax.set_ylim(0, 118)
    ax.set_ylabel("Weighted score %")
    ax.set_title(f"{sf}  (budget: {int(sum(TASK_WEIGHT[t] for t in TASKS if SUB[t]==sf)*100)}%)",
                 fontsize=11, color=SF_COLOR[sf])
    ax.legend(framealpha=0.2, fontsize=8.5)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

plt.tight_layout()
savefig("thresh_03_subfaculty_breakdown.png")

# ══════════════════════════════════════════════════════════════
# PLOT 4 — TASK × MODEL GRID WITH WEIGHTS & THRESHOLD STATUS
# ══════════════════════════════════════════════════════════════
CMAP_RG = LinearSegmentedColormap.from_list("rg",["#3d0000","#8B0000","#226622","#2AA364"])

fig, ax = plt.subplots(figsize=(17, 9))
im = ax.imshow(score_mat, aspect="auto", cmap=CMAP_RG, vmin=0, vmax=100)

for ti, task in enumerate(TASKS):
    for mi, model in enumerate(MODELS):
        s = score_mat[ti, mi]
        above_cur = s >= THRESH_CURRENT[task]
        above_rec = s >= THRESH_RECOMMENDED[task]
        if above_rec:
            marker, mcol = "●", "#00ff88"
        elif above_cur:
            marker, mcol = "◐", "#ffcc00"
        else:
            marker, mcol = "○", "#ff5555"
        ax.text(mi, ti, f"{s:.0f}%\n{marker}", ha="center", va="center",
                fontsize=7, color=mcol, fontweight="bold" if above_rec else "normal")

# Task labels with weight and status colour
STATUS_COL = {"✓":"#55ff55","⚠":"#ffcc00","✗":"#ff5555"}
for ti, task in enumerate(TASKS):
    st = TASK_STATUS[task]
    scol = STATUS_COL.get(st[0], "#aaa")
    w_pct = TASK_WEIGHT[task] * 100
    ax.text(-0.55, ti,
            f"{TASK_SHORT[task]}  {w_pct:.1f}%  {st}",
            ha="right", va="center", fontsize=8, color=scol)

ax.set_xticks(range(NM))
ax.set_xticklabels(MODELS, rotation=38, ha="right", fontsize=9)
ax.set_yticks([])
ax.set_xlim(-7.5, NM - 0.5)

plt.colorbar(im, ax=ax, label="Assertion-pass %", shrink=0.6, pad=0.01)
ax.set_title("Task × Model Grid  |  ● Passes recommended  ◐ Passes current only  ○ Fails both\n"
             "Task label format:  Name  Weight%  Doc Status", fontsize=11, pad=10, color="#eee")

# Sub-faculty colour strip on left
for ti, task in enumerate(TASKS):
    col = SF_COLOR[SUB[task]]
    ax.add_patch(plt.Rectangle((-7.4, ti-0.45), 0.25, 0.9, color=col, clip_on=False))

patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches, loc="lower left", bbox_to_anchor=(0, -0.22),
          ncol=4, framealpha=0, fontsize=8.5, labelcolor="#cccccc")
plt.tight_layout()
savefig("thresh_04_task_weight_grid.png")

# ══════════════════════════════════════════════════════════════
# PLOT 5 — GAP TO PASSING THRESHOLD  (how far each model needs to go)
# ══════════════════════════════════════════════════════════════
gap_cur = OVERALL_THRESH_CUR - weighted_score   # negative = already passing
gap_rec = OVERALL_THRESH_REC - weighted_score

order = np.argsort(gap_rec)   # sort by hardest gap first (most to gain last)

fig, ax = plt.subplots(figsize=(13, 7))
x = np.arange(NM)
w = 0.35

for xi, mi in enumerate(order):
    model = MODELS[mi]
    g_c = gap_cur[mi]
    g_r = gap_rec[mi]
    col = MODEL_COLOR[model]

    col_c = "#2AA364" if g_c <= 0 else "#CC2200"
    col_r = "#2AA364" if g_r <= 0 else "#FF8800"

    b1 = ax.bar(xi - w/2, g_c, w, color=col_c, alpha=0.75, edgecolor="#111",
                label="Current gap" if xi == 0 else "")
    b2 = ax.bar(xi + w/2, g_r, w, color=col_r, alpha=0.55, edgecolor="#111",
                label="Recommended gap" if xi == 0 else "")

    for bar_container, val, cc in [(b1, g_c, col_c), (b2, g_r, col_r)]:
        bar = bar_container[0]
        ypos = val + 0.5 if val >= 0 else val - 2
        ax.text(bar.get_x()+bar.get_width()/2, ypos,
                f"{'+'if val>0 else ''}{val:.1f}",
                ha="center", va="bottom" if val >= 0 else "top",
                fontsize=7.5, color=cc, fontweight="bold")

ax.axhline(0, color="#555", lw=1.2)
ax.set_xticks(x)
ax.set_xticklabels([MODELS[i] for i in order], rotation=38, ha="right", fontsize=9)
ax.set_ylabel("Score gap to threshold  (negative = already passing ✓)")
ax.set_title(
    "Gap to Passing Threshold\n"
    f"Bars below zero line = PASS  |  Current threshold: {OVERALL_THRESH_CUR}%  |  Recommended: {OVERALL_THRESH_REC}%",
    fontsize=12, pad=10, color="#eee", linespacing=1.6
)
ax.legend(framealpha=0.2, fontsize=9)
ax.yaxis.grid(True, alpha=0.4)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("thresh_05_threshold_gap.png")

# ══════════════════════════════════════════════════════════════
# PRINTED SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  METACAL WEIGHTED SCORECARD — THRESHOLD ANALYSIS")
print("="*70)
print(f"\n  Task weights:  Meta Sensitivity 30%  |  Calibration 25%")
print(f"                 Error Detection 25%   |  Thinking Path 20%")
print(f"\n  Current overall pass threshold:     >= {OVERALL_THRESH_CUR}%")
print(f"  Recommended overall pass threshold: >= {OVERALL_THRESH_REC}%")
print()
print(f"  {'#':<3}  {'Model':<22}  {'Score':>7}  {'Current':>10}  {'Recommended':>12}  {'Delta':>7}")
print(f"  {'-'*65}")
for rank, mi in enumerate(np.argsort(-weighted_score), 1):
    m = MODELS[mi]
    sc = weighted_score[mi]
    vc, _ = verdict(sc, OVERALL_THRESH_CUR)
    vr, _ = verdict(sc, OVERALL_THRESH_REC)
    delta = sc - OVERALL_THRESH_REC
    print(f"  {rank:<3}  {m:<22}  {sc:>6.1f}%  {vc:>10}  {vr:>12}  {delta:>+6.1f}%")

print(f"\n  Sub-faculty weighted pass rates (recommended thresholds):")
for sf in ["Calibration","Error Detection","Meta Sensitivity","Thinking Path"]:
    vals = sf_score[sf]
    thr  = SF_THRESH[sf]
    n_pass = sum(1 for v in vals if v >= thr)
    print(f"    {sf:<20}  threshold {thr}%  |  {n_pass}/{NM} models pass")

print(f"\n  Tasks flagged as broken/too-weak in doc (lowest reliability):")
for t, s in TASK_STATUS.items():
    if s.startswith("✗"):
        print(f"    {t}: {TASK_SHORT[t]:<22}  weight={TASK_WEIGHT[t]*100:.1f}%  [{s}]")
print("="*70)
