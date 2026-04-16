"""
MetaCal — Score-Only Visualization (no True/False pass logic)
=============================================================
Re-generates all 12 extended plots using ONLY the numeric assertion
scores (0–100). There is no binary pass/fail anywhere — every metric
is a continuous percentage derived from the scores themselves.

Output folder: score_only/

Plots produced
--------------
  1.  heatmap_score        — assertion % per model × task (continuous colour)
  2.  overall_bar          — mean score per model
  3.  subfaculty_grouped   — mean assertion % per sub-faculty per model
  4.  radar                — model profiles (polar, score-based)
  5.  score_stacked        — passed vs missed assertion estimates (from scores)
  6.  leaderboard          — ranked table by mean score
  7.  task_difficulty      — tasks sorted by cross-model mean score
  8.  model_consistency    — std-dev of scores across tasks (error-bar)
  9.  gap_from_leader      — how far each model is behind the best per task
  10. subfaculty_matrix    — compact sub-faculty × model heatmap
  11. calibration_deep     — calibration task deep-dive
  12. bubble_grid          — bubble size = assertion %, shaded by score tier

Run: python visualize_score_only.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from raw_results import RAW as _RAW

OUT = Path(__file__).parent / "score_only"
OUT.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# SHARED DATA — derived from raw_results.py (single source of truth)
# ──────────────────────────────────────────────────────────────────────────────
TASKS  = list(_RAW.keys())
MODELS = list(next(iter(_RAW.values())).keys())

TASK_LABEL = {
    "T-01": "T-01 Graded Confidence",
    "T-02": "T-02 Strategy Selection",
    "T-03": "T-03 Uncertainty Injection",
    "T-04": "T-04 Post-Answer Error Flag",
    "T-05": "T-05 Injected Error Detection",
    "T-06": "T-06 Difficulty Prediction",
    "T-07": "T-07 Accuracy-Matched Conf",
    "T-08": "T-08 Confabulation vs Correction",
    "T-09": "T-09 Thinking Path Quality",
    "T-10": "T-10 Hallucination & Abstention",
    "T-11": "T-11 Logical Consistency",
    "T-12": "T-12 Abstention Capability",
}

SUB = {
    "T-01":"Calibration","T-03":"Calibration",
    "T-04":"Error Detection","T-05":"Error Detection",
    "T-07":"Meta Sensitivity","T-08":"Meta Sensitivity",
    "T-02":"Thinking Path","T-06":"Thinking Path",
    "T-09":"Thinking Path","T-10":"Thinking Path","T-11":"Thinking Path","T-12":"Thinking Path",
}

SF_COLOR = {
    "Calibration":     "#4C72B0",
    "Error Detection": "#DD8452",
    "Meta Sensitivity":"#55A868",
    "Thinking Path":   "#C44E52",
}

MODEL_COLOR = {
    "Claude Opus 4.6":               "#D4762A",
    "Claude Sonnet 4.6":             "#E8A850",
    "Gemini 2.5 Flash":              "#378ADD",
    "Gemini 3.1 Flash-Lite Preview": "#5BA0E8",
    "Gemma 4 31B":                   "#7EA8CC",
    "GLM-5":                         "#E84393",
    "GPT-5.4":                       "#2AA364",
    "Qwen 3 Next 80B Thinking":      "#55A868",
    "Deepseek V3.1":                 "#7B5CF5",
    "GPT-5.4 mini":                  "#60C090",
    "DeepSeek-R1":                   "#FF6B6B",
}

# ── Build score matrix from raw_results.py ───────────────────
# score = num/denom * 100  (precise fraction, not rounded bucket)
NM, NT = len(MODELS), len(TASKS)
score_mat = np.zeros((NT, NM))

for ti, task in enumerate(TASKS):
    for mi, model in enumerate(MODELS):
        _, num, denom = _RAW[task][model]
        score_mat[ti, mi] = num / denom * 100

model_mean_score = score_mat.mean(axis=0)   # per-model mean assertion %
task_mean_score  = score_mat.mean(axis=1)   # per-task mean assertion %
overall_mean     = score_mat.mean()

# Sub-faculty score matrices
SUBFACS = ["Calibration", "Error Detection", "Meta Sensitivity", "Thinking Path"]
SF_TASKS = {sf: [t for t in TASKS if SUB[t] == sf] for sf in SUBFACS}

sf_score_mat = np.zeros((len(SUBFACS), NM))
for sfi, sf in enumerate(SUBFACS):
    tindices = [TASKS.index(t) for t in SF_TASKS[sf]]
    sf_score_mat[sfi] = score_mat[tindices].mean(axis=0)

# ── Global style ─────────────────────────────────────────────
BG   = "#0f0f0f"
AXES = "#1a1a1a"
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": AXES,
    "axes.edgecolor": "#333", "axes.labelcolor": "#ccc",
    "xtick.color": "#888", "ytick.color": "#888",
    "text.color": "#eee", "grid.color": "#2a2a2a",
    "grid.linewidth": 0.5, "font.family": "monospace", "font.size": 10,
})

def savefig(name: str, fig=None):
    path = OUT / name
    (fig or plt).savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close("all")
    print(f"[SAVED] {path}")

CMAP_RG = LinearSegmentedColormap.from_list("rg", ["#3d0000","#8B0000","#226622","#2AA364"])

# ══════════════════════════════════════════════════════════════
# PLOT 1 — SCORE HEATMAP  (assertion % — continuous, no PASS/FAIL)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(score_mat.T, aspect="auto", cmap=CMAP_RG, vmin=0, vmax=100)

for mi in range(NM):
    for ti in range(NT):
        val = score_mat[ti, mi]
        # colour the text by score tier instead of pass/fail
        if val >= 80:
            tc = "#88ff88"
        elif val >= 50:
            tc = "#ffdd66"
        else:
            tc = "#ff7777"
        ax.text(ti, mi, f"{val:.0f}%", ha="center", va="center",
                fontsize=8.5, color=tc, fontweight="bold")

ax.set_xticks(range(NT))
ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=42, ha="right", fontsize=8)
ax.set_yticks(range(NM))
ax.set_yticklabels(MODELS, fontsize=9)
plt.colorbar(im, ax=ax, label="Assertion score %", shrink=0.7)

# Sub-faculty colour strip on top
for ti, task in enumerate(TASKS):
    col = SF_COLOR[SUB[task]]
    ax.add_patch(plt.Rectangle((ti - 0.5, -1.0), 1.0, 0.5, color=col, clip_on=False))

ax.set_title(f"MetaCal — Score Heatmap  |  overall mean score {overall_mean:.1f}%",
             fontsize=13, pad=24, color="#eee")
patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(0, -0.22),
          ncol=4, framealpha=0, fontsize=8, labelcolor="#cccccc")
plt.tight_layout()
savefig("s01_heatmap.png")

# ══════════════════════════════════════════════════════════════
# PLOT 2 — OVERALL BAR  (mean assertion % per model)
# ══════════════════════════════════════════════════════════════
sorted_idx   = np.argsort(-model_mean_score)
s_models     = [MODELS[i] for i in sorted_idx]
s_mean_score = model_mean_score[sorted_idx]
s_colors     = [MODEL_COLOR[m] for m in s_models]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(s_models))
bars = ax.bar(x, s_mean_score, 0.6, color=s_colors, alpha=0.88, edgecolor="#111")

for bar, val in zip(bars, s_mean_score):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=9, color="#ddd")

ax.set_xticks(x)
ax.set_xticklabels(s_models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean Assertion Score (%)")
ax.set_ylim(0, 115)
ax.set_title("MetaCal — Mean Assertion Score per Model", fontsize=13, pad=10)
ax.axhline(50, color="#555", lw=0.8, ls="--", alpha=0.6, label="50% line")
ax.axhline(overall_mean, color="#aaa", lw=1.0, ls=":", alpha=0.7,
           label=f"Overall mean ({overall_mean:.1f}%)")
ax.legend(framealpha=0.15, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s02_overall_bar.png")

# ══════════════════════════════════════════════════════════════
# PLOT 3 — SUB-FACULTY GROUPED BAR  (mean assertion % per sub-faculty)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
x    = np.arange(NM)
w    = 0.21
sf_c = [SF_COLOR[sf] for sf in SUBFACS]

for i, (sf, col) in enumerate(zip(SUBFACS, sf_c)):
    vals  = sf_score_mat[i]
    rects = ax.bar(x + (i-1.5)*w, vals, w, label=sf, color=col, alpha=0.85, edgecolor="#111")
    for r, v in zip(rects, vals):
        if v > 5:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color="#ccc")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Mean Assertion Score (%)")
ax.set_ylim(0, 120)
ax.set_title("Mean Assertion Score by Sub-Faculty per Model", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s03_subfaculty_grouped.png")

# ══════════════════════════════════════════════════════════════
# PLOT 4 — RADAR CHART  (sub-faculty score profiles)
# ══════════════════════════════════════════════════════════════
N = len(SUBFACS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor(AXES)
fig.patch.set_facecolor(BG)

highlight = ["Claude Opus 4.6", "GPT-5.4", "Deepseek V3.1", "GLM-5", "Gemma 4 31B"]
for mi, model in enumerate(MODELS):
    vals = [sf_score_mat[sfi, mi] / 100 for sfi in range(N)] + [sf_score_mat[0, mi] / 100]
    col  = MODEL_COLOR[model]
    lw   = 2.5 if model in highlight else 1.0
    al   = 0.9 if model in highlight else 0.35
    ax.plot(angles, vals, color=col, lw=lw, label=model, alpha=al)
    ax.fill(angles, vals, color=col, alpha=0.04 if model not in highlight else 0.10)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(SUBFACS, fontsize=10, color="#bbb")
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25%","50%","75%","100%"], fontsize=7, color="#555")
ax.grid(color="#2a2a2a", lw=0.6)
ax.spines["polar"].set_color("#333")
ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15),
          framealpha=0.15, fontsize=8, labelcolor="#ccc")
ax.set_title("Sub-Faculty Radar — Model Profiles (score-based)", fontsize=13, pad=20, color="#eee")
plt.tight_layout()
savefig("s04_radar.png")

# ══════════════════════════════════════════════════════════════
# PLOT 5 — SCORED ASSERTION STACKED BAR
# Assertion counts estimated from continuous scores (no binary pass/fail)
# ══════════════════════════════════════════════════════════════
ASSERTS_PER_TASK = 4
total_asserts = NT * ASSERTS_PER_TASK

model_earned  = (model_mean_score / 100) * total_asserts
model_missed  = total_asserts - model_earned
order = np.argsort(-model_earned)

fig, ax = plt.subplots(figsize=(11, 5))
ypos = np.arange(NM)
ax.barh(ypos, model_earned[order], color="#2AA364", label="Scored (proportional)",
        alpha=0.85, edgecolor="#111")
ax.barh(ypos, model_missed[order], left=model_earned[order],
        color="#8B0000", label="Missed", alpha=0.85, edgecolor="#111")

for i, mi in enumerate(order):
    p = model_earned[mi]
    f = model_missed[mi]
    ax.text(p/2,   i, f"{p:.1f}", ha="center", va="center", fontsize=9, color="white")
    ax.text(p+f/2, i, f"{f:.1f}", ha="center", va="center", fontsize=9, color="white")

ax.set_yticks(ypos)
ax.set_yticklabels([MODELS[i] for i in order], fontsize=9)
ax.set_xlabel(f"Estimated assertion weight (scaled to {total_asserts} total)")
ax.set_title("Assertion Score Distribution — Earned vs Missed per Model", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s05_score_stacked.png")

# ══════════════════════════════════════════════════════════════
# PLOT 6 — LEADERBOARD TABLE  (ranked by mean score)
# ══════════════════════════════════════════════════════════════
lb_order = np.argsort(-model_mean_score)
col_hdrs = ["Rank","Model","Mean Score%","Calibration","Error Det.","Meta Sens.","Thinking"]
rows = []
for rank, mi in enumerate(lb_order, 1):
    model = MODELS[mi]
    row = [f"#{rank}", model, f"{model_mean_score[mi]:.1f}%"]
    for sfi in range(len(SUBFACS)):
        row.append(f"{sf_score_mat[sfi, mi]:.0f}%")
    rows.append(row)

fig, ax = plt.subplots(figsize=(15, max(3, NM*0.75 + 1.5)))
ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=col_hdrs, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.7)

for j in range(len(col_hdrs)):
    tbl[0, j].set_facecolor("#1e3a5f")
    tbl[0, j].set_text_props(color="#eee", fontweight="bold")

for i, mi_ord in enumerate(lb_order, 1):
    model = MODELS[mi_ord]
    bg = "#1e1a00" if i == 1 else ("#1a1a1a" if i%2==0 else "#141414")
    for j in range(len(col_hdrs)):
        tbl[i, j].set_facecolor(bg)
        tbl[i, j].set_text_props(color="#ddd")
    tbl[i, 1].set_text_props(color=MODEL_COLOR[model], fontweight="bold")

ax.set_title("MetaCal Final Leaderboard  (by mean assertion score)", fontsize=13, pad=16, color="#eee")
plt.tight_layout()
savefig("s06_leaderboard.png")

# ══════════════════════════════════════════════════════════════
# PLOT 7 — TASK DIFFICULTY RANKING  (by mean score across models)
# ══════════════════════════════════════════════════════════════
diff_order = np.argsort(-task_mean_score)   # easiest → hardest
d_labels   = [TASK_LABEL[TASKS[i]] for i in diff_order]
d_scores   = task_mean_score[diff_order]
d_colors   = [SF_COLOR[SUB[TASKS[i]]] for i in diff_order]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(NT), d_scores, color=d_colors, edgecolor="#111", height=0.7, alpha=0.88)
for bar, val in zip(bars, d_scores):
    ax.text(val + 0.8, bar.get_y()+bar.get_height()/2,
            f"{val:.1f}%", va="center", fontsize=9, color="#ccc")

ax.set_yticks(range(NT))
ax.set_yticklabels(d_labels, fontsize=9)
ax.set_xlabel("Mean Assertion Score across Models (%)")
ax.set_xlim(0, 110)
ax.axvline(50, color="#555", lw=0.8, ls="--", alpha=0.6, label="50% line")
ax.axvline(overall_mean, color="#aaa", lw=0.8, ls=":", alpha=0.7,
           label=f"Overall mean ({overall_mean:.1f}%)")
ax.set_title("Task Difficulty Ranking  (highest → lowest mean score)", fontsize=13, pad=10)
patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches + [
    mpatches.Patch(color="#555", label="50% line"),
    mpatches.Patch(color="#aaa", label=f"mean {overall_mean:.1f}%"),
], framealpha=0.15, fontsize=9)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s07_task_difficulty.png")

# ══════════════════════════════════════════════════════════════
# PLOT 8 — MODEL CONSISTENCY  (mean ± std across tasks)
# ══════════════════════════════════════════════════════════════
m_mean = score_mat.mean(axis=0)
m_std  = score_mat.std(axis=0)
cons_order = np.argsort(-m_mean)

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(NM)
c_colors = [MODEL_COLOR[MODELS[i]] for i in cons_order]
ax.bar(x, m_mean[cons_order], color=c_colors, alpha=0.75, edgecolor="#111",
       width=0.6, label="Mean score")
ax.errorbar(x, m_mean[cons_order], yerr=m_std[cons_order],
            fmt="none", color="#eee", capsize=5, capthick=1.2, elinewidth=1.2,
            label="±1 std dev")

for xi, mi in enumerate(cons_order):
    ax.text(xi, m_mean[mi] + m_std[mi] + 2,
            f"σ={m_std[mi]:.0f}", ha="center", fontsize=7.5, color="#999")

ax.set_xticks(x)
ax.set_xticklabels([MODELS[i] for i in cons_order], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Assertion score % (mean ± std across tasks)")
ax.set_ylim(0, 120)
ax.set_title("Model Consistency — Mean Score ± Variability Across Tasks", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s08_consistency.png")

# ══════════════════════════════════════════════════════════════
# PLOT 9 — GAP FROM LEADER  (how far behind the top score)
# ══════════════════════════════════════════════════════════════
best_per_task = score_mat.max(axis=1)
gap_mat = best_per_task[:, None] - score_mat

fig, ax = plt.subplots(figsize=(16, 7))
gap_cmap = LinearSegmentedColormap.from_list("gap", ["#006600","#aaaa00","#cc0000","#660000"])
im = ax.imshow(gap_mat.T, aspect="auto", cmap=gap_cmap, vmin=0, vmax=100)

for mi in range(NM):
    for ti in range(NT):
        g = gap_mat[ti, mi]
        ax.text(ti, mi, f"-{g:.0f}%", ha="center", va="center",
                fontsize=7.5, color="white" if g > 30 else "#111")

ax.set_xticks(range(NT))
ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=42, ha="right", fontsize=8)
ax.set_yticks(range(NM))
ax.set_yticklabels(MODELS, fontsize=9)
plt.colorbar(im, ax=ax, label="Gap below best score (%)", shrink=0.7)
ax.set_title("Gap from Best Score per Task  (green=near-best, red=far behind)", fontsize=13, pad=10)
plt.tight_layout()
savefig("s09_gap_from_leader.png")

# ══════════════════════════════════════════════════════════════
# PLOT 10 — SUB-FACULTY HEATMAP  (compact mean score overview)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(sf_score_mat, aspect="auto", cmap=CMAP_RG, vmin=0, vmax=100)

for sfi in range(len(SUBFACS)):
    for mi in range(NM):
        val = sf_score_mat[sfi, mi]
        ax.text(mi, sfi, f"{val:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white" if val < 60 else "#111")

ax.set_xticks(range(NM))
ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(SUBFACS)))
ax.set_yticklabels(SUBFACS, fontsize=10)
plt.colorbar(im, ax=ax, label="Mean assertion score %", shrink=0.8)
ax.set_title("Sub-Faculty × Model Heatmap  (mean assertion score %)", fontsize=13, pad=10)
plt.tight_layout()
savefig("s10_subfaculty_matrix.png")

# ══════════════════════════════════════════════════════════════
# PLOT 11 — CALIBRATION DEEP-DIVE
# ══════════════════════════════════════════════════════════════
cal_tasks  = SF_TASKS["Calibration"]
cal_idx    = [TASKS.index(t) for t in cal_tasks]
cal_labels = [TASK_LABEL[t] for t in cal_tasks]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: per-task scores within calibration, per model
cal_score = score_mat[cal_idx]
bar_x     = np.arange(NM)
w         = 0.21
n_cal = len(cal_labels)
for ci, (tlabel, col) in enumerate(zip(cal_labels,
                                        ["#4C72B0","#6896CA","#8cbce0","#b0d4f0"])):
    vals  = cal_score[ci]
    offset = (ci - (n_cal - 1) / 2) * w
    rects = axes[0].bar(bar_x + offset, vals, w,
                        label=tlabel[:20], color=col, alpha=0.85, edgecolor="#111")
    for r, v in zip(rects, vals):
        if v > 8:
            axes[0].text(r.get_x()+r.get_width()/2, r.get_height()+1,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#bbb")

axes[0].set_xticks(bar_x)
axes[0].set_xticklabels(MODELS, rotation=35, ha="right", fontsize=8)
axes[0].set_ylabel("Assertion score %")
axes[0].set_ylim(0, 120)
axes[0].set_title("Calibration Task Scores per Model", fontsize=11)
axes[0].legend(framealpha=0.2, fontsize=8)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# Right: calibration vs error-detection scatter
ed_idx  = [TASKS.index(t) for t in SF_TASKS["Error Detection"]]
cal_avg = score_mat[cal_idx].mean(axis=0)
ed_avg  = score_mat[ed_idx].mean(axis=0)

for mi, model in enumerate(MODELS):
    col = MODEL_COLOR[model]
    axes[1].scatter(cal_avg[mi], ed_avg[mi], color=col, s=110, zorder=5,
                    edgecolors="#111", lw=1)
    axes[1].annotate(model, (cal_avg[mi], ed_avg[mi]),
                     textcoords="offset points", xytext=(6,3), fontsize=7.5, color=col)

axes[1].plot([0,100],[0,100], color="#444", lw=0.8, ls="--", alpha=0.5)
axes[1].set_xlabel("Calibration mean score %")
axes[1].set_ylabel("Error Detection mean score %")
axes[1].set_title("Calibration vs Error Detection\n(diagonal = balanced)", fontsize=11)
axes[1].set_xlim(0,110); axes[1].set_ylim(0,110)
axes[1].grid()

fig.suptitle("Calibration Sub-Faculty Deep-Dive  (score-based)", fontsize=13, color="#eee", y=1.01)
plt.tight_layout()
savefig("s11_calibration_deepdive.png")

# ══════════════════════════════════════════════════════════════
# PLOT 12 — BUBBLE GRID  (size & shade by continuous score)
# No binary pass/fail — colour intensity encodes score tier
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

for mi, model in enumerate(MODELS):
    for ti, task in enumerate(TASKS):
        s   = score_mat[ti, mi]
        size = max(20, s * 2.8)
        # shade: green ≥80, yellow 50-79, red <50
        if s >= 80:
            col = "#2AA364"
        elif s >= 50:
            col = "#CC9900"
        else:
            col = SF_COLOR[SUB[task]]
        alpha = 0.2 + 0.7 * (s / 100)
        ax.scatter(ti, mi, s=size, color=col, alpha=alpha,
                   edgecolors="#aaa" if s >= 80 else "#333",
                   linewidths=0.8 if s >= 80 else 0.4)

    # Annotate the worst-scoring task per model
    worst_ti = int(np.argmin(score_mat[:, mi]))
    ax.annotate("", xy=(worst_ti, mi),
                xytext=(worst_ti + 0.6, mi + 0.35),
                arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.2))

ax.set_xticks(range(NT))
ax.set_xticklabels([t for t in TASKS], fontsize=9)
ax.set_yticks(range(NM))
ax.set_yticklabels(MODELS, fontsize=9)
ax.set_title("Bubble Grid — size & colour = assertion score, arrow = worst task per model",
             fontsize=11, pad=10)
legend_patches = [
    mpatches.Patch(color="#2AA364", label="Score ≥ 80%"),
    mpatches.Patch(color="#CC9900", label="Score 50–79%"),
    mpatches.Patch(color="#8B0000", label="Score < 50%"),
]
ax.legend(handles=legend_patches, loc="upper right", framealpha=0.15, fontsize=9)
ax.xaxis.grid(True, lw=0.3)
ax.yaxis.grid(True, lw=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("s12_bubble_grid.png")

# ══════════════════════════════════════════════════════════════
# PRINTED SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  METACAL — SCORE-ONLY SUMMARY")
print("="*65)

print(f"\n[1] OVERALL MEAN ASSERTION SCORE: {overall_mean:.1f}%")

print("\n[2] MODEL RANKING (by mean assertion score):")
for rank, mi in enumerate(np.argsort(-model_mean_score), 1):
    print(f"    #{rank:2d}  {MODELS[mi]:<22}  {model_mean_score[mi]:5.1f}%")

print("\n[3] SUB-FACULTY MEAN SCORES:")
sf_overall = [sf_score_mat[i].mean() for i in range(len(SUBFACS))]
for sf, val in sorted(zip(SUBFACS, sf_overall), key=lambda x: x[1], reverse=True):
    bar = "#" * int(val / 5)
    print(f"    {sf:<20}  {val:5.1f}%  {bar}")

print("\n[4] TASK DIFFICULTY (highest score = easiest):")
for ti in np.argsort(-task_mean_score):
    print(f"    {TASKS[ti]}  {task_mean_score[ti]:5.1f}%  ({SUB[TASKS[ti]]})  "
          f"{TASK_LABEL[TASKS[ti]][5:]}")

print(f"\n[5] 12 score-only plots saved to: {OUT}")
print("="*65)
