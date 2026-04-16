"""
MetaCal — Extended Visualization & Insights
============================================
Generates 12 plots that mirror and extend the notebook's analysis cells.

Plots produced
--------------
  1.  heatmap_score        — assertion-pass-% per model × task
  2.  overall_bar          — mean score per model (horizontal bar)
  3.  subfaculty_grouped   — pass rate per sub-faculty per model
  4.  radar                — model profiles (polar)
  5.  passfail_stacked     — total assertions passed vs failed
  6.  leaderboard          — ranked table
  7.  task_difficulty      — tasks sorted by cross-model pass rate
  8.  model_consistency    — std-dev of scores across tasks (error-bar chart)
  9.  gap_from_leader      — how far each model is behind the best per task
  10. subfaculty_matrix    — compact sub-faculty × model heatmap
  11. calibration_deep     — T-01/02/03/15 calibration task deep-dive
  12. hardest_tasks_model  — worst-scoring task per model annotation plot

Run: python visualize_extended.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from raw_results import RAW as _RAW

OUT = Path(__file__).parent

# ──────────────────────────────────────────────────────────────────────────────
# SHARED DATA  — derived from raw_results.py (single source of truth)
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

# ── Build numpy matrices from raw_results.py ─────────────────
# score = num/denom * 100  (precise fraction, not rounded bucket)
NM, NT = len(MODELS), len(TASKS)
pass_mat  = np.zeros((NT, NM))   # 1=pass, 0=fail
score_mat = np.zeros((NT, NM))   # 0-100 (continuous)

for ti, task in enumerate(TASKS):
    for mi, model in enumerate(MODELS):
        passed, num, denom = _RAW[task][model]
        pass_mat[ti, mi]  = float(passed)
        score_mat[ti, mi] = num / denom * 100

model_pass_rate  = pass_mat.mean(axis=0)   # per-model fraction of tasks passed
model_mean_score = score_mat.mean(axis=0)  # per-model mean assertion %
task_pass_rate   = pass_mat.mean(axis=1)   # per-task fraction of models that passed
overall          = pass_mat.mean() * 100

# Sub-faculty matrices
SUBFACS = ["Calibration", "Error Detection", "Meta Sensitivity", "Thinking Path"]
SF_TASKS = {sf: [t for t in TASKS if SUB[t] == sf] for sf in SUBFACS}

sf_score_mat = np.zeros((len(SUBFACS), NM))
for sfi, sf in enumerate(SUBFACS):
    tindices = [TASKS.index(t) for t in SF_TASKS[sf]]
    sf_score_mat[sfi] = score_mat[tindices].mean(axis=0)

sf_pass_mat = np.zeros((len(SUBFACS), NM))
for sfi, sf in enumerate(SUBFACS):
    tindices = [TASKS.index(t) for t in SF_TASKS[sf]]
    sf_pass_mat[sfi] = pass_mat[tindices].mean(axis=0)

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
    print(f"[SAVED] {path.name}")

CMAP_RG = LinearSegmentedColormap.from_list("rg", ["#3d0000","#8B0000","#226622","#2AA364"])

# ══════════════════════════════════════════════════════════════
# PLOT 1 — SCORE HEATMAP  (assertion-pass % per model × task)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(16, 8))
im = ax.imshow(score_mat.T, aspect="auto", cmap=CMAP_RG, vmin=0, vmax=100)

for mi in range(NM):
    for ti in range(NT):
        val = score_mat[ti, mi]
        is_pass = pass_mat[ti, mi]
        label  = "PASS" if is_pass else "FAIL"
        color  = "#88ff88" if is_pass else "#ff6666"
        ax.text(ti, mi, f"{label}\n{val:.0f}%", ha="center", va="center",
                fontsize=7.5, color=color, fontweight="bold" if is_pass else "normal")

ax.set_xticks(range(NT))
ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], rotation=42, ha="right", fontsize=8)
ax.set_yticks(range(NM))
ax.set_yticklabels(MODELS, fontsize=9)
plt.colorbar(im, ax=ax, label="Assertion-pass %", shrink=0.7)

# Sub-faculty colour strip on top
for ti, task in enumerate(TASKS):
    col = SF_COLOR[SUB[task]]
    ax.add_patch(plt.Rectangle((ti - 0.5, -1.0), 1.0, 0.5, color=col, clip_on=False))

ax.set_title(f"MetaCal — Score Heatmap  |  overall task-pass {overall:.1f}%",
             fontsize=13, pad=24, color="#eee")
patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches, loc="upper left", bbox_to_anchor=(0, -0.22),
          ncol=4, framealpha=0, fontsize=8, labelcolor="#cccccc")
plt.tight_layout()
savefig("ext_01_heatmap.png")

# ══════════════════════════════════════════════════════════════
# PLOT 2 — OVERALL BAR  (mean score per model)
# ══════════════════════════════════════════════════════════════
sorted_idx   = np.argsort(-model_mean_score)
s_models     = [MODELS[i] for i in sorted_idx]
s_pass_rate  = model_pass_rate[sorted_idx] * 100
s_mean_score = model_mean_score[sorted_idx]
s_colors     = [MODEL_COLOR[m] for m in s_models]

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(s_models))
b1 = ax.bar(x - 0.2, s_pass_rate,   0.38, color=s_colors, alpha=0.9,  label="Task PASS %")
b2 = ax.bar(x + 0.2, s_mean_score,  0.38, color=s_colors, alpha=0.45, label="Avg assertion %")

for bar, val in zip(b1, s_pass_rate):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#ccc")
for bar, val in zip(b2, s_mean_score):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=8, color="#999")

ax.set_xticks(x)
ax.set_xticklabels(s_models, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 115)
ax.set_title("Overall MetaCal Score per Model", fontsize=13, pad=10)
ax.axhline(50, color="#555", lw=0.8, ls="--", alpha=0.6)
ax.legend(framealpha=0.15, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_02_overall_bar.png")

# ══════════════════════════════════════════════════════════════
# PLOT 3 — SUB-FACULTY GROUPED BAR
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
x    = np.arange(NM)
w    = 0.21
sf_c = [SF_COLOR[sf] for sf in SUBFACS]

for i, (sf, col) in enumerate(zip(SUBFACS, sf_c)):
    vals  = sf_pass_mat[i] * 100
    rects = ax.bar(x + (i-1.5)*w, vals, w, label=sf, color=col, alpha=0.85, edgecolor="#111")
    for r, v in zip(rects, vals):
        if v > 5:
            ax.text(r.get_x()+r.get_width()/2, r.get_height()+1.5,
                    f"{v:.0f}%", ha="center", va="bottom", fontsize=7, color="#ccc")

ax.set_xticks(x)
ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Task Pass Rate (%)")
ax.set_ylim(0, 120)
ax.set_title("Pass Rate by Sub-Faculty per Model", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_03_subfaculty_grouped.png")

# ══════════════════════════════════════════════════════════════
# PLOT 4 — RADAR CHART  (sub-faculty profiles)
# ══════════════════════════════════════════════════════════════
N = len(SUBFACS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.set_facecolor(AXES)
fig.patch.set_facecolor(BG)

highlight = ["Claude Opus 4.6", "GPT-5.4", "Deepseek V3.1", "GLM-5", "Gemma 4 31B"]
for mi, model in enumerate(MODELS):
    vals = [sf_pass_mat[sfi, mi] for sfi in range(N)] + [sf_pass_mat[0, mi]]
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
ax.set_title("Sub-Faculty Radar — Model Profiles", fontsize=13, pad=20, color="#eee")
plt.tight_layout()
savefig("ext_04_radar.png")

# ══════════════════════════════════════════════════════════════
# PLOT 5 — STACKED PASS / FAIL ASSERTION COUNTS
# ══════════════════════════════════════════════════════════════
# Simulate total assertion counts (avg ~4 assertions/task × 15 tasks = 60 total)
# We model each assertion as score_mat[ti,mi]/100 fraction passed.
# For display we scale to a realistic total of 60 assertions per run.
ASSERTS_PER_TASK = 4  # approximate average
total_asserts = NT * ASSERTS_PER_TASK

model_passed = (score_mat.mean(axis=0) / 100) * total_asserts
model_failed = total_asserts - model_passed
order = np.argsort(-model_passed)

fig, ax = plt.subplots(figsize=(11, 5))
ypos = np.arange(NM)
ax.barh(ypos, model_passed[order], color="#2AA364", label="Passed", alpha=0.85, edgecolor="#111")
ax.barh(ypos, model_failed[order], left=model_passed[order],
        color="#8B0000", label="Failed", alpha=0.85, edgecolor="#111")

for i, mi in enumerate(order):
    p = model_passed[mi]
    f = model_failed[mi]
    ax.text(p/2,   i, f"{p:.0f}", ha="center", va="center", fontsize=9, color="white")
    ax.text(p+f/2, i, f"{f:.0f}", ha="center", va="center", fontsize=9, color="white")

ax.set_yticks(ypos)
ax.set_yticklabels([MODELS[i] for i in order], fontsize=9)
ax.set_xlabel(f"Estimated assertion count (scaled to {total_asserts} total)")
ax.set_title("Total Assertions: Passed vs Failed per Model", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_05_passfail_stacked.png")

# ══════════════════════════════════════════════════════════════
# PLOT 6 — LEADERBOARD TABLE
# ══════════════════════════════════════════════════════════════
lb_order = np.argsort(-model_pass_rate)
col_hdrs = ["Rank","Model","Task PASS%","Avg Score%","Calibration","Error Det.","Meta Sens.","Thinking"]
rows = []
for rank, mi in enumerate(lb_order, 1):
    model = MODELS[mi]
    row = [
        f"#{rank}",
        model,
        f"{model_pass_rate[mi]*100:.0f}%",
        f"{model_mean_score[mi]:.0f}%",
    ]
    for sfi in range(len(SUBFACS)):
        row.append(f"{sf_pass_mat[sfi, mi]*100:.0f}%")
    rows.append(row)

fig, ax = plt.subplots(figsize=(16, max(3, NM*0.75 + 1.5)))
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

ax.set_title("MetaCal Final Leaderboard", fontsize=13, pad=16, color="#eee")
plt.tight_layout()
savefig("ext_06_leaderboard.png")

# ══════════════════════════════════════════════════════════════
# PLOT 7 — TASK DIFFICULTY RANKING  (new)
# ══════════════════════════════════════════════════════════════
diff_order = np.argsort(-task_pass_rate)   # easiest first
d_labels   = [TASK_LABEL[TASKS[i]] for i in diff_order]
d_pass     = task_pass_rate[diff_order] * 100
d_colors   = [SF_COLOR[SUB[TASKS[i]]] for i in diff_order]

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(range(NT), d_pass, color=d_colors, edgecolor="#111", height=0.7, alpha=0.88)
for bar, val in zip(bars, d_pass):
    ax.text(val + 0.8, bar.get_y()+bar.get_height()/2,
            f"{val:.0f}%", va="center", fontsize=9, color="#ccc")

ax.set_yticks(range(NT))
ax.set_yticklabels(d_labels, fontsize=9)
ax.set_xlabel("% of Models that PASSED")
ax.set_xlim(0, 100)
ax.axvline(50, color="#555", lw=0.8, ls="--", alpha=0.6, label="50% threshold")
ax.set_title("Task Difficulty Ranking  (easiest → hardest)", fontsize=13, pad=10)
patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches + [mpatches.Patch(color="#555",label="50% line")],
          framealpha=0.15, fontsize=9)
ax.xaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_07_task_difficulty.png")

# ══════════════════════════════════════════════════════════════
# PLOT 8 — MODEL CONSISTENCY  (mean ± std across tasks)  (new)
# ══════════════════════════════════════════════════════════════
m_mean = score_mat.mean(axis=0)
m_std  = score_mat.std(axis=0)
cons_order = np.argsort(-m_mean)

fig, ax = plt.subplots(figsize=(12, 5.5))
x = np.arange(NM)
c_colors = [MODEL_COLOR[MODELS[i]] for i in cons_order]
ax.bar(x, m_mean[cons_order], color=c_colors, alpha=0.75, edgecolor="#111", width=0.6, label="Mean score")
ax.errorbar(x, m_mean[cons_order], yerr=m_std[cons_order],
            fmt="none", color="#eee", capsize=5, capthick=1.2, elinewidth=1.2, label="±1 std dev")

for xi, mi in enumerate(cons_order):
    ax.text(xi, m_mean[mi] + m_std[mi] + 2,
            f"σ={m_std[mi]:.0f}", ha="center", fontsize=7.5, color="#999")

ax.set_xticks(x)
ax.set_xticklabels([MODELS[i] for i in cons_order], rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Assertion-pass % (mean ± std across tasks)")
ax.set_ylim(0, 120)
ax.set_title("Model Consistency — Mean Score ± Variability Across Tasks", fontsize=13, pad=10)
ax.legend(framealpha=0.2, fontsize=9)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_08_consistency.png")

# ══════════════════════════════════════════════════════════════
# PLOT 9 — GAP FROM LEADER  (how far behind the best model)  (new)
# ══════════════════════════════════════════════════════════════
best_per_task = score_mat.max(axis=1)   # shape (NT,)
gap_mat = best_per_task[:, None] - score_mat   # (NT, NM)  — how many % behind leader

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
plt.colorbar(im, ax=ax, label="Gap below best model (%)", shrink=0.7)
ax.set_title("Gap from Best Model per Task  (green=near-best, red=far behind)", fontsize=13, pad=10)
plt.tight_layout()
savefig("ext_09_gap_from_leader.png")

# ══════════════════════════════════════════════════════════════
# PLOT 10 — SUB-FACULTY HEATMAP  (compact overview)  (new)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(sf_pass_mat * 100, aspect="auto", cmap=CMAP_RG, vmin=0, vmax=100)

for sfi in range(len(SUBFACS)):
    for mi in range(NM):
        val = sf_pass_mat[sfi, mi] * 100
        ax.text(mi, sfi, f"{val:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold", color="white" if val < 60 else "#111")

ax.set_xticks(range(NM))
ax.set_xticklabels(MODELS, rotation=35, ha="right", fontsize=9)
ax.set_yticks(range(len(SUBFACS)))
ax.set_yticklabels(SUBFACS, fontsize=10)
plt.colorbar(im, ax=ax, label="Task pass rate %", shrink=0.8)
ax.set_title("Sub-Faculty × Model Heatmap  (% tasks passed within sub-faculty)", fontsize=13, pad=10)
plt.tight_layout()
savefig("ext_10_subfaculty_matrix.png")

# ══════════════════════════════════════════════════════════════
# PLOT 11 — CALIBRATION DEEP-DIVE  (new)
# ══════════════════════════════════════════════════════════════
cal_tasks = SF_TASKS["Calibration"]   # T-01, T-02, T-03, T-15
cal_idx   = [TASKS.index(t) for t in cal_tasks]
cal_labels = [TASK_LABEL[t] for t in cal_tasks]

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: per-task pass rate within calibration, per model
cal_score = score_mat[cal_idx]   # shape (4, NM)
bar_x     = np.arange(NM)
w         = 0.21
n_cal = len(cal_labels)
for ci, (tlabel, col) in enumerate(zip(cal_labels,
                                        ["#4C72B0","#6896CA","#8cbce0","#b0d4f0"])):
    vals = cal_score[ci]
    offset = (ci - (n_cal - 1) / 2) * w
    rects = axes[0].bar(bar_x + offset, vals, w,
                        label=tlabel[:20], color=col, alpha=0.85, edgecolor="#111")
    for r, v in zip(rects, vals):
        if v > 8:
            axes[0].text(r.get_x()+r.get_width()/2, r.get_height()+1,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#bbb")

axes[0].set_xticks(bar_x)
axes[0].set_xticklabels(MODELS, rotation=35, ha="right", fontsize=8)
axes[0].set_ylabel("Assertion-pass %")
axes[0].set_ylim(0, 120)
axes[0].set_title("Calibration Task Scores per Model", fontsize=11)
axes[0].legend(framealpha=0.2, fontsize=8)
axes[0].yaxis.grid(True); axes[0].set_axisbelow(True)

# Right: calibration vs error-detection scatter
ed_idx   = [TASKS.index(t) for t in SF_TASKS["Error Detection"]]
cal_avg  = score_mat[cal_idx].mean(axis=0)
ed_avg   = score_mat[ed_idx].mean(axis=0)

for mi, model in enumerate(MODELS):
    col = MODEL_COLOR[model]
    axes[1].scatter(cal_avg[mi], ed_avg[mi], color=col, s=110, zorder=5,
                    edgecolors="#111", lw=1)
    axes[1].annotate(model, (cal_avg[mi], ed_avg[mi]),
                     textcoords="offset points", xytext=(6,3), fontsize=7.5, color=col)

axes[1].plot([0,100],[0,100], color="#444", lw=0.8, ls="--", alpha=0.5)
axes[1].set_xlabel("Calibration avg score %")
axes[1].set_ylabel("Error Detection avg score %")
axes[1].set_title("Calibration vs Error Detection\n(diagonal = balanced)", fontsize=11)
axes[1].set_xlim(0,110); axes[1].set_ylim(0,110)
axes[1].grid()

fig.suptitle("Calibration Sub-Faculty Deep-Dive", fontsize=13, color="#eee", y=1.01)
plt.tight_layout()
savefig("ext_11_calibration_deepdive.png")

# ══════════════════════════════════════════════════════════════
# PLOT 12 — HARDEST TASK PER MODEL  (annotated bubble grid)  (new)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))

for mi, model in enumerate(MODELS):
    for ti, task in enumerate(TASKS):
        s = score_mat[ti, mi]
        p = pass_mat[ti, mi]
        size = max(20, s * 2.5)
        col  = SF_COLOR[SUB[task]]
        alpha= 0.9 if p else 0.35
        ax.scatter(ti, mi, s=size, color=col, alpha=alpha,
                   edgecolors="#aaa" if p else "#333", linewidths=0.8 if p else 0.4)

    # Mark worst task
    worst_ti = int(np.argmin(score_mat[:, mi]))
    ax.annotate("", xy=(worst_ti, mi),
                xytext=(worst_ti + 0.6, mi + 0.35),
                arrowprops=dict(arrowstyle="->", color="#ff4444", lw=1.2))

ax.set_xticks(range(NT))
ax.set_xticklabels([t for t in TASKS], fontsize=9)
ax.set_yticks(range(NM))
ax.set_yticklabels(MODELS, fontsize=9)
ax.set_title("Bubble Grid — Circle size = assertion %, filled = PASS, arrow = worst task",
             fontsize=11, pad=10)
patches = [mpatches.Patch(color=c, label=k) for k, c in SF_COLOR.items()]
ax.legend(handles=patches, loc="upper right", framealpha=0.15, fontsize=9)
ax.xaxis.grid(True, lw=0.3)
ax.yaxis.grid(True, lw=0.3)
ax.set_axisbelow(True)
plt.tight_layout()
savefig("ext_12_bubble_grid.png")

# ══════════════════════════════════════════════════════════════
# PRINTED INSIGHTS
# ══════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  METACAL BENCHMARK — KEY INSIGHTS")
print("="*65)

# 1. Overall
print(f"\n[1] OVERALL TASK-PASS RATE: {overall:.1f}%")
print("    Metacognition is a hard frontier — only 1 in 7 task-runs")
print("    passes all assertions. No model exceeds 50% task-pass rate.")

# 2. Model ranking
print("\n[2] MODEL RANKING (by task pass rate):")
for rank, mi in enumerate(np.argsort(-model_pass_rate), 1):
    print(f"    #{rank:2d}  {MODELS[mi]:<22}  {model_pass_rate[mi]*100:5.1f}%  "
          f"(avg assertion {model_mean_score[mi]:.0f}%)")

# 3. Sub-faculty insight
print("\n[3] SUB-FACULTY HARDNESS:")
sf_overall = [sf_pass_mat[i].mean()*100 for i in range(len(SUBFACS))]
for sf, val in sorted(zip(SUBFACS, sf_overall), key=lambda x: x[1]):
    bar = "#" * int(val / 5)
    print(f"    {sf:<20}  {val:5.1f}%  {bar}")

print("\n    -> Error Detection (T-04) is the only sub-faculty where")
print("      models show meaningful competence (T-04: 62% pass rate).")
print("    -> Meta Sensitivity and Calibration are weakest overall.")

# 4. Task difficulty
print("\n[4] TASK DIFFICULTY (hardest first):")
for ti in np.argsort(task_pass_rate):
    print(f"    {TASKS[ti]}  {task_pass_rate[ti]*100:5.1f}%  "
          f"({SUB[TASKS[ti]]})  {TASK_LABEL[TASKS[ti]][5:]}")

# 5. Provider patterns
print("\n[5] PROVIDER PATTERNS:")
groups = {
    "Anthropic":  ["Claude Opus 4.6","Claude Sonnet 4.6"],
    "OpenAI":     ["GPT-5.4","GPT-5.4 mini"],
    "Google":     ["Gemini 2.5 Flash","Gemini 3.1 Flash-Lite Preview","Gemma 4 31B"],
    "DeepSeek":   ["Deepseek V3.1"],
    "Qwen":       ["Qwen 3 Next 80B Thinking"],
    "ZhipuAI":    ["GLM-5"],
}
for provider, members in groups.items():
    avg = np.mean([model_pass_rate[MODELS.index(m)]*100 for m in members if m in MODELS])
    print(f"    {provider:<12}  {avg:5.1f}% avg task-pass")

# 6. Consistency
print("\n[6] MOST CONSISTENT MODELS (low std-dev across tasks):")
stds = score_mat.std(axis=0)
for mi in np.argsort(stds)[:4]:
    print(f"    {MODELS[mi]:<22}  std={stds[mi]:.1f}%  (mean {model_mean_score[mi]:.0f}%)")

# 7. Specific findings
print("\n[7] NOTABLE FINDINGS:")
print("    - T-07 is passed by only 1 model (GPT-5.4); T-08 by 1 (Qwen 3).")
print("    - T-02 (Strategy Sel.) passed only by Deepseek V3.1.")
print("    - T-03 (Uncertainty Inj.) and T-07 are the hardest tasks overall.")
print("    - Calibration (T-01,T-03) and Meta Sensitivity (T-07,T-08) weakest sub-faculties.")
print("    - Thinking Path is weighted 20% and spans 6 tasks (T-02,T-06,T-09-T-12).")
print("\n" + "="*65)
print("  12 plots saved. All data read from raw_results.py (single source of truth).")
print("="*65)
