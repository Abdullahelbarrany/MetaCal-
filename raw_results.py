# Model column order (positions 1-10):
# 1. Claude Opus 4.6
# 2. Claude Sonnet 4.6
# 3. Deepseek V3.1
# 4. Gemini 2.5 Flash
# 5. Gemini 3.1 Flash-Lite Preview
# 6. Gemma 4 31B
# 7. GPT-5.4
# 8. Qwen 3 Next 80B Thinking
# 9. GLM-5
# 10. GPT-5.4 mini

# Structure: RAW[task_id][model_name] = (passed: bool, score_num: int, score_denom: int)

RAW: dict[str, dict[str, tuple[bool, int, int]]] = {

    "T-01": {
        # Graded Confidence — Factual Trivia (v10)
        "Claude Opus 4.6":               (True,  25, 25),
        "Claude Sonnet 4.6":             (True,  25, 25),
        "Deepseek V3.1":                 (False, 22, 25),
        "Gemini 2.5 Flash":              (False, 24, 25),
        "Gemini 3.1 Flash-Lite Preview": (False, 23, 25),
        "Gemma 4 31B":                   (False, 24, 25),
        "GPT-5.4":                       (False, 24, 25),
        "Qwen 3 Next 80B Thinking":      (False, 23, 25),
        "GLM-5":                         (False, 24, 25),
        "GPT-5.4 mini":                  (False, 24, 25),
    },

    "T-02": {
        # Strategy Selection & Execution (v4) — shown as "T-2" on Kaggle
        "Claude Opus 4.6":               (False,  8, 13),
        "Claude Sonnet 4.6":             (False,  9, 13),
        "Deepseek V3.1":                 (True,  13, 13),
        "Gemini 2.5 Flash":              (False,  0, 13),
        "Gemini 3.1 Flash-Lite Preview": (False,  6, 13),
        "Gemma 4 31B":                   (False,  8, 13),
        "GPT-5.4":                       (False,  7, 13),
        "Qwen 3 Next 80B Thinking":      (False,  9, 13),
        "GLM-5":                         (False,  8, 13),
        "GPT-5.4 mini":                  (False,  8, 13),
    },

    "T-03": {
        # Uncertainty Injection — Confidence Drop (v1)
        "Claude Opus 4.6":               (False, 22, 25),
        "Claude Sonnet 4.6":             (False, 20, 25),
        "Deepseek V3.1":                 (False, 13, 24),
        "Gemini 2.5 Flash":              (False, 13, 25),
        "Gemini 3.1 Flash-Lite Preview": (False, 14, 25),
        "Gemma 4 31B":                   (False,  5, 25),
        "GPT-5.4":                       (False, 21, 25),
        "Qwen 3 Next 80B Thinking":      (False,  8, 25),
        "GLM-5":                         (False, 16, 25),
        "GPT-5.4 mini":                  (False, 10, 25),
    },

    "T-04": {
        # Post-Answer Error Flag (v4, mixed totals)
        "Claude Opus 4.6":               (False, 20, 26),
        "Claude Sonnet 4.6":             (False, 21, 26),
        "Deepseek V3.1":                 (False, 18, 26),
        "Gemini 2.5 Flash":              (False, 17, 23),
        "Gemini 3.1 Flash-Lite Preview": (False, 19, 23),
        "Gemma 4 31B":                   (False, 17, 23),
        "GPT-5.4":                       (False, 19, 23),
        "Qwen 3 Next 80B Thinking":      (False, 25, 26),
        "GLM-5":                         (False, 17, 23),
        "GPT-5.4 mini":                  (False, 19, 23),
    },

    "T-05": {
        # Injected Error Detection — Sophisticated Stable (v2, total=19)
        "Claude Opus 4.6":               (False, 18, 19),
        "Claude Sonnet 4.6":             (False, 18, 19),
        "Deepseek V3.1":                 (False, 16, 19),
        "Gemini 2.5 Flash":              (False, 16, 19),
        "Gemini 3.1 Flash-Lite Preview": (False, 16, 19),
        "Gemma 4 31B":                   (False, 14, 19),
        "GPT-5.4":                       (False, 18, 19),
        "Qwen 3 Next 80B Thinking":      (False, 16, 19),
        "GLM-5":                         (False, 18, 19),
        "GPT-5.4 mini":                  (False, 16, 19),
    },

    "T-06": {
        # Difficulty Prediction & Calibration (v4) — shown as "T-6" on Kaggle
        "Claude Opus 4.6":               (False, 14, 15),
        "Claude Sonnet 4.6":             (False, 14, 15),
        "Deepseek V3.1":                 (False, 12, 15),
        "Gemini 2.5 Flash":              (False, 13, 15),
        "Gemini 3.1 Flash-Lite Preview": (True,  15, 15),
        "Gemma 4 31B":                   (False,  6, 15),
        "GPT-5.4":                       (False,  9, 15),
        "Qwen 3 Next 80B Thinking":      (False, 14, 15),
        "GLM-5":                         (False,  6, 15),
        "GPT-5.4 mini":                  (False, 13, 15),
    },

    "T-07": {
        # Accuracy-Matched Confidence Discrimination (v4, mixed totals)
        "Claude Opus 4.6":               (False,  4,  5),
        "Claude Sonnet 4.6":             (False,  2,  4),
        "Deepseek V3.1":                 (False,  2,  5),
        "Gemini 2.5 Flash":              (False,  2,  4),
        "Gemini 3.1 Flash-Lite Preview": (False,  2,  4),
        "Gemma 4 31B":                   (False,  2,  4),
        "GPT-5.4":                       (True,   5,  5),
        "Qwen 3 Next 80B Thinking":      (False,  2,  4),
        "GLM-5":                         (False,  2,  4),
        "GPT-5.4 mini":                  (False,  2,  5),
    },

    "T-08": {
        # Confabulation vs. Genuine Correction (v2, total=19)
        "Claude Opus 4.6":               (False, 18, 19),
        "Claude Sonnet 4.6":             (False, 18, 19),
        "Deepseek V3.1":                 (False, 18, 19),
        "Gemini 2.5 Flash":              (False, 18, 19),
        "Gemini 3.1 Flash-Lite Preview": (False, 18, 19),
        "Gemma 4 31B":                   (False, 18, 19),
        "GPT-5.4":                       (False, 18, 19),
        "Qwen 3 Next 80B Thinking":      (True,  19, 19),
        "GLM-5":                         (False, 18, 19),
        "GPT-5.4 mini":                  (False, 18, 19),
    },

    "T-09": {
        # Thinking Path Quality (v2, total=62)
        "Claude Opus 4.6":               (False, 60, 62),
        "Claude Sonnet 4.6":             (False, 60, 62),
        "Deepseek V3.1":                 (False, 60, 62),
        "Gemini 2.5 Flash":              (True,  62, 62),
        "Gemini 3.1 Flash-Lite Preview": (False, 60, 62),
        "Gemma 4 31B":                   (True,  62, 62),
        "GPT-5.4":                       (False, 59, 62),
        "Qwen 3 Next 80B Thinking":      (False, 60, 62),
        "GLM-5":                         (False, 60, 62),
        "GPT-5.4 mini":                  (False, 60, 62),
    },

    "T-10": {
        # Hallucination & Abstention Detection (v2, total=22)
        "Claude Opus 4.6":               (False, 19, 22),
        "Claude Sonnet 4.6":             (False, 20, 22),
        "Deepseek V3.1":                 (False, 21, 22),
        "Gemini 2.5 Flash":              (False, 20, 22),
        "Gemini 3.1 Flash-Lite Preview": (False, 20, 22),
        "Gemma 4 31B":                   (False, 20, 22),
        "GPT-5.4":                       (False, 20, 22),
        "Qwen 3 Next 80B Thinking":      (False, 20, 22),
        "GLM-5":                         (False, 20, 22),
        "GPT-5.4 mini":                  (False, 20, 22),
    },

    "T-11": {
        # Logical Consistency Detection (v2, total=35)
        "Claude Opus 4.6":               (False, 24, 35),
        "Claude Sonnet 4.6":             (False, 24, 35),
        "Deepseek V3.1":                 (False, 24, 35),
        "Gemini 2.5 Flash":              (False, 24, 35),
        "Gemini 3.1 Flash-Lite Preview": (False, 24, 35),
        "Gemma 4 31B":                   (False, 24, 35),
        "GPT-5.4":                       (False, 23, 35),
        "Qwen 3 Next 80B Thinking":      (False, 22, 35),
        "GLM-5":                         (False, 23, 35),
        "GPT-5.4 mini":                  (False, 24, 35),
    },

    "T-12": {
        # Abstention Capability (v2, mixed totals)
        "Claude Opus 4.6":               (False, 11, 16),
        "Claude Sonnet 4.6":             (False, 11, 16),
        "Deepseek V3.1":                 (False, 11, 16),
        "Gemini 2.5 Flash":              (False, 10, 15),
        "Gemini 3.1 Flash-Lite Preview": (False, 11, 15),
        "Gemma 4 31B":                   (False, 11, 16),
        "GPT-5.4":                       (False, 10, 16),
        "Qwen 3 Next 80B Thinking":      (False, 11, 15),
        "GLM-5":                         (False, 10, 14),
        "GPT-5.4 mini":                  (False, 11, 16),
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_str(task: str, model: str) -> str:
    """Return e.g. '22/25'."""
    _, num, denom = RAW[task][model]
    return f"{num}/{denom}"

def pct(task: str, model: str) -> float:
    """Return accuracy as a 0-100 float."""
    _, num, denom = RAW[task][model]
    return round(num / denom * 100, 1)

def passed(task: str, model: str) -> bool:
    return RAW[task][model][0]


# ── Quick summary print ───────────────────────────────────────────────────────
if __name__ == "__main__":
    MODELS = list(next(iter(RAW.values())).keys())
    TASKS  = list(RAW.keys())

    col_w = 7
    header = f"{'Model':<35}" + "".join(f"{t:>{col_w}}" for t in TASKS) + f"{'AVG':>{col_w}}  PASSES"
    print(header)
    print("-" * (len(header) + 4))

    for model in MODELS:
        scores = [pct(t, model) for t in TASKS]
        avg    = round(sum(scores) / len(scores), 1)
        n_pass = sum(passed(t, model) for t in TASKS)
        row    = (f"{model:<35}"
                  + "".join(f"{s:>{col_w-1}.0f}%" for s in scores)
                  + f"{avg:>{col_w-1}.1f}%  {n_pass}/{len(TASKS)}")
        print(row)