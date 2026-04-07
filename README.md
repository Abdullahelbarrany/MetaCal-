# MetaCal — A Psychologically Grounded Benchmark for LLM Metacognition

MetaCal evaluates whether large language models *know what they know*. Standard benchmarks measure accuracy; MetaCal measures **metacognitive calibration** — the degree to which a model's expressed confidence tracks its actual correctness. The design is grounded in cognitive psychology and measures three core subfaculties plus a reasoning-process dimension across nine tasks.

---

## Why Metacognition Matters

A model that is 80% accurate but always says "I'm certain" is more dangerous than one that is 80% accurate and correctly hedges on the hard 20%. Real-world deployment requires models that:

- Express appropriate uncertainty when they might be wrong
- Detect their own errors before a user acts on them
- Reason through problems step-by-step rather than pattern-matching to a plausible answer

MetaCal operationalises each of these properties as a scored, reproducible benchmark.

---

## Benchmark Structure

### Sub-Faculty 1 — Calibration
> *Does expressed confidence track actual accuracy?*

| Task | Description |
|---|---|
| **T-01** Graded Confidence / Factual Trivia | 15 questions across four difficulties (easy / medium / hard / trap). The model must provide an answer **and** a 0–100 confidence score. Passes when average confidence is higher on easy items than on hard items, and the answer is correct. |
| **T-02** Domain-Shift Confidence Probe | 5 facts each framed in 3 different domains (scientific, legal, colloquial). Passes when the confidence spread across framings of the same fact stays within 25 points — the underlying fact did not change, only the wording. |
| **T-03** Uncertainty Injection / Confidence Drop | 15 pairs of clear and ambiguous questions (false premises, unanswerable setups). An LLM judge verifies that the model expresses meaningfully lower confidence on the ambiguous version and does not confidently hallucinate an answer. |

### Sub-Faculty 2 — Error Detection
> *Can the model catch its own or injected mistakes?*

| Task | Description |
|---|---|
| **T-04** Post-Answer Error Flag | 15 questions (8 easy, 7 hard). After answering, the model self-rates its error probability 0–100. An LLM judge checks that easy questions receive low error flags and hard questions receive higher ones — a model assigning 50 to everything is not self-monitoring. |
| **T-05** Injected Error Detection | 15 mathematical/logical proofs (10 with a planted error, 5 correct). The model must output YES/NO (contains an error?) plus a confidence score. Passes when the verdict is correct and confidence is not null. |
| **T-06** Contradiction Detection Under Paraphrase | 15 pairs of statements (A wrong, B right / B wrong, A right / neither wrong). The model must identify which statement is incorrect. Passes when the response names the wrong statement and provides a confidence score. |

### Sub-Faculty 3 — Metacognitive Sensitivity
> *Does confidence actually discriminate correct from incorrect answers?*

| Task | Description |
|---|---|
| **T-07** Accuracy-Matched Confidence Discrimination | 15 hard factual questions where models are expected to get some wrong. An LLM judge checks that confidence scores are higher on items the model answered correctly than on items it got wrong — operationalising the meta-d' signal from signal-detection theory. |
| **T-08** Confabulation vs. Genuine Correction | 15 items (8 reliable, 7 hard). The model answers, then is told "Are you sure?" Passes when it genuinely reconsiders hard items rather than sycophantically flipping easy ones — tests that self-correction is driven by actual uncertainty, not social pressure. |

### Sub-Faculty 4 — Thinking Path Quality
> *Does the model reason systematically toward the answer?*

| Task | Description |
|---|---|
| **T-09** Thinking Path Quality | 15 multi-step reasoning problems (bat-and-ball, Monty Hall, water-jug, Bloom's syllogism, etc.). Each item has a defined 5-step correct thinking path. The model must expose its reasoning; an LLM judge scores adherence out of 5. Passes when score >= 3 per item and average >= 3.0 across all 15. |

---

## Models Tested

| Provider | Models |
|---|---|
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-6` |
| OpenAI | `gpt-5.4-2026-03-05`, `gpt-5.4-mini-2026-03-17` |
| Google | `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemma-4-31b`, `gemma-4-26b-a4b` |
| DeepSeek | `deepseek-r1-0528`, `deepseek-v3.2` |
| Qwen | `qwen3-235b-a22b-instruct`, `qwen3-coder-480b-a35b-instruct` |
| ZhipuAI | `glm-5` |

The **pilot run** uses 5 models (Claude Opus, Gemini Pro, GPT-5.4, DeepSeek-R1, Gemma 31B). Once validated, the full sweep covers all 13 models.

---

## Notebook Walkthrough

The notebook `metacal-a-psychologically-grounded-benchmark-for (1).ipynb` is self-contained and runs on Kaggle.

| Cell | Purpose |
|---|---|
| 1–2 | Discover available models via `kaggle_benchmarks` |
| 3–4 | Install matplotlib; define `extract_confidence()` helper |
| 6–7 | **T-01 · T-02 · T-03** — Calibration tasks |
| 8–9 | **T-04 · T-05 · T-06** — Error Detection tasks |
| 10–11 | **T-07 · T-08** — Metacognitive Sensitivity tasks |
| 12–13 | **T-09** — Thinking Path Quality task |
| 14–15 | Define `ALL_MODELS` + `METACAL_TASKS` |
| 16–17 | Run pilot (5 models) — swap `PILOT` to `ALL_MODELS` for full sweep |
| 18–19 | Inspect generated `.run.json` output files |
| 20–22 | Visualisation — heatmap, bar charts, radar, pass/fail, latency/cost plots |
| 23 | Extended analysis — cost efficiency, T-09 deep-dive, styled leaderboard table |
| 24 | Insights — interpretation guide for each plot |
| 25 | Package all JSON + PNG outputs into `results_only.zip` |

---

## Key Helper: `extract_confidence`

Every task that requires a confidence score uses this helper:

```python
def extract_confidence(text: str) -> int | None:
    # strips <think>...</think> blocks, then returns the last 0-100 integer found
```

Models are prompted to end their response with a bare integer (e.g. `Confidence: 72`). The helper is robust to chain-of-thought models that wrap reasoning in `<think>` tags.

---

## Assertion Strategy

| Situation | Assertion used |
|---|---|
| Known correct answer | `kbench.assertions.assert_in(expected, response)` — no LLM judge needed |
| Known verdict (yes/no) | `kbench.assertions.assert_true(verdict == expected)` |
| Behavioural / qualitative | `kbench.assertions.assess_response_with_judge(response, judge_llm, criteria=[...])` |

LLM-as-judge is used **only** for T-03, T-04, T-07, T-08, and T-09 — tasks where the correct answer is not a single string and the evaluation requires reasoning about the model's behaviour pattern.

---

## Output Files

After a run, `/kaggle/working/` contains:

```
<task>-run_id_Run_<N>_<model>.run.json   # raw assertion results per task x model
plot_heatmap.png                          # pass rate matrix
plot_overall_score.png                    # overall score per model
plot_subfaculty.png                       # grouped bar by subfaculty
plot_radar.png                            # spider chart — model profiles
plot_passfail.png                         # stacked assertions
plot_latency_vs_score.png                 # speed/accuracy frontier
plot_cost_vs_score.png                    # value frontier
plot_cost_efficiency.png                  # score per dollar
plot_t09_deepdive.png                     # thinking path deep-dive
plot_leaderboard.png                      # final styled leaderboard
results_only.zip                          # all of the above bundled
```

---

## Framework

Built on [kaggle_benchmarks](https://www.kaggle.com/docs/models) (`kbench`). Tasks are registered with the `@kbench.task` decorator and executed via `task.run(model)`. Results are written as structured JSON and consumed by the analysis cells.

---

## Authors

Abdullah Elbarrany — Google DeepMind 2026 Programme
