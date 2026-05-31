# DeepEval to LangSmith Evaluation Pipeline

This project runs DeepEval metrics with a Groq judge model, writes the evaluation results to JSON on disk, and then pushes those results to LangSmith as experiment feedback.

The flow is split into standalone scripts so it can run locally or from GitHub Actions.

## Files

- `1_custom_groq_class.py` defines the custom DeepEval judge model backed by Groq.
- `2_test_cases.py` defines the `LLMTestCase` examples to evaluate.
- `3_evaluation_deepEval.py` runs DeepEval and writes JSON artifacts.
- `4_push_langmsmith.py` reads the JSON artifacts and uploads them to LangSmith.
- `.github/workflows/rag-eval.yml` runs the full pipeline in GitHub Actions.

There is also an older `2.test_cases.py` file from the notebook split. Use `2_test_cases.py` for the runnable pipeline because Python modules cannot normally be imported with a dot in the filename.

## Setup

Install dependencies with `uv`:

```bash
uv sync
```

Create a `.env` file for local runs:

```bash
GROQ_API_KEY=your_groq_key
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_PROJECT="Agents Evaluation 1"
GROQ_EVAL_MODEL=llama-3.1-8b-instant
```

## Run Locally

Run DeepEval and write JSON artifacts:

```bash
uv run python 3_evaluation_deepEval.py
```

This creates:

```text
eval_outputs/deepeval_metrics_YYYY-MM-DD_HH-MM-SS_UTC.json
eval_outputs/langsmith_payload_YYYY-MM-DD_HH-MM-SS_UTC.json
outputs/evaluation_results.json
```

Example `deepeval_metrics` row:

```json
{
  "question": "What is Docker?",
  "answer": "Docker is used for containers.",
  "expected_output": "Docker is a platform used to build, package, and run applications inside containers.",
  "latency": 1.2,
  "model": "llama-3.1-8b-instant",
  "metrics": {
    "correctness": 0.2
  },
  "metric_reasons": {
    "correctness": "The actual output is partially correct but missing details."
  },
  "metric_success": {
    "correctness": false
  },
  "correctness": 0.2
}
```

Validate the LangSmith upload payload without pushing:

```bash
uv run python 4_push_langmsmith.py --dry-run
```

Push the latest payload to LangSmith:

```bash
uv run python 4_push_langmsmith.py
```

You can also push specific files:

```bash
uv run python 4_push_langmsmith.py \
  --payload-path eval_outputs/langsmith_payload_YYYY-MM-DD_HH-MM-SS_UTC.json \
  --metrics-path eval_outputs/deepeval_metrics_YYYY-MM-DD_HH-MM-SS_UTC.json
```

## GitHub Actions

The workflow at `.github/workflows/rag-eval.yml` runs on `push` and can also be started manually with `workflow_dispatch`.

Add these repository secrets:

```text
GROQ_API_KEY
LANGSMITH_API_KEY
```

The workflow:

1. Checks out the repo.
2. Installs dependencies with `uv sync`.
3. Runs `3_evaluation_deepEval.py`.
4. Uploads the JSON artifacts.
5. Runs `4_push_langmsmith.py` to push the reviewed payload shape to LangSmith.

## Add More Metrics

Add metrics in `3_evaluation_deepEval.py` inside `build_metrics()`:

```python
def build_metrics():
    correctness_metric = GEval(...)
    return [correctness_metric]
```

Return additional DeepEval metrics from that list. They will automatically appear in:

- `deepeval_metrics_*.json`
- `langsmith_payload_*.json`
- LangSmith feedback columns

## Enhanced Metric Suites

`3.1_enhanced_eval.py` supports metric suites so CI can run a faster subset or the full enhanced set.

Run the faster core suite:

```bash
uv run python 3.1_enhanced_eval.py --metric-suite core --print-json
```

Run all enhanced metrics:

```bash
uv run python 3.1_enhanced_eval.py --metric-suite all --print-json
```

Available suites:

| Suite | Metrics | Notes |
| --- | --- | --- |
| `core` | `Correctness`, `Completeness`, `AnswerRelevancy`, `ExactMatch` | Recommended for CI. Faster and focused on answer quality. |
| `safety` | `Hallucination`, `Bias`, `Toxicity` | Safety and risk checks. These are lower-is-better metrics. |
| `all` | Everything in `core` plus `Faithfulness`, `Hallucination`, `Bias`, `Toxicity` | Full enhanced run. Slower because several metrics make extra judge-model calls. |

Approximate Groq judge calls with the current 5 test cases:

| Suite | Approximate Groq Calls |
| --- | --- |
| `core` | About 15-20 calls |
| `safety` | About 15+ calls |
| `all` | About 40+ calls |

`ExactMatch` does not call the LLM. Most other DeepEval metrics call the Groq judge model at least once per test case, and some metrics may call it more than once internally.

## Notes

- `3_evaluation_deepEval.py` disables LangSmith tracing during local DeepEval execution. Only `4_push_langmsmith.py` uploads to LangSmith.
- Generated artifacts are ignored by git via `.gitignore`.
- If a DeepEval metric fails, the JSON still records the row with `null` score and a failure reason in `metric_reasons`.
