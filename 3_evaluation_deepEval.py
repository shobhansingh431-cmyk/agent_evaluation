import argparse
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

# Keep this step local: JSON is uploaded by 4_push_langmsmith.py after review/artifact creation.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

groq_judge = importlib.import_module("1_custom_groq_class").groq_judge
test_cases = importlib.import_module("2_test_cases").test_cases


def clean_text(value):
    return value.strip() if isinstance(value, str) else value


def metric_key(metric):
    return str(getattr(metric, "name", metric.__class__.__name__)).strip().replace(" ", "_").lower()


def judge_model_name(metric):
    model = getattr(metric, "model", None)
    if model is not None and hasattr(model, "get_model_name"):
        return model.get_model_name()
    return str(getattr(metric, "evaluation_model", ""))


def case_to_eval_kwargs(case: LLMTestCase) -> dict:
    kwargs = {
        "input": clean_text(case.input),
        "actual_output": clean_text(case.actual_output),
        "expected_output": clean_text(case.expected_output),
    }
    for optional_field in ("context", "retrieval_context"):
        value = getattr(case, optional_field, None)
        if value is not None:
            kwargs[optional_field] = value
    return kwargs


def build_metrics():
    correctness_metric = GEval(
        name="Correctness",
        criteria=(
            "Grade semantic correctness from 0 to 1 using this rubric: "
            "1.0 = fully correct and includes all important expected details. "
            "0.8 = correct and includes most important expected details, with only minor omissions. "
            "0.6 = directionally correct and captures the main idea, but misses several important details. "
            "0.4 = partially related but vague, incomplete, or missing the main purpose. "
            "0.2 = minimally related, mostly incomplete, or only contains a keyword match. "
            "0.0 = incorrect, unsupported, or unrelated. "
            "Use intermediate values like 0.7, 0.5, or 0.3 when the answer falls between bands. "
            "Do not require exact wording."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.5,
        model=groq_judge,
    )
    return [correctness_metric]


def build_payload(deepeval_metrics_json: list[dict], run_timestamp: str) -> dict:
    return {
        "dataset_name": f"DeepEval Groq Metrics {run_timestamp}",
        "examples": [
            {
                "inputs": {
                    "question": item["question"],
                    "actual_output": item["answer"],
                    "deepeval_metrics": item["metrics"],
                },
                "outputs": {"expected_output": item.get("expected_output")},
                "metadata": {
                    "source": "deepeval_metrics_json",
                    "model": item.get("model"),
                    "latency": item.get("latency"),
                },
            }
            for item in deepeval_metrics_json
        ],
        "feedback": [
            {
                "question": item["question"],
                "answer": item["answer"],
                "results": [
                    {
                        "key": f"deepeval_{metric_name}",
                        "score": score,
                        "comment": item["metric_reasons"].get(metric_name),
                        "metadata": {
                            "success": item["metric_success"].get(metric_name),
                            "judge_model": item.get("model"),
                            "latency": item.get("latency"),
                        },
                    }
                    for metric_name, score in item["metrics"].items()
                ],
            }
            for item in deepeval_metrics_json
        ],
    }


def run_evaluation() -> list[dict]:
    deepeval_metrics = build_metrics()
    deepeval_metrics_json = []

    for case in test_cases:
        eval_case_kwargs = case_to_eval_kwargs(case)
        row = {
            "question": eval_case_kwargs["input"],
            "answer": eval_case_kwargs["actual_output"],
            "expected_output": eval_case_kwargs.get("expected_output"),
            "latency": None,
            "model": None,
            "metrics": {},
            "metric_reasons": {},
            "metric_success": {},
        }

        started_at = time.perf_counter()
        for metric in deepeval_metrics:
            metric_name = metric_key(metric)
            try:
                metric.measure(LLMTestCase(**eval_case_kwargs))
                row[metric_name] = float(metric.score) if metric.score is not None else None
                row["metrics"][metric_name] = row[metric_name]
                row["metric_reasons"][metric_name] = str(getattr(metric, "reason", ""))
                row["metric_success"][metric_name] = bool(getattr(metric, "success", False))
                row["model"] = row["model"] or judge_model_name(metric)
            except Exception as exc:
                row[metric_name] = None
                row["metrics"][metric_name] = None
                row["metric_reasons"][metric_name] = f"DeepEval failed: {exc}"
                row["metric_success"][metric_name] = False
                row["model"] = row["model"] or judge_model_name(metric)

        row["latency"] = round(time.perf_counter() - started_at, 3)
        deepeval_metrics_json.append(row)

    return deepeval_metrics_json


def main():
    parser = argparse.ArgumentParser(description="Run DeepEval metrics and write JSON payloads.")
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--artifact-path", default="outputs/evaluation_results.json")
    parser.add_argument("--print-json", action="store_true", help="Print DeepEval metrics JSON to stdout.")
    args = parser.parse_args()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    deepeval_metrics_json = run_evaluation()
    langsmith_payload = build_payload(deepeval_metrics_json, run_timestamp)

    metrics_path = output_dir / f"deepeval_metrics_{run_timestamp}.json"
    payload_path = output_dir / f"langsmith_payload_{run_timestamp}.json"
    metrics_path.write_text(json.dumps(deepeval_metrics_json, indent=2), encoding="utf-8")
    payload_path.write_text(json.dumps(langsmith_payload, indent=2), encoding="utf-8")
    artifact_path.write_text(json.dumps(deepeval_metrics_json, indent=2), encoding="utf-8")

    print(f"Saved DeepEval metrics JSON to: {metrics_path.resolve()}")
    print(f"Saved LangSmith payload JSON to: {payload_path.resolve()}")
    print(f"Saved GitHub artifact JSON to: {artifact_path.resolve()}")
    if args.print_json:
        print("DEEPEVAL_METRICS_JSON_START")
        print(json.dumps(deepeval_metrics_json, indent=2))
        print("DEEPEVAL_METRICS_JSON_END")


if __name__ == "__main__":
    main()
