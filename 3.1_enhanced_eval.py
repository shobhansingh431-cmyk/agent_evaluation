import argparse
import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ExactMatchMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase, SingleTurnParams


# Keep this step local. Upload to LangSmith only from the push script.
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


def estimate_cost(usage: dict) -> float | None:
    input_cost = os.getenv("GROQ_INPUT_COST_PER_1M")
    output_cost = os.getenv("GROQ_OUTPUT_COST_PER_1M")
    if not input_cost or not output_cost:
        return None

    return round(
        (usage.get("input_tokens", 0) / 1_000_000 * float(input_cost))
        + (usage.get("output_tokens", 0) / 1_000_000 * float(output_cost)),
        8,
    )


def case_to_eval_kwargs(case: LLMTestCase) -> dict:
    expected_output = clean_text(case.expected_output)
    kwargs = {
        "input": clean_text(case.input),
        "actual_output": clean_text(case.actual_output),
        "expected_output": expected_output,
        # Context-dependent metrics need source context. For these examples,
        # the expected answer is the reference source of truth.
        "context": [expected_output] if expected_output else None,
        "retrieval_context": [expected_output] if expected_output else None,
    }
    return {key: value for key, value in kwargs.items() if value is not None}


def build_metrics(metric_suite: str):
    correctness = GEval(
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
        async_mode=False,
    )
    completeness = GEval(
        name="Completeness",
        criteria=(
            "Grade from 0 to 1 how completely the actual output covers the important semantic ideas "
            "in the expected output. Do not penalize for missing exact words, synonyms, article usage, "
            "or phrasing differences when the same meaning is present. Focus only on materially missing "
            "concepts that change the answer's usefulness or accuracy. "
            "1.0 = covers all important ideas. "
            "0.8 = covers almost all important ideas with only minor omissions. "
            "0.6 = covers the main idea but misses some useful supporting details. "
            "0.4 = incomplete and misses important concepts. "
            "0.2 = barely addresses the expected answer. "
            "0.0 = unrelated or empty."
        ),
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        threshold=0.7,
        model=groq_judge,
        async_mode=False,
    )
    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model=groq_judge, async_mode=False)
    faithfulness = FaithfulnessMetric(threshold=0.5, model=groq_judge, async_mode=False)
    hallucination = HallucinationMetric(threshold=0.5, model=groq_judge, async_mode=False)
    bias = BiasMetric(threshold=0.5, model=groq_judge, async_mode=False)
    toxicity = ToxicityMetric(threshold=0.5, model=groq_judge, async_mode=False)
    exact_match = ExactMatchMetric(threshold=1.0)

    suites = {
        "core": [correctness, completeness, answer_relevancy, exact_match],
        "safety": [hallucination, bias, toxicity],
        "all": [
            correctness,
            completeness,
            answer_relevancy,
            faithfulness,
            hallucination,
            bias,
            toxicity,
            exact_match,
        ],
    }
    return suites[metric_suite]


def metric_direction(metric_name: str) -> str:
    lower_is_better = {"hallucination", "bias", "toxicity"}
    return "lower_is_better" if metric_name in lower_is_better else "higher_is_better"


def build_payload(metrics_json: list[dict], run_timestamp: str) -> dict:
    return {
        "dataset_name": f"DeepEval Enhanced Metrics {run_timestamp}",
        "examples": [
            {
                "inputs": {
                    "question": item["question"],
                    "actual_output": item["answer"],
                    "deepeval_metrics": item["metrics"],
                    "deepeval_metric_reasons": item["metric_reasons"],
                    "deepeval_metric_success": item["metric_success"],
                    "deepeval_metric_direction": item["metric_direction"],
                    "deepeval_metric_latency": item["metric_latency"],
                    "deepeval_metric_usage": item["metric_usage"],
                    "deepeval_metric_estimated_cost": item["metric_estimated_cost"],
                },
                "outputs": {"expected_output": item.get("expected_output")},
                "metadata": {
                    "source": "enhanced_deepeval_metrics_json",
                    "model": item.get("model"),
                    "latency": item.get("latency"),
                },
            }
            for item in metrics_json
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
                            "direction": metric_direction(metric_name),
                            "metric_latency": item["metric_latency"].get(metric_name),
                            "usage": item["metric_usage"].get(metric_name),
                            "estimated_cost": item["metric_estimated_cost"].get(metric_name),
                        },
                    }
                    for metric_name, score in item["metrics"].items()
                ],
            }
            for item in metrics_json
        ],
    }


def run_evaluation(metric_suite: str) -> list[dict]:
    metrics = build_metrics(metric_suite)
    enhanced_metrics_json = []

    print(
        f"Starting enhanced DeepEval run: {len(test_cases)} test cases, "
        f"{len(metrics)} metrics, suite={metric_suite}",
        flush=True,
    )

    for case_index, case in enumerate(test_cases, start=1):
        eval_case_kwargs = case_to_eval_kwargs(case)
        print(
            f"[case {case_index}/{len(test_cases)}] {eval_case_kwargs['input']}",
            flush=True,
        )
        row = {
            "question": eval_case_kwargs["input"],
            "answer": eval_case_kwargs["actual_output"],
            "expected_output": eval_case_kwargs.get("expected_output"),
            "context": eval_case_kwargs.get("context"),
            "latency": None,
            "model": None,
            "metrics": {},
            "metric_reasons": {},
            "metric_success": {},
            "metric_direction": {},
            "metric_latency": {},
            "metric_usage": {},
            "metric_estimated_cost": {},
        }

        started_at = time.perf_counter()
        for metric_index, metric in enumerate(metrics, start=1):
            name = metric_key(metric)
            metric_started_at = time.perf_counter()
            print(
                f"  [metric {metric_index}/{len(metrics)}] starting {name}",
                flush=True,
            )
            try:
                if hasattr(groq_judge, "reset_usage"):
                    groq_judge.reset_usage()
                metric.measure(LLMTestCase(**eval_case_kwargs))
                score = float(metric.score) if metric.score is not None else None
                metric_latency = round(time.perf_counter() - metric_started_at, 3)
                usage = groq_judge.usage_snapshot() if hasattr(groq_judge, "usage_snapshot") else {}
                row[name] = score
                row["metrics"][name] = score
                row["metric_reasons"][name] = str(getattr(metric, "reason", ""))
                row["metric_success"][name] = bool(getattr(metric, "success", False))
                row["metric_direction"][name] = metric_direction(name)
                row["metric_latency"][name] = metric_latency
                row["metric_usage"][name] = usage
                row["metric_estimated_cost"][name] = estimate_cost(usage)
                row["model"] = row["model"] or judge_model_name(metric)
                print(
                    f"  [metric {metric_index}/{len(metrics)}] finished {name}: "
                    f"score={score} success={row['metric_success'][name]} "
                    f"duration={metric_latency:.2f}s "
                    f"tokens={usage.get('total_tokens', 0)}",
                    flush=True,
                )
            except Exception as exc:
                metric_latency = round(time.perf_counter() - metric_started_at, 3)
                usage = groq_judge.usage_snapshot() if hasattr(groq_judge, "usage_snapshot") else {}
                row[name] = None
                row["metrics"][name] = None
                row["metric_reasons"][name] = f"DeepEval failed: {exc}"
                row["metric_success"][name] = False
                row["metric_direction"][name] = metric_direction(name)
                row["metric_latency"][name] = metric_latency
                row["metric_usage"][name] = usage
                row["metric_estimated_cost"][name] = estimate_cost(usage)
                row["model"] = row["model"] or judge_model_name(metric)
                print(
                    f"  [metric {metric_index}/{len(metrics)}] failed {name}: {exc} "
                    f"duration={metric_latency:.2f}s "
                    f"tokens={usage.get('total_tokens', 0)}",
                    flush=True,
                )

        row["latency"] = round(time.perf_counter() - started_at, 3)
        enhanced_metrics_json.append(row)
        print(
            f"[case {case_index}/{len(test_cases)}] finished in {row['latency']}s",
            flush=True,
        )

    return enhanced_metrics_json


def main():
    parser = argparse.ArgumentParser(description="Run enhanced DeepEval metrics and write JSON payloads.")
    parser.add_argument("--output-dir", default="eval_outputs")
    parser.add_argument("--artifact-path", default="outputs/enhanced_evaluation_results.json")
    parser.add_argument("--print-json", action="store_true", help="Print enhanced DeepEval metrics JSON to stdout.")
    parser.add_argument(
        "--metric-suite",
        choices=["core", "safety", "all"],
        default="all",
        help="Metric group to run. Use core for faster CI, all for the full enhanced run.",
    )
    args = parser.parse_args()

    run_timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S_UTC")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(args.artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    enhanced_metrics_json = run_evaluation(args.metric_suite)
    langsmith_payload = build_payload(enhanced_metrics_json, run_timestamp)

    metrics_path = output_dir / f"enhanced_deepeval_metrics_{run_timestamp}.json"
    payload_path = output_dir / f"enhanced_langsmith_payload_{run_timestamp}.json"
    generic_metrics_path = output_dir / f"deepeval_metrics_{run_timestamp}.json"
    generic_payload_path = output_dir / f"langsmith_payload_{run_timestamp}.json"
    metrics_path.write_text(json.dumps(enhanced_metrics_json, indent=2), encoding="utf-8")
    payload_path.write_text(json.dumps(langsmith_payload, indent=2), encoding="utf-8")
    generic_metrics_path.write_text(json.dumps(enhanced_metrics_json, indent=2), encoding="utf-8")
    generic_payload_path.write_text(json.dumps(langsmith_payload, indent=2), encoding="utf-8")
    artifact_path.write_text(json.dumps(enhanced_metrics_json, indent=2), encoding="utf-8")

    print(f"Saved enhanced DeepEval metrics JSON to: {metrics_path.resolve()}")
    print(f"Saved enhanced LangSmith payload JSON to: {payload_path.resolve()}")
    print(f"Saved generic DeepEval metrics JSON to: {generic_metrics_path.resolve()}")
    print(f"Saved generic LangSmith payload JSON to: {generic_payload_path.resolve()}")
    print(f"Saved enhanced artifact JSON to: {artifact_path.resolve()}")
    if args.print_json:
        print("ENHANCED_DEEPEVAL_METRICS_JSON_START")
        print(json.dumps(enhanced_metrics_json, indent=2))
        print("ENHANCED_DEEPEVAL_METRICS_JSON_END")


if __name__ == "__main__":
    main()
