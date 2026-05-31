import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langsmith import Client


def latest_file(pattern: str) -> Path:
    files = sorted(Path("eval_outputs").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for eval_outputs/{pattern}")
    return files[-1]


def load_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Push DeepEval JSON payload to LangSmith.")
    parser.add_argument("--payload-path", default=None)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate JSON files without uploading to LangSmith.")
    args = parser.parse_args()

    load_dotenv()
    os.environ["LANGSMITH_TRACING"] = "true"

    payload_path = Path(args.payload_path) if args.payload_path else latest_file("langsmith_payload_*.json")
    metrics_path = Path(args.metrics_path) if args.metrics_path else latest_file("deepeval_metrics_*.json")

    langsmith_payload = load_json(payload_path)
    deepeval_metrics_json = load_json(metrics_path)

    if args.dry_run:
        print(f"Validated payload file: {payload_path.resolve()}")
        print(f"Validated metrics file: {metrics_path.resolve()}")
        print(f"Examples: {len(langsmith_payload['examples'])}")
        print(f"Metric rows: {len(deepeval_metrics_json)}")
        return

    client = Client()
    dataset = client.create_dataset(
        dataset_name=langsmith_payload["dataset_name"],
        description="DeepEval JSON payload from GitHub Actions",
        metadata={
            "source": "github_actions",
            "payload_path": str(payload_path),
            "metrics_path": str(metrics_path),
        },
    )
    client.create_examples(dataset_id=dataset.id, examples=langsmith_payload["examples"])

    def target(inputs: dict) -> dict:
        return {
            "answer": inputs["actual_output"],
            "deepeval_metrics": inputs.get("deepeval_metrics", {}),
        }

    def feedback_from_deepeval_json(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        metrics = inputs.get("deepeval_metrics", {})
        matching_item = next(
            item
            for item in deepeval_metrics_json
            if item["question"] == inputs["question"] and item["answer"] == outputs["answer"]
        )
        return {
            "results": [
                {
                    "key": f"deepeval_{metric_name}",
                    "score": score,
                    "comment": matching_item["metric_reasons"].get(metric_name),
                    "metadata": {
                        "success": matching_item["metric_success"].get(metric_name),
                        "judge_model": matching_item.get("model"),
                        "latency": matching_item.get("latency"),
                        "source": "deepeval_metrics_json",
                    },
                }
                for metric_name, score in metrics.items()
            ]
        }

    experiment_results = client.evaluate(
        target,
        data=langsmith_payload["dataset_name"],
        evaluators=[feedback_from_deepeval_json],
        experiment_prefix="deepeval-json-groq",
        metadata={"source": "github_actions", "source_payload": str(payload_path)},
        max_concurrency=1,
    )
    print(experiment_results.to_pandas())


if __name__ == "__main__":
    main()
