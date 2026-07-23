"""fp16 vs INT8 vs INT4 benchmark: accuracy, latency, and estimated
cost-per-verification at hypothetical production volume.

Same train/eval injection pattern as leave_one_out_eval.py/adversarial_rounds.py:
load_fn/eval_fn are the only parts that touch a real model (Kaggle only, needs
the trained checkpoint reloaded at each precision); the precision-comparison
orchestration and cost-per-verification math are pure logic, unit-tested
locally against fake callables.
"""

import json
from pathlib import Path

from src.eval.metrics import bootstrap_ci
from src.utils.config_utils import load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger("quantization_bench")

# REASONED ASSUMPTION for a portfolio project, not a real cloud GPU pricing
# figure (same labeling convention as cost_matrix_config.yaml): an on-demand
# T4-class GPU-hour, used only to translate measured per-example latency into
# an illustrative "cost per 1M verifications" figure alongside the
# accuracy/latency tradeoff. What actually matters for the argument this
# benchmark makes is the RELATIVE cost across precisions (e.g. int4 costing
# some fraction of fp16's cost at the same volume), not the absolute dollar
# figure, which depends heavily on provider/region/reserved-vs-on-demand
# pricing this project has no access to real data for.
ASSUMED_GPU_COST_PER_HOUR_USD = 0.35


def cost_per_million_verifications(avg_latency_seconds: float,
                                    gpu_cost_per_hour: float = ASSUMED_GPU_COST_PER_HOUR_USD) -> float:
    gpu_cost_per_second = gpu_cost_per_hour / 3600.0
    return avg_latency_seconds * gpu_cost_per_second * 1_000_000


def run(model_config_path: str, training_config_path: str, eval_examples: list[dict],
        load_fn=None, eval_fn=None, out_path: str | None = None) -> dict:
    """load_fn(precision: str, model_config: dict) -> model_handle — loads the
    trained checkpoint at the given precision (Kaggle: real reload per
    precision; local smoke test: a stub returning anything).
    eval_fn(model_handle, eval_examples) -> list[dict] of {"correct": bool,
    "latency_seconds": float} per example.

    If either callable is omitted, stops after logging what WOULD run — same
    "local environment never touches model weights" contract as every other
    Kaggle-only eval script in this project.
    """
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    precisions = training_config["quantization_bench"]["precisions"]

    if load_fn is None or eval_fn is None:
        logger.info(f"load_fn/eval_fn not provided — stopping before benchmarking. "
                     f"Would compare {precisions} over {len(eval_examples)} examples.")
        return {"precisions": precisions, "results": None}

    results = {}
    for precision in precisions:
        logger.info(f"Benchmarking precision={precision} over {len(eval_examples)} examples")
        model_handle = load_fn(precision, model_config)
        eval_results = eval_fn(model_handle, eval_examples)
        correctness = [float(r["correct"]) for r in eval_results]
        latencies = [r["latency_seconds"] for r in eval_results]
        avg_latency = sum(latencies) / len(latencies) if latencies else float("nan")
        results[precision] = {
            "accuracy": bootstrap_ci(correctness),
            "avg_latency_seconds": avg_latency,
            "estimated_cost_per_million_verifications_usd": cost_per_million_verifications(avg_latency),
        }

    output = {"precisions": precisions, "n_examples": len(eval_examples), "results": results}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(output, indent=2), encoding="utf-8")
        logger.info(f"Wrote quantization benchmark results to {out_path}")
    return output
