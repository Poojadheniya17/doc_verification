"""3 rounds of targeted retraining driven by mined failure cases; tracks the
accuracy curve across rounds (config/training_config.yaml's
adversarial_rounds section: num_rounds=3, failure_sample_cap=200).

Flow per round: evaluate the current model on a fixed eval set -> mine the
incorrect examples (capped at failure_sample_cap) -> retrain targeted on
those failures -> re-evaluate -> record this round's accuracy. Same
train_fn/eval_fn injection pattern as leave_one_out_eval.py: actually training
each round's model needs Kaggle GPU time, but failure-mining and the
round-by-round accuracy-curve bookkeeping are pure logic, testable locally
against fake callables.
"""

import json
from pathlib import Path

from src.eval.metrics import bootstrap_ci
from src.utils.config_utils import load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger("adversarial_rounds")


def mine_failures(eval_results: list[dict], cap: int) -> list[dict]:
    """Selects incorrect examples for the next round's targeted retraining.

    eval_results: list of {"image_path", "correct": bool, ...} (any extra keys
    like "tier"/"true_label" pass through untouched — useful for later
    breaking failure counts down by tier). Caps at `cap` rather than
    retraining on every failure at once: training_config.yaml's rationale is
    VRAM/time bound, same constraint as everything else that runs on Kaggle's
    free tier.

    Order is preserved (first `cap` failures in eval_results' own order, not
    resampled) — deterministic given a deterministic eval_fn, which matters
    for reproducing a specific round's retraining set later.
    """
    failures = [r for r in eval_results if not r.get("correct", False)]
    return failures[:cap]


def build_accuracy_curve(round_scores: list[list[float]], confidence: float = 0.95,
                          n_resamples: int = 1000) -> list[dict]:
    """round_scores[i] = per-example correctness (0/1) for round i's eval pass.

    Returns one bootstrap_ci dict per round, each tagged with its round
    number, in round order — the thing results/charts/ actually plots as the
    accuracy-over-rounds curve.
    """
    return [
        {"round": i, **bootstrap_ci(scores, confidence=confidence, n_resamples=n_resamples)}
        for i, scores in enumerate(round_scores)
    ]


def run(training_config_path: str, eval_set: list[dict], train_fn=None, eval_fn=None,
        out_path: str | None = None) -> dict:
    """Orchestrates num_rounds of eval -> mine-failures -> retrain.

    eval_set: the fixed held-out examples every round is scored against (kept
    constant across rounds so the accuracy curve reflects the model
    improving, not the eval set changing).

    train_fn(failure_examples: list[dict], round_num: int) -> model_handle —
    round 0 has no failures yet (round_num == 0 gets an empty list; a real
    implementation trains round 0's baseline from the Phase 4 SFT checkpoint
    with no additional targeted data). eval_fn(model_handle, eval_set) ->
    list[dict] of {"image_path", "correct": bool, ...} per example.

    Same "stop after logging what would run" contract as sft_train.py /
    clean_eval.py / leave_one_out_eval.py when the callables needed to
    actually touch a model aren't provided.
    """
    training_config = load_yaml(training_config_path)
    adv_cfg = training_config["adversarial_rounds"]
    num_rounds = adv_cfg["num_rounds"]
    cap = adv_cfg["failure_sample_cap"]

    if train_fn is None or eval_fn is None:
        logger.info(f"train_fn/eval_fn not provided — stopping before round 0. Would run {num_rounds} rounds "
                     f"against {len(eval_set)} eval examples, capping mined failures at {cap}/round.")
        return {"num_rounds": num_rounds, "rounds": None}

    round_scores = []
    round_records = []
    failures = []
    for round_num in range(num_rounds):
        logger.info(f"Round {round_num}: training on {len(failures)} mined failures from the previous round")
        model_handle = train_fn(failures, round_num)
        eval_results = eval_fn(model_handle, eval_set)
        scores = [float(r.get("correct", False)) for r in eval_results]
        round_scores.append(scores)

        failures = mine_failures(eval_results, cap)
        round_records.append({
            "round": round_num,
            "num_failures_mined": len(failures),
            "accuracy": bootstrap_ci(scores, confidence=adv_cfg.get("confidence_level", 0.95)),
        })
        logger.info(f"Round {round_num}: accuracy={round_records[-1]['accuracy']['mean']:.3f}, "
                     f"mined {len(failures)} failures for round {round_num + 1}")

    curve = build_accuracy_curve(round_scores, confidence=adv_cfg.get("confidence_level", 0.95))
    results = {"num_rounds": num_rounds, "rounds": round_records, "accuracy_curve": curve}

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Wrote adversarial-rounds results to {out_path}")

    return results
