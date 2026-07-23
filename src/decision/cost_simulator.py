"""Cost-tradeoff simulation: sweeps decision thresholds against an eval set
and config/cost_matrix_config.yaml's cost matrix, to find the empirically
cost-minimizing (auto_approve_min_confidence, auto_reject_max_confidence)
pair rather than trusting the config's starting-guess thresholds.

Pure decision-layer logic — no model involved, so (like risk_tiering.py) this
is fully real in local unit tests, tested against mocked confidence scores
rather than deferred to Kaggle.
"""

from src.decision.risk_tiering import route
from src.utils.config_utils import load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger("cost_simulator")


def compute_cost(eval_results: list[dict], thresholds: dict, cost_config: dict) -> dict:
    """eval_results: list of {"confidence": float in [0,1], "true_label": "genuine"|"tampered"}.

    A false accept is a tampered doc routed to auto_approve; a false reject is
    a genuine doc routed to auto_reject. Every human_review costs the flat
    manual-review fee regardless of true label (a human still has to look at
    it). Correct auto-approve/auto-reject decisions cost nothing — the cost
    matrix models the price of getting it wrong or punting, not of routing
    itself.
    """
    costs = cost_config["costs"]
    counts = {"auto_approve": 0, "auto_reject": 0, "human_review": 0, "false_accept": 0, "false_reject": 0}
    total_cost = 0.0

    cost_config_for_route = {"thresholds": thresholds}
    for r in eval_results:
        decision = route(r["confidence"], cost_config_for_route)
        counts[decision] += 1
        if decision == "auto_approve" and r["true_label"] == "tampered":
            total_cost += costs["false_accept"]["value"]
            counts["false_accept"] += 1
        elif decision == "auto_reject" and r["true_label"] == "genuine":
            total_cost += costs["false_reject"]["value"]
            counts["false_reject"] += 1
        elif decision == "human_review":
            total_cost += costs["manual_review"]["value"]

    n = len(eval_results)
    return {
        "thresholds": thresholds,
        "total_cost": total_cost,
        "avg_cost_per_doc": total_cost / n if n else float("nan"),
        "n_examples": n,
        "counts": counts,
    }


def sweep_thresholds(eval_results: list[dict], cost_config: dict,
                      approve_candidates: list[float] | None = None,
                      reject_candidates: list[float] | None = None) -> dict:
    """Tries every (approve, reject) pair from the candidate grids (defaulting
    to cost_config["threshold_sweep"]'s grid), skipping invalid pairs where
    reject >= approve (auto-reject threshold must sit below auto-approve —
    otherwise the human-review band inverts or vanishes). Returns the full
    curve plus the argmin by avg_cost_per_doc.
    """
    sweep_cfg = cost_config.get("threshold_sweep", {})
    approve_candidates = approve_candidates or sweep_cfg.get("auto_approve_candidates", [0.95])
    reject_candidates = reject_candidates or sweep_cfg.get("auto_reject_candidates", [0.05])

    curve = []
    for approve in approve_candidates:
        for reject in reject_candidates:
            if reject >= approve:
                continue
            thresholds = {"auto_approve_min_confidence": approve, "auto_reject_max_confidence": reject}
            curve.append(compute_cost(eval_results, thresholds, cost_config))

    if not curve:
        raise ValueError("No valid (approve, reject) threshold pairs — check the candidate grids "
                          "(every reject candidate was >= every approve candidate)")

    best = min(curve, key=lambda r: r["avg_cost_per_doc"])
    logger.info(f"Swept {len(curve)} threshold pairs over {len(eval_results)} examples; "
                f"best avg_cost_per_doc={best['avg_cost_per_doc']:.2f} at {best['thresholds']}")
    return {"curve": curve, "best": best, "n_examples": len(eval_results)}


def sweep_thresholds_from_paths(eval_results_path: str, cost_config_path: str) -> dict:
    import json
    from pathlib import Path

    eval_results = json.loads(Path(eval_results_path).read_text(encoding="utf-8"))
    cost_config = load_yaml(cost_config_path)
    return sweep_thresholds(eval_results, cost_config)
