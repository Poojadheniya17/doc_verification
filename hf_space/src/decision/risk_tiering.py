"""Confidence score -> auto-approve / auto-reject / human-review routing,
driven by config/cost_matrix_config.yaml's thresholds.

Pure decision logic — no model involved, so this is fully real (not mocked)
even in local unit tests. `confidence` is the model's confidence that a
document is GENUINE (i.e. P(genuine), in [0, 1]): high confidence routes to
auto-approve, low confidence routes to auto-reject, the uncertain middle
routes to human review.
"""

from src.utils.config_utils import load_yaml

DECISIONS = ("auto_approve", "auto_reject", "human_review")


def route(confidence: float, cost_config: dict) -> str:
    """Takes an already-loaded cost_config dict, not a path — this gets called
    once per document in a batch (cost_simulator.py sweeps it across an entire
    eval set per threshold pair), and re-parsing the same YAML file on every
    call would be wasteful. Use route_from_path() for one-off/CLI use.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    thresholds = cost_config["thresholds"]
    if confidence >= thresholds["auto_approve_min_confidence"]:
        return "auto_approve"
    if confidence <= thresholds["auto_reject_max_confidence"]:
        return "auto_reject"
    return "human_review"


def route_from_path(confidence: float, cost_config_path: str) -> str:
    return route(confidence, load_yaml(cost_config_path))
