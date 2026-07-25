"""Generates a short natural-language risk recommendation per flagged document
(not just accept/reject) — the human-facing output this project's decision
layer builds toward, combining risk_tiering.py's routing, cost_simulator.py's
cost matrix, and (optionally) retrieval.case_index.py's similar past cases.

Pure string/logic assembly — no model call happens here. The confidence score
it consumes comes from src/eval/finetuned_eval.py's run_single()
(generation_confidence_to_p_genuine()), computed once per document at
inference time; this module's only job is turning an already-scored document
into a routed decision plus a defensible written rationale, so it's fully
real and unit-tested locally like risk_tiering.py/cost_simulator.py.
"""

from src.decision.risk_tiering import route


def explain_decision(document_result: dict, cost_config: dict, similar_cases: list[dict] | None = None) -> str:
    """document_result: {"image_path", "parsed": {...schema, incl. tamper_verdict
    and explanation...}, "confidence": float (P(genuine), in [0, 1])} — the
    shape src/eval/finetuned_eval.run_single() returns.

    similar_cases (optional): output of src/retrieval/case_index.py's query()
    — used here purely as supporting context surfaced to a human reviewer,
    not as a second vote that changes the routing decision itself (the
    routing tier is confidence-driven only, via risk_tiering.route()).
    """
    confidence = document_result["confidence"]
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    parsed = document_result.get("parsed") or {}
    tamper_verdict = parsed.get("tamper_verdict", "unknown")
    model_explanation = parsed.get("explanation") or "no explanation provided"

    tier = route(confidence, cost_config)
    thresholds = cost_config["thresholds"]
    costs = cost_config["costs"]

    if tier == "auto_approve":
        tier_rationale = (
            f"confidence {confidence:.2f} is at or above the auto-approve threshold "
            f"({thresholds['auto_approve_min_confidence']:.2f}) — approving automatically costs nothing if "
            f"correct, but risks a ${costs['false_accept']['value']:.0f} false-accept loss if this document "
            f"is actually tampered."
        )
    elif tier == "auto_reject":
        tier_rationale = (
            f"confidence {confidence:.2f} is at or below the auto-reject threshold "
            f"({thresholds['auto_reject_max_confidence']:.2f}) — rejecting automatically risks a "
            f"${costs['false_reject']['value']:.0f} false-reject cost (churn/support) if this document is "
            f"actually genuine."
        )
    else:
        tier_rationale = (
            f"confidence {confidence:.2f} falls in the uncertain band between "
            f"{thresholds['auto_reject_max_confidence']:.2f} and {thresholds['auto_approve_min_confidence']:.2f} "
            f"— routed to a human reviewer at a flat ${costs['manual_review']['value']:.0f} cost rather than "
            f"risking either automatic-decision error."
        )

    lines = [
        f"Decision: {tier.replace('_', ' ')}.",
        f"Model verdict: {tamper_verdict} (confidence {confidence:.2f} that this document is genuine). "
        f"{model_explanation}",
        f"Why this routing: {tier_rationale}",
    ]

    if similar_cases:
        case_summaries = "; ".join(
            f"{c.get('case_id', c.get('image_path', 'unknown case'))} "
            f"({c.get('tamper_verdict', 'unknown verdict')}, similarity {c.get('similarity', 0.0):.2f})"
            for c in similar_cases[:3]
        )
        lines.append(f"Similar past-flagged cases for context: {case_summaries}.")

    return "\n".join(lines)
