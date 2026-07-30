"""Post-hoc calibration analysis for the VLM's confidence score.

Real motivation: finetuned_eval.generation_confidence_to_p_genuine()'s own
docstring calls it "a documented simplification, not a claim of true
field-level calibration" -- a real, admitted, never-actually-checked gap.
risk_tiering.py's entire cost-aware routing (auto_approve_min_confidence=0.95,
auto_reject_max_confidence=0.05) assumes `confidence` genuinely tracks
P(correct). If the model is systematically over- or under-confident, that
assumption is wrong, and cost_simulator.py's threshold sweep was optimized
against a distorted signal without anyone checking. This module checks it
directly instead of leaving it assumed.

A real subtlety, worth getting right rather than assumed: `confidence` means
P(genuine) specifically, not P(the model's actual prediction is correct). If
predicted_verdict == "tampered", the model's real confidence IN THAT
PREDICTION is (1 - confidence), not confidence itself. confidence_in_prediction()
does this conversion explicitly -- every function below expects its output,
not raw P(genuine) values, or the calibration curve checks the wrong thing.

Pure logic, no model involved -- fully real and unit-tested locally, same
split as risk_tiering.py/cost_simulator.py. Needs real (confidence, correct)
pairs at meaningful scale to produce a real finding: the existing captured
eval outputs (LOO/adversarial-rounds raw JSON, see results/tables/) predate
score_prediction() passing `confidence` through and don't have it saved --
only the 7-example demo-gallery capture does, far too small for a real
estimate. This module is fully built and tested so it's ready to run for
real the moment a large-enough eval run (v28, few-shot-exposure, or any
future one) produces real confidence-carrying output.
"""

import numpy as np

_EPS = 1e-6


def confidence_in_prediction(p_genuine: float, predicted_verdict: str) -> float:
    """Converts P(genuine) into "confidence in whatever the model actually
    predicted" -- P(genuine) itself if it predicted genuine, (1 - P(genuine))
    if it predicted tampered. This is the value calibration must be checked
    against, not raw P(genuine).
    """
    if not 0.0 <= p_genuine <= 1.0:
        raise ValueError(f"p_genuine must be in [0, 1], got {p_genuine}")
    return p_genuine if predicted_verdict == "genuine" else 1.0 - p_genuine


def expected_calibration_error(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> dict:
    """Standard ECE: bins predictions into n_bins equal-width confidence bins,
    and for each bin sums |avg_confidence - avg_accuracy| weighted by the
    bin's share of all examples. 0.0 = perfectly calibrated; higher = worse.

    confidences must already be confidence_in_prediction() values, not raw
    P(genuine) -- see module docstring.
    """
    if len(confidences) != len(corrects):
        raise ValueError(f"confidences and corrects must be the same length, got "
                          f"{len(confidences)} and {len(corrects)}")
    if not confidences:
        return {"ece": float("nan"), "n": 0, "bins": []}

    conf = np.asarray(confidences, dtype=float)
    corr = np.asarray(corrects, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    bins = []
    n_total = len(conf)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is closed on both ends so confidence==1.0 lands somewhere.
        in_bin = (conf >= lo) & (conf < hi) if i < n_bins - 1 else (conf >= lo) & (conf <= hi)
        count = int(in_bin.sum())
        if count == 0:
            bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": 0,
                         "avg_confidence": None, "avg_accuracy": None})
            continue
        avg_confidence = float(conf[in_bin].mean())
        avg_accuracy = float(corr[in_bin].mean())
        ece += (count / n_total) * abs(avg_confidence - avg_accuracy)
        bins.append({"bin_lo": float(lo), "bin_hi": float(hi), "count": count,
                      "avg_confidence": avg_confidence, "avg_accuracy": avg_accuracy})

    return {"ece": ece, "n": n_total, "bins": bins}


def _logit(p: np.ndarray) -> np.ndarray:
    p_clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p_clipped / (1.0 - p_clipped))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_platt_scaling(confidences: list[float], corrects: list[bool]) -> dict:
    """Fits p_calibrated = sigmoid(a * logit(p) + b) via 1D logistic
    regression on the logit-transformed confidence.

    Platt scaling, not classic single-parameter temperature scaling: temperature
    scaling technically operates on the model's pre-softmax logits, which
    aren't saved anywhere in this project's eval output (only the final
    scalar confidence_in_prediction() value is) -- Platt's 2-parameter
    (scale + shift) logistic fit is the standard, correct adaptation when only
    a final scalar score is available to calibrate post hoc.
    """
    from sklearn.linear_model import LogisticRegression

    if len(confidences) != len(corrects):
        raise ValueError(f"confidences and corrects must be the same length, got "
                          f"{len(confidences)} and {len(corrects)}")
    if len(set(corrects)) < 2:
        raise ValueError("Platt scaling needs both correct and incorrect examples to fit against "
                          f"-- got only {set(corrects)}")

    x = _logit(np.asarray(confidences, dtype=float)).reshape(-1, 1)
    y = np.asarray(corrects, dtype=int)
    clf = LogisticRegression()
    clf.fit(x, y)
    a, b = float(clf.coef_[0][0]), float(clf.intercept_[0])
    return {"a": a, "b": b}


def apply_platt_scaling(confidences: list[float], a: float, b: float) -> list[float]:
    """Applies a fitted Platt-scaling (a, b) pair to raw confidence_in_prediction()
    values, returning calibrated probabilities.
    """
    x = _logit(np.asarray(confidences, dtype=float))
    return _sigmoid(a * x + b).tolist()
