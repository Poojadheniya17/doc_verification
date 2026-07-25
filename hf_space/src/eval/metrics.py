"""Shared metric + confidence-interval utilities used by every eval script.

Every eval script in this project reports (mean, ci_low, ci_high, n) — never a
bare point accuracy — per the project's quality bar: a single number from a
handful of samples is not defensible under technical questioning.
"""

import difflib
from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], confidence: float = 0.95, n_resamples: int = 1000,
                  seed: int = 42) -> dict:
    """Percentile bootstrap CI over a list of per-sample scores (e.g. 0/1 correctness,
    or a continuous similarity score). Bootstrap rather than a normal-approximation
    CI because per-sample scores here are often 0/1 (binomial-ish, skewed at small n)
    where the normal approximation breaks down.

    Returns {"mean": float, "ci_low": float, "ci_high": float, "n": int} — always
    includes n so a reader can judge whether the CI itself is trustworthy at that
    sample size (a tight CI from n=5 is not something to trust either).
    """
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    if n == 1:
        # A CI is not meaningful from a single sample — report the point value
        # and n=1 explicitly rather than fabricating a false interval.
        return {"mean": float(values[0]), "ci_low": float(values[0]), "ci_high": float(values[0]), "n": 1}

    rng = np.random.default_rng(seed)
    resample_means = np.empty(n_resamples)
    for i in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        resample_means[i] = sample.mean()

    alpha = 1 - confidence
    ci_low, ci_high = np.percentile(resample_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "ci_low": float(ci_low), "ci_high": float(ci_high), "n": n}


def field_similarity(predicted: str | None, ground_truth: str | None) -> float:
    """Normalized string similarity in [0, 1] for a single extracted field.

    Uses difflib's SequenceMatcher ratio rather than exact match — extraction
    fields (dates, addresses) have legitimate formatting variance (e.g.
    "04.11.2026" vs "2026-11-04") that exact-match would unfairly zero out, and
    rather than exact-vs-wrong we want a graded signal for "close but
    reformatted" vs "actually wrong".
    """
    if predicted is None and ground_truth is None:
        return 1.0
    if predicted is None or ground_truth is None:
        return 0.0
    predicted, ground_truth = predicted.strip().lower(), ground_truth.strip().lower()
    if not predicted and not ground_truth:
        return 1.0
    return difflib.SequenceMatcher(None, predicted, ground_truth).ratio()


def field_exact_match(predicted: str | None, ground_truth: str | None) -> float:
    """Stricter companion to field_similarity — reported alongside it, not instead
    of it, since exact-match and fuzzy-similarity can diverge a lot for something
    like ID numbers where "close" is not actually acceptable (a single transposed
    digit is a different, valid-looking ID number, not a minor formatting slip).
    """
    if predicted is None and ground_truth is None:
        return 1.0
    if predicted is None or ground_truth is None:
        return 0.0
    return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
