"""Core generalization experiment: leave-one-attack-tier-out, rotated across
however many forgery tiers have a real manifest, with bootstrap confidence
intervals (config/training_config.yaml's leave_one_out section).

Design intent: for each tier T, a model gets trained on genuine + every OTHER
tier's forgeries, then evaluated ONLY on T's held-out forgeries — the question
being asked is "does tamper detection generalize to an attack type the model
never saw," not "what's overall accuracy." That requires actually training N
separate QLoRA models (N = number of tiers), which needs Kaggle GPU time — the
same RAM-constraint wall documented in sft_train.py's module docstring applies
here too. What's pure logic and testable locally: which examples go in which
fold, and how per-fold results get aggregated into per-tier + overall CIs.
train_fn/eval_fn are injected exactly so the orchestration in run() can be
unit-tested against fake callables, without a real model, the same pattern
sft_train.py and clean_eval.py use for their local/Kaggle split.

Tier availability note: config/training_config.yaml's leave_one_out.tiers
lists all 5 tiers as the intended final scope, but only tiers with an actual
manifest on disk (tier1/2/4/5 as of Phase 5 — Tier 3's diffusion inpainting
inference is deferred to Kaggle) can run a fold. run() rotates over whatever
manifests it's actually given, not a hardcoded 5, and logs which tiers (if
any) from the config are missing rather than silently pretending they ran.
"""

import json
import random
from pathlib import Path

from src.eval.metrics import bootstrap_ci
from src.utils.config_utils import load_yaml
from src.utils.logging_utils import get_logger

logger = get_logger("leave_one_out_eval")


def split_few_shot_manifest(manifest_path: str, k: int, seed: int = 42) -> tuple[dict, dict]:
    """Splits one tier's manifest into a tiny (k-example) training slice and
    an evaluation slice holding everything else -- the few-shot exposure
    diagnostic complementary to the zero-shot leave-one-out experiment.

    The question this answers: does the LOO gap (13.3% tier2, 0.000% tier4 —
    see results/tables/phase6_leave_one_out_summary.md) shrink once the model
    sees even a handful of the held-out tier's real examples during training?
    If k=2-3 closes most of the gap, this is primarily a "the model never
    saw this concept at all" data-quantity problem. If it doesn't move much,
    that's evidence for something more structural -- which is exactly the
    case regularity_disruption.py's tier-agnostic augmentation is designed to
    address without needing real examples of every possible tampering
    category. Seeded, deterministic sampling (not the first k in file order,
    which could accidentally be a biased document-code sample.

    Returns (few_shot_manifest, eval_manifest), each shaped identically to
    the source manifest ({"num_attempted", "num_success", "entries"}) so
    both can be written to disk and fed straight into build_sft_examples()'s
    tier_manifest_paths (few_shot_manifest as the held-out tier's manifest —
    now not fully absent, just k-shot) and load_tier_examples (eval_manifest,
    for measuring accuracy on what's left).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"]
    successful = [e for e in entries if e.get("success")]
    if k > len(successful):
        raise ValueError(f"Requested k={k} few-shot examples but only {len(successful)} "
                          f"successful entries exist in {manifest_path}")

    few_shot_entries = random.Random(seed).sample(successful, k)
    few_shot_paths = {e["forged_image"] for e in few_shot_entries}
    eval_entries = [e for e in successful if e["forged_image"] not in few_shot_paths]

    few_shot_manifest = {"num_attempted": k, "num_success": k, "entries": few_shot_entries}
    eval_manifest = {"num_attempted": len(eval_entries), "num_success": len(eval_entries), "entries": eval_entries}
    return few_shot_manifest, eval_manifest


def load_tier_examples(manifest_path: str, tier_name: str) -> list[dict]:
    """Loads successful forged examples from one tier manifest into the
    {"image_path", "tier", "true_label"} shape used throughout eval/, so
    downstream code doesn't need to know each tier manifest's own field names
    (tier1/2/5 use "forged_image"; all four use "success" as the filter).
    """
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    return [
        {"image_path": e["forged_image"], "tier": tier_name, "true_label": "tampered"}
        for e in manifest["entries"] if e.get("success")
    ]


def build_folds(tier_examples_by_tier: dict[str, list[dict]]) -> list[dict]:
    """One fold per tier present in the input: that tier is held out (its
    examples become the eval set), every other present tier is the "seen"
    training composition for that fold's model.

    Needs at least 2 tiers — with only 1, there's nothing to train on that
    excludes it (the whole point of the experiment), so that tier is skipped
    with a warning rather than silently producing a meaningless fold.
    """
    tier_names = list(tier_examples_by_tier.keys())
    folds = []
    for held_out in tier_names:
        train_tiers = [t for t in tier_names if t != held_out]
        if not train_tiers:
            logger.info(f"Skipping fold for '{held_out}': no other tiers available to train on")
            continue
        folds.append({
            "held_out_tier": held_out,
            "held_out_examples": tier_examples_by_tier[held_out],
            "train_tiers": train_tiers,
        })
    return folds


def aggregate_fold_results(per_fold_scores: dict[str, list[float]], confidence: float = 0.95,
                            n_resamples: int = 1000) -> dict:
    """per_fold_scores: {held_out_tier: [0/1 correctness per held-out example]}.

    Reports both per-tier CIs (so a reader can see whether generalization is
    uniform across attack types or the model just happens to do well on one)
    and a pooled "overall" CI across every fold's examples combined — neither
    alone is the whole picture: per-tier alone hides small-n tiers behind a
    wide CI, pooled alone hides a tier that's dramatically worse than the rest.
    """
    per_tier = {
        tier: bootstrap_ci(scores, confidence=confidence, n_resamples=n_resamples)
        for tier, scores in per_fold_scores.items()
    }
    pooled = [score for scores in per_fold_scores.values() for score in scores]
    overall = bootstrap_ci(pooled, confidence=confidence, n_resamples=n_resamples)
    return {"per_tier": per_tier, "overall": overall}


def run(training_config_path: str, tier_manifest_paths: dict[str, str],
        train_fn=None, eval_fn=None, out_path: str | None = None) -> dict:
    """Orchestrates the full leave-one-out experiment.

    train_fn(train_tiers: list[str]) -> model_handle — trains (Kaggle: real
    QLoRA SFT restricted to the given tiers' examples; local smoke test: a
    stub returning anything).
    eval_fn(model_handle, held_out_examples: list[dict]) -> list[float] — runs
    the trained model on each held-out example, returns 1.0/0.0 correctness
    (or a graded score) per example, same convention bootstrap_ci expects.

    If either callable is omitted, stops after logging what WOULD run — same
    "local environment never touches model weights" contract as
    sft_train.train() and clean_eval.run(), just parameterized via injection
    instead of an environment string, since here the thing that needs Kaggle
    is training N separate models, not one.
    """
    training_config = load_yaml(training_config_path)
    loo_cfg = training_config["leave_one_out"]
    configured_tiers = set(loo_cfg["tiers"])
    available_tiers = set(tier_manifest_paths.keys())
    missing = configured_tiers - available_tiers
    if missing:
        logger.info(f"leave_one_out.tiers configures {sorted(configured_tiers)} but manifests exist for "
                     f"{sorted(available_tiers)} — running only the available tiers, not skipping the experiment. "
                     f"Missing: {sorted(missing)}")

    tier_examples_by_tier = {
        tier: load_tier_examples(path, tier) for tier, path in tier_manifest_paths.items()
    }
    folds = build_folds(tier_examples_by_tier)
    logger.info(f"Built {len(folds)} leave-one-out folds: {[f['held_out_tier'] for f in folds]}")

    if train_fn is None or eval_fn is None:
        logger.info("train_fn/eval_fn not provided — stopping after fold construction. "
                     "Real per-fold training needs Kaggle GPU time (N models, N = number of tiers); "
                     "see module docstring.")
        return {"folds": folds, "results": None}

    per_fold_scores = {}
    for fold in folds:
        logger.info(f"Fold: holding out '{fold['held_out_tier']}', training on {fold['train_tiers']}")
        model_handle = train_fn(fold["train_tiers"])
        per_fold_scores[fold["held_out_tier"]] = eval_fn(model_handle, fold["held_out_examples"])

    results = aggregate_fold_results(
        per_fold_scores,
        confidence=loo_cfg["confidence_level"],
        n_resamples=loo_cfg["bootstrap_resamples"],
    )

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
        logger.info(f"Wrote leave-one-out results to {out_path}")

    return {"folds": folds, "results": results}
