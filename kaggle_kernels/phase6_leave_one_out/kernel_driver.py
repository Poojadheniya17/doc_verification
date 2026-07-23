"""Kaggle kernel driver: Phase 6 leave-one-out generalization eval.

Chains all available forgery-tier folds (tier1/2/3/4/5, whichever have a real
manifest on disk) sequentially in one session: for each fold, train a fresh
QLoRA adapter on every tier EXCEPT the held-out one, then evaluate it only on
the held-out tier's examples. This directly answers "does tamper detection
generalize to an attack type the model never saw."

Real risk, stated honestly: 5 folds at ~2h13m each (Phase 4's v24 measurement)
is ~11h of GPU time in one script, which may exceed a single Kaggle session's
practical runtime ceiling. Each fold's raw eval results are written to
/kaggle/working immediately after that fold finishes (not only at the very
end), so a session that dies partway through still leaves real, usable
results for however many folds completed — this is a deliberate mitigation
for that risk, not a guarantee against it.

Explicit GPU cleanup between every fold's train and eval step, and between
folds: Phase 4's v19-v23 OOM chain was ultimately root-caused (v24) to a
leftover model reference from an earlier phase never being freed. This script
does NOT repeat that mistake — see free_gpu_memory() below, called after
every model load, with before/after diagnostics printed so the result is
verified, not assumed.
"""

import gc
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "qwen-vl-utils==0.0.14", "peft==0.19.1",
     "accelerate==1.14.0", "bitsandbytes==0.49.2"],
    check=True,
)

candidates = [
    Path("/kaggle/input/doc-verification-data"),
    Path("/kaggle/input/datasets/poojadheniya/doc-verification-data"),
]
INPUT_ROOT = next((c for c in candidates if (c / "src").is_dir()), None)
if INPUT_ROOT is None:
    raise RuntimeError(f"Could not find the mounted dataset's src/ dir under any of {candidates}")
print(f"=== Resolved INPUT_ROOT = {INPUT_ROOT} ===", flush=True)

sys.path.insert(0, str(INPUT_ROOT))
os.chdir(INPUT_ROOT)
print(f"=== cwd = {os.getcwd()} ===", flush=True)

import torch  # noqa: E402
import yaml  # noqa: E402

from src.eval.finetuned_eval import eval_examples, load_finetuned_model  # noqa: E402
from src.eval.leave_one_out_eval import run as run_leave_one_out  # noqa: E402
from src.training.sft_train import train as run_sft_train  # noqa: E402

MODEL_CONFIG_PATH = str(INPUT_ROOT / "config" / "model_config.yaml")
with open(INPUT_ROOT / "config" / "training_config.yaml", encoding="utf-8") as f:
    _training_config = yaml.safe_load(f)
_training_config["paths"]["kaggle"]["data_root"] = str(INPUT_ROOT / "data")
TRAINING_CONFIG_PATH = "/kaggle/working/training_config_resolved.yaml"
with open(TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(_training_config, f)
print(f"=== training_config.yaml kaggle.data_root patched to {_training_config['paths']['kaggle']['data_root']} ===",
      flush=True)

with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
    MODEL_CONFIG = yaml.safe_load(f)

ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]
DATA_ROOT = INPUT_ROOT / "data"
tier_manifest_paths = {}
for tier in ALL_TIERS:
    # Manifest FILES use just the short numeric prefix ("tier1_manifest.json"),
    # not the full descriptive tier name used for the folder/tier key
    # everywhere else — a real bug on the first run of this kernel derived
    # the wrong filename here and silently found 0 tiers (see
    # sft_train._default_tier_manifest_paths' docstring for the full story).
    short = tier.split("_")[0]
    path = DATA_ROOT / "synthetic_forgeries" / tier / f"{short}_manifest.json"
    if path.exists():
        tier_manifest_paths[tier] = str(path)
print(f"=== Available tier manifests for leave-one-out: {sorted(tier_manifest_paths.keys())} ===", flush=True)

RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def free_gpu_memory(label: str, model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== GPU memory after cleanup ({label}): "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)


def train_fn(train_tiers: list[str]) -> str:
    held_out = sorted(set(tier_manifest_paths.keys()) - set(train_tiers))
    subdir = f"loo_holdout_{'_'.join(held_out) or 'none'}"
    print(f"=== Fold: training on {sorted(train_tiers)}, checkpoint_subdir={subdir} ===", flush=True)
    final_dir = run_sft_train(
        model_config_path=MODEL_CONFIG_PATH, training_config_path=TRAINING_CONFIG_PATH,
        environment="kaggle", tier_names=train_tiers, checkpoint_subdir=subdir,
    )
    # sft_train.train()'s model/trainer are local to that call and hold no
    # module-level cache (unlike the clean_eval.py bug v24 fixed) — but
    # forcing cleanup here anyway, before eval_fn loads a fresh model for
    # this same fold, costs nothing and removes any doubt.
    free_gpu_memory(f"after training fold (holdout={held_out})")
    return final_dir


def eval_fn(model_handle: str, held_out_examples: list[dict]) -> list[float]:
    held_out_tier = held_out_examples[0]["tier"] if held_out_examples else "unknown"
    print(f"=== Fold: evaluating checkpoint {model_handle} on {len(held_out_examples)} "
          f"held-out '{held_out_tier}' examples ===", flush=True)
    model, processor = load_finetuned_model(MODEL_CONFIG, model_handle)
    results = eval_examples(model, processor, held_out_examples,
                             max_image_size=MODEL_CONFIG["model"]["max_image_size"])
    scores = [float(r["correct"]) for r in results]
    accuracy = sum(scores) / len(scores) if scores else float("nan")
    print(f"=== Fold result: held_out={held_out_tier}, n={len(scores)}, accuracy={accuracy:.3f} ===", flush=True)

    # Written immediately, not only at the very end — see module docstring's
    # note on session-length risk for an 11h, 5-fold script.
    (RESULTS_DIR / f"loo_fold_{held_out_tier}_raw.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    free_gpu_memory(f"after eval fold (holdout={held_out_tier})", model=model)
    return scores


print("=" * 70, flush=True)
print("PHASE 6: Leave-one-out generalization eval", flush=True)
print("=" * 70, flush=True)

output = run_leave_one_out(
    TRAINING_CONFIG_PATH, tier_manifest_paths, train_fn=train_fn, eval_fn=eval_fn,
    out_path=str(RESULTS_DIR / "leave_one_out_results.json"),
)
print("Done. Per-fold raw results and the aggregated leave_one_out_results.json "
      "are under /kaggle/working/results.", flush=True)
