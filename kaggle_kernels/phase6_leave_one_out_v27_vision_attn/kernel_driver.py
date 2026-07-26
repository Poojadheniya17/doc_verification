"""Kaggle kernel driver: v26 leave-one-out fold, but with LoRA extended to
cover the vision encoder's attention layers.

Real motivation: v26's real leave-one-out fold 1 (tier2_splicing held out)
scored only 13.3% (2/15), 95% CI [0.0%, 33.3%] -- see
results/tables/phase6_leave_one_out_summary.md. A follow-up diagnostic
(kaggle_kernels/diagnostic_vision_modules/) found LoRA's target_modules only
partially reach the vision encoder: the vision blocks' MLP layers happen to
share naming with the LLM decoder's projections (already adapted), but the
vision blocks' attention layers (attn.qkv, attn.proj) are not. This kernel
tests whether closing that specific gap helps -- a real, speculative
experiment, not a proven fix (see sft_train.load_model_for_training()'s
include_vision_attention param, added specifically for this).

Held out tier is deliberately the SAME one that already failed
(tier2_splicing), for a direct, apples-to-apples comparison against the real
13.3% baseline -- not a different, easier fold that would make any
improvement ambiguous.

Same resilience pattern as phase6_leave_one_out_v26: one fold per kernel
push, real per-fold result written to disk immediately.
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

HELD_OUT_TIER = "tier2_splicing"

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
from src.eval.leave_one_out_eval import load_tier_examples  # noqa: E402
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
print(f"=== class_balance config: {_training_config['sft'].get('class_balance')} ===", flush=True)

with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
    MODEL_CONFIG = yaml.safe_load(f)
print(f"=== model config this run uses: r={MODEL_CONFIG['lora']['r']}, "
      f"max_image_size={MODEL_CONFIG['model']['max_image_size']}, "
      f"base target_modules={MODEL_CONFIG['lora']['target_modules']} ===", flush=True)
print("=== include_vision_attention=True: extending target_modules with real vision-block "
      "attn.qkv/attn.proj modules (see load_model_for_training() docstring) ===", flush=True)

ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]
DATA_ROOT = INPUT_ROOT / "data"
tier_manifest_paths = {}
for tier in ALL_TIERS:
    short = tier.split("_")[0]
    path = DATA_ROOT / "synthetic_forgeries" / tier / f"{short}_manifest.json"
    if path.exists():
        tier_manifest_paths[tier] = str(path)
print(f"=== Available tier manifests: {sorted(tier_manifest_paths.keys())} ===", flush=True)

if HELD_OUT_TIER not in tier_manifest_paths:
    raise RuntimeError(f"HELD_OUT_TIER={HELD_OUT_TIER!r} has no manifest among {sorted(tier_manifest_paths.keys())}")

train_tiers = [t for t in tier_manifest_paths if t != HELD_OUT_TIER]
held_out_examples = load_tier_examples(tier_manifest_paths[HELD_OUT_TIER], HELD_OUT_TIER)
print(f"=== Fold: held_out={HELD_OUT_TIER} (n={len(held_out_examples)}), "
      f"training on {sorted(train_tiers)} ===", flush=True)

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


subdir = f"loo_v27_vision_attn_holdout_{HELD_OUT_TIER}"
final_dir = run_sft_train(
    model_config_path=MODEL_CONFIG_PATH, training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle", tier_names=train_tiers, checkpoint_subdir=subdir,
    include_vision_attention=True,
)
free_gpu_memory(f"after training fold (holdout={HELD_OUT_TIER})")

model, processor = load_finetuned_model(MODEL_CONFIG, final_dir)
results = eval_examples(model, processor, held_out_examples,
                         max_image_size=MODEL_CONFIG["model"]["max_image_size"])
scores = [float(r["correct"]) for r in results]
accuracy = sum(scores) / len(scores) if scores else float("nan")
print(f"=== Fold result (vision-attention LoRA): held_out={HELD_OUT_TIER}, n={len(scores)}, "
      f"accuracy={accuracy:.3f} (v26 baseline for this same fold was 0.133) ===", flush=True)

(RESULTS_DIR / f"loo_fold_{HELD_OUT_TIER}_vision_attn_raw.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")

free_gpu_memory(f"after eval fold (holdout={HELD_OUT_TIER})", model=model)
print(f"Done. Fold {HELD_OUT_TIER} (vision-attention LoRA) results under /kaggle/working/results.", flush=True)
