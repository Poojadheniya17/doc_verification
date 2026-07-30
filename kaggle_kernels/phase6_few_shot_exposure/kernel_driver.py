"""Kaggle kernel driver: the few-shot exposure diagnostic, complementary to
the v26 leave-one-out re-run and the v28 regularity-disruption re-run.

Real question this answers: the zero-shot LOO gap (13.3% tier2_splicing held
out, 0.000% tier4_full_synthetic held out -- see
results/tables/phase6_leave_one_out_summary.md) could mean two different
things, and they call for different fixes:
  1. "The model never saw this concept at all" (a data-quantity problem) --
     if so, even a HANDFUL of real examples of the held-out tier should close
     most of the gap.
  2. "This is a structurally different concept the model can't pick up from
     a few examples" -- if so, few-shot exposure won't move the number much,
     which is exactly the case for regularity_disruption's tier-agnostic
     augmentation (kaggle_kernels/phase6_leave_one_out_v28_regularity/):
     it's designed to help WITHOUT needing real examples of every possible
     tampering category.

Run together, v28 and this kernel's result triangulate which explanation is
closer to the truth. Deliberately does NOT include regularity_disruption
here -- keeping this a single-variable comparison against the original v26
LOO fold numbers (same composition otherwise, so the only difference is
k-shot vs zero-shot exposure to the held-out tier).

Mechanism: src/eval/leave_one_out_eval.py's split_few_shot_manifest() splits
the held-out tier's manifest into FEW_SHOT_K examples (seeded, deterministic
sample -- not just the first k in file order) that get folded into training,
and everything else, which becomes the eval set. A real, silent-failure risk
was found and guarded against while designing this (not just assumed away):
build_sft_examples() filters tier1/2/3/5 examples by their SOURCE genuine
image's split=="train" -- if a randomly-sampled few-shot entry's source
happens to be in the val/test split, it would silently get dropped from
training, and this kernel would think it trained on FEW_SHOT_K examples when
it actually trained on fewer. tier4_full_synthetic has no such filtering (no
source lookup at all, included unconditionally) -- the default choice below
sidesteps the issue entirely, and the assertion after build_sft_examples()
would catch it loudly if this is ever pointed at a different tier instead of
silently under-training.
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

# Valid values: tier1_field_tamper, tier2_splicing, tier3_inpainting,
# tier4_full_synthetic, tier5_recapture. Defaults to tier4_full_synthetic --
# the fold that collapsed hardest (0.000%), the most informative first test,
# and the only tier with no split-filtering edge case (see module docstring).
HELD_OUT_TIER = "tier4_full_synthetic"
FEW_SHOT_K = 3
FEW_SHOT_SEED = 42

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
from src.eval.leave_one_out_eval import split_few_shot_manifest  # noqa: E402
from src.training.sft_train import build_sft_examples  # noqa: E402
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
print(f"=== model config this run uses: r={MODEL_CONFIG['lora']['r']}, "
      f"max_image_size={MODEL_CONFIG['model']['max_image_size']} ===", flush=True)

ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]
DATA_ROOT = INPUT_ROOT / "data"
GENUINE_MANIFEST = str(DATA_ROOT / "processed" / "genuine_manifest_templates.json")
tier_manifest_paths = {}
for tier in ALL_TIERS:
    short = tier.split("_")[0]
    path = DATA_ROOT / "synthetic_forgeries" / tier / f"{short}_manifest.json"
    if path.exists():
        tier_manifest_paths[tier] = str(path)
print(f"=== Available tier manifests: {sorted(tier_manifest_paths.keys())} ===", flush=True)

if HELD_OUT_TIER not in tier_manifest_paths:
    raise RuntimeError(f"HELD_OUT_TIER={HELD_OUT_TIER!r} has no manifest among {sorted(tier_manifest_paths.keys())}")

few_shot_manifest, eval_manifest = split_few_shot_manifest(
    tier_manifest_paths[HELD_OUT_TIER], k=FEW_SHOT_K, seed=FEW_SHOT_SEED)
print(f"=== Split {HELD_OUT_TIER}: {len(few_shot_manifest['entries'])} few-shot (into training), "
      f"{len(eval_manifest['entries'])} eval (held out) ===", flush=True)

FEW_SHOT_MANIFEST_PATH = "/kaggle/working/few_shot_manifest.json"
Path(FEW_SHOT_MANIFEST_PATH).write_text(json.dumps(few_shot_manifest, indent=2), encoding="utf-8")

train_tier_manifest_paths = {**tier_manifest_paths, HELD_OUT_TIER: FEW_SHOT_MANIFEST_PATH}
train_examples = build_sft_examples(GENUINE_MANIFEST, train_tier_manifest_paths, split="train")

n_few_shot_in_train = sum(1 for e in train_examples if e["tier"] == HELD_OUT_TIER)
if n_few_shot_in_train != FEW_SHOT_K:
    raise RuntimeError(
        f"Expected exactly {FEW_SHOT_K} {HELD_OUT_TIER} examples in the built training set, "
        f"got {n_few_shot_in_train} -- likely a source-image split mismatch (see module docstring). "
        f"Fix by choosing few-shot entries whose source images are in the 'train' split, or pick a "
        f"tier without source-based split filtering (tier4_full_synthetic)."
    )
print(f"=== Built {len(train_examples)} SFT training examples "
      f"({n_few_shot_in_train} of them the {HELD_OUT_TIER} few-shot examples) ===", flush=True)

held_out_eval_examples = [
    {"image_path": e["forged_image"], "tier": HELD_OUT_TIER, "true_label": "tampered"}
    for e in eval_manifest["entries"]
]
print(f"=== Evaluating on {len(held_out_eval_examples)} remaining {HELD_OUT_TIER} examples "
      f"(the {FEW_SHOT_K} few-shot ones are excluded from eval, they were used in training) ===", flush=True)

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


subdir = f"few_shot_{FEW_SHOT_K}shot_holdout_{HELD_OUT_TIER}"
final_dir = run_sft_train(
    model_config_path=MODEL_CONFIG_PATH, training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle", train_examples=train_examples, checkpoint_subdir=subdir,
)
free_gpu_memory(f"after training (few-shot k={FEW_SHOT_K}, holdout={HELD_OUT_TIER})")

model, processor = load_finetuned_model(MODEL_CONFIG, final_dir)
results = eval_examples(model, processor, held_out_eval_examples,
                         max_image_size=MODEL_CONFIG["model"]["max_image_size"])
scores = [float(r["correct"]) for r in results]
accuracy = sum(scores) / len(scores) if scores else float("nan")
print(f"=== Few-shot result: k={FEW_SHOT_K}, held_out={HELD_OUT_TIER}, n={len(scores)}, "
      f"accuracy={accuracy:.3f} (compare against the zero-shot LOO baseline for this tier) ===", flush=True)

(RESULTS_DIR / f"few_shot_{FEW_SHOT_K}shot_{HELD_OUT_TIER}_raw.json").write_text(
    json.dumps(results, indent=2), encoding="utf-8")

free_gpu_memory(f"after eval (few-shot k={FEW_SHOT_K}, holdout={HELD_OUT_TIER})", model=model)
print("Done. Few-shot exposure results under /kaggle/working/results.", flush=True)
