"""Kaggle kernel driver: class-balanced SFT retrain (v26).

Real motivation: three independent evaluations (adversarial-rounds,
quantization-bench, leave-one-out — see results/tables/phase6_leave_one_out_summary.md)
all show every prior trained checkpoint (v24, and v25's capacity-restoration
run) collapsing to a single predicted class ("genuine", always). All of
those runs trained on a severely imbalanced dataset (v24/v25: 700 genuine vs
19 tampered from tier1+2 only; even using all 5 tiers real data is only 700
genuine vs 62 tampered, ~11:1). The user's explicit, reasoned direction: test
class imbalance as the dominant hypothesis next, via
src.training.sft_train.balance_examples() (oversamples the minority/tampered
class to a 1:1 ratio by repetition) — a single, clean, isolated test, not
combined with further LoRA-rank experimentation.

Deliberately uses r=8/max_image_size=512 (the config's current, extensively
validated, real-footprint-backed "safe" values — NOT the r=16 that v25
happened to succeed with), to avoid conflating this test with the still-
unresolved v25 memory anomaly (see phase4_sft_summary.md's "v25" section).
Whether capacity (r=16) helps is being tested separately and independently
via adversarial-rounds against the v25 checkpoint — this run isolates the
class-imbalance variable alone.

Uses all 5 forgery tiers (not v24/v25's tier1+2-only composition) to give
the largest real, natural minority-class pool before oversampling (62 real
tampered examples across 5 tiers, vs only 19 from tier1+2 alone) — less
exact duplication needed to reach 1:1, more real diversity in the
oversampled data.
"""

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

import yaml  # noqa: E402

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
    _model_config = yaml.safe_load(f)
print(f"=== LoRA config this run uses: r={_model_config['lora']['r']}, "
      f"alpha={_model_config['lora']['alpha']}, max_image_size={_model_config['model']['max_image_size']} ===",
      flush=True)

import torch  # noqa: E402
torch.cuda.reset_peak_memory_stats()

ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]

print("=" * 70, flush=True)
print("PHASE 4 (v26): class-balanced SFT retrain, all 5 tiers", flush=True)
print("=" * 70, flush=True)
final_dir = run_sft_train(
    model_config_path=MODEL_CONFIG_PATH,
    training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle",
    tier_names=ALL_TIERS,
    checkpoint_subdir="sft_v26_balanced",
)

print(f"=== Final checkpoint: {final_dir} ===", flush=True)
print(f"=== Phase 4 (v26) peak GPU memory: "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)
print("Done. Outputs under /kaggle/working/results and /kaggle/working/checkpoints.", flush=True)
