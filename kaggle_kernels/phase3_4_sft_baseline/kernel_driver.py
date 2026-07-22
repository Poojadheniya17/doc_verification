"""Kaggle kernel driver: real zero-shot baseline (Phase 3) + SFT QLoRA training
(Phase 4), sharing one GPU session per the project's compute-budget plan
(Kaggle free tier gives ~30 GPU-hrs/week; running both in one session avoids
spending two separate sessions on things that can share one).

Mounted dataset: poojadheniya/doc-verification-data (code + the 719-example
SFT training set + 9-example zero-shot eval sample, packaged by
src/utils/kaggle_package.py from this project's local repo).

Order: zero-shot baseline FIRST, then SFT training — so the "before" number
comes from the genuinely untouched base model, not a model that's already
seen any training step.
"""

import subprocess
import sys

# Kaggle's base image doesn't have a new enough transformers for Qwen2.5-VL,
# or qwen_vl_utils at all.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers>=4.49.0", "qwen-vl-utils", "peft", "accelerate", "bitsandbytes"],
    check=True,
)

DATASET_ROOT = "/kaggle/input/doc-verification-data"
sys.path.insert(0, DATASET_ROOT)

from src.eval.clean_eval import run as run_clean_eval  # noqa: E402
from src.training.sft_train import train as run_sft_train  # noqa: E402

# The uploaded dataset's training_config.yaml has kaggle.data_root correctly
# set to f"{DATASET_ROOT}/data" as of the version pushed alongside this kernel
# (see config/training_config.yaml's comment for why that path matters) — no
# in-memory override needed here.
TRAINING_CONFIG_PATH = f"{DATASET_ROOT}/config/training_config.yaml"
MODEL_CONFIG_PATH = f"{DATASET_ROOT}/config/model_config.yaml"

print("=" * 70)
print("PHASE 3: zero-shot baseline (Qwen2.5-VL-7B-Instruct, 4-bit)")
print("=" * 70)
run_clean_eval(
    genuine_manifest_path=f"{DATASET_ROOT}/data/processed/genuine_manifest_templates.json",
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    tier1_manifest_path=f"{DATASET_ROOT}/data/synthetic_forgeries/tier1_field_tamper/tier1_manifest.json",
    tier2_manifest_path=f"{DATASET_ROOT}/data/synthetic_forgeries/tier2_splicing/tier2_manifest.json",
    n_per_category=3,
    device="cuda",
    load_in_4bit=True,
    out_path="/kaggle/working/results/clean_eval_baseline_7b.json",
)

print("=" * 70)
print("PHASE 4: SFT + QLoRA fine-tuning (Qwen2.5-VL-7B-Instruct)")
print("=" * 70)
run_sft_train(
    model_config_path=MODEL_CONFIG_PATH,
    training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle",
)

print("Done. Outputs under /kaggle/working/results and /kaggle/working/checkpoints.")
