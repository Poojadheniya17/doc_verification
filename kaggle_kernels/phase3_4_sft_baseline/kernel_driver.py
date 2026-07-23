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

import os
import subprocess
import sys
from pathlib import Path

# Must be set before torch's CUDA allocator initializes (i.e. before torch is
# imported at all, transitively or otherwise) — found necessary after kernel
# version 9 (image-resize + paged-optimizer fix) got all the way through the
# forward pass and OOM'd in the backward pass instead, needing 1.7GB more with
# only 1.1GB free. The error message itself flagged "reserved but unallocated
# memory is large" (fragmentation), which expandable_segments directly
# targets — zero accuracy cost, unlike shrinking image size or LoRA rank would
# be. Setting both names since this torch build's own warning printed
# "PYTORCH_ALLOC_CONF" but the classic CUDA-specific name is
# "PYTORCH_CUDA_ALLOC_CONF" — harmless to set both.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

# Kaggle's base image doesn't have a new enough transformers for Qwen2.5-VL,
# or qwen_vl_utils at all.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "transformers>=4.49.0", "qwen-vl-utils", "peft", "accelerate", "bitsandbytes"],
    check=True,
)

# Two prior attempts both failed at `from src.eval.clean_eval import ...` with
# `ModuleNotFoundError: No module named 'src'`, for reasons that weren't
# obvious from the traceback alone (kaggle datasets files/download -f both
# confirm src/eval/clean_eval.py genuinely exists in the uploaded dataset).
# Printing exactly what's mounted, right before the import that keeps
# failing, so this doesn't take a third blind guess.
def _walk_print(root: Path, max_depth: int, prefix: str = "") -> None:
    if max_depth < 0 or not root.exists():
        return
    try:
        entries = sorted(root.iterdir())
    except (PermissionError, NotADirectoryError):
        return
    for entry in entries:
        print(f"{prefix}{entry}", flush=True)
        if entry.is_dir():
            _walk_print(entry, max_depth - 1, prefix + "  ")


print("=== /kaggle/input recursive listing (depth 4) ===", flush=True)
_walk_print(Path("/kaggle/input"), max_depth=4)

# Second Kaggle CLI ("kaggle" 2.2.3, pulled in "kagglesdk" as a dependency)
# mounts datasets one level deeper than the classic convention this project's
# configs assumed: /kaggle/input/datasets/<owner>/<slug> instead of
# /kaggle/input/<slug> — found from the recursive listing above after two
# prior runs both hit ModuleNotFoundError with the classic path. Trying both,
# in order, rather than hardcoding the new one in case this varies by kernel
# type or changes again.
candidates = [
    Path("/kaggle/input/doc-verification-data"),
    Path("/kaggle/input/datasets/poojadheniya/doc-verification-data"),
]
INPUT_ROOT = next((c for c in candidates if (c / "src").is_dir()), None)
if INPUT_ROOT is None:
    raise RuntimeError(f"Could not find the mounted dataset's src/ dir under any of {candidates} — "
                        f"see the recursive /kaggle/input listing above for the real path.")
print(f"=== Resolved INPUT_ROOT = {INPUT_ROOT} ===", flush=True)

sys.path.insert(0, str(INPUT_ROOT))
print(f"=== sys.path[0] = {sys.path[0]} ===", flush=True)

# Manifests store image paths relative to the repo root (e.g.
# "data/raw/midv2020_templates/images/alb_id/04.jpg") because that's how every
# local run works — always invoked with cwd already at the repo root. Kaggle's
# script kernel runs with cwd elsewhere (/kaggle/working), so those relative
# paths 404'd on the very first image (FileNotFoundError, not a code bug in
# clean_eval.py/sft_train.py — same relative-path convention that works fine
# locally). chdir into INPUT_ROOT rather than rewriting every manifest again.
import os  # noqa: E402
os.chdir(INPUT_ROOT)
print(f"=== cwd = {os.getcwd()} ===", flush=True)

import yaml  # noqa: E402

from src.eval.clean_eval import run as run_clean_eval  # noqa: E402
from src.training.sft_train import train as run_sft_train  # noqa: E402

# training_config.yaml's kaggle.data_root is a static string baked in when the
# dataset was packaged — it assumed the classic /kaggle/input/<slug> mount
# path. Since INPUT_ROOT is now resolved dynamically (see candidates above,
# precisely because that assumption already broke once), data_root must be
# patched to match whichever candidate actually matched, in memory, rather
# than trusting the static value in the uploaded file.
MODEL_CONFIG_PATH = str(INPUT_ROOT / "config" / "model_config.yaml")
with open(INPUT_ROOT / "config" / "training_config.yaml", encoding="utf-8") as f:
    _training_config = yaml.safe_load(f)
_training_config["paths"]["kaggle"]["data_root"] = str(INPUT_ROOT / "data")
TRAINING_CONFIG_PATH = "/kaggle/working/training_config_resolved.yaml"
with open(TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(_training_config, f)
print(f"=== training_config.yaml kaggle.data_root patched to {_training_config['paths']['kaggle']['data_root']} ===",
      flush=True)

GENUINE_MANIFEST = str(INPUT_ROOT / "data" / "processed" / "genuine_manifest_templates.json")
TIER1_MANIFEST = str(INPUT_ROOT / "data" / "synthetic_forgeries" / "tier1_field_tamper" / "tier1_manifest.json")
TIER2_MANIFEST = str(INPUT_ROOT / "data" / "synthetic_forgeries" / "tier2_splicing" / "tier2_manifest.json")

print("=" * 70, flush=True)
print("PHASE 3: zero-shot baseline (Qwen2.5-VL-7B-Instruct, 4-bit)", flush=True)
print("=" * 70, flush=True)
run_clean_eval(
    genuine_manifest_path=GENUINE_MANIFEST,
    model_name="Qwen/Qwen2.5-VL-7B-Instruct",
    tier1_manifest_path=TIER1_MANIFEST,
    tier2_manifest_path=TIER2_MANIFEST,
    n_per_category=3,
    device="cuda",
    load_in_4bit=True,
    out_path="/kaggle/working/results/clean_eval_baseline_7b.json",
)

# Read directly rather than hardcoding the model name in the print below —
# model_config.yaml's SFT target changed from 7B to 3B after the 7B hang saga
# (see model_config.yaml's DECISION comment); a hardcoded string here already
# went stale once, silently claiming 7B in this print after the config moved
# to 3B, until caught while making that exact change.
with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
    _sft_model_name = yaml.safe_load(f)["model"]["name"]

print("=" * 70, flush=True)
print(f"PHASE 4: SFT + QLoRA fine-tuning ({_sft_model_name})", flush=True)
print("=" * 70, flush=True)
run_sft_train(
    model_config_path=MODEL_CONFIG_PATH,
    training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle",
)

print("Done. Outputs under /kaggle/working/results and /kaggle/working/checkpoints.", flush=True)
