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

# v17 (device_map={"":0}) crashed with a RuntimeError inside
# torch.nn.DataParallel's parallel_apply — direct evidence this Kaggle
# instance has 2 visible GPUs (a "T4 x2" instance), not the single T4 every
# run through v17 implicitly assumed. transformers' Trainer auto-wraps the
# model in nn.DataParallel whenever torch.cuda.device_count() > 1; it skips
# that wrapping when it detects an existing accelerate multi-device map,
# which device_map="auto" (v8-v16) produced but device_map={"":0} (v17,
# a trivial single-device map) apparently didn't count as one — so v17
# uniquely hit this. Restricting CUDA_VISIBLE_DEVICES to "0" here, before
# any CUDA-touching import, makes torch.cuda.device_count() report 1
# regardless of device_map's value, removing DataParallel from
# consideration entirely rather than depending on Trainer's internal
# detection logic.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Kaggle's base image doesn't have a new enough transformers for Qwen2.5-VL,
# or qwen_vl_utils at all.
#
# Pinned to exact versions instead of "-U" (unbounded upgrade), found
# necessary after checking current PyPI state mid-debugging: unpinned
# transformers now resolves to 5.14.1, a major-version jump past the 4.x
# line this project was built and tested against (requirements.txt requires
# >=4.49.0, the first release with Qwen2.5-VL support, with no ceiling).
# Every kernel push in this whole debugging saga (v8-v16) used "-U", so the
# exact transformers/peft/accelerate/bitsandbytes versions could have
# silently drifted between runs without ever being controlled for or
# recorded -- pinning here removes that as an uncontrolled variable in this
# and every future test. 4.57.6 is the latest release still on the 4.x line
# (deliberately not jumping to 5.x in the same change that's also testing the
# device_map fix); qwen-vl-utils/peft/accelerate/bitsandbytes pinned to
# whatever was current on PyPI when this was written, no version-jump
# concern found for those three.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "qwen-vl-utils==0.0.14", "peft==0.19.1",
     "accelerate==1.14.0", "bitsandbytes==0.49.2"],
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

from src.eval.clean_eval import run as run_clean_eval, unload_model as unload_clean_eval_model  # noqa: E402
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

# v19-v23 all OOM'd on Phase 4's 3B training with a ceiling that barely moved
# despite three independent memory cuts (image size, LoRA rank, max_seq_length)
# — suspected cause: clean_eval.py caches its model at module scope (see
# unload_model()'s docstring), so the 7B model from Phase 3 was plausibly
# still resident on the GPU for all of Phase 4. Printing real before/after
# numbers here rather than assuming the fix worked, given this project's
# history of memory behavior that didn't match expectations.
import torch  # noqa: E402
print(f"=== GPU memory before Phase 3 cleanup: "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)
unload_clean_eval_model()
print(f"=== GPU memory after Phase 3 cleanup: "
      f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)
# v25: restored max_image_size/lora.r/max_seq_length toward their pre-panic
# values now that the real leak above is fixed (see model_config.yaml's
# DECISION comments). Resetting peak-memory tracking here, right before
# Phase 4 starts, so the post-training peak print below reflects 3B
# training's own real footprint only — real evidence for how much headroom
# actually exists now, not a number contaminated by Phase 3's already-freed
# 7B usage.
torch.cuda.reset_peak_memory_stats()

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
# checkpoint_subdir="sft_v25" so this run's checkpoint lands in its own
# directory rather than overwriting v24's (already committed at
# checkpoints/sft_v24_final/) — v24 stays available as the documented,
# proven-working fallback regardless of how this higher-capacity attempt
# turns out.
run_sft_train(
    model_config_path=MODEL_CONFIG_PATH,
    training_config_path=TRAINING_CONFIG_PATH,
    environment="kaggle",
    checkpoint_subdir="sft_v25",
)

print(f"=== Phase 4 peak GPU memory (this training run only, Phase 3's usage excluded): "
      f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB allocated, "
      f"{torch.cuda.max_memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)
print("Done. Outputs under /kaggle/working/results and /kaggle/working/checkpoints.", flush=True)
