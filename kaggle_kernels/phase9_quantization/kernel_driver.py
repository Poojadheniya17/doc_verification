"""Kaggle kernel driver: Phase 9 quantization benchmarking (fp16 vs int8 vs int4).

Reuses the trained checkpoint (checkpoints/sft_v24_final) and a small,
deliberately balanced eval set (adversarial_rounds.build_eval_set(), reused
rather than duplicating the same sampling logic — see
config/training_config.yaml's quantization_bench.eval_sample_size comment for
why this project's real data only supports ~75 examples, not the original
200 placeholder).

Same explicit-cleanup discipline as the other Phase 6 kernels: every
precision's model load is followed by gc.collect()+torch.cuda.empty_cache(),
with before/after GPU memory diagnostics printed rather than assumed —
loading 3 separate model copies in one session is exactly the kind of
leftover-memory risk that caused Phase 4's whole v19-v23 OOM chain (root-
caused in v24).
"""

import gc
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
# Kaggle's base image ships torchao==0.10.0. peft's LoRA module dispatch
# tries dispatch_torchao as one of several candidate backends even for a
# plain fp16 (non-torchao-quantized) model, and its own version gate
# (is_torchao_available() in peft/import_utils.py) *raises* rather than
# skipping when it finds an installed-but-too-old torchao ("Found an
# incompatible version of torchao... only versions above 0.16.0 are
# supported") — confirmed by reading that function's source: it only
# returns False cleanly when torchao isn't installed at all, not when a
# stale version is present. This project never uses torchao (bitsandbytes
# is the only quantization backend used, for int8/int4; fp16 needs no
# quantization backend at all), so uninstalling it entirely is the correct
# fix, not pinning a newer version we have no other use for.
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)

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

from src.eval.adversarial_rounds import build_eval_set  # noqa: E402
from src.eval.finetuned_eval import eval_examples_with_latency, load_finetuned_model_at_precision  # noqa: E402
from src.eval.quantization_bench import run as run_quantization_bench  # noqa: E402

MODEL_CONFIG_PATH = str(INPUT_ROOT / "config" / "model_config.yaml")
with open(INPUT_ROOT / "config" / "training_config.yaml", encoding="utf-8") as f:
    _training_config = yaml.safe_load(f)
_training_config["paths"]["kaggle"]["data_root"] = str(INPUT_ROOT / "data")
TRAINING_CONFIG_PATH = "/kaggle/working/training_config_resolved.yaml"
with open(TRAINING_CONFIG_PATH, "w", encoding="utf-8") as f:
    yaml.safe_dump(_training_config, f)

with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
    MODEL_CONFIG = yaml.safe_load(f)

ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]
DATA_ROOT = INPUT_ROOT / "data"
tier_manifest_paths = {}
for tier in ALL_TIERS:
    short = tier.split("_")[0]
    path = DATA_ROOT / "synthetic_forgeries" / tier / f"{short}_manifest.json"
    if path.exists():
        tier_manifest_paths[tier] = str(path)
print(f"=== Available tier manifests: {sorted(tier_manifest_paths.keys())} ===", flush=True)

GENUINE_MANIFEST = str(DATA_ROOT / "processed" / "genuine_manifest_templates.json")
n_genuine = _training_config["quantization_bench"]["eval_sample_size"]
EVAL_EXAMPLES = build_eval_set(GENUINE_MANIFEST, tier_manifest_paths, n_genuine=n_genuine, split="test")
print(f"=== Built eval set: {len(EVAL_EXAMPLES)} examples "
      f"({sum(1 for e in EVAL_EXAMPLES if e['tier'] == 'genuine')} genuine, "
      f"{sum(1 for e in EVAL_EXAMPLES if e['tier'] != 'genuine')} tampered) ===", flush=True)

CHECKPOINT_PATH = str(INPUT_ROOT / "checkpoints" / "sft_v24_final")
if not (Path(CHECKPOINT_PATH) / "adapter_model.safetensors").is_file():
    raise RuntimeError(f"Expected Phase 4's committed checkpoint at {CHECKPOINT_PATH}")
print(f"=== Benchmarking checkpoint: {CHECKPOINT_PATH} ===", flush=True)

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


def load_fn(precision: str, model_config: dict):
    print(f"=== Loading checkpoint at precision={precision} ===", flush=True)
    model, processor = load_finetuned_model_at_precision(model_config, CHECKPOINT_PATH, precision)
    return (model, processor)


def eval_fn(model_handle, eval_examples: list[dict]) -> list[dict]:
    model, processor = model_handle
    results = eval_examples_with_latency(model, processor, eval_examples,
                                          max_image_size=MODEL_CONFIG["model"]["max_image_size"])
    accuracy = sum(r["correct"] for r in results) / len(results) if results else float("nan")
    avg_latency = sum(r["latency_seconds"] for r in results) / len(results) if results else float("nan")
    print(f"=== Precision result: n={len(results)}, accuracy={accuracy:.3f}, "
          f"avg_latency={avg_latency:.2f}s ===", flush=True)
    free_gpu_memory("after precision eval", model=model)
    return results


print("=" * 70, flush=True)
print("PHASE 9: Quantization benchmarking (fp16 vs int8 vs int4)", flush=True)
print("=" * 70, flush=True)

output = run_quantization_bench(
    MODEL_CONFIG_PATH, TRAINING_CONFIG_PATH, EVAL_EXAMPLES,
    load_fn=load_fn, eval_fn=eval_fn, out_path=str(RESULTS_DIR / "quantization_bench_results.json"),
)
print("Done. Results under /kaggle/working/results.", flush=True)
