"""Kaggle kernel driver: Phase 6 adversarial retraining rounds.

3 rounds (config/training_config.yaml's adversarial_rounds.num_rounds), each:
evaluate the current model on a fixed eval set -> mine incorrect examples
(capped at failure_sample_cap) -> retrain targeted on those failures ->
repeat. Round 0 has no failures yet, so it reuses the existing Phase 4 (v24)
checkpoint as-is rather than training on nothing; rounds 1+ resume from the
previous round's adapter and retrain only on that round's mined failures
(much smaller than a full retrain, so each of those rounds should be
meaningfully faster than Phase 4's ~2h13m).

Same explicit-cleanup discipline as the leave-one-out kernel: Phase 4's whole
v19-v23 OOM chain was root-caused (v24) to a leftover model reference from an
earlier phase never being freed, so every model load here is followed by an
explicit gc.collect()+torch.cuda.empty_cache(), with before/after diagnostics
printed rather than assumed.
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

from src.eval.adversarial_rounds import build_eval_set  # noqa: E402
from src.eval.adversarial_rounds import run as run_adversarial_rounds  # noqa: E402
from src.eval.finetuned_eval import eval_examples, load_finetuned_model  # noqa: E402
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
print(f"=== Available tier manifests: {sorted(tier_manifest_paths.keys())} ===", flush=True)

GENUINE_MANIFEST = str(DATA_ROOT / "processed" / "genuine_manifest_templates.json")
EVAL_SET = build_eval_set(GENUINE_MANIFEST, tier_manifest_paths, n_genuine=10, split="test")
print(f"=== Built fixed eval set: {len(EVAL_SET)} examples "
      f"({sum(1 for e in EVAL_SET if e['tier'] == 'genuine')} genuine, "
      f"{sum(1 for e in EVAL_SET if e['tier'] != 'genuine')} tampered) ===", flush=True)

PHASE4_CHECKPOINT = str(INPUT_ROOT / "checkpoints" / "sft_v24_final")
if not (Path(PHASE4_CHECKPOINT) / "adapter_model.safetensors").is_file():
    raise RuntimeError(f"Expected Phase 4's committed checkpoint at {PHASE4_CHECKPOINT} — "
                        f"check that checkpoints/sft_v24_final is included in the mounted dataset.")
print(f"=== Round 0 baseline checkpoint: {PHASE4_CHECKPOINT} ===", flush=True)

RESULTS_DIR = Path("/kaggle/working/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_state = {"current_checkpoint": PHASE4_CHECKPOINT}


def free_gpu_memory(label: str, model=None) -> None:
    if model is not None:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"=== GPU memory after cleanup ({label}): "
          f"{torch.cuda.memory_allocated() / 1e9:.2f} GB allocated, "
          f"{torch.cuda.memory_reserved() / 1e9:.2f} GB reserved ===", flush=True)


def train_fn(failures: list[dict], round_num: int) -> str:
    if round_num == 0:
        print(f"=== Round 0: no failures mined yet — reusing existing Phase 4 checkpoint "
              f"({PHASE4_CHECKPOINT}) as this round's baseline, no retrain ===", flush=True)
        return _state["current_checkpoint"]

    train_examples = [
        {"image_path": f["image_path"], "target": f["target"], "tier": f.get("tier", "adversarial_mined")}
        for f in failures if f.get("target")
    ]
    print(f"=== Round {round_num}: retraining on {len(train_examples)} mined failures, "
          f"resuming from {_state['current_checkpoint']} ===", flush=True)
    if not train_examples:
        print(f"=== Round {round_num}: no usable mined failures — keeping previous checkpoint unchanged ===",
              flush=True)
        return _state["current_checkpoint"]

    final_dir = run_sft_train(
        model_config_path=MODEL_CONFIG_PATH, training_config_path=TRAINING_CONFIG_PATH,
        environment="kaggle", train_examples=train_examples, checkpoint_subdir=f"adv_round{round_num}",
        resume_from_adapter=_state["current_checkpoint"],
    )
    free_gpu_memory(f"after training round {round_num}")
    _state["current_checkpoint"] = final_dir
    return final_dir


def eval_fn(model_handle: str, eval_set: list[dict]) -> list[dict]:
    print(f"=== Evaluating checkpoint {model_handle} on {len(eval_set)} fixed eval-set examples ===", flush=True)
    model, processor = load_finetuned_model(MODEL_CONFIG, model_handle)
    results = eval_examples(model, processor, eval_set, max_image_size=MODEL_CONFIG["model"]["max_image_size"])
    accuracy = sum(r["correct"] for r in results) / len(results) if results else float("nan")
    print(f"=== Eval result for {model_handle}: n={len(results)}, accuracy={accuracy:.3f} ===", flush=True)

    # Written immediately per round, not only at the very end.
    (RESULTS_DIR / f"adv_eval_raw_{Path(model_handle).parent.name}.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")

    free_gpu_memory("after eval", model=model)
    return results


print("=" * 70, flush=True)
print("PHASE 6: Adversarial retraining rounds", flush=True)
print("=" * 70, flush=True)

output = run_adversarial_rounds(
    TRAINING_CONFIG_PATH, EVAL_SET, train_fn=train_fn, eval_fn=eval_fn,
    out_path=str(RESULTS_DIR / "adversarial_rounds_results.json"),
)
print("Done. Per-round raw eval results and the aggregated adversarial_rounds_results.json "
      "are under /kaggle/working/results.", flush=True)
