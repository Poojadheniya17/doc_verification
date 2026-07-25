"""Kaggle kernel driver: capture a small set of real predictions from the
v26 checkpoint for the Streamlit demo's example gallery.

No training, no eval scoring loop — just run_single() through a handful of
real images (one genuine pair + one per forgery tier) and save the raw,
real output in the shape app/demo_app.py expects
(results/sample_outputs/captured_predictions.json).
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

import gc  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from src.eval.finetuned_eval import eval_examples, load_finetuned_model  # noqa: E402
from src.training.sft_train import build_sft_examples  # noqa: E402

MODEL_CONFIG_PATH = str(INPUT_ROOT / "config" / "model_config.yaml")
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

GENUINE_MANIFEST = str(DATA_ROOT / "processed" / "genuine_manifest_templates.json")

# tier3/tier5 only have train-split examples (see sft_train.py's docstring on
# tier4 lacking a split field, and the real test-split counts recorded in
# PROJECT_STATUS.md) -- pull from train for those two so the gallery still
# covers all 5 tiers, test split for everything else.
test_examples = build_sft_examples(GENUINE_MANIFEST, tier_manifest_paths, split="test")
train_examples = build_sft_examples(GENUINE_MANIFEST, tier_manifest_paths, split="train")

genuine = [e for e in test_examples if e["tier"] == "genuine"][:2]
gallery = list(genuine)
for tier in ALL_TIERS:
    pool = [e for e in test_examples if e["tier"] == tier] or [e for e in train_examples if e["tier"] == tier]
    if pool:
        gallery.append(pool[0])

print(f"=== Gallery: {len(gallery)} examples -- {[e['tier'] for e in gallery]} ===", flush=True)

CHECKPOINT = str(INPUT_ROOT / "checkpoints" / "sft_v26_balanced")
model, processor = load_finetuned_model(MODEL_CONFIG, CHECKPOINT)

results = eval_examples(model, processor, gallery, max_image_size=MODEL_CONFIG["model"]["max_image_size"])

for r, example in zip(results, gallery):
    image_path = Path(r["image_path"])
    r["document_code"] = image_path.parent.name if "raw" in image_path.parts else example["tier"]

del model
gc.collect()
torch.cuda.empty_cache()

OUT_DIR = Path("/kaggle/working/results/sample_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / "captured_predictions.json"
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"=== Wrote {len(results)} captured predictions to {out_path} ===", flush=True)
print("Done.", flush=True)
