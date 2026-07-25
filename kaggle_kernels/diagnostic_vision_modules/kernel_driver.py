"""Cheap, targeted diagnostic: does this project's LoRA config actually reach
the vision encoder, or only the language-model side?

Written after the real v26 leave-one-out fold 1 result (tier2_splicing held
out: 13.3% accuracy, 13/15 held-out examples confidently misclassified
"genuine") came in much weaker than the promising 86.7% round-0 number --
a real generalization gap, not just noise. Hypothesis: model_config.yaml's
lora.target_modules (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj) are standard Qwen2.5-VL *language-model* decoder projection names.
The vision transformer encoder in this architecture uses different module
names entirely (its own attn/mlp submodules under a `visual.*` prefix) --
if none of those match the current target_modules list, LoRA has been
training the LLM side to reinterpret a completely FROZEN, never-adapted set
of visual features this whole project. That would directly explain why a
held-out tier with a never-before-seen visual tell (a splice blend seam,
distinct from tier1's font mismatch or tier3's diffusion noise) generalizes
so poorly: nothing ever taught the vision tower to be sensitive to tampering
artifacts in the first place.

This kernel does NOT train anything and does NOT run inference -- it loads
the base model onto the GPU (same 4-bit config as every other real kernel in
this project), then walks every named module and reports which ones contain
"visual" (or otherwise look vision-tower-related) vs. which ones match the
current target_modules, printing a real, direct answer rather than guessing
from public architecture docs. Expected runtime: a couple of minutes, most
of it spent on model download/load, not compute.
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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

import torch  # noqa: E402
import yaml  # noqa: E402
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration  # noqa: E402

with open(INPUT_ROOT / "config" / "model_config.yaml", encoding="utf-8") as f:
    MODEL_CONFIG = yaml.safe_load(f)

CURRENT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
print(f"=== Current lora.target_modules (from model_config.yaml, hardcoded here to be explicit): "
      f"{CURRENT_TARGET_MODULES} ===", flush=True)

quant_cfg = MODEL_CONFIG["quantization"]
bnb_config = BitsAndBytesConfig(
    load_in_4bit=quant_cfg["load_in_4bit"],
    bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
)
model_cfg = MODEL_CONFIG["model"]
print(f"=== Loading base model {model_cfg['name']} (4-bit, no adapter -- just inspecting architecture) ===", flush=True)
_ = AutoProcessor.from_pretrained(model_cfg["name"], trust_remote_code=model_cfg["trust_remote_code"])
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_cfg["name"], quantization_config=bnb_config, device_map={"": 0},
    trust_remote_code=model_cfg["trust_remote_code"],
)
print("=== Model loaded ===", flush=True)

all_names = [name for name, _ in model.named_modules()]
print(f"=== Total named modules: {len(all_names)} ===", flush=True)

vision_related = [n for n in all_names if "visual" in n.lower() or "vision" in n.lower()]
print(f"=== Modules with 'visual'/'vision' in the name: {len(vision_related)} ===", flush=True)
for n in vision_related[:15]:
    print(f"  {n}", flush=True)
if len(vision_related) > 15:
    print(f"  ... and {len(vision_related) - 15} more", flush=True)

matches_current_target = [
    n for n in all_names
    if any(n.endswith(f".{t}") or n == t for t in CURRENT_TARGET_MODULES)
]
vision_matches_current_target = [n for n in matches_current_target if "visual" in n.lower() or "vision" in n.lower()]
llm_matches_current_target = [n for n in matches_current_target if n not in vision_matches_current_target]

print(f"=== Modules matching CURRENT target_modules: {len(matches_current_target)} total, "
      f"{len(llm_matches_current_target)} on the LLM side, "
      f"{len(vision_matches_current_target)} on the vision side ===", flush=True)
print(f"=== Sample LLM-side match: {llm_matches_current_target[0] if llm_matches_current_target else 'NONE'} ===",
      flush=True)
print(f"=== Sample vision-side match: {vision_matches_current_target[0] if vision_matches_current_target else 'NONE'} ===",
      flush=True)

# What WOULD a vision-tower-inclusive target list need to match? Print the actual
# unique leaf-module-type names under any 'visual'-prefixed module, so a real
# target_modules list can be built from what's actually there, not guessed.
vision_leaf_types = sorted({n.rsplit(".", 1)[-1] for n in vision_related if "." in n})
print(f"=== Unique leaf module names under vision-related modules (candidates for a real "
      f"vision-inclusive target_modules list): {vision_leaf_types} ===", flush=True)

print("=== CONCLUSION ===", flush=True)
if vision_matches_current_target:
    print("Current target_modules DOES already reach the vision encoder.", flush=True)
else:
    print("Current target_modules does NOT reach the vision encoder at all -- "
          "every LoRA adapter trained so far has left the vision tower completely frozen.", flush=True)
