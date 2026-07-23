"""Kaggle kernel driver: Tier 3 forgery generation (diffusion inpainting).

Mounted dataset: poojadheniya/doc-verification-data (code + data, packaged by
src/utils/kaggle_package.py from this project's local repo).

Only the diffusion pass itself (inpaint_forger.run_inpainting) needs a GPU —
mask-building is pure OpenCV and already validated locally (see
tests/test_pipeline_smoke.py). This kernel is deliberately separate from the
Phase 3/4 kernel (zero-shot baseline + SFT training): that kernel is already a
~2h13m run and pulls in a completely different model stack (Qwen2.5-VL +
bitsandbytes 4-bit); coupling a multi-GB Stable Diffusion download into the
same session would add risk for no benefit, since Tier 3 generation is a
short, independent step (~15 images, no training).
"""

import os
import subprocess
import sys
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
# This Kaggle instance type has 2 visible GPUs (found during Phase 3/4 debugging,
# v17) — restricting to GPU 0 here for consistency/safety, though the diffusers
# pipeline used here doesn't auto-wrap in DataParallel the way transformers'
# Trainer does, so this is precautionary rather than a known-necessary fix.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# NOT pinning exact versions here (unlike the Phase 3/4 kernel's deliberate
# hard pins) — found the hard way on the first push: diffusers==0.31.0 pinned
# against whatever transformers Kaggle's base image ships crashed with
# "cannot import name 'FLAX_WEIGHTS_NAME' from 'transformers.utils'", a real
# version mismatch between the two packages, not a Qwen2.5-VL-style
# compatibility cliff. Letting pip's own resolver pick mutually-compatible
# versions is the right tool for *this* specific failure mode; the installed
# version is printed below so the run stays reproducible/debuggable.
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "diffusers", "accelerate", "safetensors"],
    check=True,
)
import diffusers  # noqa: E402
print(f"=== diffusers version actually installed: {diffusers.__version__} ===", flush=True)

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

from src.data_generation.inpaint_forger import inpaint_manifest  # noqa: E402

GENUINE_MANIFEST = str(INPUT_ROOT / "data" / "processed" / "genuine_manifest_templates.json")
OUT_DIR = "/kaggle/working/tier3_inpainting"

print("=" * 70, flush=True)
print("PHASE 5: Tier 3 forgery generation (Stable Diffusion inpainting)", flush=True)
print("=" * 70, flush=True)

# limit=15 matches Tier 2/4/5's scale (see field_tamper.py/splice.py/
# synthetic_id_gen.py/recapture_sim.py) for a fair per-tier comparison in the
# leave-one-out eval that consumes all 5 tiers.
results = inpaint_manifest(GENUINE_MANIFEST, OUT_DIR, limit=15)
n_success = sum(r["success"] for r in results)
print(f"=== Tier 3 inpainting: {n_success}/{len(results)} succeeded ===", flush=True)
print("Done. Outputs under /kaggle/working/tier3_inpainting.", flush=True)
