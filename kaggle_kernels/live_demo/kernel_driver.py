"""Kaggle kernel: real live inference against the v26 checkpoint, exposed via
a temporary public URL through a Cloudflare quick tunnel.

Why this exists instead of a hosted Space: Hugging Face Spaces (even free
cpu-basic Gradio Spaces) currently require a PRO subscription on this brand-
new HF account, and ZeroGPU needs PRO or a community grant -- both blocked
without payment. Kaggle's free GPU tier gets the same real, live result
(upload an image, get a real verdict back) at zero cost, reusing
infrastructure this project already has fully working -- the only open
question was how to expose it publicly for free.

Real dead end, worth recording: Gradio's own `share=True` (frp-based tunnel)
does NOT work from this Kaggle environment. First attempt failed with
"Missing file: .../frpc_linux_amd64_v0.3" even though that file's URL is
directly reachable (verified: HTTP 200) -- fixed by downloading it explicitly.
Second attempt, with the binary now present, failed differently: "Could not
create share link. Please check your internet connection or our status
page" -- the frp tunnel PROTOCOL itself doesn't get through Kaggle's network
sandboxing, not just a missing file. Switched to Cloudflare's free "quick
tunnel" (`cloudflared tunnel --url ...`) instead: no account/token needed,
and it rides over standard HTTPS (443) rather than frp's custom protocol,
which is far more likely to survive a restrictive outbound network policy.

Real tradeoff, stated plainly either way: the public URL this produces is
temporary -- it lives only as long as this kernel session stays running
(Kaggle's session cap, several hours), and a fresh run produces a new URL.
Before an interview, re-run this kernel and grab the new
"https://....trycloudflare.com" line from its logs. Not a permanent hosted
endpoint, but genuinely live, real inference -- no captured/replayed data,
same as every other honesty rule in this project.

Same real pipeline as the rest of the project: finetuned_eval.run_single()
for extraction/detection, retrieval.case_index for similar-case lookup
against the 7 captured cases, decision.risk_tiering + financial_risk_reasoning
for routing and the written rationale.
"""

import base64
import io
import os
import stat
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "transformers==4.57.6", "qwen-vl-utils==0.0.14", "peft==0.19.1",
     "accelerate==1.14.0", "bitsandbytes==0.49.2", "gradio", "sentence-transformers", "faiss-cpu"],
    check=True,
)

CLOUDFLARED_PATH = Path("/kaggle/working/cloudflared")
if not CLOUDFLARED_PATH.is_file():
    print("=== Downloading cloudflared (free quick-tunnel client, no account needed) ===", flush=True)
    urllib.request.urlretrieve(
        "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        str(CLOUDFLARED_PATH),
    )
    CLOUDFLARED_PATH.chmod(CLOUDFLARED_PATH.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    print(f"=== cloudflared in place: {CLOUDFLARED_PATH} ({CLOUDFLARED_PATH.stat().st_size} bytes) ===", flush=True)

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

import gradio as gr  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

from src.decision.financial_risk_reasoning import explain_decision  # noqa: E402
from src.decision.risk_tiering import route  # noqa: E402
from src.eval.finetuned_eval import load_finetuned_model, run_single  # noqa: E402
from src.retrieval.case_index import build_index, query  # noqa: E402

MODEL_CONFIG_PATH = str(INPUT_ROOT / "config" / "model_config.yaml")
with open(MODEL_CONFIG_PATH, encoding="utf-8") as f:
    MODEL_CONFIG = yaml.safe_load(f)
with open(INPUT_ROOT / "config" / "cost_matrix_config.yaml", encoding="utf-8") as f:
    COST_CONFIG = yaml.safe_load(f)

ADAPTER_PATH = str(INPUT_ROOT / "checkpoints" / "sft_v26_balanced")
if not (Path(ADAPTER_PATH) / "adapter_model.safetensors").is_file():
    raise RuntimeError(f"Expected the v26 checkpoint at {ADAPTER_PATH}")

print("=== Loading base model + v26 adapter (real 4-bit QLoRA, same as every other real eval kernel) ===", flush=True)
MODEL, PROCESSOR = load_finetuned_model(MODEL_CONFIG, ADAPTER_PATH, device="cuda")
print("=== Model loaded ===", flush=True)

# Middle dot written as \u00b7 rather than a raw "\u00b7" character -- the real
# first live run showed this rendering as mojibake ("\u00b7") in the API response,
# most likely from kaggle-cli reading this source file with a non-UTF-8 default
# encoding during push on this Windows machine. A plain-ASCII escape sequence
# can't be mis-transcoded that way.
MIDDOT = "\u00b7"
TIER_LABELS = {
    "genuine": "Genuine",
    "tier1_field_tamper": f"Tier 1 {MIDDOT} Field tamper",
    "tier2_splicing": f"Tier 2 {MIDDOT} Photo splice",
    "tier3_inpainting": f"Tier 3 {MIDDOT} Diffusion inpaint",
    "tier4_full_synthetic": f"Tier 4 {MIDDOT} Full synthetic",
    "tier5_recapture": f"Tier 5 {MIDDOT} Recapture",
}

# The 7 real captured cases (results/sample_outputs/captured_predictions.json),
# pre-transformed via the same as_case() shape precompute_demo_data.py uses.
# Embedded directly rather than read from the mounted dataset -- results/ isn't
# part of the standard kaggle_package.py staging (only src/+config/+referenced
# images are), so depending on it being present would be a real, easy-to-miss
# failure point for a script that's meant to just work when re-run before an
# interview.
RETRIEVAL_CASES = [
    {"case_id": "04", "document_code": "alb_id", "tamper_verdict": "tampered", "tier": "genuine", "explanation": None},
    {"case_id": "54", "document_code": "alb_id", "tamper_verdict": "genuine", "tier": "genuine", "explanation": None},
    {"case_id": "42_tier1", "document_code": "tier1_field_tamper", "tamper_verdict": "tampered",
     "tier": "tier1_field_tamper", "explanation": None},
    {"case_id": "42_spliced_48_tier2", "document_code": "tier2_splicing", "tamper_verdict": "tampered",
     "tier": "tier2_splicing", "explanation": None},
    {"case_id": "alb_id_42_tier3", "document_code": "tier3_inpainting", "tamper_verdict": "tampered",
     "tier": "tier3_inpainting", "explanation": None},
    {"case_id": "synthetic_PA0265774_tier4", "document_code": "tier4_full_synthetic", "tamper_verdict": "tampered",
     "tier": "tier4_full_synthetic", "explanation": None},
    {"case_id": "alb_id_42_tier5", "document_code": "tier5_recapture", "tamper_verdict": "tampered",
     "tier": "tier5_recapture", "explanation": None},
]
print(f"=== Retrieval pool: {len(RETRIEVAL_CASES)} real captured cases ===", flush=True)
RETRIEVAL_INDEX = build_index(RETRIEVAL_CASES)


def analyze(image: Image.Image) -> dict:
    if image is None:
        raise gr.Error("Upload an image first.")

    started = time.time()
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    tmp_path = "/kaggle/working/upload.jpg"
    image.save(tmp_path, format="JPEG", quality=95)

    prediction = run_single(tmp_path, MODEL, PROCESSOR, max_image_size=MODEL_CONFIG["model"]["max_image_size"])
    parsed = prediction.get("parsed") or {}
    confidence = prediction.get("confidence")

    similar = []
    if confidence is not None:
        query_text = f"{parsed.get('tamper_verdict', '')} {parsed.get('explanation', '')}"
        similar = query(RETRIEVAL_INDEX, query_text, top_k=3)

    decision_tier, rationale = None, None
    if confidence is not None:
        decision_tier = route(confidence, COST_CONFIG)
        rationale = explain_decision(prediction, COST_CONFIG, similar_cases=similar if similar else None)

    regions_pct = []
    for x0, y0, x1, y1 in (parsed.get("tamper_regions") or []):
        regions_pct.append([
            round(x0 / orig_w * 100, 2), round(y0 / orig_h * 100, 2),
            round(x1 / orig_w * 100, 2), round(y1 / orig_h * 100, 2),
        ])

    display_img = image.copy()
    scale = 900 / max(orig_w, orig_h)
    if scale < 1:
        display_img = display_img.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    display_img.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    verdict = parsed.get("tamper_verdict", "unknown")
    elapsed = round(time.time() - started, 1)
    print(f"=== analyze(): verdict={verdict}, confidence={confidence}, {elapsed}s ===", flush=True)

    return {
        "id": f"upload-{int(started)}",
        "tier": None,
        "tier_label": "Your upload",
        "document_code": parsed.get("document_code"),
        "image_b64": image_b64,
        "regions_pct": regions_pct,
        "verdict": verdict,
        "confidence": confidence,
        "fields": {
            "name": parsed.get("name"), "dob": parsed.get("dob"),
            "id_number": parsed.get("id_number"), "address": parsed.get("address"),
            "expiry": parsed.get("expiry"),
        },
        "explanation": parsed.get("explanation"),
        "decision_tier": decision_tier,
        "rationale": rationale,
        "similar": [{"case_id": c["case_id"], "tier": c.get("tier"),
                     "tier_label": TIER_LABELS.get(c.get("tier"), c.get("tier") or "Genuine"),
                     "similarity": round(c["similarity"], 3)} for c in similar],
        "inference_seconds": elapsed,
        "parse_success": prediction.get("parse_success", False),
    }


with gr.Blocks(title="Document Verification -- live inference") as demo:
    gr.Markdown(
        "# Live inference: v26 checkpoint\n"
        "Real Qwen2.5-VL-3B (QLoRA, 4-bit) fine-tuned checkpoint, run for real on whatever "
        "you upload -- no captured/replayed data. Expect ~5-20s per image on this GPU.\n\n"
        "Portfolio project: https://github.com/Poojadheniya17/doc_verification"
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload an identity document")
        out = gr.JSON(label="Real model output")
    btn = gr.Button("Analyze", variant="primary")
    btn.click(fn=analyze, inputs=inp, outputs=out, api_name="analyze")

GRADIO_PORT = 7860
print(f"=== Launching Gradio locally on port {GRADIO_PORT} (not sharing via Gradio's own tunnel -- "
      f"see module docstring for why) ===", flush=True)
demo.queue().launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False,
                     show_error=True, prevent_thread_lock=True)

print("=== Starting cloudflared quick tunnel -- watch for a 'https://....trycloudflare.com' line below ===",
      flush=True)
tunnel_proc = subprocess.Popen(
    [str(CLOUDFLARED_PATH), "tunnel", "--url", f"http://localhost:{GRADIO_PORT}"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
)


def _stream_tunnel_output():
    for line in tunnel_proc.stdout:
        print(f"[cloudflared] {line}", end="", flush=True)


threading.Thread(target=_stream_tunnel_output, daemon=True).start()

# Keep the kernel session alive -- Gradio's own server thread (prevent_thread_lock)
# and cloudflared's subprocess both run in the background; without this the main
# script would reach EOF and Kaggle would tear the whole session (and tunnel) down.
print("=== Live. Kernel will keep running (and the tunnel stay up) until this session ends. ===", flush=True)
while True:
    time.sleep(3600)
