"""Hugging Face Space: real live inference against the v26 (class-balanced) checkpoint.

This is the ONLY place in this project that runs the fine-tuned model against
an arbitrary, user-uploaded image. Every other demo (app_static/index.html,
app/demo_app.py) deliberately replays captured predictions instead -- this
Space is what makes "upload a document and get a real verdict" honest rather
than faked, by actually running the checkpoint.

Same real pipeline as the rest of the project, just wired for one-image,
on-demand use instead of batch eval: finetuned_eval.run_single() for
extraction/detection, retrieval.case_index for similar-case lookup against
the 7 captured cases, decision.risk_tiering + financial_risk_reasoning for
routing and the written rationale. No shortcuts, no mocked outputs.

ZeroGPU note: this Space requests a GPU only inside the @spaces.GPU-decorated
function -- module-level code runs on CPU with no CUDA visible, so model
loading (which needs the GPU for 4-bit quantization) is deferred to first
call and cached in _STATE afterward, rather than attempted at import time.
"""

import base64
import io
import json
import time
from pathlib import Path

import gradio as gr
import spaces
from PIL import Image

from src.decision.financial_risk_reasoning import explain_decision
from src.decision.risk_tiering import route
from src.eval.finetuned_eval import load_finetuned_model, run_single
from src.retrieval.case_index import build_index, query
from src.utils.config_utils import load_yaml

REPO_ROOT = Path(__file__).resolve().parent
MODEL_CONFIG = load_yaml(str(REPO_ROOT / "config" / "model_config.yaml"))
COST_CONFIG = load_yaml(str(REPO_ROOT / "config" / "cost_matrix_config.yaml"))
ADAPTER_PATH = str(REPO_ROOT / "checkpoints" / "sft_v26_balanced")

TIER_LABELS = {
    "genuine": "Genuine",
    "tier1_field_tamper": "Tier 1 · Field tamper",
    "tier2_splicing": "Tier 2 · Photo splice",
    "tier3_inpainting": "Tier 3 · Diffusion inpaint",
    "tier4_full_synthetic": "Tier 4 · Full synthetic",
    "tier5_recapture": "Tier 5 · Recapture",
}

# The 7 real captured cases double as this Space's retrieval pool -- same
# cases the static demo shows, so "similar past-flagged cases" means the
# same thing in both places.
_captured = json.loads((REPO_ROOT / "captured_predictions.json").read_text(encoding="utf-8"))


def _as_case(prediction: dict) -> dict:
    parsed = prediction.get("parsed") or {}
    return {
        "case_id": Path(prediction["image_path"]).stem,
        "document_code": prediction.get("document_code"),
        "tamper_verdict": parsed.get("tamper_verdict"),
        "tier": prediction.get("tier"),
        "explanation": parsed.get("explanation"),
    }


_RETRIEVAL_CASES = [_as_case(p) for p in _captured]
_RETRIEVAL_INDEX = None  # built lazily -- SentenceTransformer download shouldn't block Space startup

# Populated on first real call; avoids re-loading the 4-bit base model + adapter per request.
_STATE = {"model": None, "processor": None}


def _get_retrieval_index():
    global _RETRIEVAL_INDEX
    if _RETRIEVAL_INDEX is None:
        _RETRIEVAL_INDEX = build_index(_RETRIEVAL_CASES)
    return _RETRIEVAL_INDEX


@spaces.GPU(duration=90)
def _run_inference(image_path: str) -> dict:
    """Everything that needs a GPU: model load (first call only, cached
    after) + one real forward/generate pass. Kept as small as possible since
    ZeroGPU grants time in short slices.
    """
    if _STATE["model"] is None:
        model, processor = load_finetuned_model(MODEL_CONFIG, ADAPTER_PATH, device="cuda")
        _STATE["model"] = model
        _STATE["processor"] = processor
    return run_single(
        image_path, _STATE["model"], _STATE["processor"],
        max_image_size=MODEL_CONFIG["model"]["max_image_size"],
    )


def analyze(image: Image.Image) -> dict:
    """image: a PIL Image from Gradio's upload widget. Returns the same
    per-case shape app_static/demo_data.json uses, so the static page's
    existing render logic can consume a live result with zero changes to
    its rendering code -- only the data source differs.
    """
    if image is None:
        raise gr.Error("Upload an image first.")

    started = time.time()
    image = image.convert("RGB")
    orig_w, orig_h = image.size

    tmp_path = "/tmp/verifydoc_upload.jpg"
    image.save(tmp_path, format="JPEG", quality=95)

    prediction = _run_inference(tmp_path)
    parsed = prediction.get("parsed") or {}
    confidence = prediction.get("confidence")

    similar = []
    if confidence is not None:
        this_case = {
            "case_id": "upload", "document_code": parsed.get("document_code"),
            "tamper_verdict": parsed.get("tamper_verdict"), "tier": None,
            "explanation": parsed.get("explanation"),
        }
        query_text = f"{parsed.get('tamper_verdict', '')} {parsed.get('explanation', '')}"
        similar = query(_get_retrieval_index(), query_text, top_k=3)

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

    # Display copy: same 900px/q85 JPEG treatment as the static demo's
    # precomputed cases, so a live result looks consistent with captured ones.
    display_img = image.copy()
    scale = 900 / max(orig_w, orig_h)
    if scale < 1:
        display_img = display_img.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    display_img.save(buf, format="JPEG", quality=85)
    image_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    verdict = parsed.get("tamper_verdict", "unknown")
    elapsed = round(time.time() - started, 1)

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
        "you upload here -- no captured/replayed data. First request after idle can take "
        "30-60s while the model loads onto the GPU; after that, expect ~5-20s per image.\n\n"
        "This is a portfolio project. Real numbers, including where it's wrong, are documented at "
        "[the GitHub repo](https://github.com/Poojadheniya17/doc_verification)."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Upload an identity document")
        out = gr.JSON(label="Real model output")
    btn = gr.Button("Analyze", variant="primary")
    btn.click(fn=analyze, inputs=inp, outputs=out, api_name="analyze")

if __name__ == "__main__":
    demo.launch()
