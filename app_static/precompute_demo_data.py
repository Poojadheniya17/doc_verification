"""One-off script: precompute real retrieval + decision output for the demo
gallery and export optimized, base64-embeddable images with percentage-based
bbox coordinates. Run once locally; output feeds the static HTML demo.
Nothing here is fabricated -- same real case_index/risk_tiering/
financial_risk_reasoning code the Streamlit app used, just executed once
instead of on every page load.
"""
import base64
import io
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image

from src.decision.financial_risk_reasoning import explain_decision
from src.decision.risk_tiering import route
from src.retrieval.case_index import build_index, query
from src.utils.config_utils import load_yaml

predictions = json.loads((REPO_ROOT / "results/sample_outputs/captured_predictions.json").read_text())
cost_config = load_yaml(str(REPO_ROOT / "config/cost_matrix_config.yaml"))

TIER_LABELS = {
    "genuine": "Genuine",
    "tier1_field_tamper": "Tier 1 \u00b7 Field tamper",
    "tier2_splicing": "Tier 2 \u00b7 Photo splice",
    "tier3_inpainting": "Tier 3 \u00b7 Diffusion inpaint",
    "tier4_full_synthetic": "Tier 4 \u00b7 Full synthetic",
    "tier5_recapture": "Tier 5 \u00b7 Recapture",
}


def as_case(prediction: dict) -> dict:
    parsed = prediction.get("parsed") or {}
    return {
        "case_id": Path(prediction["image_path"]).stem,
        "document_code": prediction.get("document_code"),
        "tamper_verdict": parsed.get("tamper_verdict"),
        "tier": prediction.get("tier"),
        "explanation": parsed.get("explanation"),
    }


out_examples = []
for i, example in enumerate(predictions):
    parsed = example.get("parsed") or {}
    confidence = example.get("confidence")

    others = [p for j, p in enumerate(predictions) if j != i]
    similar = []
    if others:
        other_cases = [as_case(p) for p in others]
        index = build_index(other_cases)
        this_case = as_case(example)
        query_text = f"{this_case.get('document_code', '')} tier: {this_case.get('tier', '')} {this_case.get('explanation', '')}"
        similar = query(index, query_text, top_k=min(3, len(other_cases)))

    tier_key, rationale = None, None
    if confidence is not None:
        tier_key = route(confidence, cost_config)
        rationale = explain_decision(example, cost_config, similar_cases=similar if similar else None)

    # real image -> resized (max 900px long edge, jpeg q85) -> base64.
    # display-only optimization, doesn't touch any prediction data.
    img_path = REPO_ROOT / example["image_path"]
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    scale = 900 / max(orig_w, orig_h)
    if scale < 1:
        img = img.resize((round(orig_w * scale), round(orig_h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # bbox -> percentages of original image dims, robust to any display size
    regions_pct = []
    for x0, y0, x1, y1 in (parsed.get("tamper_regions") or []):
        regions_pct.append([
            round(x0 / orig_w * 100, 2), round(y0 / orig_h * 100, 2),
            round(x1 / orig_w * 100, 2), round(y1 / orig_h * 100, 2),
        ])

    out_examples.append({
        "id": Path(example["image_path"]).stem,
        "tier": example.get("tier"),
        "tier_label": TIER_LABELS.get(example.get("tier"), example.get("tier")),
        "document_code": example.get("document_code"),
        "image_b64": b64,
        "image_bytes": len(buf.getvalue()),
        "regions_pct": regions_pct,
        "verdict": parsed.get("tamper_verdict", "unknown"),
        "confidence": confidence,
        "fields": {
            "name": parsed.get("name"), "dob": parsed.get("dob"),
            "id_number": parsed.get("id_number"), "address": parsed.get("address"),
            "expiry": parsed.get("expiry"),
        },
        "explanation": parsed.get("explanation"),
        "decision_tier": tier_key,
        "rationale": rationale,
        "similar": [{"case_id": c["case_id"], "tier": c.get("tier"),
                     "tier_label": TIER_LABELS.get(c.get("tier"), c.get("tier")),
                     "similarity": round(c["similarity"], 3)} for c in similar],
    })

out_path = REPO_ROOT / "app_static" / "demo_data.json"
out_path.parent.mkdir(exist_ok=True)
out_path.write_text(json.dumps(out_examples, indent=1), encoding="utf-8")
total_kb = sum(e["image_bytes"] for e in out_examples) / 1024
print(f"Wrote {len(out_examples)} examples to {out_path}, total image payload {total_kb:.0f}KB")
for e in out_examples:
    print(f"  {e['tier']:24s} verdict={e['verdict']:9s} conf={e['confidence']:.3f} decision={e['decision_tier']} regions={len(e['regions_pct'])}")
