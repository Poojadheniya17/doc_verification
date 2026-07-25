"""Streamlit demo: browse real document-verification examples through the
full pipeline — extraction, tamper verdict, localization overlay,
explanation, retrieval of similar past cases, and a risk-tiered decision.

Design decision (documented, not left implicit — see PROJECT_STATUS.md):
this dev machine has no GPU (~7.7GB RAM, an established constraint since
Phase 3) and cannot load the fine-tuned VLM. Real-time inference is
therefore NOT available in this demo. Instead it replays REAL model outputs
captured from actual Kaggle GPU runs (never fabricated), selectable from a
small gallery — clearly labeled in the UI, not hidden. Retrieval
(case_index.py) and the decision layer (financial_risk_reasoning.py) DO run
live in this session, since both are lightweight enough for this machine
(see their own module docstrings) — only the VLM inference step is replayed.

Run with: streamlit run app/demo_app.py
"""

import json
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # robust to whatever cwd `streamlit run` is launched from

from src.decision.financial_risk_reasoning import explain_decision  # noqa: E402
from src.retrieval.case_index import build_index, query  # noqa: E402
from src.utils.config_utils import load_yaml  # noqa: E402

CAPTURED_PREDICTIONS_PATH = REPO_ROOT / "results" / "sample_outputs" / "captured_predictions.json"
COST_CONFIG_PATH = REPO_ROOT / "config" / "cost_matrix_config.yaml"

EXTRACTION_FIELDS = [("name", "Name"), ("dob", "Date of birth"), ("id_number", "ID number"),
                     ("address", "Address"), ("expiry", "Expiry")]

TIER_LABELS = {
    "genuine": "Genuine",
    "tier1_field_tamper": "Tier 1 · Field tamper",
    "tier2_splicing": "Tier 2 · Photo splice",
    "tier3_inpainting": "Tier 3 · Diffusion inpaint",
    "tier4_full_synthetic": "Tier 4 · Full synthetic",
    "tier5_recapture": "Tier 5 · Recapture",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.stApp { background: #f5f6fa; }
section[data-testid="stSidebar"] { background: #14152b; }
section[data-testid="stSidebar"] * { color: #e7e8f5 !important; }
section[data-testid="stSidebar"] .stSelectbox label { color: #9a9cc4 !important; font-weight: 500; }

.hero {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 100%);
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 22px;
    color: white;
    box-shadow: 0 8px 24px rgba(67,56,202,0.25);
}
.hero h1 { font-size: 1.65rem; font-weight: 800; margin: 0 0 6px 0; color: white; }
.hero p { font-size: 0.92rem; margin: 0; opacity: 0.88; line-height: 1.5; max-width: 780px; }
.hero .badges { margin-top: 14px; }
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.03em; text-transform: uppercase;
    margin-right: 8px;
}
.pill-live { background: rgba(34,197,94,0.18); color: #22c55e; border: 1px solid rgba(34,197,94,0.4); }
.pill-replay { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }

.card {
    background: white; border-radius: 14px; padding: 20px 22px; margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(15,15,40,0.06), 0 1px 2px rgba(15,15,40,0.04);
    border: 1px solid #edeef5;
}
.card h3 {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em;
    color: #6b6d8a; margin: 0 0 14px 0;
}
.verdict-row { display: flex; align-items: center; gap: 14px; margin-bottom: 4px; }
.verdict-badge {
    font-size: 0.85rem; font-weight: 800; padding: 7px 16px; border-radius: 10px;
    letter-spacing: 0.02em;
}
.verdict-genuine { background: #dcfce7; color: #15803d; }
.verdict-tampered { background: #fee2e2; color: #b91c1c; }
.verdict-unknown { background: #f1f2f8; color: #6b6d8a; }

.conf-track { background: #eceef7; border-radius: 999px; height: 10px; width: 100%; overflow: hidden; margin-top: 10px; }
.conf-fill { height: 100%; border-radius: 999px; }

.field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px 18px; margin-top: 6px; }
.field-item { border-bottom: 1px solid #f1f2f8; padding-bottom: 8px; }
.field-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; color: #9a9cc4; letter-spacing: 0.04em; }
.field-value { font-size: 0.95rem; color: #1f2140; font-weight: 500; margin-top: 2px; }

.explanation-box {
    background: #f7f5ff; border-left: 4px solid #6d28d9; border-radius: 8px;
    padding: 14px 18px; font-size: 0.92rem; color: #2e2a52; font-style: italic; line-height: 1.5;
}

.case-card {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-radius: 10px; background: #f9f9fd; margin-bottom: 8px;
    border: 1px solid #eeeef7;
}
.case-name { font-weight: 600; color: #1f2140; font-size: 0.88rem; }
.case-tier { font-size: 0.74rem; color: #6b6d8a; }
.sim-bar-track { background: #e6e7f4; border-radius: 999px; height: 6px; width: 100px; overflow: hidden; }
.sim-bar-fill { height: 100%; background: #6d28d9; border-radius: 999px; }
.sim-score { font-size: 0.78rem; font-weight: 700; color: #4338ca; margin-left: 8px; }

.decision-banner {
    border-radius: 14px; padding: 20px 24px; margin-bottom: 14px; color: white;
}
.decision-approve { background: linear-gradient(135deg, #16a34a, #22c55e); }
.decision-reject { background: linear-gradient(135deg, #b91c1c, #ef4444); }
.decision-review { background: linear-gradient(135deg, #b45309, #f59e0b); }
.decision-banner .decision-title { font-size: 1.1rem; font-weight: 800; margin-bottom: 4px; }
.decision-banner .decision-sub { font-size: 0.85rem; opacity: 0.92; }
</style>
"""


def load_captured_predictions() -> list[dict]:
    if not CAPTURED_PREDICTIONS_PATH.is_file():
        return []
    return json.loads(CAPTURED_PREDICTIONS_PATH.read_text(encoding="utf-8"))


def draw_tamper_regions(image: Image.Image, regions: list[list[int]]) -> Image.Image:
    """Draws each tamper_regions bbox as a red rectangle overlay — pure PIL,
    no model involved, safe to run on this machine.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for bbox in regions or []:
        draw.rectangle(bbox, outline="#ef4444", width=5)
    return annotated


def _as_case(prediction: dict) -> dict:
    """Flattens a captured prediction into case_index.case_text()'s expected
    shape (top-level tamper_verdict/explanation, not nested under "parsed").
    """
    parsed = prediction.get("parsed") or {}
    return {
        "case_id": Path(prediction["image_path"]).stem,
        "document_code": prediction.get("document_code"),
        "tamper_verdict": parsed.get("tamper_verdict"),
        "tier": prediction.get("tier"),
        "explanation": parsed.get("explanation"),
    }


def _confidence_color(confidence: float) -> str:
    if confidence >= 0.7:
        return "#16a34a"
    if confidence <= 0.3:
        return "#dc2626"
    return "#d97706"


def _decision_style(tier: str) -> tuple[str, str]:
    return {
        "auto_approve": ("decision-approve", "✅ Auto-approved"),
        "auto_reject": ("decision-reject", "⛔ Auto-rejected"),
        "human_review": ("decision-review", "🕵️ Routed to human review"),
    }.get(tier, ("decision-review", tier))


def main() -> None:
    st.set_page_config(page_title="Document Verification Demo", layout="wide",
                        initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">
          <h1>🛡️ Adversarially-Robust Identity Document Verification</h1>
          <p>Fine-tuned Qwen2.5-VL extracts fields and detects and localizes tampering, backed by
          retrieval over past-flagged cases and a cost-aware risk-tiering decision layer that writes
          a natural-language rationale for every routing decision.</p>
          <div class="badges">
            <span class="pill pill-replay">VLM inference: replayed (real, captured on Kaggle)</span>
            <span class="pill pill-live">Retrieval + decision layer: live in this session</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictions = load_captured_predictions()
    if not predictions:
        st.warning(
            "No captured predictions available yet. Populate "
            "results/sample_outputs/captured_predictions.json with real output from "
            "a Kaggle eval run (see PROJECT_STATUS.md)."
        )
        return

    st.sidebar.markdown("### 📂 Example gallery")
    labels = [f"{TIER_LABELS.get(p.get('tier'), p.get('tier', 'unknown'))} — {Path(p['image_path']).name}"
              for p in predictions]
    selected_idx = st.sidebar.selectbox("Choose a document", range(len(predictions)),
                                         format_func=lambda i: labels[i])
    example = predictions[selected_idx]
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "This dev machine has no GPU (~7.7GB RAM) and cannot load the fine-tuned VLM locally — "
        "see PROJECT_STATUS.md. Every prediction shown here is real, captured from an actual "
        "Kaggle T4 inference run, never fabricated."
    )

    parsed = example.get("parsed") or {}
    tamper_verdict = parsed.get("tamper_verdict", "unknown")
    confidence = example.get("confidence")
    verdict_class = {"genuine": "verdict-genuine", "tampered": "verdict-tampered"}.get(tamper_verdict,
                                                                                        "verdict-unknown")

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown('<div class="card"><h3>Document image</h3>', unsafe_allow_html=True)
        image_path = REPO_ROOT / example["image_path"]
        if image_path.is_file():
            image = Image.open(image_path)
            regions = parsed.get("tamper_regions") or []
            st.image(draw_tamper_regions(image, regions) if regions else image, use_container_width=True)
            if regions:
                st.caption("🔴 Red box(es): model-predicted tamper region(s)")
        else:
            st.error(f"Image not found: {image_path}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        conf_pct = f"{confidence * 100:.0f}%" if confidence is not None else "—"
        conf_color = _confidence_color(confidence) if confidence is not None else "#9a9cc4"
        conf_width = f"{confidence * 100:.0f}%" if confidence is not None else "0%"

        fields_html = "".join(
            f'<div class="field-item"><div class="field-label">{label}</div>'
            f'<div class="field-value">{parsed.get(key) or "—"}</div></div>'
            for key, label in EXTRACTION_FIELDS
        )

        st.markdown(
            f"""
            <div class="card">
              <h3>Model verdict &amp; extraction</h3>
              <div class="verdict-row">
                <span class="verdict-badge {verdict_class}">{tamper_verdict.upper()}</span>
                <span style="color:#6b6d8a; font-size:0.85rem;">confidence this document is genuine</span>
              </div>
              <div style="font-size:1.6rem; font-weight:800; color:{conf_color};">{conf_pct}</div>
              <div class="conf-track"><div class="conf-fill" style="width:{conf_width}; background:{conf_color};"></div></div>
              <div class="field-grid">{fields_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        explanation = parsed.get("explanation") or "No explanation provided."
        st.markdown(
            f'<div class="card"><h3>Model explanation</h3>'
            f'<div class="explanation-box">&ldquo;{explanation}&rdquo;</div></div>',
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns([1, 1], gap="medium")

    with col3:
        st.markdown('<div class="card"><h3>Similar past-flagged cases · retrieval (live)</h3>',
                     unsafe_allow_html=True)
        other_predictions = [p for i, p in enumerate(predictions) if i != selected_idx]
        similar = []
        if other_predictions:
            other_cases = [_as_case(p) for p in other_predictions]
            index = build_index(other_cases)
            this_case = _as_case(example)
            query_text = (f"{this_case.get('document_code', '')} tier: {this_case.get('tier', '')} "
                          f"{this_case.get('explanation', '')}")
            similar = query(index, query_text, top_k=min(3, len(other_cases)))
            for c in similar:
                sim_pct = max(0, min(100, round(c["similarity"] * 100)))
                st.markdown(
                    f"""
                    <div class="case-card">
                      <div>
                        <div class="case-name">{c["case_id"]}</div>
                        <div class="case-tier">{TIER_LABELS.get(c.get("tier"), c.get("tier", ""))}</div>
                      </div>
                      <div style="display:flex; align-items:center;">
                        <div class="sim-bar-track"><div class="sim-bar-fill" style="width:{sim_pct}%;"></div></div>
                        <span class="sim-score">{c["similarity"]:.2f}</span>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("Not enough other examples in the gallery for a meaningful retrieval comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="card"><h3>Risk-tiered decision · cost-aware (live)</h3>',
                     unsafe_allow_html=True)
        if confidence is None:
            st.write("No confidence score captured for this example — cannot route a decision.")
        else:
            cost_config = load_yaml(str(COST_CONFIG_PATH))
            from src.decision.risk_tiering import route
            tier_key = route(confidence, cost_config)
            banner_class, banner_title = _decision_style(tier_key)
            rationale = explain_decision(example, cost_config, similar_cases=similar if similar else None)
            rationale_lines = rationale.split("\n")
            sub = rationale_lines[2] if len(rationale_lines) > 2 else ""
            st.markdown(
                f"""
                <div class="decision-banner {banner_class}">
                  <div class="decision-title">{banner_title}</div>
                  <div class="decision-sub">{sub}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Full rationale"):
                st.text(rationale)
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🔧 Raw model output (JSON) — for technical review"):
        st.json(parsed)


if __name__ == "__main__":
    main()
