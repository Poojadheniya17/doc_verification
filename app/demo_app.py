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
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,440;9..144,560;9..144,650&family=Public+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --paper: #EEF2F0;
    --surface: #FFFFFF;
    --surface-alt: #F6F9F7;
    --ink: #16211E;
    --ink-soft: #4B5A56;
    --ink-faint: #7C8B86;
    --border: #D7DEDA;
    --accent: #1F5C56;
    --accent-soft: #E4EEEC;
    --genuine: #3E6B4F;
    --genuine-soft: #E4EEE7;
    --tampered: #8C3B32;
    --tampered-soft: #F3E4E1;
    --review: #93701F;
    --review-soft: #F1EBDB;
}
@media (prefers-color-scheme: dark) {
    :root {
        --paper: #101614;
        --surface: #182320;
        --surface-alt: #1E2926;
        --ink: #E7EEEC;
        --ink-soft: #9FB0AB;
        --ink-faint: #6C7C77;
        --border: #2C3A36;
        --accent: #59B3A6;
        --accent-soft: #1C332F;
        --genuine: #7FB592;
        --genuine-soft: #1E2F24;
        --tampered: #D08E82;
        --tampered-soft: #33201D;
        --review: #D3AC63;
        --review-soft: #332A16;
    }
}

html, body, [class*="css"] { font-family: 'Public Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
.stApp { background: var(--paper); color: var(--ink); }
/* :where() carries zero specificity so this base reset can never outrank a
   component's own color rule below, regardless of selector type or source
   order (a real bug caught while checking computed styles: .stApp div was
   quietly beating single-class rules like .field-label). */
:where(.stApp p, .stApp span, .stApp label, .stApp div) { color: var(--ink); }
.block-container { padding-top: 2.2rem; max-width: 1180px; }

section[data-testid="stSidebar"] { background: var(--surface-alt); border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--ink) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 8px !important;
}
.sidebar-eyebrow {
    font-family: 'Public Sans', sans-serif; font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.09em; color: var(--ink-faint); margin: 0 0 10px 0;
}
.sidebar-disclosure {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; line-height: 1.6; color: var(--ink-faint);
    border-top: 1px solid var(--border); padding-top: 14px; margin-top: 18px;
}

/* quiet top bar, no gradient hero */
.topbar { border-bottom: 1px solid var(--border); padding-bottom: 18px; margin-bottom: 28px; }
.topbar .kicker {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em;
    color: var(--accent); margin: 0 0 6px 0;
}
.topbar h1 {
    font-family: 'Fraunces', Georgia, serif; font-weight: 560; font-size: 1.85rem; letter-spacing: -0.01em;
    margin: 0 0 8px 0; color: var(--ink); text-wrap: balance;
}
.topbar p { font-size: 0.92rem; margin: 0 0 12px 0; color: var(--ink-soft); line-height: 1.55; max-width: 700px; }
.status-line { display: flex; gap: 18px; flex-wrap: wrap; }
.status-item {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--ink-faint);
    display: flex; align-items: center; gap: 6px;
}
.status-dot { width: 6px; height: 6px; border-radius: 50%; display: inline-block; }
.status-dot.replay { background: var(--review); }
.status-dot.live { background: var(--genuine); }

.card {
    background: var(--surface); border-radius: 10px; padding: 22px 24px; margin-bottom: 18px;
    border: 1px solid var(--border);
}
.card h3 {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--ink-faint); margin: 0 0 16px 0; font-family: 'Public Sans', sans-serif;
}

/* the verdict panel is the dominant element on the page */
.verdict-panel { background: var(--surface); border-radius: 10px; border: 1px solid var(--border); overflow: hidden; }
.verdict-panel .eyebrow {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--ink-faint); padding: 22px 26px 0 26px; margin: 0;
}
.verdict-head { display: flex; align-items: center; gap: 16px; padding: 10px 26px 0 26px; }
.verdict-seal {
    width: 46px; height: 46px; min-width: 46px; border-radius: 50%; border: 2px solid currentColor;
    display: flex; align-items: center; justify-content: center;
    font-family: 'Fraunces', serif; font-size: 1.3rem; font-weight: 560;
}
.verdict-word {
    font-family: 'Fraunces', Georgia, serif; font-weight: 650; font-size: 2.1rem; letter-spacing: -0.01em;
    margin: 0; line-height: 1.05;
}
.verdict-genuine { color: var(--genuine); }
.verdict-tampered { color: var(--tampered); }
.verdict-unknown { color: var(--ink-faint); }

.conf-row { padding: 20px 26px 0 26px; }
.conf-label {
    font-size: 0.76rem; color: var(--ink-soft); margin: 0 0 6px 0; display: flex;
    justify-content: space-between; align-items: baseline;
}
.conf-number {
    font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 2.4rem;
    font-variant-numeric: tabular-nums; line-height: 1; margin: 0 0 10px 0;
}
.conf-track { background: var(--surface-alt); border: 1px solid var(--border); border-radius: 999px; height: 8px; width: 100%; overflow: hidden; }
.conf-fill { height: 100%; border-radius: 999px; }

.verdict-conclusion {
    margin: 20px 26px 0 26px; padding: 16px 20px; border-radius: 8px; font-size: 0.88rem; line-height: 1.55;
}
.verdict-conclusion b { font-weight: 700; }
.verdict-conclusion.approve { background: var(--genuine-soft); color: var(--genuine); }
.verdict-conclusion.reject { background: var(--tampered-soft); color: var(--tampered); }
.verdict-conclusion.review { background: var(--review-soft); color: var(--review); }

.verdict-panel .panel-foot {
    margin-top: 20px; padding: 14px 26px; border-top: 1px solid var(--border);
    background: var(--surface-alt); font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: var(--ink-faint);
}

.field-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 20px; }
.field-item { border-bottom: 1px solid var(--border); padding-bottom: 9px; }
.field-label { font-size: 0.66rem; font-weight: 700; text-transform: uppercase; color: var(--ink-faint); letter-spacing: 0.05em; }
.field-value { font-family: 'IBM Plex Mono', monospace; font-size: 0.9rem; color: var(--ink); margin-top: 3px; }

.explanation-box {
    background: var(--surface-alt); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0;
    padding: 14px 18px; font-size: 0.88rem; color: var(--ink-soft); line-height: 1.55; font-style: italic;
}

.case-card {
    display: flex; justify-content: space-between; align-items: center;
    padding: 12px 16px; border-radius: 8px; background: var(--surface-alt); margin-bottom: 8px;
    border: 1px solid var(--border);
}
.case-name { font-family: 'IBM Plex Mono', monospace; font-weight: 500; color: var(--ink); font-size: 0.82rem; }
.case-tier { font-size: 0.72rem; color: var(--ink-faint); margin-top: 2px; }
.sim-bar-track { background: var(--border); border-radius: 999px; height: 5px; width: 88px; overflow: hidden; }
.sim-bar-fill { height: 100%; background: var(--accent); border-radius: 999px; }
.sim-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; font-weight: 600; color: var(--accent); margin-left: 8px; }

hr { border-color: var(--border); }
</style>
"""


def load_captured_predictions() -> list[dict]:
    if not CAPTURED_PREDICTIONS_PATH.is_file():
        return []
    return json.loads(CAPTURED_PREDICTIONS_PATH.read_text(encoding="utf-8"))


def draw_tamper_regions(image: Image.Image, regions: list[list[int]]) -> Image.Image:
    """Draws each tamper_regions bbox as an outline overlay in the design
    system's tampered color — pure PIL, no model involved, safe to run on
    this machine.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for bbox in regions or []:
        draw.rectangle(bbox, outline="#8C3B32", width=5)
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
    # var() references, not hardcoded hex — these get used in inline `style`
    # attributes below, and a fixed hex there would ignore the dark-mode
    # token overrides that every other color in this app respects.
    if confidence >= 0.7:
        return "var(--genuine)"
    if confidence <= 0.3:
        return "var(--tampered)"
    return "var(--review)"


def _decision_style(tier: str) -> tuple[str, str]:
    return {
        "auto_approve": ("approve", "Auto-approved"),
        "auto_reject": ("reject", "Auto-rejected"),
        "human_review": ("review", "Routed to human review"),
    }.get(tier, ("review", tier))


def main() -> None:
    st.set_page_config(page_title="Document Verification Demo", layout="wide",
                        initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="topbar">
          <p class="kicker">Layer 1 + Layer 2 · working demo</p>
          <h1>Identity document verification</h1>
          <p>Fine-tuned Qwen2.5-VL extracts fields and detects and localizes tampering, backed by
          retrieval over past-flagged cases and a cost-aware risk-tiering decision layer that writes
          a natural-language rationale for every routing decision.</p>
          <div class="status-line">
            <span class="status-item"><span class="status-dot replay"></span>VLM inference — replayed, captured on Kaggle T4</span>
            <span class="status-item"><span class="status-dot live"></span>Retrieval + decision layer — live in this session</span>
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

    st.sidebar.markdown('<p class="sidebar-eyebrow">Example gallery</p>', unsafe_allow_html=True)
    labels = [f"{TIER_LABELS.get(p.get('tier'), p.get('tier', 'unknown'))} — {Path(p['image_path']).name}"
              for p in predictions]
    selected_idx = st.sidebar.selectbox("Choose a document", range(len(predictions)),
                                         format_func=lambda i: labels[i], label_visibility="collapsed")
    example = predictions[selected_idx]
    st.sidebar.markdown(
        '<p class="sidebar-disclosure">This dev machine has no GPU (~7.7GB RAM) and cannot load the '
        "fine-tuned VLM locally — see PROJECT_STATUS.md. Every prediction shown here is real, captured "
        "from an actual Kaggle T4 inference run, never fabricated.</p>",
        unsafe_allow_html=True,
    )

    parsed = example.get("parsed") or {}
    tamper_verdict = parsed.get("tamper_verdict", "unknown")
    confidence = example.get("confidence")
    verdict_class = {"genuine": "verdict-genuine", "tampered": "verdict-tampered"}.get(tamper_verdict,
                                                                                        "verdict-unknown")
    seal_glyph = {"genuine": "✓", "tampered": "✕"}.get(tamper_verdict, "?")

    # Retrieval runs before the verdict panel is rendered (even though the
    # panel itself sits above the retrieval card visually) since
    # explain_decision() below folds the retrieved cases into its rationale.
    other_predictions = [p for i, p in enumerate(predictions) if i != selected_idx]
    similar = []
    if other_predictions:
        other_cases = [_as_case(p) for p in other_predictions]
        index = build_index(other_cases)
        this_case = _as_case(example)
        query_text = (f"{this_case.get('document_code', '')} tier: {this_case.get('tier', '')} "
                      f"{this_case.get('explanation', '')}")
        similar = query(index, query_text, top_k=min(3, len(other_cases)))

    cost_config = load_yaml(str(COST_CONFIG_PATH))
    tier_key, rationale, sub = None, None, ""
    if confidence is not None:
        from src.decision.risk_tiering import route
        tier_key = route(confidence, cost_config)
        rationale = explain_decision(example, cost_config, similar_cases=similar if similar else None)
        rationale_lines = rationale.split("\n")
        sub = rationale_lines[2] if len(rationale_lines) > 2 else ""

    col1, col2 = st.columns([2, 3], gap="medium")

    with col1:
        st.markdown('<div class="card"><h3>Document image</h3>', unsafe_allow_html=True)
        image_path = REPO_ROOT / example["image_path"]
        if image_path.is_file():
            image = Image.open(image_path)
            regions = parsed.get("tamper_regions") or []
            st.image(draw_tamper_regions(image, regions) if regions else image, use_container_width=True)
            if regions:
                st.caption("Outlined region(s): model-predicted tamper location")
        else:
            st.error(f"Image not found: {image_path}")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        conf_pct = f"{confidence * 100:.0f}" if confidence is not None else "—"
        conf_color = _confidence_color(confidence) if confidence is not None else "var(--ink-faint)"
        conf_width = f"{confidence * 100:.0f}%" if confidence is not None else "0%"
        tier_label = TIER_LABELS.get(example.get("tier"), example.get("tier", "unknown"))

        conclusion_html = ""
        if tier_key is not None:
            style_class, title = _decision_style(tier_key)
            conclusion_html = (f'<div class="verdict-conclusion {style_class}"><b>{title}.</b> {sub}</div>')

        st.markdown(
            f"""
            <div class="verdict-panel">
              <p class="eyebrow">{tier_label}</p>
              <div class="verdict-head">
                <div class="verdict-seal" style="color:{conf_color};">{seal_glyph}</div>
                <p class="verdict-word {verdict_class}">{tamper_verdict.upper()}</p>
              </div>
              <div class="conf-row">
                <p class="conf-label"><span>Confidence this document is genuine</span></p>
                <p class="conf-number" style="color:{conf_color};">{conf_pct}<span style="font-size:1.1rem;">%</span></p>
                <div class="conf-track"><div class="conf-fill" style="width:{conf_width}; background:{conf_color};"></div></div>
              </div>
              {conclusion_html}
              <div class="panel-foot">document · {Path(example["image_path"]).name}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    col3, col4 = st.columns([2, 3], gap="medium")

    with col3:
        fields_html = "".join(
            f'<div class="field-item"><div class="field-label">{label}</div>'
            f'<div class="field-value">{parsed.get(key) or "—"}</div></div>'
            for key, label in EXTRACTION_FIELDS
        )
        st.markdown(
            f'<div class="card"><h3>Extracted fields</h3><div class="field-grid">{fields_html}</div></div>',
            unsafe_allow_html=True,
        )

        explanation = parsed.get("explanation") or "No explanation provided by the model for this example."
        st.markdown(
            f'<div class="card"><h3>Model explanation</h3>'
            f'<div class="explanation-box">&ldquo;{explanation}&rdquo;</div></div>',
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown('<div class="card"><h3>Similar past-flagged cases · retrieval (live)</h3>',
                     unsafe_allow_html=True)
        if similar:
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
            st.caption("Not enough other examples in the gallery for a meaningful retrieval comparison.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card"><h3>Full decision rationale</h3>', unsafe_allow_html=True)
        if rationale is not None:
            st.text(rationale)
        else:
            st.caption("No confidence score captured for this example — cannot route a decision.")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Raw model output (JSON) — for technical review"):
        st.json(parsed)


if __name__ == "__main__":
    main()
