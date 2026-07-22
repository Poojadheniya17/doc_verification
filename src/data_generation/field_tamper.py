"""Tier 1 forgery: digit/date swaps and font-mismatch edits on extracted ID fields.

Field *locations* are found via OCR (EasyOCR), not MIDV-2020's ground-truth
annotations — the templates.tar prefix we can afford to download locally doesn't
reach the annotations/ section of the archive (see acquire_dataset.py docstring),
and OCR-based localization also matches how the real inference-time pipeline
would find fields (no oracle annotations at inference either).

We deliberately render the replacement digits in a bundled DejaVuSans font rather
than trying to match the source document's actual font — this is the "font
mismatch" signal described in the tier name: an intentionally-visible tamper
artifact standing in for the many real-world cases where a forger's replacement
text doesn't perfectly match the original printing.
"""

import json
import random
import re
from pathlib import Path

import easyocr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("field_tamper")

ASSETS_FONT = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "DejaVuSansMono.ttf"

# Candidate fields are OCR tokens containing at least 2 digits — covers dates,
# passport/ID numbers, personal codes. Pure-letter tokens (names, static
# multilingual labels like "PASSPORT") are left alone: swapping them requires
# language-aware name generation, which is out of scope for Tier 1's scripted,
# format-preserving digit/date tamper.
_DIGIT_FIELD_RE = re.compile(r"(?:\d[\d./\- ]*\d|\d{2,})")

_reader = None


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _polygon_to_bbox(polygon: list) -> tuple[int, int, int, int]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _mutate_digits(text: str, rng: random.Random) -> str:
    """Replaces 1 to len(digits)//2+1 digit characters with different digits,
    preserving all separators/letters so the format (e.g. DD.MM.YYYY) still looks
    plausible at a glance.
    """
    chars = list(text)
    digit_positions = [i for i, c in enumerate(chars) if c.isdigit()]
    if not digit_positions:
        return text
    n_to_change = rng.randint(1, max(1, len(digit_positions) // 2 + 1))
    for i in rng.sample(digit_positions, min(n_to_change, len(digit_positions))):
        original = chars[i]
        new_digit = rng.choice([d for d in "0123456789" if d != original])
        chars[i] = new_digit
    return "".join(chars)


def _patch_region(image: Image.Image, bbox: tuple[int, int, int, int], new_text: str,
                   rng: random.Random) -> Image.Image:
    x0, y0, x1, y1 = bbox
    margin = 3
    np_img = np.array(image)

    # Sample background color from a thin strip just outside the box (above it,
    # falling back to below) rather than from inside it, so we don't sample the
    # original ink itself.
    strip_y0 = max(0, y0 - margin - 4)
    strip = np_img[strip_y0:max(strip_y0 + 1, y0 - margin), x0:x1]
    if strip.size == 0:
        strip = np_img[y1 + margin:y1 + margin + 4, x0:x1]
    bg_color = tuple(int(c) for c in np.median(strip.reshape(-1, strip.shape[-1]), axis=0)) \
        if strip.size else (255, 255, 255)

    draw = ImageDraw.Draw(image)
    draw.rectangle([x0 - margin, y0 - margin, x1 + margin, y1 + margin], fill=bg_color)

    box_h = max(1, y1 - y0)
    font_size = max(8, int(box_h * 0.9))
    font = ImageFont.truetype(str(ASSETS_FONT), font_size)
    ink = (rng.randint(20, 60), rng.randint(20, 60), rng.randint(20, 60))
    draw.text((x0, y0 - 1), new_text, font=font, fill=ink)
    return image


def tamper_fields(image_path: str, manifest_entry: dict, out_dir: str, seed: int | None = None,
                   max_fields_per_image: int = 2) -> dict:
    """Detects digit-bearing fields via OCR and tampers 1-2 of them per image.

    Returns a manifest entry with the forged image path and, for each tampered
    field, its bbox + before/after text — the bboxes double as the ground-truth
    tamper-localization target for training.
    """
    rng = random.Random(seed)
    reader = _get_reader()
    ocr_results = reader.readtext(image_path)

    candidates = [
        (poly, text) for poly, text, conf in ocr_results
        if _DIGIT_FIELD_RE.search(text) and conf > 0.3
    ]
    if not candidates:
        return {"source_image": image_path, "success": False, "reason": "no digit fields detected"}

    rng.shuffle(candidates)
    chosen = candidates[:max_fields_per_image]

    image = Image.open(image_path).convert("RGB")
    tampers = []
    for polygon, original_text in chosen:
        bbox = _polygon_to_bbox(polygon)
        new_text = _mutate_digits(original_text, rng)
        if new_text == original_text:
            continue
        image = _patch_region(image, bbox, new_text, rng)
        tampers.append({"bbox_xyxy": bbox, "original_text": original_text, "tampered_text": new_text})

    if not tampers:
        return {"source_image": image_path, "success": False, "reason": "no field could be mutated"}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(image_path)}_tier1.jpg"
    image.save(dest, quality=95)

    return {
        "source_image": image_path,
        "forged_image": str(dest),
        "tier": "tier1_field_tamper",
        "document_code": manifest_entry.get("document_code"),
        "split": manifest_entry.get("split"),
        "success": True,
        "tampers": tampers,
    }


def tamper_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                     limit: int | None = None) -> list[dict]:
    """`limit` caps how many genuine images get processed — EasyOCR is ~20s/image
    on CPU, so local smoke runs should pass a small limit (e.g. 10); omit it for a
    full Kaggle run over the whole manifest.
    """
    manifest = json.loads(Path(genuine_manifest_path).read_text())
    entries = manifest["entries"][:limit] if limit else manifest["entries"]
    results = []
    for i, entry in enumerate(entries):
        result = tamper_fields(entry["path"], entry, out_dir, seed=seed + i)
        results.append(result)
        status = "ok" if result["success"] else f"skipped ({result['reason']})"
        logger.info(f"[{i+1}/{len(entries)}] {entry['path']}: {status}")

    n_success = sum(r["success"] for r in results)
    out_manifest_path = Path(out_dir) / "tier1_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_attempted": len(results), "num_success": n_success,
                                              "entries": results}, indent=2))
    logger.info(f"Tier 1 field tamper: {n_success}/{len(results)} succeeded -> {out_manifest_path}")
    return results
