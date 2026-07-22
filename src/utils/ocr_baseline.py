"""Classical OCR baseline for extraction comparison against the fine-tuned VLM.

Uses EasyOCR, not Tesseract: pytesseract is in requirements.txt, but it's a thin
wrapper around a separate `tesseract` binary that has to be installed via the
OS package manager — not present on this dev machine, and not guaranteed on
Kaggle either. EasyOCR is pure-Python/torch, works identically wherever torch
does, and both are "classical OCR" for the purposes of this comparison (neither
does structured field understanding). If tesseract is available in your
environment, `engine="tesseract"` is stubbed for future use, but "easyocr" is
the one actually exercised.

This is a deliberately naive, rule-based field mapper, not a NER model: it's
supposed to be a weak "before" baseline the fine-tuned VLM should clearly beat,
not a competitive extraction system in its own right.
"""

import re

import easyocr

_reader = None

_DATE_RE = re.compile(r"\b\d{1,2}[.\-/ ]+\d{1,2}[.\-/ ]+\d{2,4}\b")
_ID_NUMBER_RE = re.compile(r"\b[A-Z]{0,3}\d{5,}\b")
_NAME_LABEL_RE = re.compile(r"surname|given name|nom|prénom|vārds|uzvārds|meno|priezvisko", re.IGNORECASE)
_ADDRESS_LABEL_RE = re.compile(r"address|adrese|adresa", re.IGNORECASE)


def _get_reader() -> easyocr.Reader:
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _bbox_center_y(polygon: list) -> float:
    ys = [p[1] for p in polygon]
    return sum(ys) / len(ys)


def _nearest_text_below(tokens: list, label_y: float, max_dy: float = 80.0) -> str | None:
    """Among OCR tokens, finds the one whose vertical center is just below
    label_y — the common ID-card layout is "LABEL:\\nVALUE" stacked vertically.
    """
    candidates = [(y, text) for y, text in tokens if 0 < (y - label_y) <= max_dy]
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def extract_fields(image_path: str, engine: str = "easyocr") -> dict:
    if engine != "easyocr":
        raise NotImplementedError(f"engine='{engine}' not available in this environment; use 'easyocr'")

    reader = _get_reader()
    ocr_results = reader.readtext(image_path)  # list of (polygon, text, confidence)

    tokens_with_y = [(_bbox_center_y(poly), text) for poly, text, conf in ocr_results if conf > 0.3]
    all_text = [(y, text) for y, text in tokens_with_y]

    dates = sorted(
        ((y, m.group()) for y, text in all_text for m in [_DATE_RE.search(text)] if m),
        key=lambda t: t[0],
    )
    dob = dates[0][1] if dates else None
    expiry = dates[-1][1] if len(dates) >= 2 else None

    id_candidates = [
        (y, m.group()) for y, text in all_text
        for m in [_ID_NUMBER_RE.search(text)] if m and "<" not in text
    ]
    id_number = id_candidates[0][1] if id_candidates else None

    name = None
    for y, text in all_text:
        if _NAME_LABEL_RE.search(text):
            name = _nearest_text_below(all_text, y)
            if name:
                break

    address = None
    for y, text in all_text:
        if _ADDRESS_LABEL_RE.search(text):
            address = _nearest_text_below(all_text, y)
            if address:
                break

    return {
        "name": name,
        "dob": dob,
        "id_number": id_number,
        "address": address,
        "expiry": expiry,
        "engine": engine,
        "num_ocr_tokens": len(all_text),
    }
