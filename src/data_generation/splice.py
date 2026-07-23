"""Tier 2 forgery: photo/field splicing with blended edges between two source documents.

Uses OpenCV's Haar cascade face detector to locate the ID photo region on both a
source and a donor document (largest detected face = the photo, which is reliably
the largest face-like region on these templates), then pastes the donor's photo
into the source at the source's photo location using Poisson blending
(cv2.seamlessClone) so the splice edge doesn't have a hard seam — a step up in
realism from Tier 1's flat rectangle patches, consistent with these being
escalating forgery tiers.

Splicing across different document_codes is intentionally allowed (not restricted
to same-code donor/source pairs): a real photo-swap forgery is exactly someone's
photo pasted into someone else's document, and requiring same-code pairs would
make this generator strictly easier than the tamper type it's meant to represent.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("splice")

_face_cascade = None


def _get_cascade() -> cv2.CascadeClassifier:
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return _face_cascade


def _detect_photo_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _get_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # Largest box by area — the genuine ID photo is bigger than any incidental
    # false-positive face-like pattern elsewhere on the document (logos, MRZ noise).
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def splice_document(source_path: str, donor_path: str, out_dir: str, margin_frac: float = 0.15) -> dict:
    source = cv2.imread(source_path)
    donor = cv2.imread(donor_path)
    if source is None or donor is None:
        return {"success": False, "reason": "could not read source or donor image"}

    source_bbox = _detect_photo_bbox(source)
    donor_bbox = _detect_photo_bbox(donor)
    if source_bbox is None or donor_bbox is None:
        return {"success": False, "reason": "no photo region detected in source and/or donor"}

    sx, sy, sw, sh = source_bbox
    dx, dy, dw, dh = donor_bbox
    mx, my = int(dw * margin_frac), int(dh * margin_frac)
    dx0, dy0 = max(0, dx - mx), max(0, dy - my)
    dx1, dy1 = min(donor.shape[1], dx + dw + mx), min(donor.shape[0], dy + dh + my)
    donor_patch = donor[dy0:dy1, dx0:dx1]
    donor_resized = cv2.resize(donor_patch, (sw, sh), interpolation=cv2.INTER_LINEAR)

    center = (sx + sw // 2, sy + sh // 2)
    mask = np.full(donor_resized.shape[:2], 255, dtype=np.uint8)

    try:
        spliced = cv2.seamlessClone(donor_resized, source, mask, center, cv2.NORMAL_CLONE)
        blend_method = "seamless_clone"
    except cv2.error as e:
        logger.info(f"seamlessClone failed ({e}), falling back to feathered alpha paste")
        spliced = source.copy()
        feather = cv2.GaussianBlur(mask, (31, 31), 0).astype(np.float32) / 255.0
        feather = feather[..., None]
        roi = spliced[sy:sy + sh, sx:sx + sw].astype(np.float32)
        blended = donor_resized.astype(np.float32) * feather + roi * (1 - feather)
        spliced[sy:sy + sh, sx:sx + sw] = blended.astype(np.uint8)
        blend_method = "feathered_alpha"

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(source_path)}_spliced_{unique_stem(donor_path)}_tier2.jpg"
    cv2.imwrite(str(dest), spliced)

    return {
        "source_image": source_path,
        "donor_image": donor_path,
        "forged_image": dest.as_posix(),
        "tier": "tier2_splicing",
        "success": True,
        "blend_method": blend_method,
        "bbox_xyxy": [sx, sy, sx + sw, sy + sh],
    }


def splice_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                     limit: int | None = None) -> list[dict]:
    """Pairs each genuine image with a random different genuine image as donor
    (any document code) and attempts a splice for each pair. `limit` caps how
    many source images get processed (donor pool is always the full manifest).
    """
    manifest = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    all_entries = manifest["entries"]
    entries = all_entries[:limit] if limit else all_entries
    rng = random.Random(seed)

    results = []
    for i, entry in enumerate(entries):
        others = [e for e in all_entries if e["path"] != entry["path"]]
        donor = rng.choice(others)
        result = splice_document(entry["path"], donor["path"], out_dir)
        result["split"] = entry["split"]
        results.append(result)
        status = "ok" if result["success"] else f"skipped ({result.get('reason')})"
        logger.info(f"[{i+1}/{len(entries)}] {entry['path']} <- {donor['path']}: {status}")

    n_success = sum(r["success"] for r in results)
    out_manifest_path = Path(out_dir) / "tier2_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_attempted": len(results), "num_success": n_success,
                                              "entries": results}, indent=2), encoding="utf-8")
    logger.info(f"Tier 2 splicing: {n_success}/{len(results)} succeeded -> {out_manifest_path}")
    return results
