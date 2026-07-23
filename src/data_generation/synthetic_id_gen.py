"""Tier 4 forgery: fully synthetic (no real source document) ID generation —
the hardest tier, since there's no genuine original at all to compare against;
every field and the layout itself are fabricated.

Scope decision: generating a synthetic face normally calls for a GAN/diffusion
model (e.g. StyleGAN), which both needs a local model load (blocked by the
Phase 3 RAM constraint) and is arguably out of scope for a scripted-forgery
generator per this project's non-goals (no learned/generative forger). Instead
we crop a face from a genuine MIDV-2020 image via the same Haar-cascade
detector splice.py uses. This isn't a real person either way — MIDV-2020's own
faces are themselves AI-generated via Generated Photos (see README attribution)
— so this is "reusing one kind of synthetic face for another kind of synthetic
document," not a privacy shortcut.

Field values (name/DOB/ID number/expiry) are drawn from small fixed lists via
Python's random module — a simple procedural generator, not a learned model of
plausible identities. That's an intentional scope cut (documented, not hidden).
"""

import random
import string
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from src.data_generation.splice import _detect_photo_bbox
from src.utils.logging_utils import get_logger

logger = get_logger("synthetic_id_gen")

FONT_REGULAR = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "DejaVuSans.ttf"
FONT_MONO = Path(__file__).resolve().parent.parent.parent / "assets" / "fonts" / "DejaVuSansMono.ttf"

_FIRST_NAMES = ["ALEX", "JORDAN", "MORGAN", "TAYLOR", "CASEY", "RILEY", "AVERY", "QUINN"]
_LAST_NAMES = ["HARRIS", "NELSON", "REYES", "COOPER", "BAILEY", "SANDERS", "FOSTER", "MYERS"]
_FAKE_COUNTRIES = ["REPUBLIC OF NORTHFIELD", "FEDERATION OF WESTMARK", "STATE OF EASTVALE"]


def _random_name(rng: random.Random) -> tuple[str, str]:
    return rng.choice(_FIRST_NAMES), rng.choice(_LAST_NAMES)


def _random_date(rng: random.Random, start_year: int, end_year: int) -> str:
    day, month, year = rng.randint(1, 28), rng.randint(1, 12), rng.randint(start_year, end_year)
    return f"{day:02d}.{month:02d}.{year}"


def _random_id_number(rng: random.Random) -> str:
    letters = "".join(rng.choices(string.ascii_uppercase, k=2))
    digits = "".join(rng.choices(string.digits, k=7))
    return f"{letters}{digits}"


def _crop_face_from_donor(donor_path: str) -> Image.Image | None:
    donor = cv2.imread(donor_path)
    if donor is None:
        return None
    bbox = _detect_photo_bbox(donor)
    if bbox is None:
        return None
    x, y, w, h = bbox
    # No margin (or a tiny one): the donor images are full ID-card scans, not
    # clean portraits, so the Haar box already tightly bounds just the face —
    # a bigger margin (tried 0.2 first) pulled in visible fragments of the
    # donor card's own printed text/background sitting right next to the photo
    # region, which made the synthetic card look obviously wrong on inspection.
    margin_x, margin_y = int(w * 0.05), int(h * 0.05)
    x0, y0 = max(0, x - margin_x), max(0, y - margin_y)
    x1, y1 = min(donor.shape[1], x + w + margin_x), min(donor.shape[0], y + h + margin_y)
    face_bgr = donor[y0:y1, x0:x1]
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_rgb)


def generate_synthetic_id(donor_face_path: str, out_dir: str, seed: int | None = None,
                           canvas_size: tuple[int, int] = (1600, 1000)) -> dict:
    """Generates one fully synthetic ID card image plus its (fabricated)
    ground-truth fields. `donor_face_path` supplies the photo (see module
    docstring for why cropping a real-dataset face is an acceptable stand-in
    for a generative one here).
    """
    rng = random.Random(seed)
    w, h = canvas_size
    canvas = Image.new("RGB", (w, h), color=(238, 236, 228))
    draw = ImageDraw.Draw(canvas)

    # Loose guilloche-style security background: overlapping fine diagonal lines,
    # not a cryptographically-real pattern, just enough texture that the card
    # doesn't look like a blank form.
    for i in range(-h, w, 14):
        draw.line([(i, 0), (i + h, h)], fill=(222, 219, 210), width=1)
    for i in range(0, w + h, 18):
        draw.line([(i, 0), (i - h, h)], fill=(226, 223, 214), width=1)

    country = rng.choice(_FAKE_COUNTRIES)
    first_name, last_name = _random_name(rng)
    dob = _random_date(rng, 1955, 2005)
    issue_date = _random_date(rng, 2018, 2023)
    expiry = _random_date(rng, 2026, 2033)
    id_number = _random_id_number(rng)

    title_font = ImageFont.truetype(str(FONT_REGULAR), 42)
    label_font = ImageFont.truetype(str(FONT_REGULAR), 20)
    value_font = ImageFont.truetype(str(FONT_MONO), 32)

    draw.text((40, 30), country, font=title_font, fill=(30, 30, 90))
    draw.text((40, 90), "IDENTITY CARD", font=title_font, fill=(120, 20, 20))

    fields = [
        ("SURNAME", last_name), ("GIVEN NAME", first_name), ("DATE OF BIRTH", dob),
        ("DOCUMENT NO.", id_number), ("DATE OF ISSUE", issue_date), ("DATE OF EXPIRY", expiry),
    ]
    text_x, text_y = 40, 220
    for label, value in fields:
        draw.text((text_x, text_y), label, font=label_font, fill=(90, 90, 90))
        draw.text((text_x, text_y + 26), value, font=value_font, fill=(20, 20, 20))
        text_y += 100

    photo_box = (w - 420, 220, w - 60, 220 + 460)
    face = _crop_face_from_donor(donor_face_path)
    if face is not None:
        face_resized = face.resize((photo_box[2] - photo_box[0], photo_box[3] - photo_box[1]))
        canvas.paste(face_resized, (photo_box[0], photo_box[1]))
    draw.rectangle(photo_box, outline=(120, 120, 120), width=2)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"synthetic_{id_number}_tier4.jpg"
    canvas.save(dest, quality=95)

    ground_truth = {
        "name": f"{first_name} {last_name}", "dob": dob, "id_number": id_number,
        "address": None, "expiry": expiry,
    }
    return {
        "donor_face_image": donor_face_path,
        "forged_image": dest.as_posix(),
        "tier": "tier4_full_synthetic",
        "success": face is not None,
        "ground_truth": ground_truth,
        # Whole-image forgery, like Tier 5 — no single localized tamper_regions bbox.
    }


def synthetic_id_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                           limit: int | None = None) -> list[dict]:
    """Generates one fully synthetic ID per donor face drawn from the genuine
    manifest (donor supplies only the face crop — see module docstring).
    """
    import json

    manifest = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"][:limit] if limit else manifest["entries"]

    results = []
    for i, entry in enumerate(entries):
        result = generate_synthetic_id(entry["path"], out_dir, seed=seed + i)
        results.append(result)
        status = "ok" if result["success"] else "skipped (no donor face detected)"
        logger.info(f"[{i+1}/{len(entries)}] donor={entry['path']}: {status}")

    n_success = sum(r["success"] for r in results)
    out_manifest_path = Path(out_dir) / "tier4_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_attempted": len(results), "num_success": n_success,
                                              "entries": results}, indent=2), encoding="utf-8")
    logger.info(f"Tier 4 synthetic generation: {n_success}/{len(results)} succeeded -> {out_manifest_path}")
    return results
