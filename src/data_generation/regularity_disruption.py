"""Regularity-disruption augmentation: a tier-agnostic tampering signal,
generated from genuine documents, that deliberately does NOT resemble any of
this project's 5 scripted tampering tiers.

Real motivation: the class-balanced checkpoint (v26) generalizes poorly to a
tampering category held entirely out of training (leave-one-out fold 2,
tier4_full_synthetic: 0.000% accuracy, a total collapse -- see
results/tables/phase6_leave_one_out_summary.md). The most likely cause,
argued in that writeup, is a genuine data/concept-coverage gap: tiers 1, 2,
3, and 5 all teach the same underlying concept ("spot a local anomaly on a
real template"); tier 4 needs a categorically different concept ("the whole
document can be fake") that the other four can't teach by analogy.

This module is a real, direct application of arXiv:2207.10402 ("Detecting
Deepfake by Creating Spatio-Temporal Regularity Disruption") to this
project's setting. That paper's core idea: train a detector to recognize
"disruption of statistical regularity" as a general signal, synthesized by
deliberately perturbing REAL genuine content -- rather than only training on
real fakes and hoping the detector generalizes past their specific artifact
signatures. The temporal half of that paper (video frame consistency) has no
analogue for single document images; the spatial half translates directly.

Three perturbation kinds are used, chosen specifically because none of them
resembles any existing tier's signature:
- color: a local brightness/contrast/hue shift, inconsistent with the
  surrounding page -- not a text edit (tier1), not a photo splice (tier2),
  not diffusion inpainting (tier3), not a global recapture effect (tier5).
- warp: a local geometric perspective warp of a small region -- introduces
  structural irregularity without altering any printed content, unlike
  tier1's text edits or tier2/3's photo replacement.
- noise_patch: a faint structured interference pattern confined to ONE small
  region -- distinct from tier5_recapture's whole-image moire simulation
  (recapture_sim.py), which affects every pixel, not a bounded patch.

Ground truth: printed fields are never altered, so target fields are carried
over from the source unchanged (same shape as tier2_splicing/tier3_inpainting
in build_sft_examples()) -- only tamper_verdict and a single localized
tamper_regions bbox differ from the genuine source.

Pure image processing (no model), like recapture_sim.py -- runs and is
tested locally; only real SFT training needs Kaggle's GPU.
"""

import json
import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("regularity_disruption")

DISRUPTION_KINDS = ("color", "warp", "noise_patch")

# A random patch this small or smaller wouldn't leave a meaningfully visible
# disruption; this large and the patch risks covering half the document.
_MIN_PATCH_FRAC = 0.15
_MAX_PATCH_FRAC = 0.32


def _random_bbox(w: int, h: int, rng: random.Random) -> tuple[int, int, int, int]:
    """A random rectangular region, fully inside the image, sized as a
    fraction of the image's own dimensions so it scales sensibly across
    MIDV-2020's varying document resolutions.
    """
    pw = int(w * rng.uniform(_MIN_PATCH_FRAC, _MAX_PATCH_FRAC))
    ph = int(h * rng.uniform(_MIN_PATCH_FRAC, _MAX_PATCH_FRAC))
    pw, ph = max(pw, 8), max(ph, 8)
    x0 = rng.randint(0, max(w - pw, 0))
    y0 = rng.randint(0, max(h - ph, 0))
    return x0, y0, x0 + pw, y0 + ph


def _disrupt_color(img: np.ndarray, bbox: tuple[int, int, int, int], rng: random.Random) -> np.ndarray:
    """First attempt at these parameters (brightness ±28, contrast 0.82-1.22,
    tint ±10) was checked visually against a real MIDV-2020 document before
    ever spending GPU time on it -- completely invisible against a
    moderately-detailed background, exactly the kind of silent, useless
    "success: true" this project's own history warns about (the Tier 4
    donor-face-margin bug, the degraded-image filename collision, both only
    caught by looking at actual output). Strengthened until genuinely visible
    on a real document, not just non-crashing.
    """
    x0, y0, x1, y1 = bbox
    patch = img[y0:y1, x0:x1].astype(np.float32)
    brightness = rng.choice([-1, 1]) * rng.uniform(45, 75)
    contrast = rng.choice([rng.uniform(0.45, 0.65), rng.uniform(1.4, 1.7)])
    # A strong per-channel tint (not just brightness) reads as a real local
    # color-consistency violation rather than a uniform lighting change.
    tint = np.array([rng.uniform(-35, 35) for _ in range(3)], dtype=np.float32)
    patch = patch * contrast + brightness + tint
    img = img.copy()
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


def _disrupt_warp(img: np.ndarray, bbox: tuple[int, int, int, int], rng: random.Random) -> np.ndarray:
    """Strengthened for the same reason as _disrupt_color -- the original
    jitter (6-14% of patch size) was checked visually and was not a
    perceptible distortion on a real document.
    """
    x0, y0, x1, y1 = bbox
    pw, ph = x1 - x0, y1 - y0
    patch = img[y0:y1, x0:x1]

    src_pts = np.float32([[0, 0], [pw, 0], [0, ph], [pw, ph]])
    jitter = min(pw, ph) * rng.uniform(0.22, 0.38)
    dst_pts = src_pts + np.float32([
        [rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)
    ])
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(patch, matrix, (pw, ph), borderMode=cv2.BORDER_REPLICATE)

    img = img.copy()
    img[y0:y1, x0:x1] = warped
    return img


def _disrupt_noise_patch(img: np.ndarray, bbox: tuple[int, int, int, int], rng: random.Random) -> np.ndarray:
    """Strengthened for the same reason as _disrupt_color -- the original
    strength (10-22 intensity units) was checked visually and blended
    invisibly into the document's own printed background pattern.
    """
    x0, y0, x1, y1 = bbox
    ph, pw = y1 - y0, x1 - x0
    freq = rng.uniform(0.2, 0.4)
    angle = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:ph, 0:pw]
    grid = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    strength = rng.uniform(55, 90)

    patch = img[y0:y1, x0:x1].astype(np.float32)
    patch += (grid * strength)[..., None]
    img = img.copy()
    img[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


_DISRUPTORS = {"color": _disrupt_color, "warp": _disrupt_warp, "noise_patch": _disrupt_noise_patch}


def generate_regularity_disruption(image_path: str, out_dir: str, seed: int | None = None) -> dict:
    """Applies exactly one randomly-chosen (seeded) disruption kind to one
    random region of a genuine source image. Deterministic given the same
    seed -- same region, same kind, same output bytes.
    """
    rng = random.Random(seed)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    bbox = _random_bbox(w, h, rng)
    kind = rng.choice(DISRUPTION_KINDS)
    result_img = _DISRUPTORS[kind](img, bbox, rng)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(image_path)}_regularity.jpg"
    cv2.imwrite(str(dest), result_img)

    return {
        "source_image": image_path,
        "forged_image": dest.as_posix(),
        "tier": "regularity_disruption",
        "success": True,
        "bbox_xyxy": list(bbox),
        "disruption_kind": kind,
    }


def regularity_disruption_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                                    limit: int | None = None) -> list[dict]:
    """Same shape as recapture_manifest()/other tier manifest builders: reads
    the genuine manifest's entries, applies the disruption to each (any
    split -- this tier is meant to always be present in training, never held
    out, so it isn't scoped to split="train" the way real tampering tiers
    are), writes a manifest.json in the same {"entries": [...]} convention
    build_sft_examples() and _default_tier_manifest_paths() expect.
    """
    manifest = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"][:limit] if limit else manifest["entries"]

    results = []
    for i, entry in enumerate(entries):
        result = generate_regularity_disruption(entry["path"], out_dir, seed=seed + i)
        result["split"] = entry["split"]
        results.append(result)
        logger.info(f"[{i + 1}/{len(entries)}] {entry['path']}: ok ({result['disruption_kind']})")

    out_manifest_path = Path(out_dir) / "regularity_manifest.json"
    out_manifest_path.write_text(
        json.dumps({"num_attempted": len(results), "num_success": len(results), "entries": results}, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Regularity disruption: {len(results)}/{len(results)} succeeded -> {out_manifest_path}")
    return results
