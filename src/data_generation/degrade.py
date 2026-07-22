"""Degradation pipeline: blur, rotation, glare, JPEG compression, low-light for
robustness eval sets (src/eval/degraded_eval.py consumes the output).

Applied to genuine AND forged images alike — degradation and forgery are
orthogonal axes (a tampered doc can also be blurry; a clean scan can still be a
forgery), so this runs as a separate pass over whatever manifest of images
(genuine or synthetic_forgeries/*) it's pointed at.
"""

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("degrade")

DEGRADATION_KINDS = ["blur", "rotation", "glare", "compression", "low_light"]


def _apply_blur(img: np.ndarray, rng: random.Random) -> np.ndarray:
    ksize = rng.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(img, (ksize, ksize), 0)


def _apply_rotation(img: np.ndarray, rng: random.Random) -> np.ndarray:
    angle = rng.uniform(-8, 8)  # small realistic capture-angle skew, not a flip/tumble
    h, w = img.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _apply_glare(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulates a specular highlight (phone-camera flash / light reflection off
    a laminated ID) as a soft bright ellipse blended additively.
    """
    h, w = img.shape[:2]
    overlay = np.zeros((h, w), dtype=np.uint8)
    cx, cy = rng.randint(0, w), rng.randint(0, h)
    axis = (rng.randint(w // 6, w // 3), rng.randint(h // 6, h // 3))
    cv2.ellipse(overlay, (cx, cy), axis, rng.uniform(0, 360), 0, 360, 255, -1)
    overlay = cv2.GaussianBlur(overlay, (51, 51), 0)
    overlay_f = (overlay.astype(np.float32) / 255.0)[..., None] * rng.uniform(0.4, 0.8)
    glared = img.astype(np.float32) + overlay_f * 255.0
    return np.clip(glared, 0, 255).astype(np.uint8)


def _apply_compression(img: np.ndarray, rng: random.Random) -> np.ndarray:
    quality = rng.randint(15, 40)  # aggressive JPEG artifacting, not a mild resave
    ok, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def _apply_low_light(img: np.ndarray, rng: random.Random) -> np.ndarray:
    gamma = rng.uniform(2.0, 3.5)  # >1 darkens; simulates dim indoor capture
    darkened = np.clip((img.astype(np.float32) / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 6, img.shape) if hasattr(rng, "normal") else np.zeros_like(darkened)
    return np.clip(darkened.astype(np.float32) + noise, 0, 255).astype(np.uint8)


_APPLIERS = {
    "blur": _apply_blur,
    "rotation": _apply_rotation,
    "glare": _apply_glare,
    "compression": _apply_compression,
    "low_light": _apply_low_light,
}


def degrade_image(image_path: str, degradation_config: dict, out_dir: str) -> dict:
    """Applies one randomly-chosen degradation kind (or a fixed `kind` from
    degradation_config) to image_path and writes the result to out_dir.

    Returns a manifest entry recording exactly what was applied, so degraded_eval.py
    can break results down by degradation kind rather than reporting one pooled number.
    """
    rng = random.Random(degradation_config.get("seed"))
    kind = degradation_config.get("kind") or rng.choice(DEGRADATION_KINDS)
    if kind not in _APPLIERS:
        raise ValueError(f"Unknown degradation kind '{kind}', expected one of {DEGRADATION_KINDS}")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # low_light's numpy-based noise needs a numpy Generator, not random.Random —
    # keep both seeded from the same value for reproducibility.
    np_rng = np.random.default_rng(degradation_config.get("seed"))
    degraded = _APPLIERS[kind](img, np_rng if kind == "low_light" else rng)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(image_path)}_{kind}.jpg"
    cv2.imwrite(str(dest), degraded)

    return {
        "source_image": image_path,
        "degraded_image": str(dest),
        "degradation_kind": kind,
    }


def degrade_manifest(manifest_path: str, out_dir: str, kinds: list[str] | None = None,
                      seed: int = 42) -> list[dict]:
    """Applies every kind in `kinds` (default: all 5) to every genuine image listed
    in a manifest produced by acquire_dataset.build_genuine_manifest, and writes a
    combined degraded-set manifest to out_dir/degraded_manifest.json.
    """
    manifest = json.loads(Path(manifest_path).read_text())
    kinds = kinds or DEGRADATION_KINDS
    results = []
    for i, entry in enumerate(manifest["entries"]):
        for kind in kinds:
            result = degrade_image(
                entry["path"],
                {"kind": kind, "seed": seed + i},
                out_dir,
            )
            result["document_code"] = entry["document_code"]
            result["split"] = entry["split"]
            results.append(result)

    out_manifest_path = Path(out_dir) / "degraded_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_images": len(results), "entries": results}, indent=2))
    logger.info(f"Degraded {len(manifest['entries'])} source images x {len(kinds)} kinds "
                f"= {len(results)} outputs -> {out_manifest_path}")
    return results
