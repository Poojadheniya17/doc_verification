"""Tier 5 forgery: screen-recapture / moire pattern simulation on an otherwise
genuine document — the "fraud" here is presenting a photo of a screen showing
the document (or a printed copy) instead of the physical document itself, which
real verification systems must catch even though every field on it is
genuinely correct (nothing was edited — the whole document is a re-photographed
copy). Escalates past Tier 1-2 (localized edits) and previews Tier 3-4
(no localized tamper region at all — the entire image is the "forgery").

Pure image processing (no model), unlike Tier 3 (diffusion inpainting).
"""

import random
from pathlib import Path

import cv2
import numpy as np

from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("recapture_sim")


def _moire_pattern(shape: tuple[int, int], rng: random.Random) -> np.ndarray:
    """A fine, slightly-rotated sinusoidal grid — approximates the aliasing
    between a display's subpixel grid and a camera sensor's own pixel grid,
    which is what produces visible moire when photographing a screen.
    """
    h, w = shape
    freq = rng.uniform(0.35, 0.55)  # cycles/pixel — high enough to alias against a photographed sensor grid
    angle = rng.uniform(0, np.pi)
    y, x = np.mgrid[0:h, 0:w]
    grid = np.sin(2 * np.pi * freq * (x * np.cos(angle) + y * np.sin(angle)))
    grid2 = np.sin(2 * np.pi * (freq * 1.07) * (x * np.cos(angle + 0.3) + y * np.sin(angle + 0.3)))
    interference = grid * grid2  # product of two close frequencies -> beat pattern, the actual moire
    return interference.astype(np.float32)


def simulate_recapture(image_path: str, out_dir: str, seed: int | None = None) -> dict:
    rng = random.Random(seed)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    h, w = img.shape[:2]

    moire = _moire_pattern((h, w), rng)
    moire_strength = rng.uniform(6, 14)  # intensity units on a 0-255 scale — subtle, not a checkerboard
    img_f = img.astype(np.float32) + moire[..., None] * moire_strength

    # Slight per-channel offset: recapturing a screen through a camera sensor
    # rarely reproduces colors in perfect registration across channels.
    for c in range(3):
        dx, dy = rng.randint(-1, 1), rng.randint(-1, 1)
        img_f[..., c] = np.roll(np.roll(img_f[..., c], dx, axis=1), dy, axis=0)

    # Screen recapture softens fine detail and slightly lifts blacks / compresses
    # contrast (backlight glow), unlike degrade.py's "low_light" (which darkens).
    img_f = cv2.GaussianBlur(img_f, (3, 3), 0)
    contrast = rng.uniform(0.85, 0.95)
    brightness_lift = rng.uniform(5, 15)
    img_f = img_f * contrast + brightness_lift

    result = np.clip(img_f, 0, 255).astype(np.uint8)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(image_path)}_tier5.jpg"
    cv2.imwrite(str(dest), result)

    return {
        "source_image": image_path,
        "forged_image": str(dest),
        "tier": "tier5_recapture",
        "success": True,
        # No localized tamper_regions — the whole document is the "forgery" here,
        # unlike Tier 1/2's bbox-localized edits.
    }


def recapture_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                        limit: int | None = None) -> list[dict]:
    import json

    manifest = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"][:limit] if limit else manifest["entries"]

    results = []
    for i, entry in enumerate(entries):
        result = simulate_recapture(entry["path"], out_dir, seed=seed + i)
        result["split"] = entry["split"]
        results.append(result)
        logger.info(f"[{i+1}/{len(entries)}] {entry['path']}: ok")

    out_manifest_path = Path(out_dir) / "tier5_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_attempted": len(results), "num_success": len(results),
                                              "entries": results}, indent=2), encoding="utf-8")
    logger.info(f"Tier 5 recapture: {len(results)}/{len(results)} succeeded -> {out_manifest_path}")
    return results
