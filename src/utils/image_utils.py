"""Shared image loading/preprocessing helpers used across data_generation, eval, and app."""

from pathlib import Path

import cv2


def load_image(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def unique_stem(image_path: str) -> str:
    """Parent-dir + filename stem, e.g. 'lva_passport_00' for
    '.../images/lva_passport/00.jpg'.

    MIDV-2020 (and our own generated tiers) reuse plain numeric filenames
    (00.jpg..99.jpg) across every document code, so the bare stem alone collides
    across codes — found the hard way when degrade_manifest() silently
    overwrote 375 of 875 outputs because two document codes both had a '00.jpg'.
    Every data_generation script should build output filenames from this, not
    from Path(image_path).stem directly.
    """
    p = Path(image_path)
    return f"{p.parent.name}_{p.stem}"
