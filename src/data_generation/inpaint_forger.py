"""Tier 3 forgery: diffusion-based (Stable Diffusion) inpainting over targeted
document regions — a step up from Tier 2's splice (literal copy-paste of
another real photo, which leaves a real donor image's noise/lighting
signature) to AI-generated content that blends seamlessly with no second
source image involved at all.

Mask construction (build_inpaint_mask) is pure OpenCV/PIL logic — reuses
splice.py's Haar-cascade photo-region detector — and is exercised locally with
no model. The actual diffusion call (run_inpainting) needs a Stable Diffusion
inpainting pipeline loaded via `diffusers`, which is several GB and — per the
Phase 3 RAM constraint (this machine has ~7.7GB RAM; even a 3B VLM caused
severe thrashing) — is not attempted locally. Same split as sft_train.py:
logic tested here, the actual model call runs on Kaggle.
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.data_generation.splice import _detect_photo_bbox
from src.utils.image_utils import unique_stem
from src.utils.logging_utils import get_logger

logger = get_logger("inpaint_forger")

DEFAULT_PROMPT = ("a professional passport-style photo of a different person's face, "
                   "neutral expression, plain background, photorealistic")


def build_inpaint_mask(image_path: str) -> dict:
    """Locates the document's photo region and builds a binary mask (255 =
    inpaint, 0 = keep) covering it. Returns the mask as a numpy array alongside
    the bbox, rather than only a file path, so callers (including tests) can
    inspect it without a round-trip through disk.
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    bbox = _detect_photo_bbox(img)
    if bbox is None:
        return {"success": False, "reason": "no photo region detected", "mask": None, "bbox_xyxy": None}

    x, y, w, h = bbox
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255
    return {"success": True, "mask": mask, "bbox_xyxy": [x, y, x + w, y + h], "image_shape": img.shape[:2]}


def run_inpainting(image_path: str, mask: np.ndarray, out_dir: str, prompt: str = DEFAULT_PROMPT,
                    model_name: str = "runwayml/stable-diffusion-inpainting",
                    num_inference_steps: int = 30, seed: int | None = None) -> dict:
    """Runs the actual Stable Diffusion inpainting pass. Requires `diffusers`
    and a GPU with enough VRAM to hold the pipeline — this is the part of Tier
    3 that only runs on Kaggle (see module docstring).
    """
    import torch
    from diffusers import StableDiffusionInpaintPipeline

    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_name, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed) if seed is not None else None

    image = Image.open(image_path).convert("RGB")
    mask_image = Image.fromarray(mask).convert("L")
    result = pipe(prompt=prompt, image=image, mask_image=mask_image,
                  num_inference_steps=num_inference_steps, generator=generator).images[0]

    # runwayml/stable-diffusion-inpainting's built-in safety checker replaces
    # the output with an all-black image on a (frequently false-positive, per
    # the diffusers project's own known behavior) NSFW flag — found for real
    # on this project's first Kaggle run: 1/15 source images (a face-region
    # inpaint, the exact kind of content this safety checker is prone to
    # over-triggering on) came back as a genuinely blank frame despite the
    # prompt being an entirely benign "professional passport-style photo".
    # Detected and rejected here (real success rate reported honestly) rather
    # than disabling the safety checker outright, which would trade a
    # legitimate safety mechanism for convenience on what's ultimately a rare
    # failure mode with a simple check.
    result_arr = np.asarray(result)
    if result_arr.max() == 0:
        return {"source_image": image_path, "success": False,
                "reason": "SD safety checker returned an all-black image (likely a false-positive NSFW flag)"}

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    dest = out_path / f"{unique_stem(image_path)}_tier3.jpg"
    result.save(dest, quality=95)
    return {"source_image": image_path, "forged_image": dest.as_posix(), "tier": "tier3_inpainting", "success": True}


def inpaint_region(image_path: str, out_dir: str, prompt: str = DEFAULT_PROMPT,
                    model_name: str = "runwayml/stable-diffusion-inpainting", seed: int | None = None) -> dict:
    """End-to-end Tier 3 entrypoint: build the mask locally, then run the
    diffusion pass (Kaggle only — see module docstring).
    """
    mask_result = build_inpaint_mask(image_path)
    if not mask_result["success"]:
        return {"source_image": image_path, "success": False, "reason": mask_result["reason"]}

    result = run_inpainting(image_path, mask_result["mask"], out_dir, prompt=prompt,
                             model_name=model_name, seed=seed)
    result["bbox_xyxy"] = mask_result["bbox_xyxy"]
    return result


def inpaint_manifest(genuine_manifest_path: str, out_dir: str, seed: int = 42,
                      limit: int | None = None) -> list[dict]:
    """Mirrors field_tamper.tamper_manifest / splice.splice_manifest's pattern for
    consistency across tiers. `limit` caps how many genuine images get processed
    (default 15 in the Kaggle driver, matching Tier 2/4/5's scale for a fair
    per-tier comparison). The actual diffusion pass (inpaint_region ->
    run_inpainting) needs a GPU, so real runs happen on Kaggle — same split as
    sft_train.py. Local calls only exercise this with inpaint_region mocked
    (see tests/test_pipeline_smoke.py).
    """
    manifest = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    entries = manifest["entries"][:limit] if limit else manifest["entries"]
    results = []
    for i, entry in enumerate(entries):
        result = inpaint_region(entry["path"], out_dir, seed=seed + i)
        result["split"] = entry["split"]
        results.append(result)
        status = "ok" if result["success"] else f"skipped ({result.get('reason')})"
        logger.info(f"[{i+1}/{len(entries)}] {entry['path']}: {status}")

    n_success = sum(r["success"] for r in results)
    out_manifest_path = Path(out_dir) / "tier3_manifest.json"
    out_manifest_path.write_text(json.dumps({"num_attempted": len(results), "num_success": n_success,
                                              "entries": results}, indent=2), encoding="utf-8")
    logger.info(f"Tier 3 inpainting: {n_success}/{len(results)} succeeded -> {out_manifest_path}")
    return results
