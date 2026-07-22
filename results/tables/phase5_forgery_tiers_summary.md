# Phase 5 Summary — Forgery Tiers 3-5

## Tier 5 — recapture simulation (fully local, real data)

`src/data_generation/recapture_sim.py`: moire interference (product of two
close-frequency sinusoidal grids — the actual beat pattern that causes visible
moire), per-channel pixel misregistration, softened contrast/lifted blacks.
Deterministic given a seed (verified by a unit test).

Smoke run: **15/15** genuine images processed successfully. Manifest:
`data/synthetic_forgeries/tier5_recapture/tier5_manifest.json`. Example:
[results/sample_outputs/example_tier5_recapture.jpg](../sample_outputs/example_tier5_recapture.jpg).

## Tier 4 — fully synthetic ID generation (fully local, real data)

`src/data_generation/synthetic_id_gen.py`: procedurally drawn card layout
(security-pattern background, labeled fields) with fabricated identity fields
(small fixed name lists / random dates / random ID numbers — a simple
procedural generator, not a learned identity model, by design). The photo is a
face crop from a genuine MIDV-2020 image via the same Haar-cascade detector
`splice.py` uses — not a privacy shortcut, since MIDV-2020's own faces are
already AI-generated (Generated Photos), so this reuses one kind of synthetic
face for a different kind of synthetic document.

Smoke run: **15/15** succeeded. Manifest:
`data/synthetic_forgeries/tier4_full_synthetic/tier4_manifest.json`.

**Bug found and fixed during this phase:** the first version cropped the donor
face with a 20%-of-bbox margin on every side, intended as headroom for
shoulders/hair. Because donor images are full ID-card scans (not clean
portraits), that margin pulled in visible fragments of the donor card's own
printed text sitting next to the photo region — plainly visible on inspection
of the first generated sample. Fixed by reducing the margin to 5%. Caught by
actually looking at the rendered output, not just checking `success: true` —
another instance (after Phase 2's filename-collision bug) of a result that
"ran successfully" while being visibly wrong.

## Tier 3 — diffusion inpainting (logic-only locally, execution deferred to Kaggle)

`src/data_generation/inpaint_forger.py` splits into two pieces, same pattern as
`sft_train.py` and `clean_eval.py`'s zero-shot baseline:

- `build_inpaint_mask()`: locates the photo region (reusing `splice.py`'s
  Haar-cascade detector) and builds a binary inpainting mask. Pure
  OpenCV/NumPy, no model — **run for real** against a genuine image (mask
  correctly matches the same photo bbox `splice.py` finds on the same image)
  and unit-tested for the graceful-failure path (no detectable face).
- `run_inpainting()`: the actual Stable Diffusion inpainting call
  (`diffusers.StableDiffusionInpaintPipeline`, ~4-5GB, needs a CUDA GPU with
  meaningful VRAM headroom). **Not executed on this machine** — per the Phase 3
  RAM constraint (~7.7GB total, thrashed on a 3B VLM), a multi-GB diffusion
  pipeline would fail the same way. Runs on Kaggle.

## Scale note

Tier 4/5 smoke runs are 15 images each, drawn from the same kind of
smoke-scale slice as Phase 2's Tier 1/2 batches — not the full ~1000-image
genuine manifest. Tier 3 has produced zero forged images so far (mask-building
logic only); the first real Tier 3 forgeries come from the Kaggle run that
also handles SFT training (Phase 4) and the real zero-shot baseline (Phase 3).
