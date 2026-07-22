# Phase 4 SFT Summary

## What was built

- `src/training/sft_train.py`: builds SFT training examples from the genuine +
  Tier 1/2 manifests, formats them as Qwen2.5-VL chat conversations (image +
  extraction/tamper-verdict/localization JSON target), and — when
  `environment=kaggle` — loads Qwen2.5-VL-7B-Instruct in 4-bit (QLoRA per
  `config/model_config.yaml`), wraps it with a LoRA adapter, and fine-tunes via
  `transformers.Trainer`.
- `src/training/checkpoint_utils.py`: adapter-only save/load (QLoRA only trains
  the LoRA weights, so checkpoints are a few tens of MB, not the full base
  model) plus step-numbered checkpoint discovery for resuming.

## What was actually run locally (and what wasn't)

Per the standing RAM constraint (see `results/tables/phase3_baseline_summary.md`,
README.md), **no model loading happens on this machine**. What *was* run for
real, locally:

- `build_sft_examples()` against the real Phase 2/3 manifests: **719 training
  examples** (700 genuine + 8 Tier 1 + 11 Tier 2, `split="train"`) built and
  inspected — real data, real JSON targets, zero mocking.
- `python -m src.training.sft_train --environment local` end-to-end: builds the
  719 examples, then exits cleanly *before* touching any model weights (see the
  `environment == "local"` early-return in `train()`).
- Unit tests (`tests/test_pipeline_smoke.py`) cover the pure logic: Tier 1
  field-override matching (does a tamper's OCR text get correctly attributed to
  the right schema field, or correctly left alone if it hits something outside
  the 5 tracked fields like an MRZ line), example construction from fixture
  manifests, conversation formatting, and checkpoint-directory discovery.

What was **not** run: `load_model_for_training()`, the actual QLoRA fine-tuning
loop, and anything touching real Qwen2.5-VL weights. That happens on Kaggle.

## Honest note on the Tier-1 field-override heuristic

`_apply_tier1_field_overrides()` matches a tamper's OCR text back to one of the
5 schema fields by fuzzy string similarity (threshold 0.6). Verified on a real
example: a genuine expiry `"04.11.2026."` tampered by `field_tamper.py` into
`"14.16.0026 ."` was correctly matched and substituted into the `expiry`
field, while a separate tamper on the MRZ line was correctly left unmatched
(not one of the 5 fields). This is a known simplification, not a guarantee —
short or heavily garbled OCR text could fail to clear the similarity threshold
even when it did hit a real field, silently leaving that field at its
pre-tamper value in the training target. Worth re-checking once real training
numbers come back from Kaggle if extraction accuracy on Tier-1 examples looks
worse than expected.

## Scale note

719 examples come from the same smoke-scale Tier 1 (10 images) / Tier 2 (15
images) batches generated in Phase 2 — not the full ~1000-genuine-image
manifest run through all forgery tiers. Real Kaggle training should regenerate
Tier 1/2 at full scale first (`tamper_manifest`/`splice_manifest` without a
`limit`) for a properly sized training set.
