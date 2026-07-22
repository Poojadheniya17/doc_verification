# Phase 2 Data Summary

*Numbers reflect the local CPU smoke-scale run described below, not a full
production dataset — see "Scale note" at the bottom.*

## Genuine base images

- Source: MIDV-2020 `templates.tar` (real public dataset, not synthetic),
  partial download (~150MB of 863MB — see `acquire_dataset.py` docstring for why
  a partial prefix still yields complete, valid images).
- 175 genuine images across 2 document codes (`lva_passport`, `svk_id`)
- Stratified 70/15/15 train/val/test split per document code
  (`data/processed/genuine_manifest_templates.json`)

## Tier 1 — field tampering

- Method: EasyOCR-detected digit-bearing fields (dates, passport/personal
  numbers), 1-2 fields mutated per image, patched with a bundled DejaVuSans font
  (intentional font-mismatch signal)
- Smoke run: 10/10 images succeeded, 2 fields tampered each (20 tampered fields
  total), each with a ground-truth bbox for tamper localization
- Manifest: `data/synthetic_forgeries/tier1_field_tamper/tier1_manifest.json`

## Tier 2 — splicing

- Method: Haar-cascade face/photo detection on source + donor, Poisson blending
  (`cv2.seamlessClone`) with feathered-alpha fallback
- Smoke run: 15/15 pairs succeeded (mix of same-document-code and
  cross-document-code donor pairs)
- Manifest: `data/synthetic_forgeries/tier2_splicing/tier2_manifest.json`
- Observation (worth remembering for later eval): same-code splices blend
  almost seamlessly (background/lighting match); cross-code splices show a
  visible tone mismatch. This predicts cross-code splices will be the easier
  Tier-2 case for the detector and same-code splices the harder one — worth
  breaking out separately in Tier-2 eval rather than reporting one pooled
  Tier-2 accuracy number.

## Degradation pipeline

- All 5 kinds (blur, rotation, glare, compression, low_light) applied to every
  genuine image
- 175 images x 5 kinds = 875 degraded outputs
- Manifest: `data/degraded/degraded_manifest.json`

## Honest failure found + fixed during this phase

The first full run of `degrade_manifest` over all 175 genuine images produced
only **500 of the expected 875** output files. Root cause: MIDV-2020 (and our
own Tier 1/2 generators) reuse plain numeric filenames (`00.jpg`...`99.jpg`)
across every document code, and `degrade_image`'s output path was built from
`Path(image_path).stem` alone — so `lva_passport/00.jpg` and `svk_id/00.jpg`
both wrote to `00_blur.jpg` and silently overwrote each other, for every one of
the 5 degradation kinds. Fixed by adding `src/utils/image_utils.py:unique_stem()`
(parent-dir + stem) and using it in `degrade.py`, `field_tamper.py`, and
`splice.py`'s output filenames; added a regression test
(`test_degrade_image_does_not_collide_across_document_codes`) so this can't
silently regress. Re-run after the fix produced the full 875/875.

This is exactly the kind of thing that would stay invisible at the 2-document-code
smoke scale if you only checked "did the script exit 0" — it only surfaced by
checking the actual output count against the expected count.

## Scale note

This is a smoke-scale run sized for local CPU dev (EasyOCR is ~1.5-2s/image on
CPU here once the reader is warm; the 20s figure in code comments is worst-case
cold-start). A full run would: (1) download the complete `templates.tar` (863MB;
~2-3 min at this session's observed ~4MB/s, but budget more on a slower link),
covering all 10 document codes instead of 2, and (2) run Tier 1/2 generation over
the full ~1000-image manifest rather than a 10/15-image slice. Both scripts
accept the full manifest with no code changes — only the `--max-bytes` /
`limit` arguments used here for the smoke run need to be omitted.
