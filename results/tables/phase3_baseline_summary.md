# Phase 3 Baseline Summary

*Smoke-scale local run — see "Scale note" at the bottom before treating any
number here as a final reported result.*

## OCR baseline (EasyOCR, rule-based field mapping)

10 genuine test-split images, drawn across the full 10-document-code MIDV-2020
sample (not just the 2 codes used in Phase 2's smoke run). Full results in
`results/tables/ocr_baseline.json`.

| Field | Fuzzy similarity | Exact match |
|---|---|---|
| name | 0.60 | 0% |
| dob | 1.00 | 100% |
| id_number | 1.00 | 100% |
| expiry | 0.89 | 0% |

Address not scored — none of the 10 sampled images had a non-null ground-truth
address (only `srb_passport`, 1 of 10 document codes, prints one).

Reading these honestly: dob/id_number look perfect at this sample size, which is
itself worth being suspicious of — it likely reflects that these two fields are
the easiest OCR case on clean, high-resolution template scans (large, isolated
printed digits) and n=10 is too small to trust a "1.00" as a real ceiling. name's
0% exact-match despite 0.60 similarity is a structural OCR-baseline limitation,
not noise: the label-proximity heuristic in `ocr_baseline.py` finds a name-like
label but returns only one of {given name, surname}, so it can never exactly
match the concatenated ground truth by construction — this is exactly the kind
of thing a fine-tuned VLM (which understands "the name" as a semantic concept,
not a text-proximity rule) should fix.

## Zero-shot VLM baseline (Qwen2.5-VL, no fine-tuning) — deferred to Kaggle

No real numbers here yet, and that's a deliberate outcome of this phase, not an
oversight. `src/eval/clean_eval.py` is fully implemented (prompt, JSON-parsing,
per-field similarity/exact-match, tamper-verdict accuracy, all reported through
`bootstrap_ci`) and its pure logic is unit-tested
(`test_extract_json_handles_markdown_fenced_output`,
`test_extract_json_returns_none_on_unparseable_output`,
`test_build_eval_sample_balances_categories`) — but it has not been run against
a real model on this machine.

**Why:** this dev machine has only ~7.7GB total RAM (often <1GB free at idle).
Qwen2.5-VL-3B-Instruct was downloaded (7.1GB, real weights, verified complete)
and a first inference attempt was made — it did not just run slowly, it caused
severe disk thrashing (system free RAM collapsed to ~0.5-0.8GB, the process's
resident memory oscillated erratically while committed/paged memory kept
climbing) and had to be killed before it risked destabilizing the machine.
Switching to bf16 would roughly halve the memory need (~6-7GB) but still leaves
next to no margin against a 7.7GB ceiling that already has <1GB free before any
model is even loaded. Retrying was judged not worth the risk versus simply
running on Kaggle, which has enough RAM/VRAM to load either 3B or 7B without a
fight. This is now a documented standing constraint (README.md,
`config/training_config.yaml`) for every later phase: no model ever loads
locally on this machine, at any size.

**What "done" looks like for this phase without the VLM number:** OCR baseline
run for real (above), VLM eval script fully built and logic-tested, real VLM
baseline numbers explicitly deferred to the first Kaggle session (which Phase 4
needs anyway for SFT training) rather than faked or hand-waved locally.

## Scale note

OCR baseline ran on 10 examples — smoke-scale, sized for local CPU iteration
speed, not a final reported number. The real reported zero-shot baseline uses
Qwen2.5-VL-**7B**-Instruct (the actual fine-tuning target in
`config/model_config.yaml`), evaluated over a much larger, randomly-drawn split,
run on Kaggle GPU. Every result file `clean_eval.py` writes is labeled with the
exact `model_name` and `device` used so a smoke number can never be mistaken for
a reported one.
