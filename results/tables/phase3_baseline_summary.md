# Phase 3 baseline summary

Smoke-scale local run — see the scale note at the bottom before treating any
number here as final.

## OCR baseline (EasyOCR, rule-based field mapping)

10 genuine test-split images, drawn across the full 10-document-code
MIDV-2020 sample (not just the 2 codes used in Phase 2's smoke run). Full
results in `results/tables/ocr_baseline.json`.

| Field | Fuzzy similarity | Exact match |
|---|---|---|
| name | 0.60 | 0% |
| dob | 1.00 | 100% |
| id_number | 1.00 | 100% |
| expiry | 0.89 | 0% |

Address not scored — none of the 10 sampled images had a non-null
ground-truth address (only `srb_passport`, 1 of 10 codes, prints one).

dob/id_number look perfect at this sample size, which is itself worth being
suspicious of — probably just reflects that these two fields are the
easiest OCR case on clean, high-res template scans, and n=10 is too small
to trust a 1.00 as a real ceiling. name's 0% exact-match despite 0.60
similarity isn't noise, it's structural: the label-proximity heuristic in
`ocr_baseline.py` finds a name-like label but only returns one of {given
name, surname}, so it can never exactly match the concatenated ground truth
by construction. Exactly the kind of thing a fine-tuned VLM should fix,
since it understands "the name" as a concept rather than a text-proximity
rule.

## Zero-shot VLM baseline (Qwen2.5-VL-7B-Instruct, no fine-tuning)

Ran as part of kernel v24's Phase 3 step, same session that produced Phase
4's first completed SFT run — see `phase4_sft_summary.md`. 9 examples
(3 genuine + 3 tier1 + 3 tier2, same scale caveat as the OCR baseline
above). Full raw output in `results/tables/phase3_clean_eval_baseline_7b.json`.

| Metric | Value | 95% CI | n |
|---|---|---|---|
| Parse success rate | 100% | [100%, 100%] | 9 |
| Tamper-verdict accuracy | 66.7% | [33.3%, 100%] | 9 |
| Field similarity (name/dob/id_number/expiry) | 1.00 each | [1.00, 1.00] | 3 each |

The 66.7% headline hides the more interesting split: the model caught all 6
real forgeries (tier1 + tier2, 6/6) but flagged all 3 genuine documents as
tampered too (0/3), hallucinating a specific-sounding but false tamper cue
each time — one example claimed "the ID number appears to be edited, as it
is not consistent with the format typically used on Albanian IDs" about an
untouched ID number. High recall, high false-positive rate: zero-shot 7B
never misses a forgery in this small sample, but it's trigger-happy on
genuine documents and invents plausible justifications instead of
abstaining when unsure. Worth checking whether the fine-tuned 3B model
(actually trained on genuine-vs-tampered examples) shows the same bias or
corrects it — see `phase6_adversarial_rounds_summary.md` for how that
turned out.

CIs are wide at n=9 (or n=3 per field) — this is a smoke-scale sample, not
a statistically powered baseline, see the scale note below.

This dev machine has ~7.7GB total RAM (often under 1GB free at idle). Tried
loading Qwen2.5-VL-3B locally once, and it wasn't just slow — it caused
severe disk thrashing (free RAM collapsed to ~0.5-0.8GB, resident memory
oscillating while committed memory kept climbing) and had to be killed
before it risked destabilizing the machine. bf16 would roughly halve the
memory need but still leaves next to no margin against a ceiling that
already has under 1GB free before any model loads. Not worth the risk
compared to just running on Kaggle, which has enough RAM/VRAM to load
either size without a fight — this is now a standing rule for every later
phase, no model loads locally on this machine at any size.

The 7B baseline stayed at the same 9-example smoke scale as the local OCR
run rather than scaling up before Phase 4 started, since Kaggle time went
to getting SFT training working first (see the debugging story in
`phase4_sft_summary.md`).

## Scale note

Both the OCR baseline (10 examples) and the zero-shot 7B baseline (9
examples) are smoke-scale, sized for quick local iteration rather than as
final reported numbers. A larger, randomly-drawn evaluation split for the
7B baseline would be a reasonable follow-up with more Kaggle time. Every
result file `clean_eval.py` writes is labeled with the exact `model_name`
and `device` used, so a smoke number can't get mistaken for a properly
sized one later.
