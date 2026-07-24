# Phase 6: Adversarial Retraining Rounds — Real Results

Real Kaggle run (kernel `doc-verification-adversarial-rounds`, dataset version
with the manifest/path-separator fixes), 3 rounds against a fixed 30-example
eval set (10 genuine + 20 tampered across tier1/tier2/tier4/tier5, per
`adversarial_rounds.build_eval_set()`).

## The headline number, reported honestly

| Round | Retrained on | Accuracy | 95% CI | Predicted-verdict distribution |
|---|---|---|---|---|
| 0 (existing v24 checkpoint, no retrain) | — | **33.3%** | [16.7%, 50.0%] | genuine: 25, *unparseable*: 5, tampered: **0** |
| 1 | 20 failures mined from round 0 | **66.7%** | [50.0%, 83.3%] | tampered: **30**, genuine: 0 |
| 2 | 10 failures mined from round 1 | **33.3%** | [16.7%, 50.0%] | genuine: **30**, tampered: 0 |

n=30 every round (fixed eval set, unchanged across rounds by design).
Bootstrap 95% CIs are wide at this sample size — stated plainly, not hidden.

## This is not a generalization curve — it's a collapsing classifier

The verdict-distribution column is the real story, not just the accuracy
number. **At no point across any of the 3 rounds did the model produce a
mixed genuine/tampered verdict distribution.** Every round, it predicted a
single class for all 30 examples (with 5 unparseable responses in round 0):

- Round 0: always "genuine" (or fails to parse) → gets the 10 actually-genuine
  examples right by coincidence, gets all 20 actually-tampered examples wrong.
  33.3% = 10/30, exactly what "always predict genuine" scores on this set.
- Round 1: retrained on round 0's 20 mined failures (all 20 were the
  actually-tampered examples it called "genuine") → the retrain didn't teach
  it to *discriminate* genuine from tampered, it taught it to always say
  "tampered" instead. 66.7% = 20/30 — better only because "always tampered"
  happens to score higher on a 20-tampered/10-genuine set than "always
  genuine" does, not because it learned anything about *why* an image is
  tampered.
- Round 2: retrained on round 1's 10 newly-mined failures (the 10
  actually-genuine examples it now incorrectly called "tampered") → flips
  straight back to always predicting "genuine". 33.3% again — back to
  exactly round 0's number.

**Honest conclusion: this adversarial retraining loop, at this scale, is not
producing a more robust model.** It is oscillating a single global bias
(genuine ↔ tampered) between two degenerate single-class predictors,
entirely determined by whichever tiny (10-20 example) mined-failure batch it
saw most recently. There is no evidence across any of these 3 rounds that
the model developed a real decision boundary that separates genuine from
tampered content based on image evidence.

## Why, honestly — ties directly to the loss-plateau finding

This result is consistent with, and gives real evidence *for*, the
hypothesis flagged in `PROJECT_STATUS.md`/`phase4_sft_summary.md` about
v24's training: the loss plateau in epochs 2-3 (stuck ~1.25-1.26 after a
fast drop in epoch 1) was flagged as a possible sign the model reached its
effective capacity given **LoRA rank 4** (cut for memory reasons during the
v19-v23 OOM chain, not because r=4 was judged sufficient) and the small
**719-example** training set. A model with genuinely limited capacity to
represent the tamper-detection task, combined with retraining rounds on
extremely small (10-20 example) batches, is exactly the failure mode that
produces this kind of oscillating single-class collapse rather than
incremental improvement — each tiny retrain has enough gradient signal to
flip the whole adapter's global bias, but not enough signal (or capacity) to
learn a real, generalizing distinction.

## What this means for the rest of the project (documented decision)

This is being reported as a real, weak, and somewhat concerning result — not
smoothed over. It directly informs:
- The writeup's Limitations section must lead with this, not bury it: the
  trained model's real tamper-detection capability, at the config forced by
  this project's compute constraints (LoRA r=4, 256px images, 719 training
  examples), does not show evidence of the kind of image-content-based
  discrimination the whole system is designed around.
- Leave-one-out's results (once available) are the more important read on
  actual generalization capability, since each fold is a genuine ~719-example
  retrain from the base model, not a 10-20-example perturbation.
- If this pattern also shows up in leave-one-out (e.g., near-chance accuracy,
  single-class collapse), the honest conclusion for the final writeup is that
  meaningfully more training data and/or a higher LoRA rank — both cut for
  Kaggle T4 memory reasons documented extensively in `phase4_sft_summary.md`
  — are the primary, well-evidenced levers for a stronger model, not further
  algorithmic changes to the adversarial-rounds or leave-one-out procedures
  themselves.

## v25 capacity-only re-test: identical collapse, real evidence against capacity alone

Once v25 (LoRA r=16, a real, verified capacity-restoration checkpoint — see
`phase4_sft_summary.md`'s "v25" section) completed, this exact adversarial-rounds
procedure was re-run against it as Round 0's baseline, holding everything else
constant — same fixed 30-example eval set, same tier1+tier2-only, ~700
genuine/19 tampered (~35:1) training-data composition as v24. This isolates
one question: does more capacity alone (without touching the class imbalance)
fix the collapse?

**No. The result is essentially identical to v24's, verified down to the
real per-example verdict distribution, not just the aggregate accuracy:**

| Round | Retrained on | Accuracy | Predicted-verdict distribution (real, from raw per-example output) |
|---|---|---|---|
| 0 (v25 checkpoint, r=16, no retrain) | — | **33.3%** | genuine: **30/30**, tampered: 0 |
| 1 | 20 failures mined from round 0 | **66.7%** | tampered: **30/30**, genuine: 0 |
| 2 | 10 failures mined from round 1 | **33.3%** | genuine: **30/30**, tampered: 0 |

The accuracy sequence (33.3% -> 66.7% -> 33.3%) matches v24's run exactly,
and — checked directly against the real per-example output this time, not
assumed from the aggregate number — every single round is still a pure
single-class predictor across all 30 examples, with the same genuine ->
tampered -> genuine oscillation pattern as v24.

**Honest conclusion: capacity alone (LoRA r=16 vs r=4) does not fix the
collapse.** This is real, direct evidence for the class-imbalance hypothesis
over the capacity hypothesis — the same severely imbalanced training data
(700 genuine vs 19 tampered, ~35:1) produces the identical failure mode
regardless of how much LoRA capacity is available to the adapter. It does
not rule out capacity as *a* contributing factor entirely (v25's own
training-loss plateau was measurably lower than v24's, a real, separate
positive signal — see `phase4_sft_summary.md`), but it demonstrates capacity
is not sufficient on its own to fix the collapse, which is the more decision-
relevant finding for this project's remaining scope. This directly motivates
and justifies the next real experiment: class-balanced training (oversample
tampered examples to 1:1 — see `sft_train.balance_examples()`), tested next
while holding the safer, extensively-validated r=8/512px config constant to
avoid conflating the balance variable with v25's own unresolved memory
anomaly.
