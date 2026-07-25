# Phase 6: adversarial retraining rounds

Kaggle run (`doc-verification-adversarial-rounds`), 3 rounds against a fixed
30-example eval set (10 genuine + 20 tampered across tier1/tier2/tier4/tier5,
built by `adversarial_rounds.build_eval_set()`).

## v24 baseline: not a generalization curve, a collapsing classifier

| Round | Retrained on | Accuracy | 95% CI | Predicted verdicts |
|---|---|---|---|---|
| 0 (v24 checkpoint, no retrain) | — | 33.3% | [16.7%, 50.0%] | genuine: 25, unparseable: 5, tampered: 0 |
| 1 | 20 failures mined from round 0 | 66.7% | [50.0%, 83.3%] | tampered: 30, genuine: 0 |
| 2 | 10 failures mined from round 1 | 33.3% | [16.7%, 50.0%] | genuine: 30, tampered: 0 |

n=30 every round, same fixed set each time. The CIs are wide at this sample
size, worth keeping in mind before reading much into the exact percentages.

The verdict column is the real story, not the accuracy number. In every
round the model predicted a single class for all 30 examples:

- Round 0 defaults to "genuine" for basically everything, so it gets the 10
  actually-genuine examples right by coincidence and misses all 20 tampered
  ones. 33.3% is exactly what "always say genuine" scores on this set.
- Round 1 gets retrained on those 20 missed tampered examples and swings to
  always predicting "tampered" instead — not discrimination, just a flipped
  global bias. 66.7% is what "always say tampered" scores on a
  20-tampered/10-genuine set.
- Round 2 retrains on round 1's new failures (the 10 genuine documents it
  now calls tampered) and flips right back to always "genuine." Same 33.3%
  as round 0.

Three rounds, two degenerate single-class predictors, no evidence the model
ever developed a real decision boundary based on image content. Each tiny
10-20 example retrain has enough gradient signal to flip the adapter's
global bias but not enough to teach it anything that generalizes.

This lines up with what the training loss showed too — v24's loss plateaus
around 1.25-1.26 in epochs 2-3 after a fast initial drop, consistent with a
model near the limit of what a rank-4 adapter and 719 training examples can
represent. See `phase4_sft_summary.md` for the loss curve.

## v25: does more LoRA capacity fix it?

v25 bumps the adapter to r=16 (from v24's r=4), same 719-example
composition, same eval set. Same collapse, checked against the real
per-example output this time, not just the aggregate number:

| Round | Retrained on | Accuracy | Predicted verdicts |
|---|---|---|---|
| 0 (v25 checkpoint, r=16, no retrain) | — | 33.3% | genuine: 30/30 |
| 1 | 20 failures mined from round 0 | 66.7% | tampered: 30/30 |
| 2 | 10 failures mined from round 1 | 33.3% | genuine: 30/30 |

Identical numbers to v24, identical oscillation pattern. Capacity alone
doesn't fix it — the imbalanced training data (700 genuine vs 19 tampered)
produces the same failure mode regardless of adapter rank. v25's training
loss did plateau lower than v24's, so rank isn't irrelevant, but it's
clearly not sufficient by itself. That's what pointed at the class ratio as
the actual thing to fix next, not further rank tuning.

## v26: fixing the class imbalance actually works, mostly

v26 (r=8, 384px, tampered examples oversampled to a real 1:1 ratio — see
`phase4_sft_summary.md`) goes through the same procedure as its Round 0
baseline.

| Round | Retrained on | Accuracy | Predicted verdicts | Confusion (true → pred) |
|---|---|---|---|---|
| 0 (v26 checkpoint, no retrain) | — | 86.7% | tampered: 24, genuine: 6 | genuine→genuine: 6, genuine→tampered: 4, tampered→tampered: 20, tampered→genuine: 0 |
| 1 | 4 failures mined from round 0 | 33.3% | genuine: 30 | genuine→genuine: 10, tampered→genuine: 20 |
| 2 | 20 failures mined from round 1 | 66.7% | tampered: 30 | genuine→tampered: 10, tampered→tampered: 20 |

This splits into two separate findings, worth keeping apart rather than
reading as one mixed result.

**Finding 1 (round 0): class-balancing fixed the base checkpoint.**
Evaluated cold, with no retraining, the model isn't defaulting to one class
anymore — it caught every tampered example in the set, got 6 of 10 genuine
documents right, and only misfired on the other 4 (calling them tampered,
not the reverse). A real false-positive bias, not a collapse, and a much
more defensible failure mode for a fraud-detection system: nothing
tampered slipped through as genuine.

**Finding 2 (rounds 1-2): the adversarial-rounds retraining loop is
separately fragile.** This undoes finding 1, but it's a different problem,
not a contradiction of it. Retraining on just 4 mined examples flips the
whole model back to single-class, and retraining on 20 flips it to the
opposite class. The base v26 checkpoint isn't the issue here — the
retraining loop is. A full 3-epoch retrain on a handful of examples is
apparently enough to overwrite most of what the 1400-example balanced set
taught it, which is a lot of forgetting for such a small update. The same
loop produced identical single-class collapses on v24 and v25 too, so this
isn't new behavior — it's just now clearly separable from the class-
imbalance problem instead of getting blamed on it. Worth fixing on its
own — either a much smaller learning rate for these mini-retrains, or
mixing the mined failures back in with a slice of the original training
set instead of training on them in isolation.

A full leave-one-out re-run on v26 wasn't done — the check above already
answers the main question (does balancing fix the collapse), and a full
5-fold retrain is a multi-hour job that isn't necessary to draw that
conclusion. It would be the obvious next step with more time, and fixing
the retraining-loop fragility above would be worth doing first so it
doesn't mask what the base checkpoint actually learned.
