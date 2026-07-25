# Phase 6: leave-one-out generalization

Kaggle run (`doc-verification-leave-one-out`, ~11h39m wall-clock), 5 full
QLoRA retrains from the base model using v24's training config (LoRA r=4,
256px images, max_seq_length=1024). Each fold trains on genuine (694
examples) plus every forgery tier except the held-out one, then evaluates
only on that held-out tier — the core question the whole project is built
around: does tamper detection generalize to an attack type the model never
saw during training.

## Every fold scored exactly 0.000 accuracy

| Held-out tier | n | Accuracy | 95% CI | Training set size |
|---|---|---|---|---|
| tier1_field_tamper | 10 | 0.000 | [0.000, 0.000] | 754 |
| tier2_splicing | 15 | 0.000 | [0.000, 0.000] | 751 |
| tier3_inpainting | 13 | 0.000 | [0.000, 0.000] | 749 |
| tier4_full_synthetic | 15 | 0.000 | [0.000, 0.000] | 747 |
| tier5_recapture | 15 | 0.000 | [0.000, 0.000] | 747 |
| Overall | 68 | 0.000 | [0.000, 0.000] | — |

The CI is exactly [0, 0] because every one of the 68 held-out examples
across all 5 folds scored 0 — no variance to bootstrap.

## Every fold predicts "genuine," always

| Held-out tier | Predicted genuine | Predicted tampered | Unparseable |
|---|---|---|---|
| tier1_field_tamper | 10/10 | 0 | 0 |
| tier2_splicing | 15/15 | 0 | 0 |
| tier3_inpainting | 8/13 | 0 | 5 |
| tier4_full_synthetic | 15/15 | 0 | 0 |
| tier5_recapture | 13/15 | 0 | 2 |

Zero "tampered" predictions across all 68 held-out examples, across all 5
independent retrains. Not noise, not specific to one tier — every fold,
trained from scratch on a real ~719-744-example dataset, converged to a
model that always says "genuine" no matter what's in the image.

## Third confirmation of the same pattern

- Adversarial rounds (`phase6_adversarial_rounds_summary.md`): every round
  is a single-class predictor.
- Quantization benchmarking (`phase9_quantization_bench_summary.md`):
  exactly 50.0% accuracy at all three precisions on a balanced 40-example
  set — same signature, independent of the retraining process entirely.
- Leave-one-out (this doc): 0.000% across 5 full retrains from the base
  model, not a small perturbation or a quantization artifact.

Three different evaluation procedures, same failure mode. That's not
coincidence.

## Most likely cause: severe class imbalance

Every fold's training set is ~694 genuine examples against only ~40-55
forgery examples spread across the 4 remaining tiers — roughly 6-8%
tampered. A model minimizing training loss under that imbalance has a very
easy shortcut available: predict "genuine" unconditionally and be right
~92-94% of the time on the training set itself, without ever learning to
use image content at all. That's a textbook class-imbalance collapse, and
it's at least as well supported here as the LoRA-rank-capacity hypothesis
from Phase 4's training-loss plateau — probably the two compound, since a
lower-capacity adapter (r=4, cut for memory reasons) has even less room to
resist the easy majority-class solution than a bigger one would.

Worth noting even before the v25 capacity test ran: if v25 trains a
higher-capacity checkpoint successfully, this result already suggests class
imbalance alone could produce the same collapse regardless of capacity —
capacity might be necessary but isn't likely to be sufficient on its own. A
real fix needs to address the imbalance directly (oversampling, a weighted
loss, or a bigger and more balanced forgery dataset per tier). See
`phase4_sft_summary.md`'s v25 and v26 sections for how that played out.

## Bottom line

The core research question — does tamper detection generalize to an unseen
attack type — has a clear answer at this training configuration: no. The
model doesn't just generalize poorly, it doesn't appear to have learned
image-content-based discrimination at all, converging to the majority-class
shortcut every time. That's the project's central finding at this point in
the training story, not something to soften into "needs more tuning."
