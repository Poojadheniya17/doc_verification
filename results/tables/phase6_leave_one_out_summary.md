# Phase 6: Leave-One-Out Generalization — Real Results

Real Kaggle run (kernel `doc-verification-leave-one-out`, ~11h 39m wall-clock,
5 full QLoRA retrains from the base model, v24's training config: LoRA r=4,
`max_image_size=256`, `max_seq_length=1024`). Each fold trains on genuine
(694 examples) + every forgery tier EXCEPT the held-out one, then evaluates
only on the held-out tier's examples — the core question this whole project
is built around: does tamper detection generalize to an attack type the
model never saw during that fold's training.

## The result, stated plainly

**Every single fold scored exactly 0.000 accuracy.**

| Held-out tier | n | Accuracy | 95% CI | Training set size (this fold, real) |
|---|---|---|---|---|
| tier1_field_tamper | 10 | **0.000** | [0.000, 0.000] | 754 |
| tier2_splicing | 15 | **0.000** | [0.000, 0.000] | 751 |
| tier3_inpainting | 13 | **0.000** | [0.000, 0.000] | 749 |
| tier4_full_synthetic | 15 | **0.000** | [0.000, 0.000] | 747 |
| tier5_recapture | 15 | **0.000** | [0.000, 0.000] | 747 |
| **Overall** | **68** | **0.000** | **[0.000, 0.000]** | — |

The CI is exactly [0, 0] because every single one of the 68 held-out
examples across all 5 folds scored 0 — there is no variance to bootstrap.

## The real per-example evidence: every fold predicts "genuine," always

Verdict distribution across each fold's held-out (all genuinely tampered)
examples:

| Held-out tier | Predicted "genuine" | Predicted "tampered" | Unparseable |
|---|---|---|---|
| tier1_field_tamper | 10/10 | **0** | 0 |
| tier2_splicing | 15/15 | **0** | 0 |
| tier3_inpainting | 8/13 | **0** | 5 |
| tier4_full_synthetic | 15/15 | **0** | 0 |
| tier5_recapture | 13/15 | **0** | 2 |

**Zero "tampered" predictions across all 68 held-out examples, in all 5
independent full retrains.** This is not noise, not a borderline result, and
not specific to one tier or one training run — every fold, trained from
scratch on a genuine, real ~719-744-example dataset, converged to a model
that always predicts "genuine" regardless of what it's shown.

## This is the third independent confirmation of the same collapse pattern

- **Adversarial rounds** (`phase6_adversarial_rounds_summary.md`): every
  round is a single-class predictor (round 0/2: always "genuine"; round 1:
  always "tampered"), on 10-20-example targeted retrains.
- **Quantization benchmarking** (`phase9_quantization_bench_summary.md`):
  exactly 50.0% accuracy at all three precisions on a balanced 40-example
  set — the same always-one-class signature, independent of the retraining
  process entirely.
- **Leave-one-out (this document)**: 0.000% accuracy across 5 independent
  **full, real ~719-744-example retrains from the base model** — not a
  small perturbation, not a quantization artifact. The model itself, trained
  fresh each time on a real, substantial dataset, always converges to
  "genuine."

Three independent evaluations, using three different real training/eval
procedures, all show the identical failure mode. This is airtight evidence
that something systematic — not noise, not an artifact of any one script —
is happening.

## The most likely real cause, reasoned honestly: severe class imbalance

Every fold's training set is **~694 genuine examples against only ~40-55
total forgery examples spread across 4 remaining tiers** (roughly 6-8%
tampered). A model minimizing training loss under this imbalance has a very
strong, very easy local optimum available: predict "genuine" unconditionally
and be right ~92-94% of the time on the training set itself, without ever
learning to use image content to discriminate. This is a textbook class-
imbalance collapse, and it is at least as well-supported by this project's
real evidence as the LoRA-rank-capacity hypothesis floated after Phase 4's
training-loss plateau (see `phase4_sft_summary.md`) — likely the two
compound: a model with genuinely limited capacity (LoRA r=4, cut for memory
reasons) has even less room to resist collapsing to the easy majority-class
solution than a higher-capacity model would.

**Honest note on the v25 capacity-restoration experiment** (see
`phase4_sft_summary.md`'s v25 section for the full real investigation): even
if v25 successfully trains a higher-capacity checkpoint, this leave-one-out
result suggests **class imbalance alone could still produce the same
collapse regardless of capacity** — capacity may be necessary but is
unlikely to be sufficient. A real fix would need to address the imbalance
directly (oversampling forgery examples, a weighted loss, or a
substantially larger and more balanced forgery dataset per tier) — flagged
here as the most concrete, evidence-backed item for Future Work, ahead of
further capacity tuning.

## What this means for the project's honest bottom line

This project's core research question — "does tamper detection generalize
to an unseen attack type" — has a clear, honest, real answer at the current
training configuration: **no evidence of generalization was found**. The
model does not merely generalize poorly to held-out tiers; it does not
appear to have learned image-content-based tamper discrimination at all,
converging instead to the majority-class shortcut every single time. This
is reported as the project's central, honest finding — not smoothed into a
softer "needs more tuning" framing — because that is what three independent,
real evaluations actually show.
