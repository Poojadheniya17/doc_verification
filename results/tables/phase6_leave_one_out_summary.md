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

## v26 (class-balanced) re-validation — 2 scoped folds

A full 5-fold re-run on v26 would cost ~27-28 GPU-hours, close to the
entire weekly Kaggle free-tier quota, so a scoped 2-fold version is running
instead: tier2_splicing (a localized-forgery case) and tier4_full_synthetic
(the most-different generalization case), each a real full retrain on the
class-balanced set (`kaggle_kernels/phase6_leave_one_out_v26/`), not a
shortcut.

| Held-out tier | n | Accuracy | 95% CI | Predicted genuine | Predicted tampered | Unparseable |
|---|---|---|---|---|---|---|
| tier2_splicing | 15 | 0.133 | [0.000, 0.333] | 13/15 | 2/15 | 0 |
| tier4_full_synthetic | 15 | 0.000 | [0.000, 0.000] | 14/15 | 0/15 | 1 |
| **Combined** | 30 | 0.067 | [0.000, 0.167] | 27/30 | 2/30 | 1 |

Both real. Before trusting either number, the same three places a pipeline
bug could hide were checked directly against each fold's own logs, not
assumed: (1) training composition -- confirmed each fold's held-out tier
was genuinely excluded (`training on [...]` printed the other 4 tiers only,
in both cases); (2) checkpoint used for eval -- confirmed each fold loaded
its own freshly-trained checkpoint (`loo_v26_holdout_tier2_splicing/final`
and `loo_v26_holdout_tier4_full_synthetic/final` respectively), never a
stale one; (3) scoring logic -- simple, already-unit-tested code, re-read
and correct in both cases; class-balancing applied correctly in both
(751→1400 and 747→1400). No pipeline bug found in either fold.

**Fold 1 (tier2_splicing): 13.3% (2/15 correct).** 13 of 15 held-out
splicing examples were misclassified "genuine," most with high confidence
in that wrong verdict (0.95-0.99+ P(genuine)). Only 2 of 15 were correctly
caught, both with strong confidence in the correct direction — not a total
collapse, but a real, significant generalization gap.

**Fold 2 (tier4_full_synthetic): 0.000 (0/15 correct) — a total collapse,
the same single-class signature as v24's original failure.** 14 of 15
held-out examples predicted "genuine," 1 unparseable, zero ever predicted
"tampered."

**Why tier4 failed harder than tier2 — a real, structural reason, not just
"generalization is hard."** Tiers 1, 2, 3, and 5 all share the same
underlying tampering concept: start from a real MIDV-2020 template and
make a *local* change (a swapped digit, a spliced photo region, a
diffusion-inpainted patch, a recapture degradation) — the document's
overall layout and security background stay genuine in every case. Tier 4
is categorically different: an entirely fabricated document generated from
scratch, with no real template underneath at all. When tier4 is held out,
the remaining four tiers can only teach "spot a local anomaly on a real
template" — they never demonstrate the concept the model actually needs
for tier4 ("the whole document can be fake, not just a piece of it").
That's a genuine data/concept-coverage gap, not a capacity or
attention-coverage issue. It also explains tier2's partial success: photo
splicing is still "a local anomaly on a real template," closer in kind to
what tier1/3/5 already taught than tier4 is.

Follow-up diagnostic (`kaggle_kernels/diagnostic_vision_modules/`): tested
whether LoRA's `target_modules` ever reach the vision encoder at all (a
plausible cause for the perceptual side of this — the frozen vision tower
would never have been taught to notice tampering-specific visual
artifacts). Result: partially yes. Qwen2.5-VL's vision-block MLP layers
(`gate_proj`/`up_proj`/`down_proj`) happen to share naming with the LLM
decoder's projections, so 96 real vision-side modules across all 32 vision
blocks do get LoRA-adapted. What's NOT covered: the vision tower's
attention modules (`attn.qkv`, `attn.proj` — different names from the LLM
side's `q/k/v/o_proj`). A narrower, more speculative gap than the clean
"vision encoder is completely frozen" hypothesis this diagnostic was built
to test — refuted, not confirmed. Given fold 2's result, this lever (if it
helps at all) is more plausible for tier2-style localized-artifact
generalization than for tier4-style wholesale-fabrication generalization,
which looks like a concept-coverage gap no amount of attention capacity
would close. A real retrain (`kaggle_kernels/phase6_leave_one_out_v27_vision_attn/`)
is queued to test this against the tier2 fold specifically, with tempered
expectations stated up front rather than after the fact.

**Reading both folds honestly**: the class-balancing fix (v26) solves the
easy-shortcut problem within its training distribution (86.7% on the
same-distribution eval set), but generalizing to a genuinely unseen
tampering technique is a separate, harder problem that balancing the data
did not solve — confirmed now with 2 real data points, not 1, one partial
failure and one total collapse. This is the project's most important open
finding as of this writing.
