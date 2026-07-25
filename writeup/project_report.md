# Adversarially-Robust Identity Document Verification System

*Paper-style writeup, populated with real results throughout. Where a
result wasn't available at time of writing, that's stated plainly rather
than filled in with a placeholder.*

## Problem

Identity document verification sits at an awkward intersection of computer
vision, fraud economics, and distribution shift. A system that only scores
well on the exact forgery techniques it was trained on isn't actually
useful in production — real attackers adapt, and a pipeline that's never
seen a new kind of tamper is exactly the pipeline that gets exploited
first. This project is framed less around "what's our accuracy on a
held-out test set from the same distribution as training" and more around
how well the system generalizes to an attack type it's genuinely never seen
— a leave-one-attack-tier-out question, not a single in-distribution
accuracy number.

Three properties shaped every design decision here:

1. **Forgery diversity.** A field tamper (a single digit swapped) and a
   fully synthetic fabricated document are extremely different signals to
   detect, yet both need to route to "tampered." A system tuned to catch
   one convincingly can be blind to the other.
2. **Class imbalance and asymmetric costs.** Genuine documents vastly
   outnumber forged ones in any real deployment, and a false accept (a
   forged document waved through) doesn't cost the same as a false reject
   (a genuine customer wrongly blocked). A single accuracy number without a
   cost-aware decision layer is reporting the wrong thing.
3. **Generalization to unseen attack types** — the central axis of this
   project, tested via a 5-tier forgery taxonomy and a leave-one-tier-out
   evaluation protocol.

This project builds, trains, and evaluates a real system against these
three properties — not a mocked pipeline or API calls to a hosted model,
but a genuinely fine-tuned open-weights VLM (Qwen2.5-VL) trained via QLoRA
on real, locally-generated forgery data, with every reported number backed
by an actual model run.

## Method

**Model.** Qwen2.5-VL-3B-Instruct, fine-tuned via QLoRA (4-bit NF4
quantization, LoRA adapters). The original plan targeted 7B; after a
debugging investigation that tested six independent hypotheses for a
persistent training hang without finding the root cause (see *Engineering
journey*), the model was switched to 3B. The 7B zero-shot baseline in
Results is therefore a different model size than the 3B fine-tuned model —
flagged explicitly everywhere the two are compared, since it's not a clean
ablation.

**Training.** Supervised fine-tuning only — DPO was scoped out once the SFT
debugging saga ate significantly more time than planned. Three checkpoints
were trained and evaluated: v24 (LoRA r=4, 256px, the original tier1+tier2
composition — this is what a T4's memory constraints allowed after the full
debugging chain below), v25 (LoRA r=16, same data, testing whether more
adapter capacity changes anything), and v26 (LoRA r=8, 384px, all 5 tiers,
tampered examples oversampled to a 1:1 ratio against genuine, testing
whether the actual class imbalance was the real problem).

**5-tier forgery taxonomy**, escalating in sophistication:

| Tier | Technique | Generation method |
|---|---|---|
| 1 | Field tamper | OCR-located text-region digit/text swap |
| 2 | Photo splicing | Haar-cascade face detection + Poisson blend (`cv2.seamlessClone`) from a donor document |
| 3 | Diffusion inpainting | Stable Diffusion inpainting over the photo region (GPU-only, Kaggle) |
| 4 | Full synthetic | Procedurally fabricated document (fields + guilloche background + a donor face crop) |
| 5 | Recapture simulation | Moire interference + channel misregistration + contrast shift, simulating a screen-recapture presentation attack |

All five tiers were generated for real — see
`results/tables/phase5_forgery_tiers_summary.md`.

**Evaluation design.** Two experiments probe generalization directly:

- **Leave-one-out**: for each forgery tier, train a fresh QLoRA adapter on
  every other tier's examples plus genuine documents, then evaluate only on
  the held-out tier. Measures whether tamper detection transfers to an
  attack type never seen during that fold's training.
- **Adversarial retraining rounds**: starting from a trained checkpoint,
  repeatedly evaluate against a fixed eval set, mine the incorrect
  examples, retrain specifically on those failures, and track the accuracy
  curve across rounds.

**Decision layer.** A cost-aware routing layer (`src/decision/`) converts a
model's confidence into auto-approve, auto-reject, or human review, via
thresholds swept against a cost matrix
(`config/cost_matrix_config.yaml`: an assumed-but-labeled $500 false-accept
/ $50 false-reject / $5 manual-review structure — reasoned portfolio
assumptions, not real industry figures).
`financial_risk_reasoning.py` turns a routed decision into a short
natural-language rationale, optionally citing similar past-flagged cases
from a retrieval layer (`src/retrieval/case_index.py`: sentence-transformer
embeddings + FAISS cosine similarity).

**Confidence signal.** The fine-tuned model doesn't output a calibrated
probability directly — it outputs a categorical verdict plus free text.
Confidence here is derived from generation statistics: average per-token
softmax probability across the response, mapped to P(genuine) based on
which verdict was produced. This is a whole-response proxy, not a claim of
true field-level calibration.

**Compute.** All model training/inference ran on Kaggle's free-tier T4 GPU
(~30 GPU-hours/week quota). Local development happened on a ~7.7GB-RAM
machine that can't load any size of the VLM without severe disk thrashing —
a standing constraint from Phase 3 onward. Pure logic gets tested locally;
every model-touching call runs on Kaggle.

## Engineering journey

Getting SFT training to complete took 24 kernel versions across two
distinct failure modes. The debugging process is worth writing up in full
rather than trimming to a one-paragraph summary — the hypotheses tested and
what each one ruled out are as informative as the eventual fix.

### A real CUDA OOM chain, solved cleanly (v8-v10)

The first real Kaggle runs hit a straightforward, explainable OOM:
MIDV-2020's source images are full-resolution scans (~2167×1521), and
feeding them in uncapped multiplied vision-token count (and activation
memory) far beyond a T4's headroom. v8 crashed 486MB short in the forward
pass with images uncapped; capping `max_image_size` (v9) moved the failure
to the backward pass (1.7GB short); adding
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (v10) didn't close the
gap but measurably improved it (free memory rose from 1.10GB to 1.36GB,
fragmentation dropped from 524MB to 291MB). Each fix's numbers moved in the
expected direction — a correctly-diagnosed problem, and a useful contrast
with what came next.

### Training stopped OOM'ing and started silently hanging (v11-v18)

Once the memory chain was resolved, training on the original 7B target
began silently hanging instead — no error, no traceback, zero progress.
Six independent, single-variable hypotheses were tested in sequence, each
one ruling out a real candidate cause without finding the actual one:

1. **v11** (paged 8-bit optimizer, r=12): stalled 39 minutes at the
   identical backward-pass line, zero progress, no crash. Hypothesis: the
   paged optimizer thrashing under memory pressure.
2. **v12** (non-paged `adamw_bnb_8bit`, tighter memory footprint): stalled
   at the identical line — confirmed via two live-log fetches minutes apart
   returning byte-identical content. Paging hypothesis falsified.
3. **v13** (`gradient_checkpointing_kwargs={"use_reentrant": False}`,
   targeting a documented reentrant-checkpointing deadlock class): didn't
   hang at the same line — the reentrant-mode warning present in every
   prior log disappeared, confirming the fix was active — but it then
   stalled ~58 minutes at an earlier point. A different failure signature,
   not a resolution.
4. **v14** (checkpointing off entirely): hung at yet another point.
5. **v16** (`use_reentrant=False` + explicit `use_cache=False`): hung
   again.
6. **v17-v18** (`device_map={"": 0}`, then `CUDA_VISIBLE_DEVICES=0` before
   any CUDA import): v17 crashed with a genuinely new error — a
   `RuntimeError` inside `torch.nn.DataParallel`, revealing this Kaggle
   instance has 2 visible GPUs, not 1. Transformers' `Trainer` auto-wraps
   in `DataParallel` whenever `device_count() > 1`; restricting visibility
   to GPU 0 before any CUDA import removed this entirely.

Three independent, well-reasoned fixes each producing a different hang
location rather than clearing the problem is a different pattern than one
unlucky bug — it points at something structural in the T4/driver/library-
stack combination in this environment. The failure point kept moving across
six hypotheses spanning both model sizes, and the actual root cause of the
original hang was never found. That investigation was closed in favor of a
pragmatic workaround: disable gradient checkpointing and manage memory
through other levers.

### OOM again, a bounded search (v19-v23)

With checkpointing off, the hang was gone, replaced by a clean, ordinary
OOM. What followed was a bounded step-down search:

| Kernel | Change | Result | Shortfall | Memory in use |
|---|---|---|---|---|
| v19 | 512px | OOM (MLP dequant) | ~44 MiB short | 14.55 GiB |
| v20 | 384px | OOM (lm_head) | ~51 MiB short | 14.03 GiB |
| v21 | 320px | OOM (lm_head) | ~163 MiB short | 14.14 GiB |
| v22 | 256px (floor) | OOM (loss/cross_entropy) | ~19 MiB short | 14.33 GiB |
| v23 | + LoRA r=4, max_seq_length=1024 | OOM (first backward pass) | ~49 MiB short | 14.36 GiB |

Memory usage wasn't monotonic with image size across this search, never
fully explained. v23 combined two new levers (rank cut, sequence-length
cut) and still OOM'd by roughly the same margin as v22 — five straight
failures across three memory levers, all landing within ~20-160 MiB of the
same ~14.3-14.5 GiB ceiling.

### The actual root cause — a leftover model in memory (v24)

A direct question reframed the investigation: was Phase 3's 7B zero-shot
baseline model actually being freed before Phase 4 loaded the 3B
fine-tuning target, given both steps run in the same Kaggle session?

It wasn't. `clean_eval.py` caches its loaded model at module scope
(`_model`/`_processor` globals, added so its own per-image eval loop
wouldn't reload weights on every call), and the kernel driver script that
runs Phase 3 then Phase 4 in one process never cleared that cache. A
diagnostic confirmed it directly:

```
=== GPU memory before Phase 3 cleanup: 5.91 GB allocated, 7.29 GB reserved ===
=== GPU memory after Phase 3 cleanup: 0.01 GB allocated, 0.03 GB reserved ===
```

~7.3GB of the ~14.5GB budget was leftover 7B model weights on every one of
v19-v23's five attempts. None of the memory levers cut in that search
(image size, LoRA rank, sequence length) were ever going to close a gap
that large — all of them were shrinking the wrong side of the memory
ledger. The fix was a single `gc.collect()` + `torch.cuda.empty_cache()`
call between phases, plus a module-level `unload_model()` added to
`clean_eval.py`.

Training completed for the first time in this project — 135/135 steps, 3
epochs, no crash, `train_runtime=7999s` (~2h13m), `train_loss=1.7296`
(running average). This is `checkpoints/sft_v24_final/`, the checkpoint
every later phase builds on.

Eleven kernel pushes (v14-v24) across two entirely different failure modes
— six hang hypotheses that never converged, followed by five OOM attempts
that all cut the wrong variable — were only resolved by a question about
process lifecycle and cross-phase state, not by more hyperparameter tuning.
The fix wasn't a bigger model, more VRAM, or a cleverer LoRA config. It was
a cached global reference outliving its scope.

### Two more real bugs, before the core experiments could run

Getting the checkpoint to exist wasn't sufficient — running leave-one-out
and adversarial-rounds against it surfaced two more bugs, both invisible
locally:

- **A manifest-filename convention mismatch.** Every forgery tier's
  manifest file is actually named with a short numeric prefix
  (`tier1_manifest.json`), but the code that derived this filename for the
  leave-one-out and adversarial-rounds Kaggle drivers used the tier's full
  descriptive name instead (`tier1_field_tamper_manifest.json`, which never
  exists). Since example-building silently skips any tier whose manifest
  path doesn't exist (a deliberate design choice, so a not-yet-generated
  tier degrades gracefully), this produced zero tier examples with no error
  at all — invisible until a Kaggle run logged "0 folds" with nothing to
  point at the cause.
- **Windows-native (backslash) paths baked into every manifest.** Every
  data-generation script serialized manifest path fields via bare
  `str(some_path)`, which produces backslashes on this Windows dev machine.
  Every path resolved fine locally (Windows accepts both slash
  conventions), invisible until a manifest reached the Linux Kaggle
  environment and hard-failed with `FileNotFoundError`. A project utility
  (`kaggle_package.py`'s `stage_package()`/`_rewrite_paths_in_place()`)
  already existed specifically to normalize this before staging — this
  exact class of problem had been solved once before, but later ad-hoc
  dataset refreshes bypassed that utility and pushed raw manifests through.

Both fixed at the source (short-prefix filename derivation; every generator
switched to `.as_posix()`), with regression tests added against real
committed data, since both bugs were invisible to fixture-based tests that
never exercised the actual default-path-construction logic.

One more, non-code observation from the same stretch: a Kaggle dataset
version reporting "ready" via the CLI doesn't reliably mean a kernel pushed
immediately afterward sees the update. Observed multiple times across this
project, resolving itself after waiting roughly 10-15 minutes with no other
action. `kaggle_kernels/diagnostic_check/` exists specifically to verify
dataset state directly and cheaply (seconds, no GPU) rather than
re-testing an assumption via a full training run.

## Results

Every number below is real, sourced from `results/tables/`. Confidence
intervals are reported wherever sample size supports one — several samples
are small, and the resulting wide CIs are stated plainly.

### Zero-shot 7B baseline (n=9)

| Metric | Value | 95% CI |
|---|---|---|
| Tamper-verdict accuracy (overall) | 66.7% | [33.3%, 100%] |
| Tamper-verdict accuracy (genuine, n=3) | 0% | — |
| Tamper-verdict accuracy (tampered, n=6) | 100% | — |
| Parse success rate | 100% | [100%, 100%] |
| Field similarity (name/dob/id_number/expiry) | 1.00 each | [1.00, 1.00] |

The aggregate 66.7% hides the real pattern: zero-shot 7B caught every real
forgery in this sample but flagged every genuine document as tampered too,
hallucinating a specific but false justification each time (e.g. flagging a
genuine, unedited ID number as "not consistent with the format typically
used"). High recall, high false positives — worth watching for in the
fine-tuned model's own behavior below.

### SFT training, v24 (135/135 steps, 719 examples)

`train_runtime=7999s` (~2h13m), final running-average `train_loss=1.7296`.
See `results/charts/phase4_loss_curve.png`. Loss dropped fast in epoch 1
(5.03 → 1.31) then plateaued around 1.25-1.26 for the remaining two
epochs — a real finding, expanded on in Failure analysis below.

### Leave-one-out generalization (5 full retrains, ~11h39m)

This is the project's central experiment — does tamper detection generalize
to an attack type never seen during training — and it produced the
project's most important result.

| Held-out tier | n | Accuracy | 95% CI | Training set size |
|---|---|---|---|---|
| tier1_field_tamper | 10 | 0.000 | [0.000, 0.000] | 754 |
| tier2_splicing | 15 | 0.000 | [0.000, 0.000] | 751 |
| tier3_inpainting | 13 | 0.000 | [0.000, 0.000] | 749 |
| tier4_full_synthetic | 15 | 0.000 | [0.000, 0.000] | 747 |
| tier5_recapture | 15 | 0.000 | [0.000, 0.000] | 747 |
| Overall | 68 | 0.000 | [0.000, 0.000] | — |

Every one of 5 independent full retrains — each on a real ~747-754-example
dataset, not a small perturbation — scored exactly 0.000 accuracy on its
held-out tier. Across all 68 held-out examples, zero were predicted
"tampered." Every fold converged to a model that predicts "genuine" for
essentially everything. Full breakdown:
`results/tables/phase6_leave_one_out_summary.md`.

### Adversarial retraining rounds, v24 (n=30/round)

| Round | Retrained on | Accuracy | 95% CI | Verdict distribution |
|---|---|---|---|---|
| 0 (v24 checkpoint) | — | 33.3% | [16.7%, 50.0%] | genuine: 25, unparseable: 5, tampered: 0 |
| 1 | 20 mined failures | 66.7% | [50.0%, 83.3%] | tampered: 30, genuine: 0 |
| 2 | 10 mined failures | 33.3% | [16.7%, 50.0%] | genuine: 30, tampered: 0 |

The verdict-distribution column matters more than the accuracy column. At
every round the model predicted a single class for all 30 examples — not
incremental improvement, just the adapter's global bias flipping between
two degenerate single-class predictors depending on whichever tiny mined
batch it retrained on most recently.

### Quantization benchmarking, v24 (n=40/precision)

| Precision | Accuracy | 95% CI | Avg latency | Est. cost / 1M verifications |
|---|---|---|---|---|
| fp16 | 50.0% | [35.0%, 65.0%] | 8.69s | $845 |
| int8 | 50.0% | [35.0%, 65.0%] | 22.17s | $2,156 |
| int4 | 50.0% | [35.0%, 65.0%] | 20.98s | $2,040 |

Two findings. First: exactly 50.0% across all three precisions isn't
evidence quantization is harmless — it's the same collapse pattern from the
adversarial-rounds table, showing up again on this balanced 20/20 eval set.
A model predicting one class for every input scores exactly 50% on any
balanced set regardless of precision.

Second: fp16 measured both faster and cheaper than int8/int4 — 8.69s vs
22.17s and 20.98s, inverting the usual quantize-for-speed assumption. This
benchmark runs at batch size 1, and `bitsandbytes` int8/int4 layers carry
real per-call dequantization overhead that a larger production batch would
amortize but a single-example batch can't. The recommendation from this
specific benchmark is fp16, not int8/int4 — the opposite of conventional
wisdom, because that's what was actually measured. A deployment with real
request batching would need to re-run this comparison. Full analysis:
`results/tables/phase9_quantization_bench_summary.md`.

## Failure analysis

At the v24/v25 training configuration, the model shows no evidence of
having learned image-content-based tamper discrimination. Three independent
evaluations — leave-one-out, adversarial rounds, and quantization
benchmarking — converge on the same failure signature, with leave-one-out
the most statistically meaningful of the three.

**Leave-one-out (primary evidence).** Every one of 5 independent full QLoRA
retrains scored exactly 0.000 accuracy on its held-out tier, across 68
total examples, zero of which were ever predicted "tampered." Every fold,
trained fresh, converged to a model that predicts "genuine" for essentially
everything.

**Adversarial rounds (corroborating).** Headline accuracy moves 33.3% →
66.7% → 33.3%, which looks like noisy improvement-then-regression, but the
per-example distribution shows every round is a single-class predictor,
flipping the adapter's global bias based on whichever tiny mined batch it
saw most recently.

**Quantization benchmarking (corroborating, no retraining involved).** The
same checkpoint scored exactly 50.0% at all three precisions on a separate
balanced 40-example set — same signature, this time ruling out "an artifact
of the retraining loop" as the explanation.

**Most likely cause: severe class imbalance.** Every leave-one-out fold's
training set is ~694 genuine examples against only ~40-55 forgery examples
spread across the 4 remaining tiers — roughly 6-8% tampered. A model
minimizing training loss under that imbalance has an easy local optimum:
predict "genuine" unconditionally and be right ~92-94% of the time on the
training set itself, without ever using image content. Combined with v24's
training-loss plateau (stuck ~1.25-1.26 in epochs 2-3), this likely
compounds with a genuine capacity limitation too — LoRA rank 4 (cut for
memory reasons during the v8-v23 fight, never chosen because it was judged
sufficient) may leave less room to resist the easy majority-class shortcut
than a higher-capacity model would.

### Testing the capacity hypothesis (v25)

With the real cause of v19-v23's ceiling understood, rank/image
size/sequence length were restored toward less aggressive values. v23's OOM
showed 14.36GiB in use at crash; subtracting the confirmed ~7.29GiB leak
puts 3B training's actual footprint at only ~7.07GiB against a ~14.56GiB
budget — roughly 2x headroom was available the entire time the v19-v23
search ran.

Three attempts at r=16 OOM'd with byte-identical numbers — attempt 1 (768px,
seq=2048) and attempt 2 (384px, seq=1536) both hit 44.00 MiB requested,
8.81 MiB free, 14.55GiB in use, despite a 4x pixel-area cut and a real
sequence-length change between them (verified via a diagnostic kernel to
rule out a stale-config artifact). Attempt 3 isolated LoRA rank alone and
produced the same identical failure a third time — which read at the time
as a clean isolation of rank r=16 as the cause.

A fourth push, intended as the safe final config (r=8, 512px, seq=2048),
actually ran with the stale r=16 config due to a Kaggle dataset-propagation-
lag issue — a dataset version reported "ready" before a freshly started
kernel's mount actually reflected the update. This run completed
successfully: 135/135 steps, all 3 epochs, no crash, peak GPU memory
10.17GB allocated / 10.51GB reserved (`train_runtime`: 8398s,
`train_loss`: 1.5125) — a ~4GB safety margin under the ceiling that killed
the three prior r=16 attempts. Confirmed two ways, not inferred: the
trainable-param count at training start (37,152,768, matching r=16
exactly) and the saved checkpoint's real `adapter_config.json`
(`"r": 16, "lora_alpha": 32`).

The earlier "conclusive" isolation of r=16 as broken doesn't hold up
against this. The same rank, same base model, same library versions, same
T4 tier failed identically three times and then succeeded with real margin
on a fourth attempt. Three byte-identical failures are real evidence; one
clean success is also real evidence; resolving that tension in either
direction would overstate what's actually known. The exact reconciling
mechanism was never found — the log doesn't record this run's actual image
size or sequence length (unlike LoRA rank, not verifiable from any saved
artifact), and the three failed attempts already showed image size and
sequence length changes made no measurable difference to their identical
OOM point, arguing against those variables being the answer here either.
The best-supported explanation is real Kaggle infrastructure/session
variance — this kernel had been deleted and freshly re-pushed partway
through the investigation — rather than a deterministic, reproducible
code-level cause, but that's the most plausible explanation, not a
confirmed one.

One genuine positive signal from this run, independent of the memory
anomaly: its training-loss plateau (~1.217-1.220 across epochs 2-3) is
measurably lower than v24's r=4 plateau (~1.25-1.26) — real evidence the
restored capacity let the model fit the training data better. This run
used the same tier1+tier2-only, 719-example composition as v24, so it
tested capacity alone, holding the same severe imbalance constant.

Running adversarial-rounds against this checkpoint answered the actual
question: capacity alone does not fix the collapse. Round 0: 33.3%
accuracy, 30/30 predictions "genuine." Round 1 (retrained on 20 mined
failures): 66.7%, 30/30 predictions "tampered." Round 2: 33.3%, 30/30
predictions "genuine" again — the exact same oscillating single-class
pattern and accuracy numbers as v24. The same imbalanced data (~700
genuine vs 19 tampered, ~35:1) produces the identical failure mode
regardless of LoRA rank. That doesn't rule out capacity as a contributing
factor (the lower training loss is still a real, separate signal), but it
made the class ratio the more decision-relevant thing to fix next, not
further rank tuning.

### Fixing the actual imbalance (v26)

Oversampled tampered examples to a real 1:1 ratio
(`sft_train.balance_examples()`) across all 5 forgery tiers instead of just
2, at the safer r=8/512px config rather than v25's still-unresolved r=16,
to keep the balance variable from tangling with that memory anomaly. Local
data showed 700 genuine vs 62 tampered before balancing (762 total), 1400
after.

The first attempt OOM'd immediately on the first training step (374MiB
short, 14.46GiB in use) — a useful surprise. The r=8/512px/seq=2048
combination had actually never been run for real before this; v25 was
supposed to test it but ran the stale r=16 config instead. The ~7.26GiB
"safe" estimate that config leaned on came from v19's data, which predates
`max_seq_length` being enforced at all — v19 trained on genuinely uncapped
sequences and still fit, so it was never a like-for-like comparison to a
run that actually truncates to 2048 tokens.

`max_image_size` was dropped 512 → 384, based on the real,
internally-consistent delta between v19 (512px) and v20 (384px)'s original
numbers (~520MiB) — not a blind guess. The second attempt completed: 264/264
steps, all 3 epochs, no crash, real margin (12.94GB reserved, ~1.6GB of
headroom under the 14.56GiB budget). `adapter_config.json` confirms r=8,
`lora_alpha=16` as intended. Checkpoint: 74,405,904 bytes (~71MB), committed
to git in full. Training wall-clock: 19553s (~5h26m) — longer than v24/v25's
~2.3h, expected given the balanced set's ~1.9x more examples.

The loss plateau (~2.38-2.45) is notably higher than v24's (~1.25-1.26) and
v25's (~1.217-1.22) — not necessarily a worse result. v24 and v25 both
trained on data where "always predict genuine" is a low-loss shortcut; this
run's balanced data removes that shortcut, so a higher loss plausibly
reflects the model actually attempting discrimination rather than being
worse-trained. That's a hypothesis, not something the loss number alone can
prove.

Running adversarial-rounds against v26 answers the main question, but the
result splits into two separate findings that shouldn't get blended
together — one about the checkpoint itself, one about the retraining
procedure used to test it.

| Round | Retrained on | Accuracy | Predicted verdicts | Confusion (true → pred) |
|---|---|---|---|---|
| 0 (v26 checkpoint, no retrain) | — | 86.7% | tampered: 24, genuine: 6 | genuine→genuine: 6, genuine→tampered: 4, tampered→tampered: 20, tampered→genuine: 0 |
| 1 | 4 failures mined from round 0 | 33.3% | genuine: 30 | genuine→genuine: 10, tampered→genuine: 20 |
| 2 | 20 failures mined from round 1 | 66.7% | tampered: 30 | genuine→tampered: 10, tampered→tampered: 20 |

**Finding 1: class-balancing fixes the base checkpoint — on this eval set.**
Round 0 is v26 evaluated cold, no retraining involved. The model isn't
defaulting to one class anymore — it caught every tampered example in the
set, got 6 of 10 genuine documents right, and only misfired on the other 4
(calling them tampered, not the reverse). A real false-positive bias, and a
much more defensible failure mode for a fraud-detection system than
"always says genuine": nothing tampered slipped through. This is a
promising, cheaply-obtained answer to the question the whole v25/v26
investigation was chasing — capacity alone (v25) didn't fix the collapse,
fixing the class ratio did — but it's promising, not conclusive: this is
one 30-example set from the same distribution as training, not the
leave-one-out, never-seen-attack-type test that v24 actually failed on.
Whether the fix generalizes the same way is a separate, harder question,
addressed directly below.

**Finding 2: the adversarial-rounds retraining loop is separately fragile,
independent of the checkpoint it starts from.** Rounds 1 and 2 are a
different experiment layered on top of Finding 1, and they undo it.
Retraining on just 4 mined examples flips the whole model back to
single-class, and retraining on 20 flips it to the opposite class. This
isn't evidence the balanced checkpoint stopped working — round 0 already
showed it discriminates — it's evidence that a full 3-epoch retrain on a
handful of examples is enough to overwrite most of what the 1400-example
balanced set taught it, regardless of how good the starting point was. The
same loop produced identical single-class collapses on v24 and v25 too, so
this fragility isn't new to v26, it's just now clearly separable from the
class-imbalance problem instead of getting attributed to it. A fix is
implemented but not yet validated on Kaggle: the kernel driver now mixes
each round's mined failures with a random replay slice of the original
balanced training set and retrains at a fraction of the full-run learning
rate (`config/training_config.yaml`'s `replay_sample_size` /
`mini_retrain_lr_scale`), instead of training on the mined failures in
isolation at the full rate. Queued to run after the current leave-one-out
re-validation, since Kaggle's free tier only reliably runs one GPU kernel
at a time.

Both findings matter for different reasons: Finding 1 says the core
detection approach works once the data is balanced. Finding 2 says the
adversarial-hardening step built on top of it needs its own fix before it's
useful. Neither should be read as evidence against the other.

A full 5-fold leave-one-out re-run on v26 would cost ~27-28 GPU-hours —
close to the entire weekly Kaggle free-tier quota and longer than a single
session reliably runs, so it wasn't attempted in full. A scoped 2-fold
version is in progress instead (tier2_splicing and tier4_full_synthetic
held out, chosen as a localized-forgery case and the most-different
generalization case respectively), each fold a real full retrain on the
balanced set — not a shortcut, just a smaller slice of the same experiment.
As of this writing, fold 1 (tier2_splicing held out) is still training on
Kaggle; neither fold's result is in yet, so Finding 1 above should be read
as promising-but-unconfirmed until real leave-one-out numbers land here.
Fixing Finding 2's retraining-loop fragility first would keep a
future full re-run from mixing that separate bug into these numbers, but
wasn't a blocker for this scoped check since no adversarial-round retraining
is involved here — each fold trains once, directly on the balanced data,
same as v26 itself.

## Limitations

- **The model never generates the "explanation" field, across every
  checkpoint (v24, v25, v26).** Confirmed on the 7 examples captured for
  the demo gallery (0/7) and re-checked against the raw eval output already
  collected for v24, v25, and v26's adversarial-rounds runs (0/30 in every
  case). Root cause, checked in order from cheapest to most involved: the
  SFT training targets built by `build_sft_examples()` never include an
  `"explanation"` key at all — every target dict is built from each
  manifest's `ground_truth` (which only ever has `name`/`dob`/`id_number`/
  `address`/`expiry`) plus `tamper_verdict` and `tamper_regions` set
  explicitly. No manifest in this project carries any authored narrative
  text to source one from either. So across all 719-1400 training examples
  in every checkpoint, the model never once saw a training target with that
  key populated — it's not a generation cutoff (every captured response
  parses as complete, valid JSON via a strict `json.loads()`, no
  truncation-repair logic involved) and it's not a prompting gap (`SFT_PROMPT`
  does explicitly ask for `"explanation": one sentence explaining your
  tamper_verdict`). The model is just doing exactly what it was trained to
  do: reproducing the JSON shape it was shown, which never included this
  field, regardless of what the prompt asks for.

  This is fixable, but not cheaply: it needs real per-tier template logic
  (e.g. "field tampering detected in `id_number`" for tier1, "photo appears
  spliced from another document" for tier2, and so on, built from metadata
  each manifest already has — tampered field names, bbox regions, tier
  type) wired into `build_sft_examples()`, then a full retrain from
  scratch, since this can't be patched into an existing checkpoint. Given
  v26's retrain alone took ~5.5 hours on a free-tier Kaggle T4, this is a
  real follow-up experiment, not a quick correction — documented here as a
  known limitation rather than fixed live.

- **The adversarial-rounds retraining loop was fragile, even on a
  checkpoint that discriminates well.** v26's base checkpoint shows real,
  varied, mostly-correct predictions, but 3-epoch retrains on 4-20 mined
  examples collapsed it back to single-class within one round. Separate
  from the original class-imbalance problem. A fix (replay-mixing + reduced
  learning rate for mini-retrains) is implemented but not yet validated on
  Kaggle — see Results above.
- **v24 and v25 show no evidence of learned tamper discrimination.**
  Leave-one-out (5 full retrains), adversarial rounds, and quantization
  benchmarking all independently show these checkpoints collapsing to a
  single predicted class rather than discriminating on image content —
  see Failure analysis for the full evidence and the class-imbalance
  explanation.
- **Severe class imbalance in the v24/v25 training data.** ~694 genuine
  examples against only ~40-55 forgery examples per leave-one-out fold
  (~6-8% tampered) — the likely dominant driver of the collapse. v26 fixed
  this for the base checkpoint; the retraining loop issue above is what's
  left.
- **Synthetic/public data only.** All training and evaluation data comes
  from MIDV-2020 (a public, CC BY-SA 2.5 dataset of synthetic documents
  with AI-generated faces) plus this project's own locally-generated
  forgeries. No real fraud data used or claimed.
- **Compute-forced hyperparameters for v24.** LoRA rank 4, 256px images,
  max_seq_length=1024, and gradient checkpointing disabled are the product
  of the Kaggle free-tier T4 memory fight above, not values independently
  judged optimal. v25 and v26 relaxed some of these once the real memory
  leak was understood.
- **The v25 memory anomaly is unresolved.** Same rank, same base model,
  same library versions, same T4 tier failed identically three times then
  succeeded with real margin on a fourth attempt. Documented as an open
  question in Failure analysis, not a solved one.
- **Small sample sizes for two of three eval procedures.** The zero-shot
  baseline (n=9) and adversarial-rounds/quantization eval sets (n=30-40)
  carry wide bootstrap CIs. Leave-one-out (n=68 across 5 full retrains) is
  the most statistically grounded of the three.
- **7B-vs-3B baseline comparison isn't a clean ablation.** The zero-shot
  baseline uses 7B; the fine-tuned models are 3B. Any comparison between
  the two conflates a model-size effect with a fine-tuning effect.
- **DPO was scoped out**, given how much of the compute/time budget the
  SFT debugging saga consumed.
- **Scripted, not learned, forgery generation.** All 5 tiers use
  deterministic/scripted generation (OCR-located edits, Poisson blending,
  diffusion inpainting with a fixed prompt, procedural fabrication, image
  filters) rather than a learned adversarial forger.
- **No autonomous orchestration layer.** Every Kaggle run in this project
  was manually triggered and monitored across many kernel pushes — there's
  no self-driving "detect a failure, diagnose it, re-launch" loop.

## Future work

- **Add real explanation text to the training targets and retrain.** The
  model never generates the `explanation` field because it never once saw
  it populated during training — see Limitations. Needs per-tier template
  logic in `build_sft_examples()` plus a full retrain; a real follow-up
  experiment, not a quick patch.
- **Validate the adversarial-rounds retraining-loop fix on Kaggle.** The
  code fix (replay-mixing + reduced learning rate for mini-retrains, see
  Results above) is implemented but hasn't been run yet — needs a real
  re-run of rounds 1-2 against v26 to confirm predictions stay varied
  instead of flipping back to single-class.
- **Full 5-fold leave-one-out re-run on v26.** A 2-fold scoped version
  (tier2_splicing, tier4_full_synthetic) is in progress at time of writing
  as a time-boxed check within a real compute budget; the remaining 3 folds
  are the natural extension once time/compute allow, to confirm the
  class-balance fix generalizes to unseen tiers and not just the fixed
  30-example eval set.
- **Resolve the v25 memory anomaly** — why LoRA rank r=16 OOM'd identically
  three times and then completed successfully with a real ~4GB margin on a
  fourth attempt, on the same base model/library versions/T4 tier.
- **RL-based self-play forger**: train an adversarial generator against the
  detector in a closed loop, rather than the fixed 5-tier scripted
  taxonomy.
- **Real fraud data partnerships**, to validate against actual attack
  distributions rather than synthetic/public data only.
- **Autonomous adversarial-round orchestration**: a system that runs
  eval → mine-failures → retrain without a human driving each Kaggle push.
- **DPO on top of the v26 checkpoint**, now that it's shown it can learn a
  real decision boundary — DPO on v24/v25 would likely have just tuned
  noise into a model that wasn't ready for it.
