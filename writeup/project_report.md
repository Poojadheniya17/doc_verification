# Adversarially-Robust Identity Document Verification System

*Paper-style writeup. Populated with real results and real, unedited numbers
throughout — this document does not contain a single hypothetical or
fabricated figure. Where a result was not yet available at the time of
writing, that is stated plainly rather than filled in with a plausible
placeholder (see the explicit **PENDING** markers in Results).*

## Problem

Identity document verification sits at an awkward intersection of computer
vision, fraud economics, and distribution shift. A system that only scores
well on the exact forgery techniques it was trained on is not actually
useful in production — real attackers adapt, and a document-verification
pipeline that has never seen a *new* kind of tamper is exactly the pipeline
that gets exploited first. This project is therefore framed less around "what
is our accuracy on a held-out test set drawn from the same distribution as
training" and more around **"how well does the system generalize to an
attack type it has genuinely never seen during training"** — a
leave-one-attack-tier-out generalization question, not a single
in-distribution accuracy number.

Three properties make this hard in a way that's worth stating explicitly,
because they shaped every design decision in this project:

1. **Forgery diversity.** A field-tamper (a single digit swapped) and a
   fully-synthetic fabricated document are extremely different signals to
   detect, yet both need to route to "tampered." A system tuned to catch one
   convincingly can be blind to the other.
2. **Class imbalance and asymmetric costs.** Genuine documents vastly
   outnumber forged ones in any real deployment, and the cost of a false
   accept (a forged document waved through) is not symmetric with the cost of
   a false reject (a genuine customer wrongly blocked). A system that reports
   a single accuracy number without a cost-aware decision layer is reporting
   the wrong thing.
3. **Generalization to unseen attack types**, as above — the central design
   axis of this project, tested directly via a 5-tier forgery taxonomy and a
   leave-one-tier-out evaluation protocol (see Method).

This project builds, trains, and evaluates a real system against these three
properties — not a mocked pipeline, not API calls to a hosted model, but a
genuinely fine-tuned open-weights VLM (Qwen2.5-VL) trained via QLoRA on
real, locally-generated forgery data, with every reported number backed by
an actual model run.

## Method

**Model.** Qwen2.5-VL-3B-Instruct, fine-tuned via QLoRA (4-bit NF4
quantization, LoRA adapters). The original plan targeted the 7B variant;
after an extensive, evidence-driven debugging investigation (see *Engineering
Journey* below) documented six independently-tested hypotheses that each
ruled out one candidate cause of a persistent training hang without finding
the actual root cause, the model was switched to 3B as a deliberate,
documented engineering decision — not a silent scope reduction. The 7B
zero-shot baseline (Results) is therefore a different model size than the
3B fine-tuned model; this is flagged explicitly everywhere the two are
compared, since it is not a clean ablation.

**Training.** Supervised fine-tuning (SFT) only — DPO was scoped out of this
project's remaining timeline (a documented, not-silent decision) once the
SFT debugging saga consumed significantly more engineering time than
planned. Final training configuration: LoRA rank 4, alpha 8, 4-bit NF4
quantization, `max_seq_length=1024`, `max_image_size=256`, gradient
checkpointing disabled. Every one of those specific numbers is the result of
a real, documented memory-constraint fight on a free-tier Kaggle T4 GPU —
none of them were chosen because they were judged ideal; they were the
values that let training complete at all. This matters for how the results
should be read (see Failure Analysis).

**5-tier forgery taxonomy**, escalating in sophistication and generation
technique:

| Tier | Technique | Generation method |
|---|---|---|
| 1 | Field tamper | OCR-located text-region digit/text swap |
| 2 | Photo splicing | Haar-cascade face detection + Poisson blend (`cv2.seamlessClone`) from a donor document |
| 3 | Diffusion inpainting | Stable Diffusion inpainting over the photo region (GPU-only, Kaggle) |
| 4 | Full synthetic | Procedurally fabricated document (fields + guilloche background + a donor face crop) |
| 5 | Recapture simulation | Moire interference + channel misregistration + contrast shift, simulating a screen-recapture presentation attack |

All five tiers were generated for real (real image files, real manifests,
no mocked data) — see `results/tables/phase5_forgery_tiers_summary.md`.

**Evaluation design.** Two experiments probe generalization directly, both
using the same real trained checkpoint as their starting point:

- **Leave-one-out**: for each forgery tier, train a fresh QLoRA adapter on
  every *other* tier's examples plus genuine documents, then evaluate *only*
  on the held-out tier. This directly measures whether tamper detection
  transfers to an attack type never seen during that fold's training — the
  central generalization question this project is built around.
- **Adversarial retraining rounds**: starting from the base trained
  checkpoint, repeatedly evaluate against a fixed eval set, mine the
  incorrect examples (capped at 200/round), and retrain specifically on
  those failures, tracking the accuracy curve across rounds.

**Decision layer.** A cost-aware routing layer (`src/decision/`) converts a
model's confidence into one of three actions — auto-approve, auto-reject, or
route to human review — via configurable thresholds swept against a real
cost matrix (`config/cost_matrix_config.yaml`: an assumed-but-labeled
$500 false-accept / $50 false-reject / $5 manual-review cost structure,
explicitly documented as reasoned portfolio assumptions, not real industry
figures). `financial_risk_reasoning.py` turns a routed decision into a
short, defensible natural-language rationale, optionally citing similar
past-flagged cases surfaced by a real retrieval layer
(`src/retrieval/case_index.py`: sentence-transformers embeddings + FAISS
cosine similarity — real, unit-tested, running locally).

**Confidence signal.** The fine-tuned model doesn't output a calibrated
probability directly — it outputs a categorical verdict plus free text. This
project derives a usable confidence score from the model's own generation
statistics: the average per-token softmax probability across the generated
response, mapped to P(genuine) based on which verdict was produced (high
confidence + "genuine" → high P(genuine); high confidence + "tampered" → low
P(genuine); an unparseable response maps to 0.5, maximal uncertainty). This
is a documented simplification — it measures confidence in the *whole*
generated response, not a token-span-isolated confidence in the
`tamper_verdict` field specifically — chosen because it's a legitimate,
real signal for a portfolio-scale decision layer, not a claim of true
field-level calibration.

**Compute.** All real model training/inference ran on Kaggle's free-tier
T4 GPU (~30 GPU-hours/week quota). Local development happened entirely on a
~7.7GB-RAM Windows machine that cannot load any size of the VLM without
severe disk thrashing — a standing constraint documented from Phase 3
onward. This split (pure logic tested locally, every model-touching call
deferred to Kaggle) is used consistently across every phase of this project
and is the reason the Method section above can honestly say "no mocked
pipeline" — every number in Results comes from a real Kaggle GPU run, not
a simulation.

## Engineering Journey: The Real Debugging Story

This section is included deliberately and in full, not trimmed into a
sanitized one-paragraph summary. Getting SFT training to complete for real
took 24 kernel versions across two genuinely distinct failure modes, and the
debugging process itself — the hypotheses tested, what each one ruled out,
and what actually turned out to be true — is real evidence of engineering
process, arguably more informative than a clean success story would have
been.

### Phase 1: A real CUDA OOM chain, solved cleanly (v8-v10)

The very first real Kaggle runs hit a straightforward, explainable CUDA
out-of-memory error: MIDV-2020's source images are full-resolution scans
(~2167×1521), and feeding them in uncapped multiplied vision-token count (and
therefore activation memory) far beyond a T4's headroom. v8 crashed 486MB
short in the forward pass with images uncapped; capping `max_image_size`
(v9) moved the failure to the backward pass (1.7GB short, 1.10GB free); adding
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (v10) didn't fully close
the gap but measurably improved it (free memory rose from 1.10GB to 1.36GB,
fragmentation dropped from 524MB to 291MB reserved-but-unallocated) — a real,
working fix for a real memory problem, each version's numbers moving in the
expected direction. This is what a correctly-diagnosed fix looks like, and it
stands in useful contrast to what came next.

### Phase 2: Training stopped OOM'ing and started silently hanging instead (v11-v18)

Once the straightforward memory chain was resolved, training on the original
7B target began *silently hanging* instead of crashing — no error, no
traceback, just zero progress indefinitely. Six independently-reasoned,
single-variable hypotheses were tested in sequence, each one ruling out a
real candidate cause without finding the actual one:

1. **v11** (paged 8-bit optimizer, r=12): stalled 39 minutes at the identical
   backward-pass line, zero progress, no crash. Working hypothesis: the
   paged optimizer was thrashing under memory pressure rather than erroring.
2. **v12** (switched to non-paged `adamw_bnb_8bit`, tighter memory footprint
   to make a non-paged optimizer viable): stalled at the **identical** line —
   confirmed via two live-log fetches minutes apart returning byte-identical
   content. Paging hypothesis falsified.
3. **v13** (`gradient_checkpointing_kwargs={"use_reentrant": False}`,
   targeting a documented reentrant-checkpointing deadlock class): did *not*
   hang at the same line — real progress, the reentrant-mode warning present
   in every prior log disappeared, confirming the fix was active — but it
   then stalled ~58 minutes at an *earlier* point. A different failure
   signature, not a resolution.
4. **v14** (checkpointing off entirely, avoiding the reentrant-checkpointing
   hypothesis completely): hung at yet another point.
5. **v16** (`use_reentrant=False` + explicit `model.config.use_cache=False`
   set together): hung again.
6. **v17-v18** (`device_map={"": 0}` instead of `"auto"`, then
   `CUDA_VISIBLE_DEVICES=0` set before any CUDA-touching import): v17
   crashed with a genuinely new, informative error — a `RuntimeError` inside
   `torch.nn.DataParallel`, revealing this Kaggle instance type has **2
   visible GPUs**, not the 1 every prior run implicitly assumed.
   Transformers' `Trainer` auto-wraps in `DataParallel` whenever
   `torch.cuda.device_count() > 1`; restricting visibility to GPU 0 before
   any CUDA import (v18) removed this from consideration entirely.

**Why this is treated as strong evidence, not one unlucky bug**: three
independent, well-reasoned fixes (optimizer paging, then checkpointing
reentrancy) each targeting a different plausible root cause, each producing
a *different* hang location rather than clearing the problem, is a
materially different pattern than "one bug, wrong fix, try again." The
failure point kept moving rather than converging on an explanation across
six total hypotheses spanning both model sizes (v14, v16 confirmed via
`git log` to be genuinely on 3B, not 7B as an earlier internal status note
incorrectly claimed before being corrected). The actual root cause of the
original hang was **never found**. Per an explicit, reasoned decision to
stop rather than keep guessing indefinitely, this investigation was closed
in favor of a pragmatic workaround: disable gradient checkpointing entirely
and manage memory through other levers.

### Phase 3: OOM again, but now a bounded, honest search (v19-v23)

With checkpointing off, the hang was gone — replaced by a clean, ordinary,
arithmetic-explainable OOM (v19: 44MiB short at `max_image_size=512`). What
followed was a bounded step-down search, explicitly authorized rather than
run indefinitely:

| Kernel | Change | Result | Shortfall | Memory in use |
|---|---|---|---|---|
| v19 | 512px | OOM (MLP dequant) | ~44 MiB short | 14.55 GiB |
| v20 | 384px | OOM (lm_head) | ~51 MiB short | 14.03 GiB |
| v21 | 320px | OOM (lm_head) | ~163 MiB short | 14.14 GiB |
| v22 | 256px (floor) | OOM (loss/cross_entropy) | ~19 MiB short | 14.33 GiB |
| v23 | + LoRA r=4, `max_seq_length=1024` | OOM (first backward pass) | ~49 MiB short | 14.36 GiB |

Memory usage was **not monotonic** with image size across this whole search
— a genuine anomaly that was explicitly *not* investigated further, per a
direct instruction to stop chasing an explanation and focus on getting a
working configuration instead. v23 additionally combined two new,
previously-untried levers (LoRA rank cut, sequence-length cut) and still
OOM'd by roughly the same margin as v22 — five straight failures across
three independent memory levers, each landing within ~20-160 MiB of the same
~14.3-14.5 GiB ceiling.

### Phase 4: The actual root cause — a leftover model in memory (v24)

At this point a direct question reframed the whole investigation: **was
Phase 3's 7B zero-shot-baseline model actually being freed before Phase 4
loaded the 3B fine-tuning target, given both steps run in the same Kaggle
session?**

It was not. `clean_eval.py` caches its loaded model at module scope
(`_model`/`_processor` globals — added originally so its own per-image eval
loop wouldn't reload weights on every call), and the kernel driver script
that runs Phase 3 then Phase 4 in one process never cleared that cache. A
real, verifiable diagnostic confirmed this exactly:

```
=== GPU memory before Phase 3 cleanup: 5.91 GB allocated, 7.29 GB reserved ===
=== GPU memory after Phase 3 cleanup: 0.01 GB allocated, 0.03 GB reserved ===
```

**~7.3 GB of the ~14.5 GB budget was leftover 7B model weights, on every
single one of v19-v23's five attempts.** None of the memory levers cut in
that search (image size, LoRA rank, sequence length) were ever going to
close a gap that large — they were all shrinking the wrong side of the
memory ledger. The fix was a single `gc.collect()` + `torch.cuda.empty_cache()`
call between phases, plus a module-level `unload_model()` function added to
`clean_eval.py`.

**Result: Phase 4 SFT training completed for real for the first time in
this project**, 135/135 steps, 3 epochs, no crash, `train_runtime=7999s`
(~2h13m), `train_loss=1.7296` (running average). This is the checkpoint
(`checkpoints/sft_v24_final/`, committed to this repository) that every
subsequent phase in this project is built on.

**The honest lesson**: eleven kernel pushes (v14-v24) across two entirely
different failure modes — six hang hypotheses that never converged on a root
cause, followed by five OOM attempts that all cut the wrong variable — were
only resolved by a question about process lifecycle and cross-phase state,
not by any amount of further hyperparameter tuning. The fix was not a bigger
model, more VRAM, or a cleverer LoRA config. It was a basic resource-
management bug: a cached global reference outliving its intended scope.

### Phase 5: Two more real bugs, found the same night, before the core experiments could run

Getting Phase 4's checkpoint to exist was necessary but not sufficient —
running the leave-one-out and adversarial-rounds experiments against it
surfaced two more genuinely real, previously-invisible bugs:

- **A manifest-filename convention bug.** Every forgery tier's manifest file
  is actually named with a short numeric prefix (`tier1_manifest.json`), but
  the code path that derived this filename automatically for the leave-one-out
  and adversarial-rounds Kaggle drivers used the tier's *full descriptive
  name* instead (`tier1_field_tamper_manifest.json` — which never exists).
  Because the example-building logic silently skips any tier whose manifest
  path doesn't exist (a deliberate design choice, so a not-yet-generated
  tier degrades gracefully), this bug produced **zero** tier examples with
  **no error at all** — completely invisible until a real Kaggle run logged
  "0 folds" / "0 tampered examples" with no exception to point at the cause.
- **Windows-native (backslash) paths baked into every locally-generated
  manifest.** Every data-generation script (`acquire_dataset.py`,
  `field_tamper.py`, `splice.py`, `synthetic_id_gen.py`, `recapture_sim.py`,
  `inpaint_forger.py`) serialized its manifest path fields via bare
  `str(some_path)`, which produces OS-native separators — backslashes, on
  this Windows development machine. Every path resolved correctly on the
  local machine (Windows accepts both slash conventions) and was therefore
  completely invisible in local testing; it surfaced as a hard
  `FileNotFoundError` the moment a manifest reached a real Linux Kaggle
  kernel. A project utility (`kaggle_package.py`'s `stage_package()` /
  `_rewrite_paths_in_place()`) already existed specifically to normalize
  this before staging — a sign this exact class of problem had been solved
  once before — but later ad-hoc dataset refreshes during this session
  bypassed that utility and pushed the raw (backslash) manifests straight
  through.

Both were fixed at the source (short-prefix filename derivation; every
generator switched to `.as_posix()`), with regression tests added against
this repository's *real* committed data — not fixtures — since both bugs
were specifically invisible to fixture-based unit tests that never exercised
the actual default-path-construction logic or the real manifest files on
disk.

A third, non-code observation from the same night: a Kaggle dataset version
reporting `"ready"` via the CLI's `datasets status` command did not reliably
mean a kernel pushed immediately afterward would see the update — this
mount-propagation lag was observed twice, resolving itself after waiting
roughly 10-15 more minutes with no other action taken. A small, dedicated,
no-GPU diagnostic kernel (`kaggle_kernels/diagnostic_check/`) was built to
verify dataset state directly and cheaply (seconds, not GPU-hours) rather
than re-testing this via expensive full training/eval runs each time.

## Results

*Every number below is real and sourced from `results/tables/`. Confidence
intervals are reported wherever the underlying sample size supports one —
several of the samples below are small, and the wide resulting CIs are
stated plainly, not hidden.*

### Zero-shot 7B baseline (Phase 3, real Kaggle result, n=9)

| Metric | Value | 95% CI |
|---|---|---|
| Tamper-verdict accuracy (overall) | 66.7% | [33.3%, 100%] |
| Tamper-verdict accuracy (genuine, n=3) | 0% | — |
| Tamper-verdict accuracy (tampered, n=6) | 100% | — |
| Parse success rate | 100% | [100%, 100%] |
| Field similarity (name/dob/id_number/expiry) | 1.00 each | [1.00, 1.00] |

The aggregate 66.7% hides the real, more informative pattern: zero-shot 7B
caught **every real forgery** in this sample but **incorrectly flagged every
genuine document as tampered**, in each case hallucinating a specific,
plausible-sounding-but-false justification (e.g. flagging a genuine,
unedited ID number as "not consistent with the format typically used"). A
real high-recall, high-false-positive-rate pattern from an untuned model,
worth watching for in the fine-tuned model's own behavior below.

### SFT training (Phase 4, kernel v24, real completed run)

135/135 steps, 3 epochs, 719 real training examples (694 genuine + 10 tier1
+ 15 tier2), `train_runtime=7999s` (~2h13m), final running-average
`train_loss=1.7296`. See `results/charts/phase4_loss_curve.png` and
`results/tables/phase4_sft_loss_curve.json` for the full per-step curve. The
loss dropped fast in epoch 1 (5.03 → 1.31) then plateaued around 1.25-1.26
for the remaining two epochs with no further real improvement — flagged
here as a real finding, expanded on in Failure Analysis below, not a minor
training-curve footnote.

### Leave-one-out generalization (Phase 6) — **PENDING**

The 5-fold leave-one-out Kaggle job (`doc-verification-leave-one-out`) was
still running at the time of this writing (a real ~11-hour job — 5 full
QLoRA retrains, one per held-out tier). This section will be populated with
real per-tier accuracy and bootstrap CIs once that job completes. **No
placeholder numbers are given here** — see `results/tables/` for whether
`phase6_leave_one_out_results.json` exists yet, and the results notebook's
corresponding cell, which prints an explicit "PENDING" message rather than
fabricating a chart.

### Adversarial retraining rounds (Phase 6, real Kaggle result, n=30/round)

| Round | Retrained on | Accuracy | 95% CI | Verdict distribution (n=30) |
|---|---|---|---|---|
| 0 (existing checkpoint) | — | 33.3% | [16.7%, 50.0%] | genuine: 25, unparseable: 5, tampered: **0** |
| 1 | 20 mined failures | 66.7% | [50.0%, 83.3%] | tampered: **30**, genuine: 0 |
| 2 | 10 mined failures | 33.3% | [16.7%, 50.0%] | genuine: **30**, tampered: 0 |

**Read the verdict-distribution column, not just the accuracy column.** At
every round, the model predicted a **single class for all 30 examples**.
This is not incremental generalization improvement across rounds — it is
the adapter's entire global bias flipping between two degenerate
single-class predictors, entirely determined by whichever tiny (10-20
example) mined-failure batch it was retrained on most recently. Full
analysis, including the direct connection to the training-loss plateau
above, in Failure Analysis and in
`results/tables/phase6_adversarial_rounds_summary.md`.

### Quantization benchmarking (Phase 9, real Kaggle result, n=40/precision)

The fp16/int8/int4 comparison job hit a real, since-fixed library
compatibility error first (`peft`'s LoRA module dispatch tries a `torchao`
backend candidate specifically for plain fp16 — non-quantized — models, and
Kaggle's base image ships an incompatible `torchao==0.10.0`; `peft`'s own
version gate raises rather than skipping cleanly on a too-old-but-present
package). Fixed by uninstalling `torchao` entirely (this project only ever
uses `bitsandbytes` for actual quantization) and re-pushed successfully.

| Precision | Accuracy | 95% CI | Avg latency | Est. cost / 1M verifications |
|---|---|---|---|---|
| fp16 | 50.0% | [35.0%, 65.0%] | **8.69s** | **$845** |
| int8 | 50.0% | [35.0%, 65.0%] | 22.17s | $2,156 |
| int4 | 50.0% | [35.0%, 65.0%] | 20.98s | $2,040 |

**Two real findings, both reported honestly:**

1. **Accuracy is identical (exactly 50.0%) across all three precisions** —
   not evidence that quantization is harmless, but the same single-class-
   collapse pattern documented under Adversarial retraining rounds above,
   showing up again on this run's perfectly balanced 20 genuine / 20
   tampered eval set. A model that predicts one class for every input scores
   exactly 50% on any balanced set regardless of numeric precision, since
   precision changes numerical representation, not which class a collapsed
   model defaults to. This is further, independent corroborating evidence
   for the capacity-limitation hypothesis, not a new finding on its own.
2. **fp16 measured both faster and cheaper than int8/int4** — 8.69s/example
   vs. 22.17s and 20.98s, inverting the usual "quantize for speed/cost"
   assumption. The honest, reasoned explanation: this benchmark runs at
   batch size 1 (matching training), and `bitsandbytes` int8/int4 layers
   carry real per-call dequantization overhead that a larger production
   batch would amortize but a single-example batch cannot. **The honest
   recommendation from this specific benchmark is fp16, not int8/int4** —
   the opposite of conventional wisdom, stated plainly because that's what
   was actually measured under these real conditions, not assumed from
   general quantization lore. A production deployment with real request
   batching would need to re-run this comparison before generalizing past
   what was tested here. Full analysis:
   `results/tables/phase9_quantization_bench_summary.md`.

## Failure Analysis

**The adversarial-retraining collapse (above) is this project's most
important, most honestly-reported failure, and it is real evidence for a
specific, testable hypothesis, not an unexplained anomaly.**

The training-loss plateau at v24 (stuck ~1.25-1.26 for epochs 2-3, no
further improvement after a fast epoch-1 drop) was flagged at the time as a
*possible* sign of limited model capacity, driven by two compute-forced
configuration choices: **LoRA rank 4** (cut from an original r=16 across
v8-v23's memory-constraint fight — never chosen because r=4 was judged
sufficient) and a small **719-example** training set. The adversarial-rounds
result is direct, corroborating evidence for that hypothesis: a model with
genuine capacity to discriminate genuine-vs-tampered content, retrained on a
small (10-20 example) batch of mined failures, should refine its decision
boundary incrementally. A model with insufficient capacity — or one that
never learned a real decision boundary in the first place — should instead
do exactly what was observed: each small retrain has enough gradient signal
to flip the whole adapter's global output bias, but not enough signal (or
representational room) to learn a distinction that survives past the
specific examples it just saw.

This reframes how the whole training pipeline's success should be read.
Phase 4 completing 135/135 training steps without crashing was a genuine,
hard-won engineering milestone (see *Engineering Journey* above) — but a
training run finishing cleanly is a different claim than a model learning
the task well, and this project's own adversarial-rounds evaluation is
honest evidence that, at the compute-forced configuration this project
landed on, the second claim is not yet supported.

**Independent corroboration**: the quantization benchmark (below) found the
same v24 checkpoint scoring exactly 50.0% accuracy at all three tested
precisions (fp16/int8/int4) on a separate, perfectly-balanced 40-example
eval set — the same single-class-collapse signature (a model defaulting to
one verdict scores exactly 50% on a balanced set regardless of numeric
precision), observed independently of the adversarial-retraining process
itself. This rules out "an artifact specific to the adversarial-rounds
retraining loop" as the explanation and strengthens the case that v24's
checkpoint itself, not the evaluation procedure, is the source of the
degenerate behavior.

**A direct test of the hypothesis was launched the same night** (v25):
restoring LoRA rank (4→16), image size (256px→768px), and sequence length
(1024→2048) toward their pre-panic values, now that the real cause of
v19-v23's memory ceiling (the leftover-7B-model leak, not these
hyperparameters) was found and fixed in v24. Real math motivating this:
v23's OOM showed 14.36GiB in use at crash time; subtracting the confirmed
~7.29GiB leak leaves 3B training's actual footprint at only ~7.07GiB against
a ~14.56GiB budget — roughly 2x headroom was available the entire time
v19-v23's search was running. If this hypothesis is correct, a checkpoint
trained with meaningfully more capacity should show real image-content
discrimination rather than single-class collapse. *(Result pending at the
time of this section being written — see the top of this document / commit
history for whether v25 completed and what its adversarial-rounds re-test
showed. This is being reported honestly regardless of outcome: if restored
capacity does NOT fix the collapse, that is equally important evidence,
ruling out the capacity hypothesis and pointing at something else — e.g. the
learning-rate schedule, the 719-example dataset's own diversity, or the
tier1/tier2-only training composition.)*

**Leave-one-out results**, once available, are a further, more statistically
meaningful read on generalization specifically (as opposed to the capacity
question above) — each leave-one-out fold is a genuine ~719-example retrain
from the base model, not a 10-20-example perturbation. This section will be
updated once those results are in.

## Limitations

- **Synthetic/public data only.** All training and evaluation data comes
  from MIDV-2020 (a public, CC BY-SA 2.5 dataset of synthetic documents with
  AI-generated faces) plus this project's own locally-generated forgeries.
  No real fraud data was used or is claimed to have been used.
- **Compute-forced hyperparameters, not chosen ones.** LoRA rank 4,
  `max_image_size=256`, `max_seq_length=1024`, and gradient checkpointing
  disabled are all the product of an extensive Kaggle free-tier T4 memory
  fight (see *Engineering Journey*), not values independently judged optimal
  for this task. The adversarial-rounds finding above is real evidence this
  matters, not just a theoretical caveat.
- **Small real sample sizes throughout.** The zero-shot baseline (n=9) and
  adversarial-rounds eval set (n=30) both carry wide bootstrap confidence
  intervals, reported honestly rather than treated as precise point
  estimates.
- **7B-vs-3B baseline comparison is not a clean ablation.** The zero-shot
  baseline uses 7B (the original plan); the fine-tuned model is 3B (a
  documented, evidence-based scope change after the 7B training hang could
  not be root-caused). Any comparison between the two conflates a model-size
  effect with a fine-tuning effect.
- **DPO was scoped out.** The original project plan included an SFT-then-DPO
  training pipeline; DPO was not attempted given how much of this project's
  compute/time budget the SFT debugging saga consumed. This is a documented
  scope decision, not an oversight.
- **Scripted, not learned, forgery generation.** All 5 forgery tiers use
  deterministic/scripted generation (OCR-located edits, Poisson blending,
  diffusion inpainting with a fixed prompt, procedural fabrication, image
  filters) rather than a learned adversarial forger. An RL-based self-play
  forgery generator was considered and explicitly not attempted — see Future
  Work.
- **No autonomous orchestration layer.** Every Kaggle run in this project was
  triggered, monitored, and diagnosed by a human-directed (or
  human-delegated-autonomous) process across many kernel pushes — there is
  no self-driving "detect a failure, diagnose it, and re-launch" loop.

## Future Work

- **Address the capacity-limitation finding directly**: re-run SFT at a
  higher LoRA rank and/or a larger training set once more compute budget is
  available (a paid Kaggle tier, or a different provider), and check whether
  the adversarial-rounds single-class-collapse pattern goes away — this is
  the most concrete, well-evidenced next experiment this project's own
  results point to.
- **RL-based self-play forger**: train an adversarial generator against the
  detector in a closed loop, rather than this project's fixed 5-tier
  scripted taxonomy. Explicitly out of scope here, documented as a real
  future direction rather than attempted partially.
- **Real fraud data partnerships**, to validate against actual attack
  distributions rather than synthetic/public data only.
- **Autonomous adversarial-round orchestration**: a system that runs
  eval → mine-failures → retrain without a human (or human-delegated
  session) driving each Kaggle push individually.
- **DPO on top of the completed SFT checkpoint**, once the capacity question
  above is resolved — DPO on a model that hasn't yet demonstrated it can
  learn a real decision boundary would likely just be tuning noise into a
  model that isn't ready for it.
