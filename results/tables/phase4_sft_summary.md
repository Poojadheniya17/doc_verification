# Phase 4 SFT Summary

## What was built

- `src/training/sft_train.py`: builds SFT training examples from the genuine +
  Tier 1/2 manifests, formats them as Qwen2.5-VL chat conversations (image +
  extraction/tamper-verdict/localization JSON target), and — when
  `environment=kaggle` — loads Qwen2.5-VL-7B-Instruct in 4-bit (QLoRA per
  `config/model_config.yaml`), wraps it with a LoRA adapter, and fine-tunes via
  `transformers.Trainer`.
- `src/training/checkpoint_utils.py`: adapter-only save/load (QLoRA only trains
  the LoRA weights, so checkpoints are a few tens of MB, not the full base
  model) plus step-numbered checkpoint discovery for resuming.

## What was actually run locally (and what wasn't)

Per the standing RAM constraint (see `results/tables/phase3_baseline_summary.md`,
README.md), **no model loading happens on this machine**. What *was* run for
real, locally:

- `build_sft_examples()` against the real Phase 2/3 manifests: **719 training
  examples** (700 genuine + 8 Tier 1 + 11 Tier 2, `split="train"`) built and
  inspected — real data, real JSON targets, zero mocking.
- `python -m src.training.sft_train --environment local` end-to-end: builds the
  719 examples, then exits cleanly *before* touching any model weights (see the
  `environment == "local"` early-return in `train()`).
- Unit tests (`tests/test_pipeline_smoke.py`) cover the pure logic: Tier 1
  field-override matching (does a tamper's OCR text get correctly attributed to
  the right schema field, or correctly left alone if it hits something outside
  the 5 tracked fields like an MRZ line), example construction from fixture
  manifests, conversation formatting, and checkpoint-directory discovery.

What was **not** run: `load_model_for_training()`, the actual QLoRA fine-tuning
loop, and anything touching real Qwen2.5-VL weights. That happens on Kaggle.

## Honest note on the Tier-1 field-override heuristic

`_apply_tier1_field_overrides()` matches a tamper's OCR text back to one of the
5 schema fields by fuzzy string similarity (threshold 0.6). Verified on a real
example: a genuine expiry `"04.11.2026."` tampered by `field_tamper.py` into
`"14.16.0026 ."` was correctly matched and substituted into the `expiry`
field, while a separate tamper on the MRZ line was correctly left unmatched
(not one of the 5 fields). This is a known simplification, not a guarantee —
short or heavily garbled OCR text could fail to clear the similarity threshold
even when it did hit a real field, silently leaving that field at its
pre-tamper value in the training target. Worth re-checking once real training
numbers come back from Kaggle if extraction accuracy on Tier-1 examples looks
worse than expected.

## Scale note

719 examples come from the same smoke-scale Tier 1 (10 images) / Tier 2 (15
images) batches generated in Phase 2 — not the full ~1000-genuine-image
manifest run through all forgery tiers. Real Kaggle training should regenerate
Tier 1/2 at full scale first (`tamper_manifest`/`splice_manifest` without a
`limit`) for a properly sized training set.

## The real Kaggle training saga: 7B hang, decision to switch to 3B

Once local logic was validated (above), real training moved to Kaggle. This
section is the honest, complete record of what actually happened — including
the part that didn't work, per this project's stated quality bar of
documenting failures rather than burying them.

**Kernels v8-v10 — genuine CUDA OOM, each fix confirmed working:**

| Kernel | Config | Result |
|---|---|---|
| v8 | 1280px, LoRA r=16, `paged_adamw_8bit` | OOM in forward pass, 486MB short (image resize was uncapped — a real bug) |
| v9 | 1280px (resize now capped), r=16, `paged_adamw_8bit` | OOM in backward pass, 1.7GB short, 1.10GB free |
| v10 | same + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | Same OOM, 1.7GB short, but free memory rose to 1.36GB and fragmentation ("reserved but unallocated") dropped 524MB->291MB — the fix demonstrably worked, just wasn't sufficient alone |

Each fix visibly moved the numbers in the expected direction. This is what a
real, working fix for a real memory problem looks like — contrast with what
came next.

**Kernels v11-v13 — training stopped OOM'ing and started silently hanging
instead, and three different targeted fixes each moved the hang rather than
resolving it:**

| Kernel | Config | Result |
|---|---|---|
| v11 | 896px, LoRA r=12, `paged_adamw_8bit` | No OOM. Silently stalled at the backward-pass entry line for **39 minutes**, zero progress, no crash. Working hypothesis: paged optimizer paging to host memory under pressure, thrashing rather than erroring. |
| v12 | 768px, LoRA r=8, **non-paged** `adamw_bnb_8bit` (removes the paging hypothesis) | Stalled at the **identical** line, confirmed via two live-log fetches minutes apart returning byte-identical content. Paging hypothesis falsified. |
| v13 | same as v12 + `gradient_checkpointing_kwargs={"use_reentrant": False}` (targets a documented reentrant-checkpointing deadlock specific to QLoRA's frozen-base + trainable-adapter parameter mix) | Did NOT hang at the same line — progress: the "use_reentrant should be passed explicitly" warning present in every prior log disappeared, confirming the fix was active. But it then stalled for **~58 minutes** at an *earlier* point (before even reaching the backward-pass line), a different failure signature, not a resolution. |

**Why this is treated as strong evidence, not one unlucky bug:** three
independent, individually well-reasoned fixes (optimizer paging, then
checkpointing reentrancy) each targeting a different plausible root cause,
each producing a *different* hang location rather than clearing the problem,
is a materially different pattern than "one bug, wrong fix, try again." A
shifting failure point across independently-motivated fixes points at
something structural in the T4 / driver / library-version interaction at 7B
scale in this specific environment — not a config value away from working.

**Decision:** stop debugging 7B SFT training. Switch the SFT (and future DPO)
training target to Qwen2.5-VL-**3B**-Instruct (`config/model_config.yaml`).
Phase 3's 7B zero-shot baseline succeeded and is kept as the reported
baseline number as-is — it is not re-run on 3B.

**Comparison caveat, stated explicitly:** the eventual 3B fine-tuned results
will be compared against a 7B zero-shot baseline. That comparison conflates
two effects (the fine-tuning gain, and a model-size difference) and is not a
clean ablation. This must be flagged every place the numbers are shown
side by side (README, results notebook, final writeup) — not presented as if
it were a same-model before/after.

**What was deliberately NOT changed alongside the model swap:** `max_image_size`
(768px) and LoRA rank (r=8) stayed at their most-aggressive, 7B-hang-driven
values rather than being relaxed back up for 3B's much larger memory
headroom. This was a deliberate choice to change exactly one variable (the
model) in the push that's meant to finally produce a completed run, not a
belief that these are the right values for 3B long-term. Revisiting them
upward — for better OCR legibility at higher resolution, or a higher LoRA
rank now that there's VRAM to spare — is flagged here as a reasonable,
clearly-labeled follow-up once a 3B baseline run has actually completed.

## 3B training: the hang continues, then a new OOM chain (v14-v23)

**v14-v18 — same silent-hang failure mode reappeared on 3B, across five more
single-variable fixes:**

| Kernel | Config | Result |
|---|---|---|
| v14 | 3B, checkpointing off (avoids the reentrant-checkpointing hypothesis entirely) | Hung at yet another point — ruled out checkpointing as sole cause |
| v16 | 3B, `use_reentrant=False` + explicit `model.config.use_cache=False` set together (the exact combination later reconsidered for v23 — see below) | Hung again |
| v17 | 3B, `device_map={"": 0}` | Crashed with a `RuntimeError` inside `torch.nn.DataParallel` — revealed this Kaggle instance has 2 visible GPUs, not 1 |
| v18 | 3B, `CUDA_VISIBLE_DEVICES=0` set before any CUDA import (removes DataParallel auto-wrap entirely) | No more DataParallel crash, but still no completed run |

Six single-variable hypotheses (v11-v18, spanning both model sizes) were each
tested and ruled out in turn. The actual root cause of the original hang was
never found. Per explicit user direction, this investigation was stopped —
"we've spent significant time on the training hang... no more open-ended
debugging loops" — in favor of a pragmatic workaround: disable gradient
checkpointing entirely and manage memory through other levers instead.

**v19-v22 — bounded image-size step-down search, checkpointing off. Every
attempt OOM'd:**

| Kernel | max_image_size | Result | Shortfall | Total GPU memory in use |
|---|---|---|---|---|
| v19 | 512px | OOM (MLP dequant) | ~44 MiB short | 14.55 GiB |
| v20 | 384px | OOM (lm_head) | ~51 MiB short | 14.03 GiB |
| v21 | 320px | OOM (lm_head) | ~163 MiB short | 14.14 GiB |
| v22 | 256px (the floor — lower would make field-extraction meaningless) | OOM (loss/cross_entropy) | ~19 MiB short | 14.33 GiB |

Memory usage was **not monotonic** with image size (14.55 -> 14.03 -> 14.14
-> 14.33 GiB) — never explained, not investigated further per explicit user
instruction ("we don't need to understand why, we just need this to work").

**v23 — LoRA rank r=8->4 + `max_seq_length` 2048->1024, image size held at
the 256px floor, checkpointing still off:**

Before pushing v23, re-enabling checkpointing (the "untested gap" originally
listed as option (a)) was reconsidered and correctly rejected: v16 (above)
already ran the exact `use_reentrant=False` + explicit `use_cache=False`
combination on 3B, and it hung. That combination is a known-bad configuration,
not an untested one — an error in this document's earlier framing, caught
before it led to wasted GPU time.

Also found and fixed while implementing v23: `max_seq_length` in
`training_config.yaml` had been a **decorative value only** since Phase 4's
first kernel — `sft_train.py`'s `collate()` never actually passed it to the
processor, so every kernel from v8 through v22 trained with the full
untruncated sequence regardless of what the config said. Fixed by adding
`truncation=True, max_length=max_seq_length` to the `processor(...)` call.

Result: **OOM again.** `trainable params: 9,288,192 || all params:
3,763,911,168 || trainable%: 0.2468` confirms the rank cut took effect.
Crashed on the very first training step's backward pass:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 252.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 202.81 MiB is free.
Including non-PyTorch memory, this process has 14.36 GiB memory in use.
Of the allocated memory 14.03 GiB is allocated by PyTorch, and 193.47 MiB
is reserved by PyTorch but unallocated.
```

Shortfall ~49 MiB — slightly **worse** than v22's ~19 MiB, and total memory
in use (14.36 GiB) essentially unchanged from v22 (14.33 GiB) despite halving
LoRA rank and halving max sequence length. Both new levers, genuinely wired
in this time, moved the needle by less than noise.

**Where this leaves things:** five bounded, well-reasoned attempts (v19-v23)
across three different memory levers (image size, LoRA rank, sequence length)
have now landed within ~50-160 MiB of the same ~14.3-14.5 GiB ceiling on this
T4, every single time, with no clear trend as any one lever is cut further.
Combined with the six earlier hang hypotheses (v11-v18) that also never
converged on a root cause, this is a strong enough pattern to treat as a
genuine stop-and-report point rather than another autonomous cut — per this
project's own standing rule not to loop indefinitely on a resistant failure.
The remaining options are no longer cheap, low-risk config tweaks; they are
real structural decisions (a materially smaller/different base model, a
paid Kaggle tier or different compute provider, or accepting a reduced
training scope) that need the user's judgment, not another bounded search.

## v24: the real root cause, and the first completed training run

Before committing to a structural decision, the user asked one more targeted
question: was Phase 3's 7B model actually being freed before Phase 4 loaded
the 3B model, given `kaggle_kernels/phase3_4_sft_baseline/kernel_driver.py`
runs both in the same process? It was not — and worse than a simple
oversight: `src/eval/clean_eval.py` caches its model/processor at **module
scope** (`_model`/`_processor` globals, originally added so Phase 3's own
per-image loop wouldn't reload weights). Since `kernel_driver.py` keeps that
module imported for the rest of the process, the 7B model's reference never
went away after Phase 3 finished. This fully explains v19-v23's mystery:
three independently well-reasoned memory cuts to the 3B training (image
size, LoRA rank, sequence length) barely moved the OOM ceiling because none
of them touched the actual problem — the ceiling was leftover 7B weights,
not 3B training's own footprint.

**Fix (v24):** added `clean_eval.unload_model()` (clears the cached globals,
`gc.collect()`, `torch.cuda.empty_cache()`), called from `kernel_driver.py`
between phases, with real before/after GPU memory diagnostic prints so the
result would be verified rather than assumed. v23's training settings (LoRA
r=4, `max_seq_length=1024`, `max_image_size=256`, checkpointing off) were
otherwise unchanged — a single targeted test of one new hypothesis.

**The diagnostic confirmed it exactly:**

```
=== GPU memory before Phase 3 cleanup: 5.91 GB allocated, 7.29 GB reserved ===
=== GPU memory after Phase 3 cleanup: 0.01 GB allocated, 0.03 GB reserved ===
```

~7.3 GB of the ~14.5 GB budget was leftover 7B model on every one of
v19-v23's attempts — a real, root-caused, confirmed explanation, not a
guess.

**Result: Phase 4 SFT training completed successfully — the first
completed training run in this entire project.** 135/135 steps, all 3
epochs, no crash:

```
{'train_runtime': 7998.9749, 'train_samples_per_second': 0.27,
 'train_steps_per_second': 0.017, 'train_loss': 1.7295982360839843, 'epoch': 3.0}
```

- Total training wall-clock: 7999 seconds (~2h 13m).
- `train_loss` (Trainer's running average over the whole run, including the
  high-loss early steps) = 1.7296.
- Loss trajectory (from the log): 5.03 (epoch 0.22) → 3.12 (0.45) → 1.76
  (0.67) → 1.41 (0.89) → 1.31 (1.11) → 1.29 (1.33) → 1.28 (1.56) → 1.27
  (1.78) → 1.26 (2.00) → 1.25 (2.22) → 1.25 (2.45) → 1.25 (2.67) → 1.26
  (2.89). Fast drop in epoch 1, then a plateau around 1.25-1.26 for the
  remaining two epochs — an honest flag for the writeup: this could mean the
  model reached its effective capacity given LoRA r=4 and 719 training
  examples, not necessarily a red flag, but not clearly still improving
  either. Worth checking against real eval numbers once
  `leave_one_out_eval.py`/`clean_eval.py` are run against this checkpoint.
- `trainable params: 9,288,192 || all params: 3,763,911,168 || trainable%:
  0.2468` confirms the LoRA r=4 configuration was genuinely applied.

**Checkpoint:** saved to
`C:\Users\Acer\OneDrive\Desktop\hyperVerge\kaggle_run_output_v24\checkpoints\sft\`
— `checkpoint-45/`, `checkpoint-90/`, `checkpoint-135/` (~55MB each, include
optimizer/scheduler state) and `final/` (36MB, adapter weights only — the
one for downstream use). Not committed into git per the repo's existing
`.gitignore` convention (checkpoints tracked outside git deliberately,
predating this session) — flagged to the user as a real option to revisit
since 36MB is well under GitHub's 100MB file limit.

**The honest throughline for the writeup:** eleven kernel pushes (v14-v24)
across two different failure modes — six single-variable hang hypotheses
that never converged on a root cause (v11-v18, unresolved, documented as
such), followed by five OOM attempts that all shrank the wrong variable
(v19-v23) — were only resolved once the user asked a question about process
lifecycle and cross-phase state that no amount of config-level cutting could
have found. That is worth stating plainly in the final paper: the fix was
not a bigger model, more VRAM, or a cleverer LoRA config — it was a basic
resource-management bug (a cached global reference outliving its scope)
that had nothing to do with any of the levers being tuned.

## v25: capacity restoration — a real success, and a real unresolved anomaly

With the true root cause of v19-v23's ceiling understood (leftover 7B model,
not insufficient 3B capacity), LoRA rank/image size/sequence length were
restored toward pre-panic values, motivated by real math: v23's OOM showed
14.36GiB in use at crash; subtracting the confirmed ~7.29GiB leak leaves 3B
training's actual footprint at only ~7.07GiB against a ~14.56GiB budget —
roughly 2x headroom was available the whole time v19-v23's search ran.

**Three attempts at r=16, each OOM'ing with byte-identical numbers**, are
documented in `writeup/project_report.md`'s Failure Analysis (768px/seq=2048,
384px/seq=1536, and an isolated-rank retry at 256px/seq=1536 — all three:
44.00 MiB requested, 8.81 MiB free, 14.55 GiB total in use, verified via a
diagnostic kernel to rule out a stale-config artifact). This was written up
as "conclusively attributing the cause to LoRA rank r=16 itself."

**A fourth push (intended to be the safe, final r=8/512px/seq=2048 config)
was, due to a real process error, actually a fourth r=16 attempt — and this
one succeeded.** The mistake: `kaggle datasets status` was checked and
reported "ready" before the r=8 dataset push had actually finished
propagating to a freshly-started kernel's mount (a real, previously-observed
Kaggle infrastructure quirk — see PROJECT_STATUS.md). The kernel
(`poojadheniya/doc-verification-zero-shot-baseline-sft-qlora`, kernel version
9) started training against the **stale, pre-push r=16 config** — confirmed
beyond doubt by two independent pieces of ground truth, not inference:

- Trainable-param count at training start: `37,152,768` (matches r=16
  exactly; r=8 would show ~18.5M).
- The saved checkpoint's real `adapter_config.json`: `"r": 16, "lora_alpha":
  32` — read directly from the downloaded Kaggle output, not assumed.

**This run completed all 135/135 steps, all 3 epochs, no crash:**

```
{'train_runtime': 8397.8802, 'train_samples_per_second': 0.257,
 'train_steps_per_second': 0.016, 'train_loss': 1.5124582361291956, 'epoch': 3.0}
=== Phase 4 peak GPU memory (this training run only, Phase 3's usage excluded):
10.17 GB allocated, 10.51 GB reserved ===
```

- Total wall-clock: 8398 seconds (~2h20m) — comparable to v24's 2h13m.
- Loss trajectory: 4.39 (0.22) → 1.77 (0.45) → 1.31 (0.67) → 1.28 (0.89) →
  1.25 (1.11) → 1.24 (1.33) → 1.24 (1.56) → 1.23 (1.78) → 1.22 (2.00) → 1.22
  (2.22) → 1.22 (2.45) → 1.22 (2.67) → plateaus at **~1.217-1.220** for the
  last two epochs — genuinely **lower** than v24's r=4 plateau (~1.25-1.26).
  Real evidence the restored capacity (r=16 vs r=4) let the model fit the
  training data measurably better, independent of the generalization
  question tested next via adversarial-rounds.
- Peak GPU memory (10.17GB allocated / 10.51GB reserved) is a real,
  **directly-measured** number from this run's own diagnostic print — **not
  an inference**. It sits comfortably ~4GB below the ~14.5GiB ceiling that
  killed 3 prior r=16 attempts at the exact same rank.

**The honest, unresolved contradiction, stated plainly:** the same LoRA rank
(r=16), on the same base model, same library versions, same Kaggle T4 tier,
failed identically three times and then succeeded with a ~4GB safety margin
on the fourth attempt. This project's own Failure Analysis previously stated
r=16 was "conclusively" isolated as the cause of catastrophic memory growth —
that conclusion must be walked back to **not proven**, given this direct
counterexample. At the same time, three prior byte-identical failures are
also real data — this is not proof r=16 is now safe or reliable either. Both
facts are true simultaneously and both are reported here, rather than
resolving the tension by picking whichever conclusion is more convenient.

**What could not be determined, labeled honestly as a real information
gap:** `kernel_driver.py`'s Phase 4 logging does not print `max_image_size`
or `max_seq_length` directly, so — unlike LoRA rank, which is verified via
`adapter_config.json` — this run's actual image size / sequence length
cannot be confirmed from any saved artifact. The three failed r=16 attempts
already demonstrated that varying image size (768→384→256px) and sequence
length (2048→1536) made **no measurable difference** to their identical OOM
point, which argues against those two variables being the reconciling factor
here either. The best-supported honest explanations, in order of plausibility,
given the evidence available:

1. **Real Kaggle infrastructure/session variance** — a different underlying
   GPU instance, driver memory overhead, or session freshness (this kernel
   was deleted and re-pushed fresh partway through this investigation per
   PROJECT_STATUS.md's documented recovery procedure for
   "Maximum batch GPU session count reached" errors) — plausible given the
   3 failures were closely spaced in time/session state and this success came
   after an intervening delete+repush, but not verifiable after the fact.
2. **An unidentified config or environment difference** introduced by the
   dataset mid-propagation state itself — possible but unproven, since the
   demonstrated insensitivity of the 3 failures to image_size/seq_length
   argues against those specific values being the answer.

Neither explanation is confirmed. This is reported as a genuine open
question, per this project's standing rule against papering over
inconvenient or confusing real results.

**Checkpoint:** `checkpoints/sft_v25_final/` (adapter, r=16). Unlike v24's
36MB checkpoint, this adapter's `adapter_model.safetensors` is **141.8MB —
over GitHub's 100MB hard per-file push limit** (r=16 roughly quadruples
trainable params vs v24's r=4: 37.15M vs 9.29M). No Git LFS is set up in
this repo, and standing one up for a single file wasn't judged worth the
added complexity — so only the small metadata (`adapter_config.json`,
`README.md`) is committed to git; the real weights live in the Kaggle
kernel output (kernel version 9) and this dev machine's local disk. A
documented scoping decision, not an oversight.

**Real, important scope note on this run's training data:** this run used
the exact same 719-example, tier1+tier2-only composition as v24 (700
genuine + 8 tier1 + 11 tier2 — confirmed via the log's "Built 719 SFT
training examples" line, identical to v24's). It did **not** test the
class-imbalance hypothesis — it tests capacity in isolation, holding the
same severe (~35:1) class imbalance constant. The next real test
(adversarial-rounds against this checkpoint) answers: does more capacity
alone fix the single-class collapse, or does the same imbalance still win?
