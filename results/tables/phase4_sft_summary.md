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
