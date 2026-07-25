# Phase 4 SFT summary

## What was built

- `src/training/sft_train.py`: builds SFT training examples from the genuine
  and forgery-tier manifests, formats them as Qwen2.5-VL chat conversations
  (image + extraction/tamper-verdict/localization JSON target), and — when
  `environment=kaggle` — loads the model in 4-bit (QLoRA per
  `config/model_config.yaml`), wraps it with a LoRA adapter, and fine-tunes
  via `transformers.Trainer`.
- `src/training/checkpoint_utils.py`: adapter-only save/load plus
  step-numbered checkpoint discovery for resuming.

## Local validation

This dev machine has ~7.7GB RAM and can't load any size of Qwen2.5-VL, so
everything model-related runs on Kaggle. What's actually testable locally is
the data-construction logic: `build_sft_examples()` against the real Phase
2/3 manifests (719 examples for the original tier1+tier2 composition — 700
genuine, 8 tier1, 11 tier2), and `python -m src.training.sft_train
--environment local`, which builds the examples and exits cleanly before
touching any weights. `tests/test_pipeline_smoke.py` covers the pure logic:
tier1 field-override matching, example construction from fixture manifests,
conversation formatting, checkpoint-directory discovery.

One thing worth flagging: `_apply_tier1_field_overrides()` matches a
tamper's OCR text back to a schema field by fuzzy string similarity
(threshold 0.6). Verified on a real example — a genuine expiry
`"04.11.2026."` tampered into `"14.16.0026."` matched correctly to the
`expiry` field, and a separate MRZ-line tamper correctly matched nothing
(not one of the 5 tracked fields). Short or heavily garbled OCR text could
still fail to clear the threshold even when it did hit a real field, which
would silently leave that field at its pre-tamper value in the training
target.

## Getting a training run to actually complete

Real training runs on Kaggle. Getting one to finish took eleven kernel
pushes across two distinct failure modes.

**v8-v10 — genuine OOM, each fix visibly working:**

| Kernel | Config | Result |
|---|---|---|
| v8 | 1280px, LoRA r=16, `paged_adamw_8bit` | OOM in forward pass, 486MB short — image resize was uncapped |
| v9 | 1280px (resize capped), r=16 | OOM in backward pass, 1.7GB short |
| v10 | + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments` | Same OOM, but free memory rose and fragmentation dropped 524MB→291MB — real progress, just not enough |

**v11-v13 — stopped OOM'ing, started silently hanging instead, and each
targeted fix moved the hang rather than fixing it:**

| Kernel | Config | Result |
|---|---|---|
| v11 | 896px, r=12, `paged_adamw_8bit` | Stalled 39 minutes at the backward-pass entry, no crash |
| v12 | 768px, r=8, non-paged `adamw_bnb_8bit` | Stalled at the identical line — rules out the paging hypothesis |
| v13 | + `use_reentrant=False` | Stopped hanging at that line, but stalled ~58 minutes at an earlier point instead |

Three independently-motivated fixes, three different hang locations, never a
resolution — that pattern points at something structural in the T4/driver/
library-version combination at 7B scale, not a config value away from
working. Decided to stop debugging 7B SFT and move the training target to
3B instead (`config/model_config.yaml`). The 7B zero-shot baseline from
Phase 3 stays as the reported baseline number — not re-run on 3B, which
means later 3B fine-tuned numbers get compared against a 7B zero-shot
baseline. That conflates a fine-tuning gain with a model-size difference and
isn't a clean ablation; flagged everywhere the numbers appear side by side.

**v14-v18 — same hang reappeared on 3B, five more single-variable fixes,
none of them the answer:**

| Kernel | Config | Result |
|---|---|---|
| v14 | 3B, checkpointing off | Hung at a different point |
| v16 | 3B, `use_reentrant=False` + `use_cache=False` | Hung again |
| v17 | 3B, `device_map={"": 0}` | Crashed inside `torch.nn.DataParallel` — turned out this instance has 2 visible GPUs |
| v18 | 3B, `CUDA_VISIBLE_DEVICES=0` set before any CUDA import | No more DataParallel crash, still no completed run |

Six single-variable hypotheses across both model sizes, root cause never
found. Stopped that investigation and switched to a pragmatic workaround
instead: disable gradient checkpointing entirely and manage memory through
other levers.

**v19-v22 — bounded image-size search, checkpointing off, every attempt
OOM'd:**

| Kernel | max_image_size | Result | Shortfall | Total memory in use |
|---|---|---|---|---|
| v19 | 512px | OOM (MLP dequant) | ~44 MiB short | 14.55 GiB |
| v20 | 384px | OOM (lm_head) | ~51 MiB short | 14.03 GiB |
| v21 | 320px | OOM (lm_head) | ~163 MiB short | 14.14 GiB |
| v22 | 256px (floor) | OOM (cross_entropy) | ~19 MiB short | 14.33 GiB |

Memory usage wasn't monotonic with image size (14.55 → 14.03 → 14.14 →
14.33 GiB), never explained.

**v23 — LoRA r=8→4 + max_seq_length 2048→1024, 256px floor.** Also found
while implementing this: `max_seq_length` had been a decorative config
value the whole time — `collate()` never actually passed it to the
processor, so every kernel from v8 through v22 trained on the full
untruncated sequence regardless of what the config said. Fixed by adding
`truncation=True, max_length=max_seq_length`. Result: OOM again, ~49 MiB
short, total memory essentially unchanged from v22 despite halving both
rank and sequence length.

Five bounded attempts across three memory levers, all landing within
~50-160 MiB of the same ~14.3-14.5 GiB ceiling, no clear trend. Combined
with the six hang hypotheses that never converged, this was the point to
stop cutting config values and figure out what was actually happening.

## v24: the real cause, and the first completed run

`kernel_driver.py` runs Phase 3 (7B zero-shot baseline) and Phase 4 (3B SFT)
back to back in the same process — worth checking whether the 7B model was
actually being freed in between. It wasn't, and not just as a simple
oversight: `clean_eval.py` caches its model at module scope
(`_model`/`_processor` globals, added so Phase 3's own per-image loop
wouldn't reload weights), and since `kernel_driver.py` keeps that module
imported for the rest of the process, the 7B model's reference never went
away. That explains v19-v23's whole mystery — none of those memory cuts
touched the actual problem, because the ceiling was leftover 7B weights,
not 3B training's own footprint.

Fix: added `clean_eval.unload_model()` (clears the cached globals,
`gc.collect()`, `torch.cuda.empty_cache()`), called between phases, with
before/after memory diagnostics so the fix would be verified rather than
assumed:

```
=== GPU memory before Phase 3 cleanup: 5.91 GB allocated, 7.29 GB reserved ===
=== GPU memory after Phase 3 cleanup: 0.01 GB allocated, 0.03 GB reserved ===
```

~7.3GB of the ~14.5GB budget was leftover 7B model on every one of v19-v23's
attempts. Training completed — 135/135 steps, all 3 epochs, no crash:

```
{'train_runtime': 7998.9749, 'train_samples_per_second': 0.27,
 'train_steps_per_second': 0.017, 'train_loss': 1.7295982360839843, 'epoch': 3.0}
```

2h13m wall-clock. Loss trajectory: 5.03 (epoch 0.22) → 3.12 (0.45) → 1.76
(0.67) → 1.41 (0.89) → 1.31 (1.11) → 1.29 (1.33) → 1.28 (1.56) → 1.27 (1.78)
→ 1.26 (2.00) → 1.25 (2.22) → 1.25 (2.45) → 1.25 (2.67) → 1.26 (2.89). Fast
drop in epoch 1, then a plateau around 1.25-1.26 — could mean the model hit
its effective capacity given LoRA r=4 and 719 training examples, not
necessarily a problem but not obviously still improving either.
`trainable params: 9,288,192 || all params: 3,763,911,168 || trainable%:
0.2468` confirms r=4 was actually applied. Checkpoint committed at
`checkpoints/sft_v24_final/` (36MB, well under GitHub's limit).

The eleven pushes it took to get here split into two failure modes: six hang
hypotheses that never converged (v11-v18), then five OOM attempts that kept
shrinking the wrong variable (v19-v23). The actual fix wasn't a bigger
model, more VRAM, or a cleverer LoRA config — it was a cached global
reference outliving its scope, something no amount of config-level cutting
could have found.

## v25: does more LoRA capacity help?

With the real cause of v19-v23's ceiling understood, it was worth restoring
rank/image size/sequence length toward less aggressive values — v23's OOM
showed 14.36GiB in use at crash, and subtracting the confirmed ~7.29GiB leak
puts 3B training's actual footprint at only ~7.07GiB against a ~14.56GiB
budget. Roughly 2x headroom was available the whole time the v19-v23 search
ran.

Three attempts at r=16 all OOM'd with byte-identical numbers (768px/seq2048,
384px/seq1536, and an isolated-rank retry at 256px/seq1536 — all three:
44.00 MiB requested, 8.81 MiB free, 14.55GiB total in use, verified via a
diagnostic kernel to rule out a stale-config artifact). That looked like a
clean isolation of LoRA rank as the cause.

A fourth push, meant to be the safer r=8/512px/seq2048 config, actually ran
as a fourth r=16 attempt by accident — a Kaggle dataset version reported
"ready" before it had actually finished propagating to a freshly-started
kernel's mount. This one completed. Confirmed two ways, not assumed:
trainable-param count at training start was 37,152,768 (matches r=16
exactly), and the saved checkpoint's real `adapter_config.json` shows `"r":
16, "lora_alpha": 32`.

```
{'train_runtime': 8397.8802, 'train_samples_per_second': 0.257,
 'train_steps_per_second': 0.016, 'train_loss': 1.5124582361291956, 'epoch': 3.0}
=== Phase 4 peak GPU memory: 10.17 GB allocated, 10.51 GB reserved ===
```

2h20m wall-clock, comparable to v24. Loss trajectory: 4.39 (0.22) → 1.77
(0.45) → 1.31 (0.67) → 1.28 (0.89) → 1.25 (1.11) → 1.24 (1.33) → 1.24 (1.56)
→ 1.23 (1.78) → 1.22 (2.00) → 1.22 (2.22) → 1.22 (2.45) → 1.22 (2.67),
plateauing around 1.217-1.220 for the last two epochs — measurably lower
than v24's r=4 plateau. Peak memory of 10.51GB reserved sits comfortably
~4GB below the ceiling that killed the three earlier r=16 attempts at the
identical rank.

That's a real contradiction worth sitting with rather than resolving in
whichever direction is more convenient: the same rank, same base model,
same library versions, same T4 tier failed identically three times and then
succeeded with a 4GB margin on the fourth try. The earlier conclusion that
r=16 was "conclusively" the cause of the memory blowup doesn't hold up
against this — it should read as not proven either way, since three
identical failures are also real data. `kernel_driver.py`'s logging doesn't
print image size or sequence length directly, so unlike LoRA rank (verified
via `adapter_config.json`), there's no way to confirm exactly what this
particular run used. The three failed r=16 attempts already showed that
varying image size and sequence length made no measurable difference to
where they crashed, which argues against those variables being the
reconciling factor here either. Best guess: real session-to-session
variance in the underlying Kaggle GPU instance or driver overhead — this
kernel had been deleted and re-pushed fresh partway through the
investigation, so it wasn't running in the same session state as the three
failures. Not verifiable after the fact, and left as an open question
rather than a resolved one.

Checkpoint: `checkpoints/sft_v25_final/`. Unlike v24, this adapter's
`adapter_model.safetensors` is 141.8MB — over GitHub's 100MB push limit
(r=16 roughly quadruples trainable params vs r=4: 37.15M vs 9.29M). Only the
metadata (`adapter_config.json`, `README.md`) is committed; the real
weights live in the Kaggle kernel output and on local disk.

This run used the exact same 719-example, tier1+tier2-only composition as
v24, so it tests capacity in isolation while holding the same ~35:1 class
imbalance constant. Running adversarial-rounds against it gives the answer:
identical single-class collapse to v24, confirmed prediction-by-prediction
(see `phase6_adversarial_rounds_summary.md`). More capacity alone doesn't
fix it.

## v26: fixing the class imbalance

Next experiment: oversample tampered examples to a real 1:1 ratio
(`sft_train.balance_examples()`), across all 5 forgery tiers instead of
just 2, at the safer r=8/512px config rather than v25's still-unresolved
r=16, to keep the balance variable from getting tangled up with that
memory anomaly.

First attempt (r=8, 512px, seq=2048, all 5 tiers, balanced to 1:1) OOM'd
immediately on the first training step:

```
2026-07-24 12:57:00,572 [INFO] sft_train: Built 762 SFT training examples
2026-07-24 12:57:00,572 [INFO] sft_train: Class-balanced 762 -> 1400 examples
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 374.00 MiB.
GPU 0 has a total capacity of 14.56 GiB of which 106.81 MiB is free.
Including non-PyTorch memory, this process has 14.46 GiB memory in use.
```

Balancing worked correctly (762 → 1400 examples, matching the local test) —
this was a real memory issue, unrelated to the balancing feature itself. It
turned out the r=8/512px/seq=2048 combination had actually never been run
for real before this — v25 was supposed to test it but ran the stale r=16
config instead, as described above. The ~7.26GiB "safe" estimate that
config was based on came from v19's data, which predates
`max_seq_length` being enforced at all — v19 trained on genuinely uncapped
sequences and still fit, so it was never a like-for-like comparison to a
run that actually truncates to 2048 tokens.

Dropped `max_image_size` 512 → 384, based on the real, internally
consistent delta between v19 (512px) and v20 (384px)'s original numbers
(14.55GiB vs 14.03GiB, ~520MiB difference — trustworthy as a delta even
though both absolute numbers included the since-fixed 7.29GiB leak
equally). That should close the 374MiB gap with real margin to spare.

Second attempt completed — 264/264 steps, all 3 epochs, no crash:

```
{'train_runtime': 19553.0763, 'train_samples_per_second': 0.215,
 'train_steps_per_second': 0.014, 'train_loss': 2.675908681118127, 'epoch': 3.0}
=== Phase 4 (v26) peak GPU memory: 12.36 GB allocated, 12.94 GB reserved ===
```

5h26m wall-clock — considerably longer than v24/v25's ~2.3h, expected given
the balanced set has ~1.9x more examples (1400 vs 719/762). Peak memory
12.94GB reserved, ~1.6GB of real headroom under budget. `trainable params:
18,576,384 || all params: 3,773,199,360` confirms r=8, exactly half of
v25's r=16. Loss trajectory: 7.16 (0.11) → 4.07 (0.23) → 2.68 (0.34) → 2.50
(0.46), plateauing around 2.38-2.45 from roughly epoch 0.6 onward, ending at
2.42. `adapter_config.json` confirms `r: 8, lora_alpha: 16`. Checkpoint file
size: 74,405,904 bytes (~71MB), under the 100MB limit, committed in full.

That plateau (~2.38-2.45) is notably higher than both v24's (~1.25-1.26)
and v25's (~1.217-1.22). Worth being careful about what that does and
doesn't mean — v24 and v25 both trained on data where "always predict
genuine" is a low-loss shortcut, and this run's data doesn't have that
shortcut anymore, so a higher loss here plausibly reflects the model
actually attempting real discrimination rather than being worse-trained.
That's a hypothesis, not something the loss number alone can prove — the
real test is whether the resulting predictions are actually varied on held
out examples, covered in `phase6_adversarial_rounds_summary.md`.
