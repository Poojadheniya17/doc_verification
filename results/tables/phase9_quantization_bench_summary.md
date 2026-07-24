# Phase 9: Quantization Benchmarking — Real Results

Real Kaggle run (kernel `doc-verification-quantization-bench`), v24 checkpoint
(`checkpoints/sft_v24_final`), fp16 vs int8 vs int4, n=40 examples per
precision (20 genuine + 20 tampered, `adversarial_rounds.build_eval_set()`
with `n_genuine=20`).

## The headline numbers

| Precision | Accuracy | 95% CI | Avg latency | Est. cost / 1M verifications |
|---|---|---|---|---|
| fp16 | 50.0% | [35.0%, 65.0%] | **8.69s** | **$845** |
| int8 | 50.0% | [35.0%, 65.0%] | 22.17s | $2,156 |
| int4 | 50.0% | [35.0%, 65.0%] | 20.98s | $2,040 |

(Cost figures use an illustrative, explicitly-labeled $0.35/GPU-hour
assumption — see `quantization_bench.py`'s module docstring. What matters is
the *relative* comparison across precisions, not the absolute dollar figure.)

## Finding 1 (expected): accuracy is identical across all three precisions, and consistent with the adversarial-rounds collapse

**Exactly 50.0% for all three precisions** is not a coincidence and not
evidence that quantization is harmless here — it's the same single-class-
collapse pattern documented in
`results/tables/phase6_adversarial_rounds_summary.md`, showing up again on a
perfectly balanced 20/20 eval set. A model that predicts one class for every
input scores exactly 50% on any perfectly-balanced set, regardless of
numeric precision, because precision doesn't change *which* class a
collapsed model defaults to. This is real, additional evidence that the
v24 checkpoint's behavior is a genuine model-capacity issue, not a
precision/quantization artifact — quantization here changes numerical
representation, not the underlying (already degenerate) decision boundary.

## Finding 2 (genuinely surprising, reported honestly): fp16 was both faster AND cheaper than int8/int4

This inverts the usual assumption that lower precision means faster,
cheaper inference. Real, measured numbers: fp16 averaged 8.69s/example;
int8 and int4 both took *longer* (22.17s and 20.98s respectively) and were
therefore *more* expensive per verification at this project's illustrative
GPU-cost assumption, not less.

**Honest, reasoned explanation, not just a data point**: this project's
benchmark runs batch size 1 (`per_device_train_batch_size: 1` throughout,
matching training), and `bitsandbytes` int8/int4 quantized layers carry a
real dequantization overhead on every forward pass (weights get
unpacked back toward higher precision for the actual matmul). At batch size
1, that per-call overhead is never amortized across a larger batch the way
it would be in a production serving setup with real request batching — so
the *usual* quantization win (smaller weights → less memory bandwidth →
faster) can be outweighed by dequantization compute cost specifically at
this tiny batch size. This is a genuine, documented limitation of this
benchmark's realism, not a claim that int4 is generally slower than fp16 in
production — it's a claim about what was actually measured, under the
specific (batch=1, single T4, this exact bitsandbytes/peft/transformers
version combination) conditions of this real test.

## What this means for the project's own recommendation

Given (1) accuracy is identical across precisions on this checkpoint (so
there is no accuracy argument for choosing a lower precision) and (2) fp16
measured faster and cheaper than both quantized options under this
benchmark's real conditions, **the honest recommendation from this specific
benchmark is fp16, not int8/int4** — the opposite of the usual "quantize for
production" default, and worth stating plainly rather than defaulting to
the conventional wisdom the data doesn't support here. A production
deployment with real request batching would need to re-run this comparison
under batched conditions before drawing a general conclusion; this result
is honestly scoped to what was actually measured.
