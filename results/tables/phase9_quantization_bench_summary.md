# Phase 9: quantization benchmarking

Kaggle run (`doc-verification-quantization-bench`), v24 checkpoint
(`checkpoints/sft_v24_final`), fp16 vs int8 vs int4, n=40 examples per
precision (20 genuine + 20 tampered).

## Headline numbers

| Precision | Accuracy | 95% CI | Avg latency | Est. cost / 1M verifications |
|---|---|---|---|---|
| fp16 | 50.0% | [35.0%, 65.0%] | 8.69s | $845 |
| int8 | 50.0% | [35.0%, 65.0%] | 22.17s | $2,156 |
| int4 | 50.0% | [35.0%, 65.0%] | 20.98s | $2,040 |

Cost figures use an illustrative $0.35/GPU-hour assumption (see
`quantization_bench.py`'s docstring) — what matters is the relative
comparison across precisions, not the absolute dollar number.

## Accuracy is identical across all three precisions

Exactly 50.0% for all three isn't a coincidence — it's the same
single-class collapse from `phase6_adversarial_rounds_summary.md` showing
up again on a perfectly balanced 20/20 set. A model that predicts one class
for every input scores exactly 50% on any balanced set regardless of
numeric precision, since precision changes the representation, not which
class a collapsed model defaults to. Additional evidence the v24
checkpoint's behavior is a real capacity/training issue, not a quantization
artifact.

## fp16 was faster and cheaper than int8/int4

This inverts the usual assumption that lower precision means faster and
cheaper. fp16 averaged 8.69s/example; int8 and int4 both took longer
(22.17s and 20.98s) and were therefore more expensive per verification at
the cost assumption above, not less.

The likely reason: this benchmark runs at batch size 1 (matching training),
and bitsandbytes int8/int4 layers carry real dequantization overhead on
every forward pass — weights get unpacked back toward higher precision for
the actual matmul. At batch size 1, that per-call overhead never gets
amortized across a bigger batch the way it would in a production serving
setup with real request batching, so the usual quantization win (smaller
weights, less memory bandwidth, faster) can get outweighed by
dequantization cost at this specific tiny batch size. This is a real
limitation of the benchmark as measured, not a general claim that int4 is
slower than fp16 in production — just what was actually observed under
these specific conditions (batch=1, single T4, this exact library version
combination).

## Recommendation from this benchmark

Accuracy is identical across precisions here, so there's no accuracy
argument for going lower, and fp16 measured faster and cheaper than both
quantized options under these conditions — so the recommendation from this
specific benchmark is fp16, not int8/int4, which is the opposite of the
usual "quantize for production" default. A deployment with real request
batching would need to re-run this comparison under batched conditions
before drawing a general conclusion.
