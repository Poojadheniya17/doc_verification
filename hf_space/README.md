---
title: Document Verification Live Inference
emoji: 🛡️
colorFrom: teal
colorTo: orange
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: apache-2.0
---

Real live inference for the [doc_verification_system](https://github.com/Poojadheniya17/doc_verification) project.

Loads the actual v26 checkpoint (Qwen2.5-VL-3B, QLoRA, 4-bit) and runs it for
real on whatever image you upload -- no captured/replayed data, no mocked
output. Same extraction, retrieval, and decision-routing code as the rest of
the project.

First request after idle takes 30-60s (model load onto GPU); after that,
expect roughly 5-20s per image.

**Status: blocked, not deployed.** Hugging Face requires a PRO subscription
to create a Gradio Space (even free cpu-basic) on this account, and ZeroGPU
needs PRO or a community grant -- both blocked without payment. The actual
live demo currently runs via `kaggle_kernels/live_demo/` instead (free,
reuses this project's existing Kaggle GPU setup, exposed through Gradio's
own `share=True` temporary public link). Revisit this Space if/when PRO or
a community grant becomes available -- the code here is otherwise complete
and was validated locally (pure decision/retrieval logic, not the GPU call).

Before pushing this as a real Space: `cp -r ../checkpoints/sft_v26_balanced
checkpoints/` (excluded from git here to avoid a duplicate 71MB copy while
this path is blocked).
