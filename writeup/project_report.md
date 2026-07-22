# Adversarially-Robust Identity Document Verification System

*Paper-style writeup. Populated incrementally as each phase produces real results —
sections below are placeholders, not filled with hypothetical numbers.*

## Problem

*(Phase 11)* What identity-document fraud detection has to get right, why it's hard
(forgery diversity, class imbalance, generalization to unseen attack types), and why
this project frames it as a leave-one-attack-tier-out generalization problem rather
than a single in-distribution accuracy number.

## Method

*(Phase 11)* Model choice (Qwen2.5-VL-7B + QLoRA) and why; SFT then DPO training
pipeline; 5-tier forgery taxonomy and generation approach; adversarial retraining
loop; decision layer cost-tradeoff framing; retrieval-augmented verdicts.

## Results

*(Phase 11)* Clean vs. degraded accuracy, leave-one-out generalization gap per tier
(with confidence intervals), adversarial-round accuracy curve, cost-tradeoff curve,
quantization (fp16/INT8/INT4) accuracy-latency-cost table. All numbers sourced from
`results/tables/`.

## Failure Analysis

*(Phase 11)* At least one honest documented failure case with a hypothesis for why
it failed — not buried, not hand-waved.

## Limitations

*(Phase 11)* Synthetic/public data only (no real fraud data), Kaggle free-tier
compute bounds on dataset size / experiment repetition, scripted (not learned)
forgery generation, no autonomous orchestration layer.

## Future Work

*(Phase 11)* RL-based self-play forger (explicitly not attempted here — documented
as future work only, per project scope decision). Real fraud data partnerships.
Autonomous adversarial-round orchestration.
