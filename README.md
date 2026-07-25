# Adversarially-Robust Identity Document Verification System

*(Working name — a real name gets picked once the system takes shape. Generic
package name for now: `doc_verification_system`.)*

A fine-tuned vision-language model system that extracts identity document fields,
detects and localizes tampering, reasons about risk in natural language, retrieves
similar past-flagged cases as supporting evidence, and recommends a risk-tiered
decision (auto-approve / auto-reject / human-review) under a configurable cost
tradeoff — with production benchmarking (fp16/INT8/INT4) reported alongside.

Built as a research-grade portfolio project: real fine-tuning (SFT + QLoRA), not
API calls to a hosted model. Every experiment reports sample sizes and confidence
intervals instead of a single accuracy number, and failures get written up instead
of quietly dropped.

**Status:** three fine-tuned checkpoints trained on Kaggle, full generalization
testing (leave-one-out, adversarial rounds, quantization benchmarking) run
against them, and a class-imbalance fix that actually changed the results. See
[Build status](#build-status) and [Results so far](#results-so-far) below.

## System overview

```
LAYER 3: Operating System (monitoring, cost, drift)      [optional extension]
  LAYER 2: Decision Layer (risk tiers, cost tradeoff)
    LAYER 1: Detection Core (VLM, forgery, eval)
```

Layer 1 needed to be solid before Layer 2 started; Layer 2 needed to work end
to end before touching Layer 3 (optional).

## What it does

1. Extracts structured fields (name, DOB, ID number, address, expiry) as JSON
2. Detects whether the document is tampered/forged, and localizes where
3. Explains its reasoning in natural language
4. Retrieves similar past-flagged cases to support its verdict
5. Recommends a risk-tiered decision based on a configurable cost matrix
6. Reports its own production cost/latency tradeoffs at different quantization levels

## Core technical approach

- **Model:** Qwen2.5-VL-3B-Instruct, fine-tuned with QLoRA (4-bit). Originally
  targeted 7B; switched to 3B after Kaggle debugging surfaced an unresolved
  T4/library-stack training issue at 7B scale (the Phase 3 zero-shot baseline
  is still 7B — see [Model size](#model-size-3b-trained-7b-baseline) below).
- **Forgery generation:** 5 escalating tiers — field tampering, image
  splicing, diffusion-based inpainting, fully synthetic generation,
  recapture/moiré simulation.
- **Core experiment:** leave-one-attack-tier-out generalization test across
  all 5 tiers, with confidence intervals.
- **Adversarial rounds:** targeted retraining based on observed failure
  modes, tracking accuracy across rounds.
- **Retrieval:** embedding-based similarity search (sentence-transformers)
  against past-flagged cases.
- **Decision layer:** cost matrix (false-accept / false-reject /
  manual-review, assumptions clearly labeled) driving a threshold sweep and
  cost curve.
- **Production benchmarking:** fp16 vs. INT8 vs. INT4 — accuracy, latency,
  and estimated cost-per-verification at hypothetical volume.

## Model size: 3B trained, 7B baseline

The original plan was to fine-tune Qwen2.5-VL-7B throughout — the largest
VLM expected to fit QLoRA 4-bit training on a 16GB T4 (Kaggle's free tier).
That held for the Phase 3 zero-shot baseline, which ran fine on 7B and
whose numbers are kept as the reported baseline. It didn't hold for SFT
training. Training on 7B kept silently hanging (not crashing) mid-backward
pass across several single-variable fixes that each moved where the hang
happened rather than fixing it — a pattern that pointed at a T4/library-
stack compatibility issue in this specific environment rather than a config
value away from working. Full kernel-by-kernel numbers in
[results/tables/phase4_sft_summary.md](results/tables/phase4_sft_summary.md).

Decision: stop debugging 7B and move SFT training to 3B. That does mean
Phase 3's reported baseline (7B, zero-shot) and every fine-tuned result
(3B) aren't directly comparable — the difference conflates a fine-tuning
gain with a model-size difference, not a clean ablation. Flagged wherever
those numbers show up side by side.

## Compute

- Local dev (VS Code, CPU): all code, data generation, eval scripts, and the
  Streamlit app are written and smoke-tested on tiny samples here.
- Real GPU training runs on Kaggle's free tier (T4/P100, ~30 GPU-hrs/week)
  via the Kaggle CLI.
- Every training/eval script is config-driven
  (`config/training_config.yaml` → `environment: local|kaggle`) so it runs
  identically either way — see `src/utils/config_utils.py`.

This dev machine has ~7.7GB total RAM, often under 1GB free at idle.
Attempting to load even the 3B model in fp32 caused severe disk thrashing
rather than graceful slowness and had to be killed. No VLM loading,
inference, or training happens locally at all, at any size — local dev is
scoped to logic that doesn't need model weights in memory (data generation,
config/manifest handling, metric math, prompt/parsing logic, UI wiring),
validated with unit tests and mocked outputs. See
[results/tables/phase3_baseline_summary.md](results/tables/phase3_baseline_summary.md)
for how the zero-shot baseline numbers were actually produced.

## Data

Public/synthetic only — no real fraud data. Base documents from MIDV-2020;
all forgeries are generated, not sourced.

Phase 2 acquired a real (partial-download, local-dev-scale) MIDV-2020 sample
and ran Tier 1 (field tamper), Tier 2 (splicing), and a 5-kind degradation
pipeline against it end to end. See
[results/tables/phase2_data_summary.md](results/tables/phase2_data_summary.md)
for exact counts and what scaling up to a full run requires.

**Attribution:** MIDV-2020 (Bulatov et al., 2022, CC BY-SA 2.5) — synthetic
faces via [Generated Photos](https://generated.photos/). Dataset files
aren't committed to this repo (see `.gitignore`);
`src/data_generation/acquire_dataset.py` re-downloads them from source.

| Genuine | Tier 1: field tamper | Tier 2: splice (same code) | Tier 2: splice (cross code) |
|---|---|---|---|
| ![genuine](results/sample_outputs/example_genuine_lva_passport_00.jpg) | ![tier1](results/sample_outputs/example_tier1_field_tamper.jpg) | ![tier2 same](results/sample_outputs/example_tier2_splicing_samecode.jpg) | ![tier2 cross](results/sample_outputs/example_tier2_splicing_crosscode.jpg) |

See [results/sample_outputs/README.md](results/sample_outputs/README.md) for captions.

## Non-goals (scope cuts, not oversights)

- No RL-based self-play forger — forgery generation is scripted/targeted
  (future work, see [writeup/project_report.md](writeup/project_report.md))
- No real fraud data sourcing
- No autonomous orchestrating agent layer — adversarial rounds are
  manually triggered
- Layer 3 (drift simulation, cost dashboard) is optional, time-boxed behind
  Layers 1 and 2
- DPO training — retrieval (the other half of Phase 8) is done, DPO wasn't
  attempted given time already spent elsewhere

## Repo layout

`src/` mirrors the system's layers (`data_generation/`, `training/`,
`retrieval/`, `eval/`, `decision/`, `monitoring/`), `config/` holds
hyperparameters and cost assumptions, `results/` holds every generated
chart and table, `writeup/` holds the paper-style report.

## Setup

```bash
python -m venv venv
venv\Scripts\activate       # Windows; source venv/bin/activate on macOS/Linux
pip install -r requirements.txt
cp .env.example .env        # fill in HF_TOKEN / WANDB_API_KEY as needed
```

Kaggle CLI auth (only needed once you reach real GPU training in Phase 4+):
create an API token at kaggle.com/settings, place `kaggle.json` at
`~/.kaggle/kaggle.json` (or `%USERPROFILE%\.kaggle\kaggle.json` on Windows).

## Build status

| Phase | What | Status |
|---|---|---|
| 1 | Scaffold (folders, configs, requirements, README) | Done |
| 2 | Data foundation (MIDV-2020, Tier 1-2 forgeries, degradation) | Done |
| 3 | Baselines (zero-shot VLM, OCR) | Done — see [Results so far](#results-so-far) |
| 4 | Core SFT + QLoRA fine-tuning | Done — three checkpoints trained (v24, v25, v26), see [Model size](#model-size-3b-trained-7b-baseline) and the debugging story in [writeup/project_report.md](writeup/project_report.md) |
| 5 | Forgery tiers 3-5 (inpainting, synthetic, recapture) | Done — all 5 tiers generated |
| 6 | Core experiments (leave-one-out, adversarial rounds) | Done — see below |
| 7 | Decision layer (risk tiering, cost matrix, cost-tradeoff sim, financial risk reasoning) | Done, unit-tested, wired into the demo app |
| 8 | DPO + retrieval | Retrieval done, live in the demo app; DPO not attempted |
| 9 | Quantization benchmarking | Done |
| 10 | Layer 3 (optional) | Not attempted, as planned |
| 11 | Demo + writeup | Streamlit demo built; writeup written |

## Results so far

See `results/tables/` for the underlying data and
[writeup/project_report.md](writeup/project_report.md) for the full
analysis.

- **Zero-shot Qwen2.5-VL-7B baseline** (n=9): 66.7% tamper-verdict accuracy,
  but the interesting split is underneath — 6/6 real forgeries caught, 0/3
  genuine documents correctly identified as genuine. High recall, high
  false positives: the model hallucinates plausible-sounding tamper
  justifications rather than abstaining.
- **First fine-tuned checkpoint (v24, LoRA r=4)**: trained end to end after
  eleven earlier kernel pushes failed across two distinct failure modes
  (see the writeup). But three independent evaluations — adversarial
  rounds, leave-one-out across all 5 tiers, and quantization benchmarking —
  all show the same problem: the model collapses to predicting a single
  class regardless of input. Root cause: the training data is ~700 genuine
  documents against ~19-60 tampered ones, and always guessing "genuine" is
  the easiest way to minimize loss.
- **Testing whether more model capacity fixes it (v25, LoRA r=16)**: no.
  Identical collapse, confirmed prediction by prediction, on the same
  imbalanced data.
- **Fixing the actual imbalance (v26)**: oversampled tampered examples to a
  real 1:1 ratio and retrained across all 5 tiers. Evaluated cold, it gets
  86.7% on a 30-example set and — more importantly — the predictions are
  actually varied: it catches every tampered example and only misfires on
  genuine documents (calling them tampered, never the reverse). The
  adversarial-rounds retraining loop itself turned out to be fragile even
  starting from this checkpoint (a handful of mined examples is enough to
  collapse it back to single-class), which is a separate, real finding —
  full writeup in
  [results/tables/phase6_adversarial_rounds_summary.md](results/tables/phase6_adversarial_rounds_summary.md).
