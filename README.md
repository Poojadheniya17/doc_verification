# Adversarially-Robust Identity Document Verification System

*(Working name — a real name gets picked once the system takes shape. Generic
package name for now: `doc_verification_system`.)*

A fine-tuned vision-language model system that extracts identity document fields,
detects and localizes tampering, reasons about risk in natural language, retrieves
similar past-flagged cases as supporting evidence, and recommends a risk-tiered
decision (auto-approve / auto-reject / human-review) under a configurable cost
tradeoff — with production benchmarking (fp16/INT8/INT4) reported alongside.

Built as a research-grade portfolio project: real fine-tuning (SFT + QLoRA, then
DPO), not API calls to a hosted model; every experiment reports sample sizes and
confidence intervals, not single-point accuracy; failures are documented, not
hidden.

**Status:** scaffolding complete (Phase 1). No trained model yet — see
[Build Status](#build-status) below.

## System Overview

```
LAYER 3: Operating System (monitoring, cost, drift)      [optional extension]
  LAYER 2: Decision Layer (risk tiers, cost tradeoff)
    LAYER 1: Detection Core (VLM, forgery, eval)
```

Layer 1 must be fully solid before Layer 2 starts; Layer 2 must work end to end
before Layer 3 (optional) is attempted.

## What it does

1. Extracts structured fields (name, DOB, ID number, address, expiry) as JSON
2. Detects whether the document is tampered/forged, and localizes where
3. Explains its reasoning in natural language
4. Retrieves similar past-flagged cases to support its verdict
5. Recommends a risk-tiered decision based on a configurable cost matrix
6. Reports its own production cost/latency tradeoffs at different quantization levels

## Core technical approach

- **Model:** Qwen2.5-VL-7B-Instruct, fine-tuned with QLoRA (4-bit)
- **Training:** SFT for extraction + tamper detection, then DPO for explanation
  quality (chosen vs. rejected reasoning pairs)
- **Forgery generation:** 5 escalating tiers — field tampering, image splicing,
  diffusion-based inpainting, fully synthetic generation, recapture/moiré simulation
- **Core experiment:** leave-one-attack-tier-out generalization test across all 5
  tiers, with confidence intervals
- **Adversarial rounds:** 3 rounds of targeted retraining based on observed failure
  modes, with an accuracy curve tracked across rounds
- **Retrieval:** embedding-based similarity search (sentence-transformers) against
  past-flagged cases
- **Decision layer:** cost matrix (false-accept / false-reject / manual-review,
  reasoned assumptions, clearly labeled) driving a threshold sweep and cost curve
- **Production benchmarking:** fp16 vs. INT8 vs. INT4 — accuracy, latency, and
  estimated cost-per-verification at hypothetical volume

## Compute

- Local dev (VS Code, CPU): all code, data generation, eval scripts, and the
  Streamlit app are written and smoke-tested on tiny samples here.
- Real GPU training (SFT, DPO, adversarial rounds) runs remotely on Kaggle's free
  tier (T4/P100, ~30 GPU-hrs/week) via the Kaggle CLI.
- Every training/eval script is config-driven (`config/training_config.yaml` ->
  `environment: local|kaggle`) so it runs identically either way — see
  `src/utils/config_utils.py`.

**Standing constraint, found in Phase 3, applies to every later phase:** this
dev machine has only ~7.7GB total RAM (often <1GB free at idle). That's not
just "slow for local inference" — attempting to load even Qwen2.5-VL-**3B** in
fp32 (~12-14GB) caused severe disk thrashing rather than graceful slowness, and
had to be killed. **No VLM loading, inference, or training happens locally on
this machine at all, at any model size or quantization level** — not just the
7B fine-tuning target, but even lightweight local "smoke tests" of model-loading
code must run on Kaggle instead. Local dev is scoped to logic that doesn't need
model weights in memory: data generation, config/manifest handling, metric
math, prompt/parsing logic, and Streamlit UI wiring — validated with unit tests
and mocked model outputs, not a real local forward pass. This is why Phase 3's
zero-shot baseline numbers come from a script that's tested-but-not-yet-executed
locally, with the real numbers deferred to a Kaggle run (see
[results/tables/phase3_baseline_summary.md](results/tables/phase3_baseline_summary.md)).

## Data

Public/synthetic only — **no real fraud data**. Base documents from MIDV-2020 (or
equivalent public ID dataset); all forgeries are generated, not sourced. Clearly
labeled throughout.

Phase 2 acquired a real (partial-download, local-dev-scale) MIDV-2020 sample and
ran Tier 1 (field tamper), Tier 2 (splicing), and the 5-kind degradation pipeline
against it end to end. See [results/tables/phase2_data_summary.md](results/tables/phase2_data_summary.md)
for exact counts, an example of a bug found and fixed mid-phase, and what scaling
up to a full run requires.

**Attribution:** MIDV-2020 (Bulatov et al., 2022, CC BY-SA 2.5) — synthetic faces
via [Generated Photos](https://generated.photos/). Dataset files themselves are
not committed to this repo (see `.gitignore`); `src/data_generation/acquire_dataset.py`
re-downloads them from the official source.

| Genuine | Tier 1: field tamper | Tier 2: splice (same code) | Tier 2: splice (cross code) |
|---|---|---|---|
| ![genuine](results/sample_outputs/example_genuine_lva_passport_00.jpg) | ![tier1](results/sample_outputs/example_tier1_field_tamper.jpg) | ![tier2 same](results/sample_outputs/example_tier2_splicing_samecode.jpg) | ![tier2 cross](results/sample_outputs/example_tier2_splicing_crosscode.jpg) |

See [results/sample_outputs/README.md](results/sample_outputs/README.md) for captions.

## Non-goals (explicit scope cuts, not oversights)

- No RL-based self-play forger — forgery generation is scripted/targeted (future
  work, see [writeup/project_report.md](writeup/project_report.md))
- No real fraud data sourcing
- No autonomous orchestrating agent layer — adversarial rounds are manually/scriptedly
  triggered
- Layer 3 (drift simulation, cost dashboard) is optional/extension, time-boxed
  behind Layers 1 and 2

## Repo layout

See folder tree in [writeup/project_report.md](writeup/project_report.md) methods
section once populated; for now, `src/` mirrors the system's layers
(`data_generation/`, `training/`, `retrieval/`, `eval/`, `decision/`, `monitoring/`),
`config/` holds all hyperparameters and cost assumptions, `results/` holds every
generated chart/table, `writeup/` holds the paper-style report.

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

## Build Status

| Phase | What | Status |
|---|---|---|
| 1 | Scaffold (folders, configs, requirements, README) | Done |
| 2 | Data foundation (MIDV-2020, Tier 1-2 forgeries, degradation) | Done |
| 3 | Baselines (zero-shot VLM, OCR) | OCR done; VLM eval built + tested, real numbers deferred to Kaggle |
| 4 | Core SFT + QLoRA fine-tuning | Script + data pipeline built and tested; training run itself deferred to Kaggle |
| 5 | Forgery tiers 3-5 (inpainting, synthetic, recapture) | Tier 4/5 done; Tier 3 mask logic done, diffusion inference deferred to Kaggle |
| 6 | Core experiments (leave-one-out, adversarial rounds) | Orchestration/aggregation logic built + tested; real per-fold/per-round training deferred to Kaggle |
| 7 | Decision layer (risk tiering, cost simulation) | Done — fully real (no model involved), see [results/tables/phase6_7_groundwork_summary.md](results/tables/phase6_7_groundwork_summary.md) |
| 8 | DPO + retrieval | Retrieval (case_index.py) done, real end-to-end verified locally; DPO not started |
| 9 | Quantization benchmarking | Not started |
| 10 | Layer 3 (optional) | Not started |
| 11 | Demo + writeup | Not started |

Results, charts, and the full paper-style writeup will be embedded here as each
phase completes.
