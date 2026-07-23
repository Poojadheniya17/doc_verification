# Project Status — resume point for a fresh session

Last updated: 2026-07-23 (v24). **Phase 4 SFT training completed successfully
for the first time.** Root cause of the entire v19-v23 OOM chain: Phase 3's
7B model was never freed before Phase 4 loaded the 3B model (clean_eval.py
caches its model at module scope; kernel_driver.py never cleared it), so
every "memory cut" from v19-v23 was shrinking the wrong side of the ledger.
Fixed in v24 by adding explicit cleanup between phases — see "Where Phase 4
actually stands" below for the full story and real training numbers.

**If you are a fresh Claude session picking this up:** read this whole file
first, then check `git log --oneline -10` and the Kaggle kernel status
(command below) before doing anything else — this file may be slightly
stale if the session ended mid-task.

## Repo / infra pointers

- GitHub: https://github.com/Poojadheniya17/doc_verification (`master` branch).
  **Standing rule: every commit gets pushed immediately, no exceptions.**
- Kaggle dataset (code + training data): `poojadheniya/doc-verification-data`
- Kaggle kernel (Phase 3 baseline + Phase 4 SFT training):
  `poojadheniya/doc-verification-zero-shot-baseline-sft-qlora`
  - Local driver source: `kaggle_kernels/phase3_4_sft_baseline/kernel_driver.py`
  - Push dataset: `kaggle datasets version -p <staging_dir> --dir-mode zip -m "<msg>"`
    (staging dir is `../kaggle_package_staging` relative to this repo, i.e.
    `C:\Users\Acer\OneDrive\Desktop\hyperVerge\kaggle_package_staging` —
    refresh via `src/utils/kaggle_package.py`'s `stage_package()`, or just
    `cp -r` the changed `config/`/`src/` folders over and rewrite
    `dataset-metadata.json` each time, matching the pattern used throughout
    this session)
  - Push kernel: `kaggle kernels push -p kaggle_kernels/phase3_4_sft_baseline`
  - **If push fails with "Maximum batch GPU session count of 2 reached":**
    a previous session is stuck/hung. Fix: `kaggle kernels delete
    poojadheniya/doc-verification-zero-shot-baseline-sft-qlora -y` then push
    again fresh (this has worked reliably every time it's been needed).
  - kaggle.exe is at `/c/Users/Acer/AppData/Roaming/Python/Python314/Scripts/kaggle.exe`
    (not on PATH in this Bash tool's shell snapshot even though it's on the
    real Windows PATH — always use the full path in Bash tool calls)
  - `kaggle kernels logs -f <kernel>` gives live logs while running (not just
    `kernels output`, which only works after the run ends). Windows console
    can't print non-ASCII log characters — always redirect to a file and
    `Read` it, never let raw output hit the terminal directly.

## Where Phase 4 (SFT training) actually stands

Six single-variable hang hypotheses (v11-v18) were each tested and ruled out
in turn — full blow-by-blow chain is in `config/model_config.yaml`'s top
DECISION comment and `results/tables/phase4_sft_summary.md`. Final decision:
gradient checkpointing disabled entirely + `max_image_size` cut to 512 for
real OOM margin (v15's approach, hardened). This was pushed as v19.

**BOUNDED STEP-DOWN SEARCH, THEN v23's COMBINED LEVERS — ALL 5 OOM'D
(2026-07-23).** After v19's confusing OOM (memory usage went UP despite a
smaller image, never fully explained), the user explicitly said not to
investigate why and instead run a bounded brute-force search: cut
`max_image_size` in steps until training succeeds or a 256px floor is hit.
That search (v19-v22) exhausted without success, so the user then directed
two more levers combined for v23: LoRA rank r=8→r=4, and `max_seq_length`
2048→1024 (the latter previously a **decorative config value only** — found
and fixed while implementing v23: `sft_train.py`'s `collate()` never
actually passed it to the processor, so v8-v22 all trained on the full
untruncated sequence regardless of what the config said). Full result table:

| Kernel | Change | Result | Shortfall | Total GPU memory in use |
|---|---|---|---|---|
| v19 | max_image_size 512px | OOM (MLP dequant) | ~44 MiB short | 14.55 GiB |
| v20 | max_image_size 384px | OOM (lm_head) | ~51 MiB short | 14.03 GiB |
| v21 | max_image_size 320px | OOM (lm_head) | ~163 MiB short | 14.14 GiB |
| v22 | max_image_size 256px (floor) | OOM (loss/cross_entropy) | ~19 MiB short | 14.33 GiB |
| v23 | + LoRA r=4, max_seq_length=1024 (genuinely wired in) | OOM (first backward pass) | ~49 MiB short | 14.36 GiB |

**Every single attempt OOM'd, including v23.** v23 confirmed the rank cut
took effect (`trainable params: 9,288,192 || all params: 3,763,911,168 ||
trainable%: 0.2468`) but total memory in use barely moved from v22 (14.36 vs
14.33 GiB) and the shortfall was slightly *worse* (~49 MiB vs ~19 MiB), not
better. Two genuinely different, correctly-implemented memory levers moved
the needle by less than noise. Memory usage has never been monotonic with
any of these levers across the whole v19-v23 search — never explained, not
investigated further per the user's explicit instruction ("we don't need to
understand why, we just need this to work").

**Re-enabling gradient checkpointing was considered for v23 and correctly
rejected before pushing** — not because it was untested (an earlier version
of this document wrongly claimed that), but because kernel v16 already ran
the exact `use_reentrant=False` + explicit `use_cache=False` combination on
3B (confirmed via `git log`: v16's commit came after the 3B model-swap
commit) and it hung. That's a known-bad configuration, not an untested gap.

**This is now a genuine stop-and-report point — five bounded attempts across
three independent memory levers (image size, LoRA rank, sequence length),
on top of six earlier unresolved hang hypotheses (v11-v18), all failing to
close a ~50-160 MiB gap on a ~14.3-14.5 GiB ceiling.** Do NOT push a further
cheap-lever attempt (e.g. LoRA r=2, which would likely make the adapter too
low-capacity to learn anything useful, or seq length below 1024, which risks
truncating real training targets). Remaining options, laid out for the user
(their call, not picked autonomously):
(a) ~~Re-enable gradient checkpointing~~ — ruled out, already tested on 3B
    (v16) and hung. Not viable.
(b) ~~Reduce LoRA rank further~~ — tried in v23 (r=4), didn't meaningfully
    help; r=2 would risk an adapter too small to learn the task.
(c) ~~Reduce `max_seq_length` further~~ — tried in v23 (1024), didn't
    meaningfully help; further cuts risk truncating real training examples.
(d) Batch/accumulation math — `per_device_train_batch_size` already at the
    floor of 1, so low expected value, but mentioned for completeness.
(e) Something structural — a materially smaller base model (e.g. a 1-2B VLM
    if one exists with adequate document-understanding capability), a paid
    Kaggle tier or different compute provider with more VRAM, or accepting
    and clearly documenting a reduced training scope (e.g. fewer target
    modules in the LoRA config, a shorter training set, or reporting Phase 4
    as a documented infrastructure-limited negative result alongside the
    honest debugging story — which the user has already said is some of the
    strongest material in this project).

**BEFORE the structural decision was made, the user asked one more question
that changed everything: was Phase 3's 7B model actually being freed before
Phase 4 loaded the 3B model, given both run in the same kernel process?**

It was not. `kernel_driver.py` calls `run_clean_eval(...)` (Phase 3) then
`run_sft_train(...)` (Phase 4) back-to-back with zero cleanup between them —
no `del`, no `torch.cuda.empty_cache()`, no `gc.collect()`. Worse:
`clean_eval.py` caches its model/processor at **module scope**
(`_model`/`_processor` globals, added originally so Phase 3's own loop
wouldn't reload weights per image) — since `kernel_driver.py` keeps that
module imported for the rest of the process, the 7B model's reference never
went away. This fully explains why v19-v23's three independent memory cuts
(image size, LoRA rank, sequence length) barely moved the OOM ceiling: none
of them addressed the actual problem. The ceiling was leftover 7B weights,
not 3B training's own footprint.

**v24 — fix: explicit cleanup between phases, v23's training settings
otherwise unchanged.** Added `clean_eval.unload_model()` (clears the cached
globals, `gc.collect()`, `torch.cuda.empty_cache()`), called from
`kernel_driver.py` between Phase 3 and Phase 4, with real before/after GPU
memory diagnostic prints so the result would be verified, not assumed.

**Diagnostic confirmed the hypothesis exactly:**
```
=== GPU memory before Phase 3 cleanup: 5.91 GB allocated, 7.29 GB reserved ===
=== GPU memory after Phase 3 cleanup: 0.01 GB allocated, 0.03 GB reserved ===
```
~7.3 GB of the ~14.5 GB budget was leftover 7B model, on every single one of
v19-v23's attempts.

**PHASE 4 TRAINING COMPLETED SUCCESSFULLY (2026-07-23, kernel v24, Kaggle
kernel version 6).** 135/135 steps, all 3 epochs, no crash:
```
{'train_runtime': 7998.9749, 'train_samples_per_second': 0.27,
 'train_steps_per_second': 0.017, 'train_loss': 1.7295982360839843, 'epoch': 3.0}
```
- Total training wall-clock: **7999 seconds ≈ 2h 13m** (matches the progress
  bar's 135/135 steps at ~58-60s/step).
- `train_loss` (the Trainer's running average over the *entire* run,
  including the high-loss early steps) = **1.7296**.
- The loss trajectory itself (from the log, `{'loss': ..., 'epoch': ...}`
  lines): 5.03 (0.22) → 3.12 (0.45) → 1.76 (0.67) → 1.41 (0.89) → 1.31 (1.11)
  → 1.29 (1.33) → 1.28 (1.56) → 1.27 (1.78) → 1.26 (2.00) → 1.25 (2.22) →
  1.25 (2.45) → 1.25 (2.67) → 1.26 (2.89). Loss dropped fast in epoch 1, then
  plateaued around 1.25-1.26 for the rest of training — worth flagging
  honestly in the writeup as a sign the model may be near its capacity given
  LoRA r=4 and 719 training examples, not necessarily evidence of a
  problem, but not obviously still improving either.
- `[NbConvertApp] Writing 301965 bytes` confirms the definitive completion
  marker.
- `trainable params: 9,288,192 || all params: 3,763,911,168 || trainable%: 0.2468`
  confirms the LoRA r=4 config was actually applied.

**Checkpoint location:** downloaded to
`C:\Users\Acer\OneDrive\Desktop\hyperVerge\kaggle_run_output_v24\checkpoints\sft\`
— `checkpoint-45/`, `checkpoint-90/`, `checkpoint-135/` (each ~55MB, includes
optimizer/scheduler/rng state for potential resume) and `final/` (36MB,
adapter weights only — this is the one downstream phases should load).
**Not committed into the git repo** — `.gitignore` already has a deliberate,
pre-existing rule (`checkpoints/ # large binaries — track via Kaggle output
/ external storage, not git`) that predates this session. Respected that
convention rather than overriding it silently; the checkpoint is safe on
disk (OneDrive-synced) and also remains in this Kaggle kernel version's
output for now. **Flag to the user:** the adapter is only 36MB, well under
GitHub's 100MB file limit, so committing it is a real option if you'd
rather have it versioned in git for portfolio completeness — just say so.

**What's now unblocked** (previously blocked on a real trained checkpoint):
Tier 3 diffusion inpainting can proceed independently; wiring the real
checkpoint into `leave_one_out_eval.py`/`adversarial_rounds.py`;
`financial_risk_reasoning.py`; quantization benchmarking; the Streamlit demo;
the results notebook; the paper-style writeup (which must include this
entire saga — v8-v24, hangs then OOMs then the memory-leak fix — as the
honest debugging story, per the user's standing instruction).

**Immediate next step when resuming:** confirm with the user whether to
proceed into the next phases (Tier 3, leave-one-out, decision layer, etc.)
and whether to commit the checkpoint into git. Do not assume either without
asking — this was flagged as a genuine milestone worth a pause.

## What's authorized for autonomous continuation (per user's explicit direction)

Once Phase 4 has a real completed training run and checkpoint:
1. Tier 3 forgery generation (diffusion inpainting) — run for real on Kaggle
   (`src/data_generation/inpaint_forger.py`'s `run_inpainting()`, not yet
   executed anywhere; mask-building logic is done and tested locally)
2. Wire the real trained checkpoint into `src/eval/leave_one_out_eval.py` and
   `src/eval/adversarial_rounds.py` — run actual leave-one-out folds (use
   judgment on 3 vs 5 tiers based on compute/time realities at that point;
   document whichever is chosen as a deliberate, reasoned decision)
3. `src/decision/financial_risk_reasoning.py` using the trained model (still
   a stub — this is the one piece of Phase 7 not yet built)
4. Quantization benchmarking (fp16/INT8/INT4) — `src/eval/quantization_bench.py`
   (still a stub)
5. Streamlit demo app — `app/demo_app.py` (still a stub)
6. Results notebook with real charts — `notebooks/03_results_analysis.ipynb`
7. Paper-style writeup — `writeup/project_report.md` (skeleton exists with
   placeholder section markers). **Must include the real debugging story
   honestly** — P100 incompatibility, the OOM chain, the six-hypothesis hang
   investigation, the model-size and checkpointing tradeoffs — the user
   explicitly called this out as some of the strongest material in the
   project, not something to sanitize into a clean narrative.

**Explicitly NOT in current scope** (flagged to the user, not yet confirmed):
DPO training (Phase 8's other half; retrieval/`case_index.py` is done). The
user's own phase-by-phase list omitted it; proceeding on the assumption it's
deprioritized given time already spent, until told otherwise.

## Standards holding throughout (non-negotiable, per user)

- Every commit → immediately pushed to GitHub, no exceptions
- Every real number reported honestly, including weak/small-sample results
- Every scoping decision documented with real reasoning (README/config/writeup),
  never left implicit
- Local logic unit-tested before any GPU run (same pattern as every phase so far —
  see `tests/test_pipeline_smoke.py`, currently 50 tests passing)
- Stop and report (don't loop indefinitely) on: a failure that resists a few
  isolated diagnostic attempts, any change to core project scope/claims, or a
  genuine compute/time tradeoff decision

## Task tracker state (see TaskList tool)

- #1-3: done (scaffold, data foundation, baselines)
- #4: in_progress — Phase 4 SFT training (this file's main subject)
- #5: in_progress — Tier 3 diffusion inpainting not yet run for real
- #6-9: pending — leave-one-out/adversarial rounds, decision layer
  (financial_risk_reasoning.py), DPO+retrieval (retrieval done, DPO
  deprioritized per above), quantization
- #10: pending, optional/time-boxed (Layer 3 — drift sim, cost dashboard)
- #11: pending — demo + writeup
