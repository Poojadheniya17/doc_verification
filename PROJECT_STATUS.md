# Project Status — resume point for a fresh session

Last updated: 2026-07-23, mid Phase 4 (v19 push in progress).

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

**v19 RESULT (2026-07-23): ERROR — a new, confusing OOM, not the hang.**
`torch.OutOfMemoryError: Tried to allocate 44.00 MiB ... of which 8.81 MiB is
free ... 14.55 GiB memory in use.` Crashed inside an MLP dequantize+matmul
(bitsandbytes), step 0, 4 seconds in — NOT a repeat of the hang signature.

**The confusing part:** v15 (768px images, no version pins) used 14.02 GiB
and was short by ~40MiB. v19 (512px images — nearly half the pixel area,
should need LESS memory) used 14.55 GiB — **~530MB MORE**, despite the
image-size cut. That's backwards from what the change should have done.

**Working hypothesis, NOT verified — flagged to the user, no fix pushed yet
per their explicit instruction for this exact situation:** v19 is the first
run using the library versions pinned in v17/v18 (transformers==4.57.6,
bitsandbytes==0.49.2, etc.) — v15 ran before those pins existed, on whatever
unpinned "latest" resolved to at the time. It's plausible the pin itself
increased baseline memory overhead (e.g. a different default attention
implementation, or more scratch memory in the 4-bit dequantization path)
enough to swamp the image-size savings. This needs the user's input on how
to proceed (test the version-overhead hypothesis specifically vs. just
cutting image size further vs. something else) — DO NOT push another
autonomous kernel version without checking with the user first, per their
explicit instruction given after v19's diagnosis.

**Immediate next step when resuming (if the user hasn't yet responded to
this finding):** re-read the conversation for the user's direction on v19's
OOM before doing anything else. Do not assume a fix and push it.

Once Phase 4 genuinely reaches a completed run (whenever that happens):
```bash
"/c/Users/Acer/AppData/Roaming/Python/Python314/Scripts/kaggle.exe" kernels status poojadheniya/doc-verification-zero-shot-baseline-sft-qlora
```
- If `COMPLETE`: download output (`kernels output ... -p kaggle_run_output`),
  check `/kaggle/working/results/` and `/kaggle/working/checkpoints/` in the
  downloaded output for the real loss curve and final adapter checkpoint.
  This is the actual trained model everything downstream depends on —
  **download and commit it locally** (adapter checkpoints are small, LoRA-only,
  a few tens of MB) before it's lost when the Kaggle session recycles.
- If `ERROR`: diagnose from `kaggle_run_output`'s log. If it's the SAME hang
  signature yet again even with checkpointing off, that would be a genuinely
  new and confusing result worth stopping and reporting to the user rather
  than guessing further (per the user's own stated discipline: a few
  isolated attempts, then stop and report, no indefinite loops). If it's an
  OOM still short by some margin, a further `max_image_size` cut (already at
  512, next step down would be ~384-448) is a reasonable single next lever.
- If `RUNNING`: use the stall-detection pattern established throughout this
  session — poll every ~4 min, compare live log snapshots, escalate to the
  user only on a confirmed stall (10+ min identical) or a genuinely new
  failure mode.

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
