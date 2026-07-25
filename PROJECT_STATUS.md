# Project Status — resume point for a fresh session

Last updated: 2026-07-24. **This is the current, most up-to-date summary —
read this section first.**

**LATEST DIRECTION FROM USER (2026-07-24, explicit, supersedes pure rank-
chasing as the priority):** the leave-one-out result (0.000% accuracy across
all 5 folds, every fold a single-class "genuine" predictor, ~15:1 genuine-to-
tampered imbalance per fold) is a third independent confirmation of the
collapse (alongside adversarial-rounds and quantization-bench) and points
strongly at **class imbalance as the dominant cause, not just capacity/LoRA
rank**. Agreed plan, explicitly requested by the user:
1. ~~Let the currently-running v25 kernel finish~~ — **DONE.** Completed
   successfully: 135/135 steps, 3 epochs, no crash. Real, verified via the
   saved `adapter_config.json`: **r=16, lora_alpha=32** (this was actually a
   stale-config re-run of an r=16 attempt, due to a real dataset-propagation-
   lag process error — see `results/tables/phase4_sft_summary.md`'s "v25"
   section for the full honest account). Peak GPU memory 10.17GB
   allocated/10.51GB reserved — a real ~4GB margin under the ~14.5GiB
   ceiling that killed 3 prior identical r=16 attempts. **This directly
   contradicts this project's own earlier "conclusively isolated r=16 as
   broken" claim** — walked back honestly in both `phase4_sft_summary.md`
   and `writeup/project_report.md` to "unresolved anomaly, not proven either
   way." Checkpoint saved as `checkpoints/sft_v25_final/` — metadata only
   committed to git (safetensors is 141.8MB, over GitHub's 100MB limit; real
   weights live in Kaggle kernel output + local disk).
2. ~~Verify its real config~~ — **DONE**, see above.
**CAPACITY-ONLY TEST RESULT (2026-07-24): capacity alone does NOT fix the
collapse.** Adversarial-rounds re-run against v25 (r=16), holding the same
tier1+tier2-only imbalanced training data constant: identical collapse to
v24, verified via real per-example verdict distributions — Round 0: 30/30
"genuine" (33.3% acc), Round 1: 30/30 "tampered" (66.7% acc), Round 2: 30/30
"genuine" again (33.3% acc). Same numbers, same oscillation pattern as v24.
Real evidence FOR the class-imbalance hypothesis over capacity — see
`results/tables/phase6_adversarial_rounds_summary.md`'s "v25 capacity-only
re-test" section. This directly motivates the balanced-training run below,
deliberately at r=8/512px (not v25's r=16) to isolate the balance variable.

3. **Next single test after that (not started yet): class-balanced
   training** — either oversampling tampered examples, undersampling
   genuine examples, or a class-weighted loss — so the model can no longer
   win by unconditionally predicting "genuine." This is now considered more
   likely to be the real fix than further LoRA-rank tuning, per the user's
   explicit reasoning and this project's own real evidence
   (`results/tables/phase6_leave_one_out_summary.md` already independently
   flagged class imbalance as the leading Future Work item before this
   direction was confirmed by the user).

**v26 balanced-retrain ATTEMPT 1: real OOM, honestly corrected (2026-07-24).**
r=8/512px/seq=2048 (the "final decision" config that v25 was supposed to
test but actually ran with stale r=16 instead) OOM'd on its first real test:
374MiB short, 14.46GiB in use. Balancing itself worked correctly (762 -> 1400
examples, confirmed in the log). Honest correction: the "~7.26GiB safe"
estimate this config was based on used v19's OLD data, which predates
`max_seq_length` even being wired in — never actually a like-for-like
comparison. Single evidence-based fix applied: `max_image_size` 512 -> 384
(based on the real v19-vs-v20 delta, ~520MiB) — see
`results/tables/phase4_sft_summary.md`'s "v26" section for full reasoning.
**Attempt 2 (384px) SUCCEEDED (2026-07-25): 264/264 steps, all 3 epochs, no
crash.** Real peak memory 12.94GB reserved (real margin under the 14.56GiB
budget, confirming the fix's reasoning). r=8 confirmed via real
adapter_config.json. Checkpoint: 74,405,904 bytes (~71MB), committed to git
in full (under the 100MB limit). Loss plateaued ~2.38-2.45 — notably higher
than v24/v25's imbalanced-data plateaus (~1.22-1.26), plausibly because the
"always genuine" shortcut no longer exists on balanced data — a real
hypothesis, not a claim; the decisive test is the adversarial-rounds
validation next. See `results/tables/phase4_sft_summary.md`'s "v26" section
for full real numbers.

**VALIDATION SCOPE DECISION (2026-07-24, explicit user instruction):** once
class-balanced training completes, validate using **adversarial-rounds only**
(fast, ~30 examples), NOT a full leave-one-out re-run (~11h). Reasoning
(user's, stated as a deliberate scoping call, to be repeated as such in the
writeup): one complete, rigorous LOO result is already documented; re-running
the full 5-fold version isn't necessary to confirm whether the collapse
resolves. If adversarial-rounds shows real, varied predictions (not
always-genuine): document as validation, and explicitly note in the writeup
that a full LOO re-run on the balanced model is out of scope for now but
would be the natural next step given more time — stated as a reasoned scoping
decision, not an omission. If adversarial-rounds still shows collapse: report
honestly, do NOT chase further fixes autonomously — stop and check with the
user first. User's explicit priority: move fast but accurately — spend extra
time only where it's actually necessary, no gold-plating, no quality
compromise.

**What's real and done as of this update:**
- Phase 4 SFT training completed for real (kernel v24) — checkpoint committed
  at `checkpoints/sft_v24_final/`. Root cause of the entire v19-v23 OOM chain:
  a leftover 7B model in memory, never freed between Phase 3 and Phase 4 in
  the same Kaggle session (confirmed via a real before/after GPU diagnostic).
- Two more real bugs found and fixed the same night, both invisible locally,
  both surfaced only on real Kaggle runs: (1) tier manifest filename
  convention mismatch (silently produced 0 tier examples, no error), (2)
  Windows-native backslash paths baked into every locally-generated manifest
  (hard `FileNotFoundError` on Linux Kaggle). Both root-caused, fixed at the
  source, regression-tested against real repo data.
- **Adversarial rounds: real result in hand.** Headline accuracy 33.3% →
  66.7% → 33.3% across 3 rounds — but the real finding is every round is a
  single-class predictor (see `results/tables/phase6_adversarial_rounds_summary.md`).
  Real evidence for a capacity-limitation hypothesis (LoRA r=4 + 719 training
  examples), not just a training-curve curiosity.
- Phase 3 zero-shot 7B baseline real result recovered and committed (was run
  weeks ago but never saved into the repo) — 66.7% accuracy, n=9, real
  high-recall/high-false-positive pattern documented.
- `financial_risk_reasoning.py` (Phase 7 decision layer): already fully
  implemented and unit-tested (found already done from earlier work).
- Streamlit demo app built with a polished, professional custom-CSS UI
  (verdict badges, confidence gauge, field grid, retrieval case cards,
  color-coded decision banner) — tested working end-to-end in a browser.
  Replays real captured Kaggle predictions (documented design decision: this
  dev machine has no GPU); retrieval + decision layer run live.
- Results notebook (`notebooks/03_results_analysis.ipynb`) populated with 3
  real charts (Phase 3 baseline, Phase 4 loss curve, Phase 6 adversarial
  rounds) — executed end-to-end, no fabricated data. Leave-one-out and
  quantization cells are real, working code that prints an honest PENDING
  message until their real result files exist.
- Full paper-style writeup (`writeup/project_report.md`) written with real
  results throughout and the complete honest debugging saga as its own
  section, per explicit standing instruction.
- README.md updated (was badly stale, said "no trained model yet").

**What's still running / pending as of this update:**
- `doc-verification-leave-one-out` Kaggle kernel: RUNNING (5-fold leave-one-out,
  ~11h job, started ~19:15). Check status before assuming it's still going.
- `doc-verification-quantization-bench` Kaggle kernel: RUNNING (re-pushed
  after fixing a real `torchao`/`peft` version-compatibility crash — see
  kernel_driver.py's comment for the exact fix and why).
- Demo app's `results/sample_outputs/captured_predictions.json` is NOT yet
  populated with real data (a placeholder was tested locally, then deleted —
  never committed). `finetuned_eval.score_prediction()` was extended to
  capture full parsed output for this purpose, but neither currently-running
  Kaggle job's code was already loaded with that extension when they started
  (code changes don't hot-reload into a running kernel) — a small dedicated
  capture kernel run (a handful of images through the trained checkpoint)
  once a GPU slot frees would populate this properly. Low priority relative
  to the two jobs above.

**If you are a fresh Claude session picking this up:** read this whole
section, then check `git log --oneline -10` and both Kaggle kernels' status
(commands below) before doing anything else.

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

**UPDATE (2026-07-23, post-v24): Phase 4 completed, and the project moved into
Phase 5/6/7/9 the same day.** Real status of each:

1. **Tier 3 (diffusion inpainting) — DONE.** First Kaggle run silently
   corrupted 1/15 images: `runwayml/stable-diffusion-inpainting`'s safety
   checker replaced a face-region inpaint with an all-black frame on a
   false-positive NSFW flag, but the manifest still said `success: true`.
   Caught by literally opening the image and checking pixel stats (mean=0,
   max=0). Fixed in `inpaint_forger.run_inpainting()`: detects an all-black
   result and marks it a real failure with a stated reason, rather than
   disabling the safety checker. Re-ran clean: **13/15 real successes**
   (2 correctly rejected: `alb_id/10.jpg`, `alb_id/72.jpg`). Real data now in
   `data/synthetic_forgeries/tier3_inpainting/` (not committed to git, like
   every other tier — pushed via the Kaggle dataset only, see repo's
   `.gitignore`). Tier 4 (full synthetic) and Tier 5 (recapture) turned out
   to have ALREADY been generated in an earlier phase (15/15 each, no GPU
   needed) — this was incorrectly believed to still be pending before a
   codebase survey corrected it.

2. **`sft_train.py` generalized to all 5 tiers.** `build_sft_examples()` now
   takes an arbitrary `tier_manifest_paths: dict[str, str]` instead of
   hardcoded tier1/tier2 params (each tier's manifest shape differs — see the
   function's docstring). `train()` gained `tier_names`/`train_examples`
   (bypass override)/`checkpoint_subdir`/`resume_from_adapter` params and now
   *returns* the final checkpoint path — the injection points leave-one-out
   (arbitrary tier subset per fold) and adversarial rounds (retrain resumed
   from a prior adapter, on mined-failure examples only) both need.

3. **New `src/eval/finetuned_eval.py`**: loads the REAL trained checkpoint
   (base 3B + LoRA adapter — distinct from `clean_eval.py`, which is
   zero-shot-7B-baseline-only) and evaluates it using `SFT_PROMPT` (the exact
   schema it was fine-tuned on). Also computes a confidence signal:
   `generate(..., output_scores=True)`'s average per-token probability,
   mapped to P(genuine) via `generation_confidence_to_p_genuine()` — a
   documented whole-response proxy, not a claim of true per-field
   calibration.

4. **Phase 7 decision layer — DONE.** `financial_risk_reasoning.py`'s
   `explain_decision()` routes a scored document via `risk_tiering.route()`
   and produces a written rationale combining the model's own
   verdict/explanation, the dollar cost tradeoff behind that tier (from
   `cost_matrix_config.yaml`), and optionally similar past cases from
   `case_index.py` as supporting context (not a second vote).

5. **Phase 9 quantization benchmarking — code DONE, not yet run.**
   `quantization_bench.py` compares fp16/int8/int4 via the same load_fn/
   eval_fn injection pattern as leave-one-out/adversarial rounds. Cost-per-
   verification uses a labeled, stated GPU-cost assumption
   (`ASSUMED_GPU_COST_PER_HOUR_USD = 0.35`) — the relative cost across
   precisions is the real argument, not the absolute dollar figure.

6. **Kaggle drivers built for leave-one-out and adversarial rounds**
   (`kaggle_kernels/phase6_leave_one_out/`, `kaggle_kernels/
   phase6_adversarial_rounds/`). User confirmed **5-tier** leave-one-out
   (not 4) after being shown the real time estimate: ~11h05m total Kaggle
   GPU time (5 folds × Phase 4's measured ~2h13m/fold), vs ~8h52m for 4
   tiers — worth the extra ~2h13m since Tier 3 was being generated anyway.
   Both drivers apply the v24-established discipline of explicit
   `gc.collect()`+`torch.cuda.empty_cache()` after every model load, with
   before/after GPU memory diagnostics printed, not assumed.

7. **REAL BUG FOUND AND FIXED (2026-07-23): the Kaggle dataset's `data/`
   folder was badly stale — only ~700-730 of the full 1000 raw MIDV-2020
   images had ever been staged** (missing ~300, roughly the val+test splits
   minus a handful used in Phase 3's smoke eval sample). This dataset was
   set up once, early in the project, and every subsequent push in this
   session only ever refreshed `config/`/`src/` — `data/` was never
   wholesale-resynced. Silent until the adversarial-rounds kernel's first run
   crashed on `FileNotFoundError: data/raw/.../alb_id/11.jpg` (a genuine
   test-split image `build_eval_set()` legitimately needed but the dataset
   never had). Fixed by wholesale `rm -rf` + `cp -r` of the entire local
   `data/` folder into staging (verified byte-identical file listing before
   pushing) rather than patching around the specific missing file — this
   exact class of bug (a stale subset silently substituting for the real
   thing) could have also broken leave-one-out folds (some tier1/2's source
   images are val/test-split) or quantization benchmarking's eval set the
   same way, so a full resync was the only real fix, not a one-off patch.
   **Anyone resuming this session should verify the current dataset version
   actually has all 1000 raw images before trusting any further Kaggle run**
   (`kaggle datasets files poojadheniya/doc-verification-data`, count per
   document-code subfolder).

8. **TWO MORE REAL BUGS FOUND AND FIXED (2026-07-23 night, same debugging
   session):**
   - **Tier manifest filename convention bug**: `_default_tier_manifest_paths()`
     (and both Kaggle drivers' own copy of the same logic) derived each
     tier's manifest filename from its full descriptive name
     (`tier1_field_tamper_manifest.json`), but every tier's manifest is
     actually written with just a short numeric prefix
     (`tier1_manifest.json` — see field_tamper.py/splice.py/etc's own
     manifest-writing code). `build_sft_examples()` silently skips
     nonexistent tier paths by design, so this produced **zero** tier
     examples/folds with no error at all — invisible until a real Kaggle run
     logged "0 folds" / "0 tampered examples". Fixed in
     `sft_train._default_tier_manifest_paths()` and both kernel drivers;
     added a regression test that exercises real repo data (not fixtures),
     since the old test suite only ever passed explicit manifest paths and
     never exercised the default path-construction logic for real.
   - **Backslash (Windows-native) paths baked into every locally-generated
     manifest**: `acquire_dataset.py`/`field_tamper.py`/`splice.py`/
     `synthetic_id_gen.py`/`recapture_sim.py`/`inpaint_forger.py` all
     serialized manifest path fields via bare `str(some_path)`, which
     produces backslash separators on this Windows dev machine. Every path
     resolved fine locally (Windows accepts both slash styles) —
     completely invisible until a manifest reached a real Linux Kaggle
     kernel, where it's a hard `FileNotFoundError`. `kaggle_package.py`'s
     `stage_package()`/`_rewrite_paths_in_place()` already existed
     specifically to normalize this before staging (a sign this was a
     known, previously-solved problem) — but every manual `cp -r` dataset
     refresh done during this session's later work bypassed that utility
     entirely, pushing raw (backslash) local manifests straight through.
     Fixed at the source (every generator now uses `.as_posix()`); existing
     manifests normalized in place via the project's own
     `_rewrite_paths_in_place()`; added a regression test asserting no
     manifest contains a backslash path.
   - **A real, separate infrastructure quirk observed twice**: a Kaggle
     dataset version reporting `"ready"` via `datasets status` did NOT mean
     a kernel pushed immediately after would see the update — both the
     data-completeness fix and the backslash-path fix showed this exact
     pattern (a kernel/diagnostic-kernel pushed right after "ready" still
     saw stale data), resolving itself after waiting ~10-15 more minutes
     with no further action. **Anyone hitting a Kaggle run that seems to
     contradict a just-completed dataset push should suspect this
     propagation lag before assuming the fix itself is wrong** — verify with
     the cheap `kaggle_kernels/diagnostic_check/` kernel (no GPU, no pip
     installs, runs in seconds) rather than burning GPU hours re-testing.
   - **Both `phase6_leave_one_out` (v3) and `phase6_adversarial_rounds` (v4)
     confirmed genuinely running past every previous failure point** as of
     this update: LOO built 754 real training examples for fold 1 and is
     training for real; adversarial-rounds cleared 18+/30 eval examples with
     no crash. This is the first time either has done real work.

**Status as of this update (overnight autonomous session, user asleep,
explicit full autonomy granted — see chat for exact wording):** both Phase 6
Kaggle jobs are running for real. Continuing to monitor them at a long
interval while building quantization benchmarking, the Streamlit demo, the
results notebook, and the writeup in parallel — see task list (`TaskList`
tool) for live status of each. Every decision made autonomously tonight is
being logged in this file as it happens, not left implicit.

**Writeup requirement, explicitly restated by the user:** the epoch 2-3
loss plateau seen in v24's training (loss stuck ~1.25-1.26 after a fast
drop in epoch 1) must be treated as a real finding, not a footnote — if
leave-one-out/adversarial-rounds eval results come back weak, the honest
hypothesis chain to flag is that LoRA rank 4 (cut for memory reasons, not
because r=4 was judged sufficient) and/or the small 719-example training
set may have limited the model's real capacity to learn the task.

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
  see `tests/test_pipeline_smoke.py`, currently 61 tests passing)
- Stop and report (don't loop indefinitely) on: a failure that resists a few
  isolated diagnostic attempts, any change to core project scope/claims, or a
  genuine compute/time tradeoff decision

## Task tracker state (see TaskList tool)

- #1-4: done (scaffold, data foundation, baselines, Phase 4 SFT training)
- #5: in_progress — Tier 3 done for real (13/15); Tier 4/5 were already done
- #6: in_progress — leave-one-out + adversarial-rounds Kaggle drivers built,
  neither has produced real results yet (blocked on the data-staleness fix
  above, now corrected — see "Immediate next steps")
- #7: done — financial_risk_reasoning.py implemented
- #8: retrieval done, DPO deprioritized per above
- #9: code done (quantization_bench.py), not yet run on Kaggle
- #10: pending, optional/time-boxed (Layer 3 — drift sim, cost dashboard)
- #11: pending — demo + writeup
