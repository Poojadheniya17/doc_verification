# Project status

Last updated 2026-07-25.

## Where things stand

Three real fine-tuned checkpoints exist, all Qwen2.5-VL-3B-Instruct + QLoRA:

- `checkpoints/sft_v24_final/` — LoRA r=4, 256px images, the original 719-example
  set (genuine + tier1 field tamper + tier2 splicing). First run that actually
  trained end to end. Committed in full.
- `checkpoints/sft_v25_final/` — LoRA r=16, same 719-example composition as v24,
  testing whether more adapter capacity helps. Metadata only committed (the
  safetensors file is 141.8MB, over GitHub's limit).
- `checkpoints/sft_v26_balanced/` — LoRA r=8, 384px, trained on all 5 forgery
  tiers with tampered examples oversampled to a 1:1 ratio against genuine
  (762 real examples before balancing, 1400 after). Committed in full (~71MB).

The core finding across the whole project: v24 and v25 both collapse to
predicting a single class regardless of what's in the image — confirmed three
separate ways (adversarial rounds, quantization benchmarking, leave-one-out).
The training data behind both is ~700 genuine documents against ~19-60
tampered ones, and the model just learns to always guess the majority class
since that's the easiest way to minimize loss. Bumping LoRA rank from 4 to 16
(v25) didn't change this at all — identical collapse pattern, confirmed
example-by-example, not just by the aggregate accuracy number.

Fixing the class imbalance (v26) did change it. Evaluated cold (no
adversarial retraining), v26 gets 86.7% on the fixed 30-example set and,
more importantly, the predictions are actually varied — it catches every
tampered example in the set and only misfires on 4 of 10 genuine documents
(calling them tampered, not missing real forgeries). That's a real,
qualitatively different failure mode from "always says genuine." Once the
adversarial-rounds loop starts retraining on tiny 4-20 example batches
though, it collapses right back to single-class predictions within one
round — so the retraining loop itself is fragile even starting from a good
checkpoint. That's a separate bug from the original one and still needs
fixing (see Known issues below).

Full numbers for all of this are in `results/tables/phase4_sft_summary.md`
and `results/tables/phase6_adversarial_rounds_summary.md`.

## The training debugging story

Getting a training run to actually complete took a while. Worth writing down
because most of the interesting engineering here was in the debugging, not
the final config.

**Started on 7B, moved to 3B.** The original plan was Qwen2.5-VL-7B end to
end. It trained fine as a zero-shot baseline but SFT training on it kept
hanging silently (not crashing) partway through the backward pass, across
six different single-variable fixes (checkpointing mode, num_workers,
use_cache, device_map, etc — see `config/model_config.yaml`'s history
comments for the full list). The hang followed the model down to 3B too,
which ruled out model size as the cause and pointed at something in the
checkpointing/accelerate interaction on this specific library stack.
Disabling gradient checkpointing entirely fixed the hang and turned it into
an ordinary, explainable OOM instead — a real, fixable problem instead of an
unexplained one.

**Then five OOM attempts in a row, one bug hiding underneath another.**
Cutting image size step by step (512 -> 384 -> 320 -> 256px) and then LoRA
rank (8 -> 4) barely moved the memory ceiling at all — each attempt failed
by roughly the same 20-160MB, regardless of which lever was pulled. That's
not what real memory pressure from those variables should look like. Turned
out the actual bug was upstream: the zero-shot 7B baseline (Phase 3) and the
3B training run (Phase 4) run in the same kernel process, and the 7B
model's weights were never being freed between the two phases —
`clean_eval.py` caches its model at module scope, so as long as that module
stayed imported, so did 7 fully-loaded billion parameters. None of the image
size or rank cuts touched the actual problem. Adding an explicit unload
between phases dropped GPU memory from 7.29GB down to 0.03GB, confirmed with
a before/after diagnostic print, and training finally completed
(`checkpoints/sft_v24_final/`).

**Testing whether more capacity fixes the collapse (v25).** Once v24 showed
the single-class collapse, the obvious next question was whether the model
just didn't have enough LoRA capacity to learn the task (r=4 is small).
Bumping to r=16 hit the exact same OOM three times in a row, byte-identical
down to the megabyte, regardless of image size or sequence length — a real,
reproducible wall. A dataset push mid-investigation landed a stale config on
a kernel by accident (the Kaggle dataset mount doesn't always reflect a
just-pushed version immediately, more on this below), and that stale r=16
run happened to complete successfully with real headroom to spare, which
directly contradicted the "r=16 always OOMs" conclusion from the three
identical failures. Verified against the real `adapter_config.json` from the
completed run rather than guessing: it really was r=16, and it really did
work that time. The honest conclusion is this is still not fully explained —
documented as an open question rather than either "r=16 is broken" or "r=16
is fine," since the evidence supports neither cleanly.

What v25 did answer clearly: running the same adversarial-rounds test
against it gave the identical single-class collapse as v24, confirmed
prediction-by-prediction. More capacity alone doesn't fix the problem.

**Fixing the actual imbalance (v26).** Oversampled tampered examples to a
real 1:1 ratio against genuine and trained across all 5 forgery tiers
instead of just 2. First attempt OOM'd immediately — it turned out the
r=8/512px/seq=2048 config that v25 was supposed to test (before the stale
config accident above) had never actually been run for real, and the memory
estimate it was based on came from an older run that predates
`max_seq_length` even being enforced during training. Dropped image size to
384px based on the real memory delta between two earlier 512px vs 384px
runs, and the retry completed cleanly with real margin to spare. That
checkpoint is the one that actually shows varied, mostly-correct predictions
described above.

## Known bugs found along the way

- **Tier manifest filename mismatch.** The code was building manifest
  filenames from each tier's long name (`tier1_field_tamper_manifest.json`)
  but the generators write them with a short prefix
  (`tier1_manifest.json`). Since missing tier files are silently skipped by
  design, this produced zero tampered training examples with no error at
  all — only surfaced once a Kaggle run logged "0 tampered examples."
- **Backslash paths in manifests.** Every manifest-writing script serialized
  paths with plain `str(path)`, which uses backslashes on Windows. Worked
  fine locally, hard-failed with `FileNotFoundError` the moment a manifest
  reached the Linux Kaggle environment. Fixed by using `.as_posix()`
  everywhere paths get serialized.
- **Stale Kaggle dataset.** The `data/` folder on the Kaggle dataset only
  had ~700 of the 1000 raw source images — it was set up once early on and
  never fully resynced after later pushes only touched `config/`/`src/`.
  Silent until a kernel needed a specific missing image and threw
  `FileNotFoundError`.
- **Kaggle dataset mount propagation lag.** A dataset version reporting
  "ready" doesn't guarantee a kernel pushed right after actually sees the
  new files — this bit multiple pushes this session, including the stale
  r=16 config incident above. `kaggle_kernels/diagnostic_check/` exists
  specifically to verify what's actually mounted (no GPU, no pip installs,
  runs in seconds) before burning real GPU time on an assumption.

## Open issue: adversarial-rounds retraining is fragile

v26's base checkpoint discriminates well, but retraining it on the 4-20
examples the adversarial-rounds script mines each round collapses it back
to a single-class predictor almost immediately — a full 3-epoch retrain on
a handful of examples appears to overwrite most of what the 1400-example
balanced set taught it. Likely fix: much smaller learning rate for these
mini-retrains, or mix the mined failures back in with a slice of the
original training set instead of training on them in isolation. Not yet
tried.

## Not done

- Full leave-one-out re-run on v26. The adversarial-rounds check above
  already answers the main question (does class balancing fix the
  collapse), and a full 5-fold retrain is a multi-hour Kaggle job — worth
  doing with more time, but not necessary to draw the current conclusions.
- DPO training (the other half of Phase 8 — retrieval is done).
- Demo app's example gallery isn't populated with real captured predictions
  yet; needs a short dedicated Kaggle run through the trained checkpoint.
- Layer 3 (drift monitoring, cost dashboard) — optional extension, not
  attempted.

## Repo / infra notes

- GitHub: https://github.com/Poojadheniya17/doc_verification, `master` branch.
- Kaggle dataset (code + data): `poojadheniya/doc-verification-data`
- Main training kernel: `poojadheniya/doc-verification-zero-shot-baseline-sft-qlora`
  (driver: `kaggle_kernels/phase3_4_sft_baseline/kernel_driver.py`)
- Push a dataset update: `kaggle datasets version -p <staging_dir> --dir-mode zip -m "<msg>"`
  (staging dir is `../kaggle_package_staging` relative to this repo)
- Push a kernel: `kaggle kernels push -p <kernel_dir>`
- If a kernel push fails with "Maximum batch GPU session count of 2 reached,"
  a previous session is stuck — `kaggle kernels delete <slug> -y` and push
  again.
- `kaggle.exe` full path on this machine:
  `/c/Users/Acer/AppData/Roaming/Python/Python314/Scripts/kaggle.exe`
  (not on the Bash tool's PATH even though it's on the real Windows PATH).
- `kaggle kernels logs -f <kernel>` streams live logs; `kernels output` only
  works after a run finishes. Redirect to a file before reading — the
  Windows console can't print some of the non-ASCII characters these logs
  contain.
- Before trusting a Kaggle run against a config or checkpoint you just
  pushed, verify it's actually mounted with `kaggle_kernels/diagnostic_check/`
  first. See "Kaggle dataset mount propagation lag" above.

## Task tracker

- Scaffold, data foundation, baselines, Phase 4 SFT training — done
- Forgery tiers 1-5 — done, all real data
- Leave-one-out + adversarial rounds — done for v24/v25/v26; full LOO re-run
  on v26 not done (see Not done above)
- Decision layer (`financial_risk_reasoning.py`) — done, unit-tested
- Retrieval — done; DPO not attempted
- Quantization benchmarking — done
- Demo app — built, example gallery needs real captured predictions
- Layer 3 — not attempted, optional
- Writeup — done, needs a final pass once the adversarial-rounds retraining
  fix (if attempted) lands
