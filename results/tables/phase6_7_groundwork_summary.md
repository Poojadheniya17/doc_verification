# Phase 6/7 Groundwork Summary — Orchestration Logic + Layer 2 Decision Layer

Built in parallel with the live Kaggle Phase 4 SFT training run, scoped to
exactly what doesn't need a trained model: orchestration/aggregation logic for
Phase 6's two core experiments, and all of Phase 7's decision layer (which
never needed a model in the first place — it operates on confidence scores).

## Leave-one-attack-tier-out (`src/eval/leave_one_out_eval.py`)

Same split as `sft_train.py`/`clean_eval.py`: fold construction and
per-tier/overall bootstrap-CI aggregation are pure logic, tested for real
against fixture manifests shaped like the actual Tier 1/2/4/5 manifests.
Actually running a fold means training a fresh QLoRA model excluding that
tier's forgeries, then evaluating on the held-out tier — that's N models'
worth of Kaggle GPU time (N = number of tiers), injected via `train_fn`/
`eval_fn` callables so `run()`'s wiring is verifiable against fakes without
touching a real model.

Tier availability: `training_config.yaml`'s `leave_one_out.tiers` lists the
intended final 5; `run()` rotates over whichever manifests it's actually
given (currently tier1/2/4/5 — Tier 3's diffusion inference is still
Kaggle-only) and logs the gap rather than silently pretending all 5 ran.

6 tests: successes-only filtering, fold construction (including the edge case
of a single tier with nothing to hold it out against), per-tier + pooled CI
aggregation, and the full `run()` orchestration both with and without
callables provided.

## Adversarial rounds (`src/eval/adversarial_rounds.py`)

Same pattern: failure-mining (cap from `training_config.yaml`, order-preserving
so a specific round's retraining set is reproducible) and the round-by-round
accuracy curve are pure logic; actual per-round retraining is Kaggle-only,
injected the same way.

4 tests, including one that drives `run()` through all 3 rounds with a fake
model whose accuracy improves each round and checks the accuracy curve comes
out monotonically increasing, matching the fake model's own behavior.

## Layer 2 decision logic (`src/decision/risk_tiering.py`, `cost_simulator.py`)

Fully real, not mocked — no model involved at any point, so local unit tests
exercise the actual logic that would run in production. `route()` takes
already-loaded config (not a path) since `cost_simulator.py` calls it once per
document across an entire eval set per threshold pair in the sweep; a
path-based `route_from_path()` wrapper covers one-off/CLI use.

`cost_matrix_config.yaml` gained a `threshold_sweep` section (candidate grids
for `auto_approve`/`auto_reject` thresholds) so the sweep itself stays
config-driven rather than hardcoding a grid in Python.

4 tests: threshold boundary behavior (inclusive on both ends), out-of-range
confidence rejection, cost accounting (false-accept/false-reject/review counts
and total cost against a hand-worked example), and the full sweep (grid size,
argmin correctness, invalid reject>=approve pairs correctly excluded).

## Retrieval layer (`src/retrieval/case_index.py`)

Not deferred to Kaggle — a sentence-embedding model (`all-MiniLM-L6-v2`,
~90MB, 384-dim) is small enough to actually run on this machine's constrained
RAM, confirmed before writing any code. FAISS (`IndexFlatIP` over normalized
embeddings = exact cosine similarity) does the actual nearest-neighbor search;
no ANN index needed at this project's scale (dozens-to-low-hundreds of
flagged cases).

**Real end-to-end verification** (not a mock) against 4 dummy flagged cases
(one per forgery tier + one genuine), querying "the ID photo looks pasted on
with a visible edge":

| Similarity | Case | Explanation |
|---|---|---|
| 0.482 | c1 (tier2_splicing) | "visible seam and inconsistent lighting" |
| 0.329 | c2 (tier4_full_synthetic) | "no physical capture artifacts" |

Correctly ranked the splicing case highest by a meaningful margin — the
retrieval layer surfaces semantically relevant precedent, not just
keyword overlap.

3 tests cover the pure logic (case-text construction, fallback when a case
has no rich fields) and a save/load roundtrip against fabricated embeddings.
`build_index`/`query` themselves are excluded from the fast automatic suite —
same reasoning as `field_tamper.py`'s EasyOCR path (real model download on
first call) — and instead verified manually as shown above.

## Test count

33 (through Phase 5) + 17 new = **50 passing**, all in
`tests/test_pipeline_smoke.py`, no new dependencies beyond what was already
in `requirements.txt` (`sentence-transformers`, `faiss-cpu` — installed and
confirmed working this phase, having sat unused since Phase 1).

## What's still Kaggle-only

- Actually running leave-one-out folds and adversarial-round retraining (both
  need real QLoRA training, N times over)
- Financial risk reasoning (`src/decision/financial_risk_reasoning.py`) — an
  LLM-generated recommendation, needs the fine-tuned model; scoped out of this
  phase since it's the one piece of Phase 7 that actually needs one
- Tier 3 diffusion-inpainting inference (unchanged from Phase 5)
