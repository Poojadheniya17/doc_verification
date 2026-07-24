"""SFT + QLoRA training entrypoint for extraction + tamper-detection + localization.

Config-driven via config/training_config.yaml (environment: local|kaggle) and
config/model_config.yaml (model/LoRA/quantization hyperparameters) — see
src/utils/config_utils.py for the environment-path resolution.

IMPORTANT (standing constraint since Phase 3, see README.md): this project's
local dev machine has ~7.7GB RAM and cannot load ANY size of Qwen2.5-VL without
severe disk thrashing. This script is therefore validated locally ONLY at the
data-construction level (build_sft_examples, target-JSON building — pure
Python/JSON logic, no model weights involved; see tests/test_pipeline_smoke.py).
The actual model loading, QLoRA setup, and training loop below have NOT been
executed on this machine — they run for real on Kaggle (python -m
src.training.sft_train --config config/training_config.yaml
--environment kaggle), which is the whole reason this project's compute split
exists in the first place.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

from src.eval.metrics import field_similarity
from src.utils.config_utils import load_yaml, resolve_paths
from src.utils.logging_utils import get_logger

logger = get_logger("sft_train")

EXTRACTION_FIELDS = ["name", "dob", "id_number", "address", "expiry"]

SFT_PROMPT = """You are an identity document verification assistant. Look at this \
identity document image and respond with ONLY a single JSON object (no other \
text, no markdown code fences) with exactly these keys:

- "name": the document holder's full name, or null if not visible
- "dob": date of birth as printed, or null if not visible
- "id_number": the document/passport/ID number as printed, or null if not visible
- "address": the printed address, or null if this document type does not show one
- "expiry": the expiry date as printed, or null if not visible
- "tamper_verdict": exactly "genuine" or "tampered"
- "tamper_regions": list of [x0, y0, x1, y1] pixel boxes for any tampered region \
(empty list if genuine)
- "explanation": one sentence explaining your tamper_verdict

Respond with the JSON object only."""


def _apply_tier1_field_overrides(ground_truth: dict, tampers: list[dict]) -> dict:
    """A Tier 1 tamper record only tells us the OCR bbox/text it edited, not
    which of our 5 schema fields (if any) that corresponds to — so we match it
    back by comparing the tamper's original_text against each ground-truth
    field value (whitespace/punctuation-insensitive) and substitute the
    tampered_text for whichever field matches best above a similarity
    threshold. Tampers that hit something outside our 5 fields (authority
    lines, MRZ, etc.) correctly leave the schema fields untouched — the model
    should still report those fields' original genuine values even though the
    image is tampered elsewhere.

    Known simplification: a tamper that doesn't clear the 0.6 similarity bar
    (e.g. very short/garbled OCR text) is silently not matched to any field,
    even if it did in fact hit one. This trades recall for not corrupting
    training targets with false-positive substitutions.
    """
    updated = dict(ground_truth)
    for tamper in tampers:
        original = "".join(ch for ch in tamper["original_text"] if ch.isalnum()).lower()
        best_field, best_score = None, 0.0
        for field in EXTRACTION_FIELDS:
            gt_value = ground_truth.get(field)
            if not gt_value:
                continue
            normalized_gt = "".join(ch for ch in gt_value if ch.isalnum()).lower()
            score = field_similarity(original, normalized_gt)
            if score > best_score:
                best_field, best_score = field, score
        if best_field and best_score > 0.6:
            updated[best_field] = tamper["tampered_text"]
    return updated


def build_sft_examples(genuine_manifest_path: str, tier_manifest_paths: dict[str, str] | None = None,
                        split: str = "train") -> list[dict]:
    """Builds {"image_path", "target": {...schema...}, "tier": ...} training
    examples from the genuine manifest (extraction + "genuine" verdict) plus
    whichever forgery tier manifests are passed in tier_manifest_paths — any
    subset of tier1_field_tamper/tier2_splicing/tier3_inpainting/
    tier4_full_synthetic/tier5_recapture. Generalized from a hardcoded
    tier1+tier2 signature (Phase 4) so leave_one_out_eval.py's train_fn can
    pass an arbitrary subset (one tier held out per fold) without this
    function needing to know about folds at all.

    Per-tier target construction differs by how each tier was generated:
    - tier1_field_tamper: text fields edited in place — source ground truth
      with _apply_tier1_field_overrides() applied, tamper_regions from each
      individual tamper's own bbox (possibly several per image).
    - tier2_splicing / tier3_inpainting: only the photo region is replaced —
      source ground truth carried over unchanged, single tamper_regions bbox
      (each manifest's own "bbox_xyxy").
    - tier4_full_synthetic: a wholly fabricated document, not derived from any
      real source — ground_truth comes directly from the manifest's own entry
      (no source lookup). tamper_regions=[] since there's no single localized
      edit region, the entire document is synthetic. Known gap: this tier's
      manifest doesn't record a train/test split (synthetic_id_gen.py never
      assigns one) — included unconditionally regardless of the `split`
      argument rather than silently dropped, since there's no principled
      holdout to apply. Flagged here rather than fixed by regenerating, since
      leave-one-out's fold design already keeps a held-out tier out of its
      own fold's training set regardless of this.
    - tier5_recapture: a screen-recapture simulation of a genuine source —
      printed fields are unchanged (only image quality is degraded), so
      ground truth is carried over from the source unchanged. tamper_regions=[]
      for the same reason as tier4 — a global quality transform, not a
      localized edit.
    """
    examples = []

    genuine = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    genuine_by_path = {e["path"]: e for e in genuine["entries"]}
    for entry in genuine["entries"]:
        if entry["split"] != split or not entry.get("ground_truth"):
            continue
        target = {**entry["ground_truth"], "tamper_verdict": "genuine", "tamper_regions": []}
        examples.append({"image_path": entry["path"], "target": target, "tier": "genuine"})

    tier_manifest_paths = tier_manifest_paths or {}

    tier1_path = tier_manifest_paths.get("tier1_field_tamper")
    if tier1_path and Path(tier1_path).exists():
        tier1 = json.loads(Path(tier1_path).read_text(encoding="utf-8"))
        for entry in tier1["entries"]:
            if not entry.get("success"):
                continue
            source_entry = genuine_by_path.get(entry["source_image"])
            if not source_entry or source_entry["split"] != split or not source_entry.get("ground_truth"):
                continue
            target = _apply_tier1_field_overrides(source_entry["ground_truth"], entry["tampers"])
            target["tamper_verdict"] = "tampered"
            target["tamper_regions"] = [t["bbox_xyxy"] for t in entry["tampers"]]
            examples.append({"image_path": entry["forged_image"], "target": target, "tier": "tier1_field_tamper"})

    # tier2_splicing and tier3_inpainting share the same shape: source lookup,
    # ground truth carried over unchanged, single bbox_xyxy region.
    for tier_name in ("tier2_splicing", "tier3_inpainting"):
        path = tier_manifest_paths.get(tier_name)
        if not path or not Path(path).exists():
            continue
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        for entry in manifest["entries"]:
            if not entry.get("success"):
                continue
            source_entry = genuine_by_path.get(entry["source_image"])
            if not source_entry or source_entry["split"] != split or not source_entry.get("ground_truth"):
                continue
            target = {**source_entry["ground_truth"], "tamper_verdict": "tampered",
                      "tamper_regions": [entry["bbox_xyxy"]]}
            examples.append({"image_path": entry["forged_image"], "target": target, "tier": tier_name})

    tier4_path = tier_manifest_paths.get("tier4_full_synthetic")
    if tier4_path and Path(tier4_path).exists():
        tier4 = json.loads(Path(tier4_path).read_text(encoding="utf-8"))
        for entry in tier4["entries"]:
            if not entry.get("success") or not entry.get("ground_truth"):
                continue
            target = {**entry["ground_truth"], "tamper_verdict": "tampered", "tamper_regions": []}
            examples.append({"image_path": entry["forged_image"], "target": target, "tier": "tier4_full_synthetic"})

    tier5_path = tier_manifest_paths.get("tier5_recapture")
    if tier5_path and Path(tier5_path).exists():
        tier5 = json.loads(Path(tier5_path).read_text(encoding="utf-8"))
        for entry in tier5["entries"]:
            if not entry.get("success"):
                continue
            source_entry = genuine_by_path.get(entry["source_image"])
            if not source_entry or source_entry["split"] != split or not source_entry.get("ground_truth"):
                continue
            target = {**source_entry["ground_truth"], "tamper_verdict": "tampered", "tamper_regions": []}
            examples.append({"image_path": entry["forged_image"], "target": target, "tier": "tier5_recapture"})

    return examples


def balance_examples(examples: list[dict], target_ratio: float = 1.0) -> list[dict]:
    """Oversamples the minority ("tampered") class by simple repetition so the
    final genuine:tampered ratio is no worse than target_ratio (1.0 = exact 1:1).

    Real motivation (post-v25): every training set this project has actually
    built is severely imbalanced (e.g. 700 genuine vs 62 tampered across all 5
    tiers, ~11:1), and three independent real evaluations (adversarial-rounds,
    quantization-bench, leave-one-out) all show the model collapsing to always
    predicting "genuine" — the easy, loss-minimizing shortcut available under
    that imbalance. Repetition (rather than a weighted loss) was chosen because
    it needs no changes to the collate/loss path — every repeated example is
    still ordinary training data, just seen more often — which keeps this
    change small and independently testable, per the project's "move fast,
    don't gold-plate" standing instruction on this specific experiment.

    Honest tradeoff, stated plainly: oversampling to exact 1:1 significantly
    grows the training set (e.g. 62 tampered examples repeated ~11x each to
    match 700 genuine -> ~1400 total, roughly double the ~719-762 examples
    prior real runs trained on), which proportionally increases real Kaggle
    training wall-clock time. This is accepted deliberately for this specific
    experiment because eliminating the trivial majority-class optimum
    entirely, not just weakening it, is the cleanest test of whether class
    imbalance (rather than capacity) is the true driver of the collapse.

    No-op (returns examples unchanged) if there are no tampered examples, or
    if the existing ratio is already at or better than target_ratio.
    """
    genuine = [e for e in examples if e["target"]["tamper_verdict"] == "genuine"]
    tampered = [e for e in examples if e["target"]["tamper_verdict"] == "tampered"]
    if not tampered or not genuine:
        return list(examples)

    target_tampered_count = round(len(genuine) / target_ratio)
    if target_tampered_count <= len(tampered):
        return list(examples)

    oversampled_tampered = [tampered[i % len(tampered)] for i in range(target_tampered_count)]
    return genuine + oversampled_tampered


def build_conversation(image_path: str, target: dict) -> list[dict]:
    """Chat-formatted SFT example: user turn is the image + instruction prompt,
    assistant turn is the target JSON serialized as the expected completion.
    """
    return [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": SFT_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
    ]


def load_model_for_training(model_config: dict, resume_from_adapter: str | None = None):
    """Loads Qwen2.5-VL in 4-bit (QLoRA) and wraps it with a LoRA adapter per
    model_config.yaml. Not exercised locally — see module docstring.

    resume_from_adapter: path to an existing saved LoRA adapter (e.g. Phase 4's
    checkpoints/sft_v24_final, or a prior adversarial round's checkpoint) to
    continue training from, instead of a freshly-initialized adapter. Needed
    for adversarial_rounds.py's targeted retraining, which builds on the
    previous round's model rather than starting over from the base model each
    time.
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    quant_cfg = model_config["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )
    model_cfg = model_config["model"]
    processor = AutoProcessor.from_pretrained(model_cfg["name"], trust_remote_code=model_cfg["trust_remote_code"])
    # device_map={"": 0} instead of "auto": every prior run (v8-v16) used "auto"
    # and every hang (v11-v14, v16) happened only when gradient checkpointing
    # was enabled, in either mode -- and every traceback in this whole saga,
    # including the successful-diagnosis OOM crashes (v9/v10), shows
    # accelerate/hooks.py's hooked forward wrapper in the call stack.
    # Checkpointing's recompute step re-enters the forward pass during
    # backward, which has to pass back through that hook -- a known class of
    # interaction issue between accelerate's multi-device dispatch hooks and
    # checkpointing recomputation. We only ever have one GPU, so "auto"'s
    # dispatch logic (and its hooks) buys nothing here; pinning explicitly to
    # device 0 removes the one variable never changed across 5 failed
    # hypotheses, without giving up checkpointing's memory savings the way
    # the v15 diagnostic did.
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name"], quantization_config=bnb_config, device_map={"": 0},
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    # FINAL DECISION on gradient checkpointing, after v11-v18 (six single-variable
    # hypotheses, each ruled out in turn: num_workers, checkpointing's
    # reentrant/non-reentrant mode, checkpointing on/off, model size 7B->3B,
    # explicit use_cache, device_map/GPU-visibility): checkpointing DISABLED
    # entirely. None of those six changes resolved the hang; each one narrowed
    # the search space without finding the actual cause, and the failure point
    # kept moving rather than converging on an explanation. Continuing to guess
    # at a seventh hypothesis was judged not worth further Kaggle time.
    #
    # v15 (the one prior run with checkpointing off) hit a real, ordinary,
    # arithmetic-explainable CUDA OOM instead of a hang (594MiB requested,
    # 554.81MiB free — short by ~40MiB) — proof checkpointing-off avoids the
    # hang entirely, it just needs a real memory margin, which
    # model_config.yaml's max_image_size cut (768->512) now provides. use_cache
    # is still set explicitly (harmless, good practice) even though it turned
    # out not to be the hang's cause.
    #
    # Real, honest cost of this decision: checkpointing trades memory for
    # recompute, so disabling it means training is faster per step but has
    # zero slack for anything that increases activation memory later
    # (larger images, higher LoRA rank, bigger batch) without hitting the OOM
    # margin recovered here. That's a real constraint on this project's
    # trained model, not a footnote — documented in README.md and
    # results/tables/phase4_sft_summary.md alongside the image-size cut's
    # accuracy tradeoff.
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    if resume_from_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, resume_from_adapter, is_trainable=True)
        logger.info(f"Resumed LoRA adapter weights from {resume_from_adapter}")
    else:
        lora_cfg = model_config["lora"]
        peft_config = LoraConfig(
            r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
            target_modules=lora_cfg["target_modules"], bias=lora_cfg["bias"], task_type=lora_cfg["task_type"],
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor


DEFAULT_TIER_NAMES = ["tier1_field_tamper", "tier2_splicing"]  # Phase 4 (v24)'s exact composition


def _default_tier_manifest_paths(paths: dict, tier_names: list[str]) -> dict[str, str]:
    """Maps tier names to their manifest paths under this environment's data_root.

    Real bug fixed here: every tier's manifest FILE is named with just the
    short numeric prefix ("tier1_manifest.json", "tier2_manifest.json", ...,
    per field_tamper.py/splice.py/inpaint_forger.py/synthetic_id_gen.py/
    recapture_sim.py's own manifest-writing code), even though the containing
    FOLDER and the "tier" key used everywhere else in this codebase is the
    full descriptive name ("tier1_field_tamper", "tier2_splicing", ...). This
    function previously derived the filename from the full name too
    ("tier1_field_tamper_manifest.json"), which never exists — and because
    build_sft_examples() silently skips any tier path that doesn't exist (by
    design, so a not-yet-generated tier degrades gracefully), this produced
    ZERO tier examples with no error at all, only surfacing when a real
    Kaggle leave-one-out/adversarial-rounds run logged "0 folds"/"0 tampered
    examples". Caught by results, not by the (previously insufficient) unit
    tests — see test_default_tier_manifest_paths_match_real_files_on_disk.
    """
    return {
        tier: str(paths["data_root"] / "synthetic_forgeries" / tier / f"{tier.split('_')[0]}_manifest.json")
        for tier in tier_names
    }


def train(model_config_path: str, training_config_path: str, environment: str | None = None,
          tier_names: list[str] | None = None, train_examples: list[dict] | None = None,
          checkpoint_subdir: str = "sft", resume_from_adapter: str | None = None) -> str | None:
    """tier_names: which forgery tiers to include when building the training set
    (defaults to Phase 4's exact tier1+tier2 composition). Ignored if
    train_examples is given directly.

    train_examples: bypasses build_sft_examples() entirely and trains on these
    examples as-is — the injection point leave_one_out_eval.py's train_fn (a
    specific tier subset per fold) and adversarial_rounds.py's train_fn (mined
    failure examples only, reconstructed with their real ground-truth target)
    both use, so this function doesn't need to know about folds or rounds.

    checkpoint_subdir: subdirectory under checkpoint_root to save into (e.g.
    "loo_fold_tier2_splicing", "adv_round1") so parallel experiments don't
    overwrite Phase 4's "sft" checkpoint or each other.

    resume_from_adapter: see load_model_for_training(). Returns the final
    checkpoint's directory path (the "model_handle" train_fn callables hand to
    eval_fn), or None if training didn't run (local environment).
    """
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    if environment:
        training_config["environment"] = environment
    paths = resolve_paths(training_config)

    if train_examples is None:
        genuine_manifest = paths["data_root"] / "processed" / "genuine_manifest_templates.json"
        tier_manifest_paths = _default_tier_manifest_paths(paths, tier_names or DEFAULT_TIER_NAMES)
        train_examples = build_sft_examples(str(genuine_manifest), tier_manifest_paths, split="train")
    logger.info(f"Built {len(train_examples)} SFT training examples "
                f"(environment={training_config['environment']})")

    class_balance_cfg = training_config["sft"].get("class_balance")
    if class_balance_cfg and class_balance_cfg.get("oversample_tampered_to_ratio"):
        pre_count = len(train_examples)
        train_examples = balance_examples(train_examples, class_balance_cfg["oversample_tampered_to_ratio"])
        logger.info(f"Class-balanced {pre_count} -> {len(train_examples)} examples "
                    f"(target ratio {class_balance_cfg['oversample_tampered_to_ratio']})")

    if training_config["environment"] == "local":
        logger.info("environment=local: stopping after data construction — this machine cannot load "
                     "Qwen2.5-VL (see module docstring). Re-run with --environment kaggle for real training.")
        return None

    from transformers import Trainer, TrainingArguments

    model, processor = load_model_for_training(model_config, resume_from_adapter=resume_from_adapter)
    sft_cfg = training_config["sft"]

    class SFTDataset:
        def __init__(self, examples: list[dict]):
            self.examples = examples

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            example = self.examples[idx]
            return build_conversation(example["image_path"], example["target"])

    max_image_size = model_config["model"]["max_image_size"]
    max_seq_length = sft_cfg["max_seq_length"]

    def collate(batch: list[list[dict]]):
        from qwen_vl_utils import process_vision_info

        texts = [processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=False) for conv in batch]
        image_inputs = []
        for conv in batch:
            imgs, _ = process_vision_info(conv)
            # MIDV-2020 source images are full-resolution scans (~2167x1521) —
            # feeding them in uncapped multiplies vision-token count (and thus
            # activation memory) far beyond what a T4 has room for alongside
            # the 4-bit base model. Found by hitting a genuine CUDA OOM 486MB
            # into the very first training step (14.24/14.56GB already used).
            # clean_eval.py already caps this for zero-shot inference
            # (max_image_size=640) — training needs the same discipline, just
            # using model_config.yaml's max_image_size instead of a hardcoded
            # eval-only constant.
            for img in imgs or []:
                img.thumbnail((max_image_size, max_image_size))
            image_inputs.extend(imgs or [])
        # max_seq_length was previously a decorative config value only (never
        # passed to the processor) — found while diagnosing v19-v22's OOMs,
        # which all ran with the full untruncated 2048-token budget regardless
        # of what training_config.yaml said. Wiring it in for real as part of
        # the v23 memory cut (see model_config.yaml/training_config.yaml
        # DECISION comments for the r=4 + max_seq_length=1024 pairing).
        inputs = processor(
            text=texts,
            images=image_inputs,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    training_args = TrainingArguments(
        output_dir=str(paths["checkpoint_root"] / checkpoint_subdir),
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        learning_rate=sft_cfg["learning_rate"],
        num_train_epochs=sft_cfg["num_epochs"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        save_strategy=sft_cfg["save_strategy"],
        logging_steps=sft_cfg["logging_steps"],
        bf16=True,
        # Switched from paged_adamw_8bit to the non-paged adamw_bnb_8bit after
        # kernel v11 (max_image_size=896, lora.r=12) silently STALLED at the
        # exact same backward-pass line for 39 minutes with zero progress and
        # no crash — never OOM'd cleanly the way v9/v10 (larger footprint,
        # still on paged_adamw_8bit) did. Working hypothesis: the paged
        # optimizer was paging optimizer state to host memory under pressure
        # and thrashing rather than erroring. v12 cuts the footprint further
        # (model_config.yaml: max_image_size 768, lora.r 8) specifically so a
        # non-paged optimizer becomes viable again — this targets the
        # suspected root cause (paging under pressure) rather than just
        # shrinking the same paged config further, which would only mask the
        # symptom if the hypothesis is right.
        optim="adamw_bnb_8bit",
        # Gated on an actual WANDB_API_KEY being present, not just
        # environment == "kaggle" — found the hard way on the first real
        # Kaggle run: Trainer tried to wandb.init() with no key configured at
        # all and crashed with UsageError before a single training step ran.
        # Set WANDB_API_KEY as a Kaggle secret to turn this back on.
        report_to=["wandb"] if os.environ.get("WANDB_API_KEY") else [],
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=SFTDataset(train_examples),
                       data_collator=collate)
    trainer.train()
    final_dir = str(paths["checkpoint_root"] / checkpoint_subdir / "final")
    trainer.save_model(final_dir)
    return final_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="config/model_config.yaml")
    parser.add_argument("--training-config", default="config/training_config.yaml")
    parser.add_argument("--environment", choices=["local", "kaggle"], default=None,
                         help="Overrides training_config.yaml's environment field")
    args = parser.parse_args()
    train(args.model_config, args.training_config, environment=args.environment)


if __name__ == "__main__":
    main()
