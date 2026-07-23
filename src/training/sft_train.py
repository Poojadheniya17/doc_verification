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


def build_sft_examples(genuine_manifest_path: str, tier1_manifest_path: str | None = None,
                        tier2_manifest_path: str | None = None, split: str = "train") -> list[dict]:
    """Builds {"image_path", "target": {...schema...}} training examples from the
    genuine manifest (extraction + "genuine" verdict) and Tier 1/2 forgery
    manifests (tamper verdict + localization bbox, extraction fields carried
    over from the source genuine document with Tier 1's overrides applied).
    """
    examples = []

    genuine = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    genuine_by_path = {e["path"]: e for e in genuine["entries"]}
    for entry in genuine["entries"]:
        if entry["split"] != split or not entry.get("ground_truth"):
            continue
        target = {**entry["ground_truth"], "tamper_verdict": "genuine", "tamper_regions": []}
        examples.append({"image_path": entry["path"], "target": target, "tier": "genuine"})

    if tier1_manifest_path and Path(tier1_manifest_path).exists():
        tier1 = json.loads(Path(tier1_manifest_path).read_text(encoding="utf-8"))
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

    if tier2_manifest_path and Path(tier2_manifest_path).exists():
        tier2 = json.loads(Path(tier2_manifest_path).read_text(encoding="utf-8"))
        for entry in tier2["entries"]:
            if not entry.get("success"):
                continue
            source_entry = genuine_by_path.get(entry["source_image"])
            if not source_entry or source_entry["split"] != split or not source_entry.get("ground_truth"):
                continue
            # Splicing only replaces the photo — text fields are untouched, so no
            # field-override logic is needed here (unlike Tier 1).
            target = {**source_entry["ground_truth"], "tamper_verdict": "tampered",
                      "tamper_regions": [entry["bbox_xyxy"]]}
            examples.append({"image_path": entry["forged_image"], "target": target, "tier": "tier2_splicing"})

    return examples


def build_conversation(image_path: str, target: dict) -> list[dict]:
    """Chat-formatted SFT example: user turn is the image + instruction prompt,
    assistant turn is the target JSON serialized as the expected completion.
    """
    return [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": SFT_PROMPT}]},
        {"role": "assistant", "content": [{"type": "text", "text": json.dumps(target)}]},
    ]


def load_model_for_training(model_config: dict):
    """Loads Qwen2.5-VL in 4-bit (QLoRA) and wraps it with a LoRA adapter per
    model_config.yaml. Not exercised locally — see module docstring.
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
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name"], quantization_config=bnb_config, device_map="auto",
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    # use_reentrant=False, not PEFT's default (unset -> reentrant=True):
    # kernel v11 and v12 (different image sizes, LoRA ranks, and optimizers)
    # both hung — not crashed — at the identical backward-pass line, only
    # after earlier OOM fixes stopped the process from running out of memory
    # first. Reentrant checkpointing is a documented deadlock source
    # specifically when a checkpointed region mixes frozen and trainable
    # parameters, which is exactly QLoRA's shape (frozen base + trainable
    # LoRA adapters) — and the warning recommending use_reentrant=False was
    # printed, unaddressed, in every single run's log. This is a targeted fix
    # for a named, evidenced cause, not another capacity cut.
    model = prepare_model_for_kbit_training(model, gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_cfg = model_config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"], bias=lora_cfg["bias"], task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor


def train(model_config_path: str, training_config_path: str, environment: str | None = None) -> None:
    model_config = load_yaml(model_config_path)
    training_config = load_yaml(training_config_path)
    if environment:
        training_config["environment"] = environment
    paths = resolve_paths(training_config)

    genuine_manifest = paths["data_root"] / "processed" / "genuine_manifest_templates.json"
    tier1_manifest = paths["data_root"] / "synthetic_forgeries" / "tier1_field_tamper" / "tier1_manifest.json"
    tier2_manifest = paths["data_root"] / "synthetic_forgeries" / "tier2_splicing" / "tier2_manifest.json"

    train_examples = build_sft_examples(str(genuine_manifest), str(tier1_manifest), str(tier2_manifest), split="train")
    logger.info(f"Built {len(train_examples)} SFT training examples "
                f"(environment={training_config['environment']})")

    if training_config["environment"] == "local":
        logger.info("environment=local: stopping after data construction — this machine cannot load "
                     "Qwen2.5-VL (see module docstring). Re-run with --environment kaggle for real training.")
        return

    from transformers import Trainer, TrainingArguments

    model, processor = load_model_for_training(model_config)
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
        inputs = processor(text=texts, images=image_inputs, padding=True, return_tensors="pt")
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

    training_args = TrainingArguments(
        output_dir=str(paths["checkpoint_root"] / "sft"),
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
    trainer.save_model(str(paths["checkpoint_root"] / "sft" / "final"))


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
