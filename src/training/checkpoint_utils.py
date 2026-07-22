"""Checkpoint save/resume helpers shared by sft_train.py and dpo_train.py.

QLoRA fine-tuning only trains the LoRA adapter weights (the 4-bit base model is
frozen) — so checkpoints save/restore just the adapter (a few tens of MB) via
PEFT's own save_pretrained/from_pretrained, not the full base model. A small
metadata.json alongside each adapter checkpoint tracks step/epoch/round so
adversarial_rounds.py (Phase 6) can resume the right checkpoint by round number
rather than by guessing directory names.
"""

import json
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger("checkpoint_utils")


def save_checkpoint(model, step: int, out_dir: str, extra_metadata: dict | None = None) -> str:
    """Saves the current PEFT adapter (not the frozen base model) plus a small
    metadata.json. `model` is expected to be a peft.PeftModel (what
    load_model_for_training in sft_train.py returns).
    """
    checkpoint_dir = Path(out_dir) / f"step_{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_dir))

    metadata = {"step": step, **(extra_metadata or {})}
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info(f"Saved adapter checkpoint to {checkpoint_dir}")
    return str(checkpoint_dir)


def load_checkpoint(base_model, checkpoint_dir: str):
    """Loads a previously-saved LoRA adapter onto `base_model` (the frozen,
    quantized base model — NOT a fresh get_peft_model call, since we're
    attaching an already-trained adapter rather than initializing a new one).
    """
    from peft import PeftModel

    metadata_path = Path(checkpoint_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    logger.info(f"Loaded adapter checkpoint from {checkpoint_dir} (metadata={metadata})")
    return model, metadata


def latest_checkpoint(checkpoint_root: str) -> str | None:
    """Finds the highest-step checkpoint under checkpoint_root, or None if empty."""
    root = Path(checkpoint_root)
    if not root.exists():
        return None
    step_dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if not step_dirs:
        return None
    return str(max(step_dirs, key=lambda d: int(d.name.removeprefix("step_"))))
