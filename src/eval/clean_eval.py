"""Baseline accuracy on the clean (non-degraded) eval set — genuine + Tier 1/2
forgeries, no blur/rotation/glare/etc. applied (that split-by-image-quality axis
is degraded_eval.py's job; this script's axis is genuine-vs-forged and
extraction accuracy). Zero-shot Qwen2.5-VL, no fine-tuning — establishes the
"before" number that SFT (Phase 4) and DPO (Phase 8) are measured against.

Model size note: the reported Phase 3 baseline is Qwen2.5-VL-7B-Instruct,
hardcoded explicitly in kaggle_kernels/phase3_4_sft_baseline/kernel_driver.py's
Phase 3 call — it does NOT read model_config.yaml. This is now a deliberate
mismatch, not an inconsistency to fix: model_config.yaml's fine-tuning target
was moved to Qwen2.5-VL-3B-Instruct after 7B SFT training hit an unresolved
T4/library-stack hang across three targeted fixes (see model_config.yaml's
DECISION comment and results/tables/phase4_sft_summary.md for the full
chain), while the 7B zero-shot baseline itself succeeded and is being kept
as-is. Comparing the two is therefore an imperfect, model-size-confounded
comparison — flag that explicitly anywhere the numbers are compared, don't
present it as a clean before/after.

`--model-name` defaults to the much lighter Qwen2.5-VL-3B-Instruct so a local
run wouldn't need the full 7B — but in practice even 3B could not be loaded
on the local dev machine (~7.7GB total RAM; loading 3B in fp32 caused severe
disk thrashing, not just slowness — see the RAM-constraint note in
README.md / training_config.yaml). run() and run_single() are therefore
validated locally only via unit tests with mocked outputs
(tests/test_pipeline_smoke.py: _extract_json, build_eval_sample) — actually
executing this script against a real model (3B or 7B, any device) happens on
Kaggle. Every result this script writes is labeled with the exact model_name and
device used, so a smoke result can never be confused with a real reported number.
"""

import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.eval.metrics import bootstrap_ci, field_exact_match, field_similarity
from src.utils.logging_utils import get_logger

logger = get_logger("clean_eval")

EXTRACTION_FIELDS = ["name", "dob", "id_number", "address", "expiry"]

PROMPT = """You are an identity document verification assistant. Look at this \
identity document image and respond with ONLY a single JSON object (no other \
text, no markdown code fences) with exactly these keys:

- "name": the document holder's full name, or null if not visible
- "dob": date of birth as printed, or null if not visible
- "id_number": the document/passport/ID number as printed, or null if not visible
- "address": the printed address, or null if this document type does not show one
- "expiry": the expiry date as printed, or null if not visible
- "tamper_verdict": exactly "genuine" or "tampered" — "tampered" if any field looks \
edited, mismatched in font, or visually inconsistent with the rest of the document
- "explanation": one sentence explaining your tamper_verdict

Respond with the JSON object only."""

_model = None
_processor = None
_loaded_model_name = None


def _get_model(model_name: str, device: str = "cpu", cache_dir: str | None = None,
               load_in_4bit: bool | None = None):
    global _model, _processor, _loaded_model_name
    if _model is None or _loaded_model_name != model_name:
        logger.info(f"Loading {model_name} on {device} (first call downloads/loads weights, can take a while)")
        # bf16 even on CPU, not fp32: found by hitting this in practice — a 3B
        # model in fp32 is ~12-14GB of weights, which doesn't fit this dev
        # machine's 7.7GB total RAM (0.5-0.8GB free at idle) and caused severe
        # disk thrashing rather than just "slow" inference. bf16 halves that to
        # ~6-7GB. Slower per-op on CPUs without native bf16 kernels, but that's
        # still far faster than swapping to disk.
        dtype = torch.bfloat16
        _processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)

        # Default to 4-bit on CUDA: Qwen2.5-VL-7B in bf16 is ~14GB, which leaves
        # almost no headroom on a T4's 16GB VRAM once activations/KV-cache/vision
        # tokens are added (real OOM risk we'd rather not discover mid-run on
        # borrowed GPU quota). This also keeps the zero-shot baseline's
        # quantization level identical to the QLoRA fine-tuned model it's being
        # compared against, so the comparison isolates the fine-tuning effect
        # rather than conflating it with a precision change.
        use_4bit = load_in_4bit if load_in_4bit is not None else (device == "cuda")
        if use_4bit:
            from transformers import BitsAndBytesConfig
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
            )
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, quantization_config=quant_config, device_map=device, cache_dir=cache_dir,
            )
        else:
            _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=dtype, device_map=device, cache_dir=cache_dir,
            )
        _loaded_model_name = model_name
    return _model, _processor


def _extract_json(raw_text: str) -> dict | None:
    """Zero-shot models often wrap JSON in markdown fences or add preamble text —
    pull out the first {...} block rather than requiring the whole response to
    be valid JSON on its own.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def run_single(image_path: str, model_name: str, device: str = "cpu",
                max_new_tokens: int = 300, max_image_size: int = 640,
                cache_dir: str | None = None, load_in_4bit: bool | None = None) -> dict:
    model, processor = _get_model(model_name, device, cache_dir=cache_dir, load_in_4bit=load_in_4bit)

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_image_size, max_image_size))  # CPU inference: keep vision-token count small

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    # processor() always builds tensors on CPU — device_map="cuda" puts the model
    # (and, for 4-bit, the quantized weights) on GPU, so inputs must follow or
    # generate() raises a device-mismatch error. Never hit locally since CPU-only
    # runs have model and inputs on the same device by coincidence.
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    parsed = _extract_json(raw_text)
    return {"image_path": image_path, "raw_output": raw_text, "parsed": parsed, "parse_success": parsed is not None}


def build_eval_sample(genuine_manifest_path: str, tier1_manifest_path: str | None,
                       tier2_manifest_path: str | None, n_per_category: int = 3) -> list[dict]:
    """Small, fixed-composition sample: n genuine + n Tier-1 + n Tier-2, so the
    tamper-verdict accuracy isn't computed on an all-genuine or all-forged set
    by chance. This is a smoke-scale sample (see class docstring) — real
    reported numbers need a much larger, randomly-drawn one.
    """
    examples = []
    genuine = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    genuine_test = [e for e in genuine["entries"] if e["split"] == "test" and e.get("ground_truth")]
    for e in genuine_test[:n_per_category]:
        examples.append({"image_path": e["path"], "true_label": "genuine", "ground_truth": e["ground_truth"]})

    for manifest_path, tier in [(tier1_manifest_path, "tier1"), (tier2_manifest_path, "tier2")]:
        if not manifest_path or not Path(manifest_path).exists():
            continue
        forged = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        successes = [e for e in forged["entries"] if e.get("success")]
        for e in successes[:n_per_category]:
            examples.append({"image_path": e["forged_image"], "true_label": "tampered", "ground_truth": None,
                              "tier": tier})
    return examples


def run(genuine_manifest_path: str, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        tier1_manifest_path: str | None = None, tier2_manifest_path: str | None = None,
        n_per_category: int = 3, device: str = "cpu", out_path: str | None = None,
        cache_dir: str | None = None, load_in_4bit: bool | None = None) -> dict:
    examples = build_eval_sample(genuine_manifest_path, tier1_manifest_path, tier2_manifest_path, n_per_category)
    resolved_4bit = load_in_4bit if load_in_4bit is not None else (device == "cuda")
    logger.info(f"Evaluating {model_name} zero-shot on {len(examples)} images "
                f"({n_per_category} per category, 4bit={resolved_4bit}) — "
                f"this is a smoke-scale sample, not the full eval set")

    per_field_similarity: dict[str, list[float]] = {f: [] for f in EXTRACTION_FIELDS}
    per_field_exact: dict[str, list[float]] = {f: [] for f in EXTRACTION_FIELDS}
    tamper_correct: list[float] = []
    parse_successes: list[float] = []
    per_example_results = []

    for i, example in enumerate(examples):
        logger.info(f"[{i+1}/{len(examples)}] {example['image_path']} (true_label={example['true_label']})")
        result = run_single(example["image_path"], model_name, device=device, cache_dir=cache_dir,
                             load_in_4bit=load_in_4bit)
        parse_successes.append(float(result["parse_success"]))

        predicted_verdict = (result["parsed"] or {}).get("tamper_verdict") if result["parse_success"] else None
        is_correct = float(predicted_verdict == example["true_label"]) if predicted_verdict else 0.0
        tamper_correct.append(is_correct)

        if example["ground_truth"] and result["parse_success"]:
            for field in EXTRACTION_FIELDS:
                gt_value = example["ground_truth"].get(field)
                if gt_value is None:
                    continue  # field not present on this document type — not scored
                pred_value = result["parsed"].get(field)
                per_field_similarity[field].append(field_similarity(pred_value, gt_value))
                per_field_exact[field].append(field_exact_match(pred_value, gt_value))

        per_example_results.append({**example, **result, "tamper_correct": bool(is_correct)})

    summary = {
        "model_name": model_name,
        "device": device,
        "load_in_4bit": resolved_4bit,
        "n_examples": len(examples),
        "n_per_category": n_per_category,
        "parse_success_rate": bootstrap_ci(parse_successes),
        "tamper_verdict_accuracy": bootstrap_ci(tamper_correct),
        "field_similarity": {f: bootstrap_ci(v) for f, v in per_field_similarity.items() if v},
        "field_exact_match": {f: bootstrap_ci(v) for f, v in per_field_exact.items() if v},
        "examples": per_example_results,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote results to {out_path}")

    return summary


def run_ocr_baseline(genuine_manifest_path: str, n_examples: int = 10,
                      out_path: str | None = None) -> dict:
    """Classical-OCR comparison point (checklist item alongside the zero-shot
    VLM): scores src/utils/ocr_baseline.py's rule-based field extraction against
    the same kind of genuine test-split ground truth clean_eval.py uses.

    Genuine images only, no tamper_verdict — the OCR baseline has no tamper
    detection story at all (it doesn't reason about the document, it just reads
    text), so comparing it on that axis wouldn't mean anything.
    """
    from src.utils.ocr_baseline import extract_fields

    genuine = json.loads(Path(genuine_manifest_path).read_text(encoding="utf-8"))
    genuine_test = [e for e in genuine["entries"] if e["split"] == "test" and e.get("ground_truth")]
    examples = genuine_test[:n_examples]
    logger.info(f"Evaluating OCR baseline on {len(examples)} genuine images")

    per_field_similarity: dict[str, list[float]] = {f: [] for f in EXTRACTION_FIELDS}
    per_field_exact: dict[str, list[float]] = {f: [] for f in EXTRACTION_FIELDS}
    per_example_results = []

    for i, entry in enumerate(examples):
        logger.info(f"[{i+1}/{len(examples)}] {entry['path']}")
        predicted = extract_fields(entry["path"])
        for field in EXTRACTION_FIELDS:
            gt_value = entry["ground_truth"].get(field)
            if gt_value is None:
                continue
            pred_value = predicted.get(field)
            per_field_similarity[field].append(field_similarity(pred_value, gt_value))
            per_field_exact[field].append(field_exact_match(pred_value, gt_value))
        per_example_results.append({"image_path": entry["path"], "predicted": predicted,
                                     "ground_truth": entry["ground_truth"]})

    summary = {
        "engine": "easyocr",
        "n_examples": len(examples),
        "field_similarity": {f: bootstrap_ci(v) for f, v in per_field_similarity.items() if v},
        "field_exact_match": {f: bootstrap_ci(v) for f, v in per_field_exact.items() if v},
        "examples": per_example_results,
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        logger.info(f"Wrote results to {out_path}")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["vlm", "ocr", "both"], default="both")
    parser.add_argument("--genuine-manifest", default="data/processed/genuine_manifest_templates.json")
    parser.add_argument("--tier1-manifest", default="data/synthetic_forgeries/tier1_field_tamper/tier1_manifest.json")
    parser.add_argument("--tier2-manifest", default="data/synthetic_forgeries/tier2_splicing/tier2_manifest.json")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--n-per-category", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cache-dir", default=None, help="Local HF snapshot dir, e.g. ./.hf_cache, to avoid re-downloading")
    parser.add_argument("--4bit", dest="load_in_4bit", action="store_true", default=None,
                         help="Force 4-bit loading regardless of device (default: on for cuda, off for cpu)")
    parser.add_argument("--no-4bit", dest="load_in_4bit", action="store_false",
                         help="Force full-precision loading even on cuda")
    parser.add_argument("--out", default="results/tables/clean_eval_baseline.json")
    parser.add_argument("--ocr-out", default="results/tables/ocr_baseline.json")
    parser.add_argument("--ocr-n-examples", type=int, default=10)
    args = parser.parse_args()

    if args.mode in ("vlm", "both"):
        run(
            genuine_manifest_path=args.genuine_manifest,
            model_name=args.model_name,
            tier1_manifest_path=args.tier1_manifest,
            tier2_manifest_path=args.tier2_manifest,
            n_per_category=args.n_per_category,
            device=args.device,
            out_path=args.out,
            cache_dir=args.cache_dir,
            load_in_4bit=args.load_in_4bit,
        )
    if args.mode in ("ocr", "both"):
        run_ocr_baseline(
            genuine_manifest_path=args.genuine_manifest,
            n_examples=args.ocr_n_examples,
            out_path=args.ocr_out,
        )


if __name__ == "__main__":
    main()
