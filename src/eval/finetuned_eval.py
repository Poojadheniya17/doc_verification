"""Runs the actual fine-tuned (QLoRA) checkpoint for real inference — distinct
from clean_eval.py, which is specifically the zero-shot 7B baseline (different
model size, different prompt schema with no tamper_regions field). Shared by
every phase that needs "load the trained model and score it against examples":
leave_one_out_eval.py's per-fold eval_fn, adversarial_rounds.py's per-round
eval_fn, and quantization_bench.py's fp16/int8/int4 comparisons.

Same RAM-constraint split as sft_train.py/clean_eval.py: model loading and
generation need a GPU and run on Kaggle only. What's pure logic and testable
locally here: matching a parsed prediction's tamper_verdict against an
example's known-correct answer (see eval_examples' docstring for the two
shapes it accepts).
"""

from PIL import Image

from src.eval.clean_eval import _extract_json
from src.training.sft_train import SFT_PROMPT
from src.utils.logging_utils import get_logger

logger = get_logger("finetuned_eval")


def load_finetuned_model(model_config: dict, adapter_path: str, device: str = "cuda"):
    """Loads the 4-bit base model (per model_config.yaml, same quantization as
    training) and attaches a trained LoRA adapter for inference. Not exercised
    locally — see module docstring.
    """
    import torch
    from peft import PeftModel
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
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_cfg["name"], quantization_config=bnb_config, device_map={"": 0} if device == "cuda" else device,
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info(f"Loaded fine-tuned model: base={model_cfg['name']}, adapter={adapter_path}")
    return model, processor


def load_finetuned_model_at_precision(model_config: dict, adapter_path: str, precision: str, device: str = "cuda"):
    """Loads the base model at a specific precision plus the trained LoRA
    adapter — quantization_bench.py's load_fn, distinct from
    load_finetuned_model() (which always uses model_config.yaml's fixed
    training-time 4-bit quantization) since varying precision is the whole
    point of this benchmark.

    precision: "fp16" (no quantization, plain fp16 weights), "int8", or
    "int4" (bitsandbytes quantization, nf4 for int4 matching training's own
    quant_type). Not exercised locally — see module docstring.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    model_cfg = model_config["model"]
    processor = AutoProcessor.from_pretrained(model_cfg["name"], trust_remote_code=model_cfg["trust_remote_code"])
    device_map = {"": 0} if device == "cuda" else device

    if precision == "fp16":
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["name"], torch_dtype=torch.float16, device_map=device_map,
            trust_remote_code=model_cfg["trust_remote_code"],
        )
    elif precision == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["name"], quantization_config=bnb_config, device_map=device_map,
            trust_remote_code=model_cfg["trust_remote_code"],
        )
    elif precision == "int4":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["name"], quantization_config=bnb_config, device_map=device_map,
            trust_remote_code=model_cfg["trust_remote_code"],
        )
    else:
        raise ValueError(f"Unknown precision: {precision!r} (expected fp16/int8/int4)")

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    logger.info(f"Loaded fine-tuned model at precision={precision}: base={model_cfg['name']}, adapter={adapter_path}")
    return model, processor


def run_single(image_path: str, model, processor, max_image_size: int = 256,
                max_new_tokens: int = 300) -> dict:
    """Mirrors clean_eval.run_single()'s shape but uses SFT_PROMPT (the exact
    schema — including tamper_regions — this checkpoint was fine-tuned on),
    not clean_eval.PROMPT (the zero-shot-baseline-only schema without
    tamper_regions). Using the training-time prompt matters: evaluating a
    fine-tuned model against a prompt it never saw during training would be
    an unfair, uninformative comparison.
    """
    import torch
    from qwen_vl_utils import process_vision_info

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((max_image_size, max_image_size))

    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": SFT_PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                  output_scores=True, return_dict_in_generate=True)
    generated = outputs.sequences
    trimmed = [out[len(inp):] for inp, out in zip(inputs["input_ids"], generated)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    parsed = _extract_json(raw_text)

    # Average per-token max-softmax-probability across the generated sequence,
    # as a rough "how confident was the model in this whole response" signal.
    # zip() truncates to the shorter of the two if generate() ever produces a
    # scores/tokens length mismatch (a known edge case with early stopping) —
    # a safe fallback that slightly under-samples rather than crashing.
    token_probs = [
        torch.softmax(step_scores[0], dim=-1)[token_id].item()
        for step_scores, token_id in zip(outputs.scores, trimmed[0])
    ]
    generation_confidence = sum(token_probs) / len(token_probs) if token_probs else 0.5

    tamper_verdict = parsed.get("tamper_verdict") if parsed else None
    p_genuine = generation_confidence_to_p_genuine(generation_confidence, tamper_verdict)

    return {"image_path": image_path, "raw_output": raw_text, "parsed": parsed,
            "parse_success": parsed is not None, "generation_confidence": generation_confidence,
            "confidence": p_genuine}


def generation_confidence_to_p_genuine(generation_confidence: float, tamper_verdict: str | None) -> float:
    """Maps the model's own confidence in whatever it just generated (see
    run_single()'s output_scores handling above) into P(genuine) — the single
    number src/decision/risk_tiering.py's route() and cost_simulator.py's
    threshold sweep operate on.

    A documented simplification, not a claim of true field-level calibration:
    this measures confidence in the WHOLE generated response (verdict +
    extracted fields + explanation together), not the tamper_verdict token
    span in isolation — isolating that span would need locating its exact
    position in the output, which is real extra work for a proxy signal this
    project's decision layer only needs to be directionally reasonable, not
    perfectly calibrated. High confidence + "genuine" -> high P(genuine);
    high confidence + "tampered" -> low P(genuine) (the model is confident
    the document IS fake); a missing/unparseable verdict maps to 0.5
    (maximally uncertain — the decision layer should route this to human
    review, not silently guess a direction).
    """
    if tamper_verdict == "genuine":
        return generation_confidence
    if tamper_verdict == "tampered":
        return 1.0 - generation_confidence
    return 0.5


def score_prediction(example: dict, prediction: dict) -> dict:
    """Pure logic, unit-tested without a model: compares a run_single() result
    against an example's known-correct tamper_verdict.

    Accepts either shape an example might carry the correct answer in:
    - {"image_path", "tier", "true_label"} — leave_one_out_eval.py's held-out
      examples (all "tampered" by construction: the held-out tier IS a
      forgery tier).
    - {"image_path", "tier", "target": {...full schema, "tamper_verdict": ...}}
      — adversarial_rounds.py's eval_set, built the same shape
      build_sft_examples() produces so the true target survives unchanged
      into a mined failure for the next round's retraining (see
      adversarial_rounds Kaggle driver — this is what makes retraining on a
      real, correct target possible instead of guessing one).
    """
    true_verdict = example.get("true_label")
    if true_verdict is None:
        true_verdict = (example.get("target") or {}).get("tamper_verdict")

    predicted_verdict = (prediction["parsed"] or {}).get("tamper_verdict") if prediction["parse_success"] else None
    correct = predicted_verdict == true_verdict

    result = {"image_path": example["image_path"], "tier": example.get("tier"),
              "correct": correct, "predicted_verdict": predicted_verdict,
              "parse_success": prediction["parse_success"],
              # Full parsed output + confidence pass through too — LOO/adversarial-rounds
              # only ever consumed "correct"/"predicted_verdict" for accuracy tracking, but
              # a richer capture (e.g. for the demo app's example gallery) needs the full
              # extraction fields, tamper_regions, explanation, and confidence, not just the
              # binary verdict comparison. Cheap to include, nothing downstream breaks by
              # ignoring the extra keys.
              "parsed": prediction.get("parsed"), "confidence": prediction.get("confidence")}
    if "target" in example:
        result["target"] = example["target"]
    return result


def eval_examples(model, processor, examples: list[dict], max_image_size: int = 256,
                   max_new_tokens: int = 300) -> list[dict]:
    """Runs the model on every example and scores each via score_prediction().
    Not exercised locally (calls run_single(), which needs a GPU) — the pure
    scoring logic (score_prediction) is what's unit-tested.
    """
    results = []
    for i, example in enumerate(examples):
        logger.info(f"[{i + 1}/{len(examples)}] {example['image_path']}")
        prediction = run_single(example["image_path"], model, processor,
                                 max_image_size=max_image_size, max_new_tokens=max_new_tokens)
        results.append(score_prediction(example, prediction))
    return results


def eval_examples_with_latency(model, processor, examples: list[dict], max_image_size: int = 256,
                                max_new_tokens: int = 300) -> list[dict]:
    """Same as eval_examples() but times each example's inference call — this
    is quantization_bench.py's eval_fn, which needs per-example
    "latency_seconds" alongside "correct" to compare fp16/int8/int4. Kept
    separate from eval_examples() rather than always timing, since
    leave-one-out/adversarial-rounds have no use for latency and timing adds
    a small amount of overhead (torch.cuda synchronization) not worth paying
    when it's not needed.
    """
    import time

    import torch

    results = []
    for i, example in enumerate(examples):
        logger.info(f"[{i + 1}/{len(examples)}] {example['image_path']}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = run_single(example["image_path"], model, processor,
                                 max_image_size=max_image_size, max_new_tokens=max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        latency = time.perf_counter() - start
        scored = score_prediction(example, prediction)
        scored["latency_seconds"] = latency
        results.append(scored)
    return results
