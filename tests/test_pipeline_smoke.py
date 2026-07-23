"""Sanity checks: does the pipeline run end to end on a handful of samples.

Phase 1: only checks scaffolding integrity (configs parse, folder tree exists).
Phase 2: exercises degrade.py and field_tamper.py's pure logic against tiny
in-memory fixtures. Deliberately does NOT call tamper_fields/splice_document
end-to-end here — those need a real downloaded MIDV-2020 sample and EasyOCR's
model weights (~20s/image on CPU), which isn't guaranteed present in a fresh
clone or CI. That path is verified manually (see README data section) instead;
this file stays fast and dependency-free so it can run on every commit.
"""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.data_generation.acquire_dataset import _load_annotations, _map_ground_truth, build_genuine_manifest
from src.data_generation.degrade import DEGRADATION_KINDS, degrade_image
from src.data_generation.field_tamper import _mutate_digits
from src.data_generation.inpaint_forger import build_inpaint_mask, inpaint_manifest
from src.data_generation.recapture_sim import simulate_recapture
from src.data_generation.synthetic_id_gen import _random_date, _random_id_number, _random_name, generate_synthetic_id
from src.decision.cost_simulator import compute_cost, sweep_thresholds
from src.decision.financial_risk_reasoning import explain_decision
from src.decision.risk_tiering import route
from src.eval.finetuned_eval import generation_confidence_to_p_genuine
from src.eval.adversarial_rounds import build_accuracy_curve, build_eval_set, mine_failures
from src.eval.adversarial_rounds import run as run_adversarial_rounds
from src.eval.clean_eval import _extract_json, build_eval_sample
from src.eval.finetuned_eval import score_prediction
from src.eval.leave_one_out_eval import aggregate_fold_results, build_folds, load_tier_examples
from src.eval.leave_one_out_eval import run as run_leave_one_out
from src.eval.metrics import bootstrap_ci, field_exact_match, field_similarity
from src.eval.quantization_bench import cost_per_million_verifications
from src.eval.quantization_bench import run as run_quantization_bench
from src.retrieval.case_index import build_index, case_text, load_index, save_index
from src.training.checkpoint_utils import latest_checkpoint
from src.training.sft_train import DEFAULT_TIER_NAMES, _apply_tier1_field_overrides, _default_tier_manifest_paths
from src.training.sft_train import build_conversation, build_sft_examples
from src.utils.config_utils import load_yaml, resolve_paths
from src.utils.image_utils import unique_stem
from src.utils.kaggle_package import collect_referenced_paths, stage_package
from src.utils.ocr_baseline import _DATE_RE

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_config_files_parse():
    config_dir = REPO_ROOT / "config"
    for name in ("model_config.yaml", "training_config.yaml", "cost_matrix_config.yaml"):
        with open(config_dir / name, "r", encoding="utf-8") as f:
            parsed = yaml.safe_load(f)
        assert parsed, f"{name} parsed empty"


def test_expected_folders_exist():
    expected = [
        "config", "data/raw", "data/synthetic_forgeries/tier1_field_tamper",
        "data/synthetic_forgeries/tier2_splicing", "data/synthetic_forgeries/tier3_inpainting",
        "data/synthetic_forgeries/tier4_full_synthetic", "data/synthetic_forgeries/tier5_recapture",
        "data/degraded", "data/processed",
        "src/data_generation", "src/training", "src/retrieval", "src/eval",
        "src/decision", "src/monitoring", "src/utils",
        "app", "notebooks", "results/charts", "results/tables", "results/sample_outputs",
        "writeup",
    ]
    for rel in expected:
        assert (REPO_ROOT / rel).is_dir(), f"missing expected folder: {rel}"


def _make_fake_image(path: Path, size=(200, 100)):
    img = Image.new("RGB", size, color=(230, 230, 230))
    img.save(path)


@pytest.mark.parametrize("kind", DEGRADATION_KINDS)
def test_degrade_image_all_kinds(tmp_path, kind):
    src = tmp_path / "genuine.jpg"
    _make_fake_image(src)
    result = degrade_image(str(src), {"kind": kind, "seed": 0}, str(tmp_path / "out"))
    assert Path(result["degraded_image"]).is_file()
    assert result["degradation_kind"] == kind


def test_degrade_image_rejects_unknown_kind(tmp_path):
    src = tmp_path / "genuine.jpg"
    _make_fake_image(src)
    with pytest.raises(ValueError):
        degrade_image(str(src), {"kind": "not_a_real_kind"}, str(tmp_path / "out"))


def test_mutate_digits_preserves_format_and_length():
    rng = random.Random(0)
    original = "04.11.2026"
    mutated = _mutate_digits(original, rng)
    assert len(mutated) == len(original)
    # separators untouched, only digit characters may differ
    for orig_c, mut_c in zip(original, mutated):
        if not orig_c.isdigit():
            assert orig_c == mut_c
    assert mutated != original


def test_mutate_digits_noop_on_text_with_no_digits():
    rng = random.Random(0)
    assert _mutate_digits("PASSPORT", rng) == "PASSPORT"


def test_degrade_image_does_not_collide_across_document_codes(tmp_path):
    """Regression test: MIDV-2020 (and our own generated tiers) reuse plain
    numeric filenames (00.jpg) across every document code. An earlier version of
    degrade.py built output filenames from Path(image_path).stem alone, which
    silently overwrote same-named outputs from different codes — found when a
    real 175-image / 2-code run produced only 500 of the expected 875 outputs.
    """
    code_a = tmp_path / "images" / "code_a"
    code_b = tmp_path / "images" / "code_b"
    code_a.mkdir(parents=True)
    code_b.mkdir(parents=True)
    _make_fake_image(code_a / "00.jpg")
    _make_fake_image(code_b / "00.jpg")

    out_dir = tmp_path / "out"
    r1 = degrade_image(str(code_a / "00.jpg"), {"kind": "blur", "seed": 0}, str(out_dir))
    r2 = degrade_image(str(code_b / "00.jpg"), {"kind": "blur", "seed": 0}, str(out_dir))

    assert r1["degraded_image"] != r2["degraded_image"]
    assert Path(r1["degraded_image"]).is_file()
    assert Path(r2["degraded_image"]).is_file()


def test_unique_stem_disambiguates_same_filename_different_parent():
    a = unique_stem("data/raw/images/code_a/00.jpg")
    b = unique_stem("data/raw/images/code_b/00.jpg")
    assert a != b


def test_date_regex_handles_mixed_separators():
    """Regression test: EasyOCR reads some MIDV-2020 dates with a stray space
    after the first separator, e.g. '28. 09.1974 .' for '28.09.1974'. An earlier
    version of _DATE_RE required exactly one separator char between groups and
    silently failed to match this, which made the OCR baseline report the
    wrong field (date of issue) as DOB on a real sample image. Fixed by
    matching one-or-more separator chars.
    """
    assert _DATE_RE.search("28. 09.1974 .").group().strip() == "28. 09.1974"
    assert _DATE_RE.search("04.11.2026").group() == "04.11.2026"
    assert _DATE_RE.search("not a date") is None


def test_bootstrap_ci_reports_n_and_bounds():
    result = bootstrap_ci([1, 1, 1, 0, 1, 0, 1, 1, 1, 1])
    assert result["n"] == 10
    assert 0.0 <= result["ci_low"] <= result["mean"] <= result["ci_high"] <= 1.0


def test_field_similarity_and_exact_match_diverge_on_reformatted_dates():
    similarity = field_similarity("04.11.2026", "2026-11-04")
    exact = field_exact_match("04.11.2026", "2026-11-04")
    assert 0.0 < similarity < 1.0
    assert exact == 0.0


def _make_via_annotation(path: Path, filename: str, fields: dict):
    via = {
        "_via_img_metadata": {
            f"{filename}0": {
                "filename": filename,
                "regions": [
                    {"region_attributes": {"field_name": k, "value": v}} for k, v in fields.items()
                ],
            }
        }
    }
    path.write_text(json.dumps(via, ensure_ascii=False), encoding="utf-8")


def test_load_annotations_handles_non_ascii_scripts(tmp_path):
    """Regression test: MIDV-2020 spans 10 countries' scripts (Cyrillic, Greek,
    Azerbaijani, extended Latin). Windows' platform-default open() encoding
    (cp1252) raised UnicodeDecodeError on 6 of the 10 real annotation files —
    found while inspecting them by hand. _load_annotations must always open
    with encoding='utf-8' regardless of platform default.
    """
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    _make_via_annotation(
        annotations_dir / "rus_internalpassport.json", "00.jpg",
        {"name": "Александр", "surname": "Иванов", "birth_date": "01.01.1990", "number": "1234567890"},
    )
    result = _load_annotations(annotations_dir, "rus_internalpassport")
    assert result["00.jpg"]["name"] == "Александр"
    mapped = _map_ground_truth(result["00.jpg"])
    assert mapped["name"] == "Александр Иванов"
    assert mapped["dob"] == "01.01.1990"


def test_map_ground_truth_prefers_universal_number_field():
    mapped = _map_ground_truth({
        "name": "AINARS", "surname": "ALKSNIS", "birth_date": "28.09.1974.",
        "number": "LV6309038", "id_number": "280974-14045", "expiry_date": "04.11.2026.",
    })
    assert mapped["id_number"] == "LV6309038"  # 'number' is present on all 10 doc codes; 'id_number' isn't
    assert mapped["address"] is None  # no residence_line0/1 present


def test_map_ground_truth_builds_address_from_residence_lines():
    mapped = _map_ground_truth({"residence_line0": "MAIN ST 1", "residence_line1": "BELGRADE"})
    assert mapped["address"] == "MAIN ST 1 BELGRADE"


def test_extract_json_handles_markdown_fenced_output():
    raw = 'Sure, here is the result:\n```json\n{"name": "JOHN DOE", "tamper_verdict": "genuine"}\n```'
    parsed = _extract_json(raw)
    assert parsed == {"name": "JOHN DOE", "tamper_verdict": "genuine"}


def test_extract_json_returns_none_on_unparseable_output():
    assert _extract_json("I cannot determine the fields from this image.") is None


def test_build_eval_sample_balances_categories(tmp_path):
    genuine_manifest = {
        "entries": [
            {"path": f"g{i}.jpg", "split": "test", "ground_truth": {"name": f"Person {i}"}}
            for i in range(5)
        ]
    }
    genuine_path = tmp_path / "genuine.json"
    genuine_path.write_text(json.dumps(genuine_manifest), encoding="utf-8")

    tier1_manifest = {"entries": [{"forged_image": f"t1_{i}.jpg", "success": True} for i in range(5)]}
    tier1_path = tmp_path / "tier1.json"
    tier1_path.write_text(json.dumps(tier1_manifest), encoding="utf-8")

    examples = build_eval_sample(str(genuine_path), str(tier1_path), None, n_per_category=2)
    assert len(examples) == 4  # 2 genuine + 2 tier1
    labels = [e["true_label"] for e in examples]
    assert labels.count("genuine") == 2
    assert labels.count("tampered") == 2


def test_apply_tier1_field_overrides_matches_tampered_field():
    ground_truth = {"name": "AINARS ALKSNIS", "dob": "28.09.1974.", "id_number": "LV6309038",
                     "address": None, "expiry": "04.11.2026."}
    tampers = [{"bbox_xyxy": [0, 0, 1, 1], "original_text": "04.11.2026 .", "tampered_text": "14.16.0026 ."}]
    updated = _apply_tier1_field_overrides(ground_truth, tampers)
    assert updated["expiry"] == "14.16.0026 ."
    assert updated["dob"] == "28.09.1974."  # untouched field stays at genuine value


def test_apply_tier1_field_overrides_ignores_tamper_outside_schema():
    ground_truth = {"name": "AINARS ALKSNIS", "dob": "28.09.1974.", "id_number": "LV6309038",
                     "address": None, "expiry": "04.11.2026."}
    # A tamper on MRZ-like text that doesn't resemble any of the 5 schema fields
    tampers = [{"bbox_xyxy": [0, 0, 1, 1], "original_text": "P<LVAALKSNIS<<AINARS", "tampered_text": "GARBAGE"}]
    updated = _apply_tier1_field_overrides(ground_truth, tampers)
    assert updated == {**ground_truth, }


def test_build_sft_examples_from_fixture_manifests(tmp_path):
    genuine_manifest = {
        "entries": [
            {"path": "g0.jpg", "split": "train", "document_code": "x",
             "ground_truth": {"name": "A B", "dob": "01.01.2000", "id_number": "X1",
                               "address": None, "expiry": "01.01.2030"}},
            {"path": "g1.jpg", "split": "test", "document_code": "x",
             "ground_truth": {"name": "C D", "dob": "02.02.2000", "id_number": "X2",
                               "address": None, "expiry": "02.02.2030"}},
        ]
    }
    genuine_path = tmp_path / "genuine.json"
    genuine_path.write_text(json.dumps(genuine_manifest), encoding="utf-8")

    tier1_manifest = {"entries": [
        {"success": True, "source_image": "g0.jpg", "forged_image": "g0_tier1.jpg",
         "tampers": [{"bbox_xyxy": [1, 2, 3, 4], "original_text": "X1", "tampered_text": "X9"}]},
        {"success": False, "source_image": "g0.jpg"},  # failed tamper attempts must be skipped
    ]}
    tier1_path = tmp_path / "tier1.json"
    tier1_path.write_text(json.dumps(tier1_manifest), encoding="utf-8")

    tier2_manifest = {"entries": [
        {"success": True, "source_image": "g0.jpg", "forged_image": "g0_tier2.jpg", "bbox_xyxy": [5, 6, 7, 8]},
    ]}
    tier2_path = tmp_path / "tier2.json"
    tier2_path.write_text(json.dumps(tier2_manifest), encoding="utf-8")

    tier3_manifest = {"entries": [
        {"success": True, "source_image": "g0.jpg", "forged_image": "g0_tier3.jpg", "bbox_xyxy": [9, 10, 11, 12]},
    ]}
    tier3_path = tmp_path / "tier3.json"
    tier3_path.write_text(json.dumps(tier3_manifest), encoding="utf-8")

    tier4_manifest = {"entries": [
        {"success": True, "forged_image": "synth0_tier4.jpg",
         "ground_truth": {"name": "E F", "dob": "03.03.2000", "id_number": "X3",
                           "address": None, "expiry": "03.03.2030"}},
    ]}
    tier4_path = tmp_path / "tier4.json"
    tier4_path.write_text(json.dumps(tier4_manifest), encoding="utf-8")

    tier5_manifest = {"entries": [
        {"success": True, "source_image": "g0.jpg", "forged_image": "g0_tier5.jpg", "split": "train"},
    ]}
    tier5_path = tmp_path / "tier5.json"
    tier5_path.write_text(json.dumps(tier5_manifest), encoding="utf-8")

    all_tiers = {
        "tier1_field_tamper": str(tier1_path), "tier2_splicing": str(tier2_path),
        "tier3_inpainting": str(tier3_path), "tier4_full_synthetic": str(tier4_path),
        "tier5_recapture": str(tier5_path),
    }

    train_examples = build_sft_examples(str(genuine_path), all_tiers, split="train")
    # g1.jpg is split=test, so only g0's genuine + its tier1/2/3/5 derivatives land in
    # train, plus tier4 (no split of its own, included unconditionally — see docstring)
    assert len(train_examples) == 6
    tiers = {e["tier"] for e in train_examples}
    assert tiers == {"genuine", "tier1_field_tamper", "tier2_splicing", "tier3_inpainting",
                      "tier4_full_synthetic", "tier5_recapture"}

    tier1_example = next(e for e in train_examples if e["tier"] == "tier1_field_tamper")
    assert tier1_example["target"]["id_number"] == "X9"  # overridden by the tamper
    assert tier1_example["target"]["tamper_verdict"] == "tampered"
    assert tier1_example["target"]["tamper_regions"] == [[1, 2, 3, 4]]

    tier3_example = next(e for e in train_examples if e["tier"] == "tier3_inpainting")
    assert tier3_example["target"]["id_number"] == "X1"  # carried over from source, unchanged
    assert tier3_example["target"]["tamper_regions"] == [[9, 10, 11, 12]]

    tier4_example = next(e for e in train_examples if e["tier"] == "tier4_full_synthetic")
    assert tier4_example["target"]["id_number"] == "X3"  # from the manifest's own ground_truth
    assert tier4_example["target"]["tamper_regions"] == []

    tier5_example = next(e for e in train_examples if e["tier"] == "tier5_recapture")
    assert tier5_example["target"]["id_number"] == "X1"  # carried over from source, unchanged
    assert tier5_example["target"]["tamper_regions"] == []

    # Only tier1_field_tamper + tier2_splicing passed -> only genuine + those two land
    two_tier_examples = build_sft_examples(
        str(genuine_path), {"tier1_field_tamper": str(tier1_path), "tier2_splicing": str(tier2_path)}, split="train")
    assert len(two_tier_examples) == 3
    assert {e["tier"] for e in two_tier_examples} == {"genuine", "tier1_field_tamper", "tier2_splicing"}

    test_examples = build_sft_examples(str(genuine_path), all_tiers, split="test")
    # g1.jpg (split=test) has no tier1/2/3/5 derivatives in these fixtures (all keyed off
    # g0), but tier4 has no split of its own and is included unconditionally regardless of
    # the split argument (see build_sft_examples' docstring) — real callers only ever pass
    # split="train" today (train_sft.train() default), so this isn't a live contamination
    # risk, just documented, honest behavior of a known data-generation gap.
    assert len(test_examples) == 2
    assert {e["tier"] for e in test_examples} == {"genuine", "tier4_full_synthetic"}


def test_default_tier_manifest_paths_match_real_files_on_disk():
    """Regression test for a real bug: _default_tier_manifest_paths() derived
    each tier's manifest FILENAME from its full descriptive name (producing
    e.g. "tier1_field_tamper_manifest.json"), but every tier's manifest is
    actually named with just the short numeric prefix ("tier1_manifest.json"
    — see field_tamper.py/splice.py/etc's own manifest-writing code).
    build_sft_examples() silently skips any tier path that doesn't exist (by
    design, for not-yet-generated tiers), so this bug produced ZERO tier
    examples with no error at all — invisible locally, only surfacing when a
    real Kaggle leave-one-out/adversarial-rounds run logged "0 folds" / "0
    tampered examples" with no exception. Exercises the REAL repo data (not
    fixtures) so a wrong filename convention actually fails this assertion.
    """
    training_config = load_yaml("config/training_config.yaml")
    training_config["environment"] = "local"
    paths = resolve_paths(training_config)
    tier_paths = _default_tier_manifest_paths(paths, DEFAULT_TIER_NAMES)
    assert tier_paths  # DEFAULT_TIER_NAMES is non-empty
    for tier, path in tier_paths.items():
        assert Path(path).is_file(), f"{tier}'s derived manifest path does not exist on disk: {path}"


def test_build_conversation_structure():
    conv = build_conversation("img.jpg", {"name": "A"})
    assert conv[0]["role"] == "user"
    assert conv[0]["content"][0] == {"type": "image", "image": "img.jpg"}
    assert conv[1]["role"] == "assistant"
    assert json.loads(conv[1]["content"][0]["text"]) == {"name": "A"}


def test_latest_checkpoint_picks_highest_step(tmp_path):
    assert latest_checkpoint(str(tmp_path / "does_not_exist")) is None
    (tmp_path / "step_10").mkdir()
    (tmp_path / "step_200").mkdir()
    (tmp_path / "step_30").mkdir()
    assert latest_checkpoint(str(tmp_path)).endswith("step_200")


def test_build_inpaint_mask_fails_gracefully_with_no_face(tmp_path):
    src = tmp_path / "blank.jpg"
    _make_fake_image(src, size=(400, 300))  # solid color, no detectable face
    result = build_inpaint_mask(str(src))
    assert result["success"] is False
    assert result["mask"] is None


def test_inpaint_manifest_builds_real_manifest_with_mocked_diffusion(tmp_path, monkeypatch):
    """inpaint_region's actual diffusion call needs a GPU (see module docstring) —
    monkeypatched here so the manifest-building logic itself (real, no mocking of
    that part) gets exercised locally, same split as sft_train.py's local/Kaggle
    divide.
    """
    genuine_path = tmp_path / "genuine_manifest.json"
    genuine_path.write_text(json.dumps({"entries": [
        {"path": "img1.jpg", "split": "train"},
        {"path": "img2.jpg", "split": "train"},
        {"path": "img3.jpg", "split": "test"},
    ]}), encoding="utf-8")

    def fake_inpaint_region(image_path, out_dir, prompt=None, model_name=None, seed=None):
        return {"source_image": image_path, "forged_image": f"{out_dir}/{image_path}_tier3.jpg",
                "tier": "tier3_inpainting", "success": True, "bbox_xyxy": [0, 0, 10, 10]}

    monkeypatch.setattr("src.data_generation.inpaint_forger.inpaint_region", fake_inpaint_region)

    results = inpaint_manifest(str(genuine_path), str(tmp_path), limit=2)
    assert len(results) == 2
    assert all(r["success"] and r["split"] == "train" for r in results)

    manifest = json.loads((tmp_path / "tier3_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_attempted"] == 2
    assert manifest["num_success"] == 2
    assert len(manifest["entries"]) == 2


def test_score_prediction_with_true_label_shape():
    example = {"image_path": "img.jpg", "tier": "tier2_splicing", "true_label": "tampered"}
    correct_prediction = {"parsed": {"tamper_verdict": "tampered"}, "parse_success": True}
    wrong_prediction = {"parsed": {"tamper_verdict": "genuine"}, "parse_success": True}
    unparseable_prediction = {"parsed": None, "parse_success": False}

    assert score_prediction(example, correct_prediction)["correct"] is True
    assert score_prediction(example, wrong_prediction)["correct"] is False
    result = score_prediction(example, unparseable_prediction)
    assert result["correct"] is False
    assert result["predicted_verdict"] is None
    assert "target" not in result


def test_score_prediction_with_target_shape_passes_target_through():
    target = {"name": "A", "tamper_verdict": "genuine", "tamper_regions": []}
    example = {"image_path": "img.jpg", "tier": "genuine", "target": target}
    prediction = {"parsed": {"tamper_verdict": "genuine"}, "parse_success": True}

    result = score_prediction(example, prediction)
    assert result["correct"] is True
    assert result["target"] == target  # passed through unchanged for later retraining use


def test_random_generators_produce_expected_formats():
    rng = random.Random(0)
    first, last = _random_name(rng)
    assert first.isalpha() and last.isalpha()
    date = _random_date(rng, 2000, 2010)
    day, month, year = date.split(".")
    assert 1 <= int(day) <= 28 and 1 <= int(month) <= 12 and 2000 <= int(year) <= 2010
    id_number = _random_id_number(rng)
    assert len(id_number) == 9
    assert id_number[:2].isalpha() and id_number[2:].isdigit()


def test_generate_synthetic_id_fails_gracefully_with_no_donor_face(tmp_path):
    donor = tmp_path / "blank_donor.jpg"
    _make_fake_image(donor, size=(400, 300))
    result = generate_synthetic_id(str(donor), str(tmp_path / "out"), seed=1)
    assert result["success"] is False


def test_simulate_recapture_preserves_shape_and_is_deterministic(tmp_path):
    src = tmp_path / "genuine.jpg"
    _make_fake_image(src, size=(300, 200))
    r1 = simulate_recapture(str(src), str(tmp_path / "out1"), seed=5)
    r2 = simulate_recapture(str(src), str(tmp_path / "out2"), seed=5)
    import numpy as np
    from PIL import Image as PILImage
    img1 = PILImage.open(r1["forged_image"])
    img2 = PILImage.open(r2["forged_image"])
    assert img1.size == (300, 200)
    assert np.array_equal(np.array(img1), np.array(img2))  # same seed -> identical output


def test_collect_referenced_paths_dedupes_across_lists():
    sft = [{"image_path": "data/a.jpg"}, {"image_path": "data/b.jpg"}]
    eval_ = [{"image_path": "data/b.jpg"}, {"path": "data/c.jpg"}]
    paths = collect_referenced_paths(sft, eval_)
    assert paths == {"data/a.jpg", "data/b.jpg", "data/c.jpg"}


def test_stage_package_copies_files_and_rewrites_manifest_paths(tmp_path):
    # Minimal fake repo layout so stage_package's REPO_ROOT-relative copies work
    import src.utils.kaggle_package as kp
    fake_repo = tmp_path / "fake_repo"
    (fake_repo / "data" / "raw").mkdir(parents=True)
    (fake_repo / "data" / "raw" / "img0.jpg").write_bytes(b"fake-image-bytes")
    (fake_repo / "src").mkdir()
    (fake_repo / "src" / "dummy.py").write_text("# placeholder", encoding="utf-8")
    (fake_repo / "config").mkdir()
    (fake_repo / "config" / "dummy.yaml").write_text("k: v", encoding="utf-8")

    manifest_path = fake_repo / "manifest.json"
    manifest_path.write_text(json.dumps({"entries": [{"path": "data\\raw\\img0.jpg", "split": "train"}]}),
                              encoding="utf-8")

    original_root = kp.REPO_ROOT
    kp.REPO_ROOT = fake_repo
    try:
        summary = stage_package(
            image_paths={"data\\raw\\img0.jpg"},
            manifest_paths={"m": str(manifest_path)},
            out_dir=str(tmp_path / "staged"),
        )
    finally:
        kp.REPO_ROOT = original_root

    staged = Path(summary["out_dir"])
    assert (staged / "data" / "raw" / "img0.jpg").is_file()
    assert (staged / "src" / "dummy.py").is_file()
    assert (staged / "config" / "dummy.yaml").is_file()

    rewritten = json.loads((staged / "manifest.json").read_text(encoding="utf-8"))
    assert rewritten["entries"][0]["path"] == "data/raw/img0.jpg"  # backslashes normalized


def test_build_genuine_manifest_stratifies_by_document_code(tmp_path):
    extract_dir = tmp_path / "midv2020_templates"
    for code, n in [("code_a", 10), ("code_b", 6)]:
        code_dir = extract_dir / "images" / code
        code_dir.mkdir(parents=True)
        for i in range(n):
            _make_fake_image(code_dir / f"{i:02d}.jpg")

    out_path = tmp_path / "genuine_manifest.json"
    manifest = build_genuine_manifest(extract_dir, out_path, split_ratios=(0.7, 0.15, 0.15), seed=1)

    assert manifest["num_images"] == 16
    assert manifest["num_document_codes"] == 2
    assert out_path.is_file()
    for code in ("code_a", "code_b"):
        splits = {e["split"] for e in manifest["entries"] if e["document_code"] == code}
        assert "train" in splits  # every code must appear in train given these ratios


# ---------------------------------------------------------------------------
# Phase 6: leave_one_out_eval.py
# ---------------------------------------------------------------------------

def test_load_tier_examples_filters_successes(tmp_path):
    manifest = {"entries": [
        {"success": True, "forged_image": "a.jpg"},
        {"success": False, "forged_image": "b.jpg"},
        {"success": True, "forged_image": "c.jpg"},
    ]}
    path = tmp_path / "tier1.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    examples = load_tier_examples(str(path), "tier1_field_tamper")
    assert [e["image_path"] for e in examples] == ["a.jpg", "c.jpg"]
    assert all(e["tier"] == "tier1_field_tamper" and e["true_label"] == "tampered" for e in examples)


def test_build_folds_creates_one_fold_per_tier_with_others_as_train():
    tier_examples = {
        "tier1": [{"image_path": "a.jpg"}],
        "tier2": [{"image_path": "b.jpg"}, {"image_path": "c.jpg"}],
        "tier4": [{"image_path": "d.jpg"}],
    }
    folds = build_folds(tier_examples)
    assert {f["held_out_tier"] for f in folds} == {"tier1", "tier2", "tier4"}

    tier2_fold = next(f for f in folds if f["held_out_tier"] == "tier2")
    assert set(tier2_fold["train_tiers"]) == {"tier1", "tier4"}
    assert len(tier2_fold["held_out_examples"]) == 2


def test_build_folds_skips_tier_with_no_others_to_train_on():
    # Only one tier present -> nothing to train on that excludes it -> no fold
    folds = build_folds({"tier1": [{"image_path": "a.jpg"}]})
    assert folds == []


def test_aggregate_fold_results_reports_per_tier_and_overall():
    per_fold_scores = {"tier1": [1.0, 1.0, 0.0], "tier2": [0.0, 0.0]}
    results = aggregate_fold_results(per_fold_scores, n_resamples=100)
    assert results["per_tier"]["tier1"]["n"] == 3
    assert results["per_tier"]["tier2"]["n"] == 2
    assert results["overall"]["n"] == 5  # pooled across both tiers
    assert results["per_tier"]["tier1"]["mean"] > results["per_tier"]["tier2"]["mean"]


def _write_leave_one_out_config(tmp_path) -> Path:
    config = {
        "leave_one_out": {
            "tiers": ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting",
                      "tier4_full_synthetic", "tier5_recapture"],
            "bootstrap_resamples": 100,
            "confidence_level": 0.95,
        },
    }
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


def _write_tier_manifest(tmp_path, name: str, n_success: int) -> Path:
    manifest = {"entries": [{"success": True, "forged_image": f"{name}_{i}.jpg"} for i in range(n_success)]}
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_run_leave_one_out_stops_without_callables(tmp_path):
    config_path = _write_leave_one_out_config(tmp_path)
    tier1_path = _write_tier_manifest(tmp_path, "tier1_field_tamper", 3)
    tier2_path = _write_tier_manifest(tmp_path, "tier2_splicing", 2)

    result = run_leave_one_out(
        str(config_path),
        {"tier1_field_tamper": str(tier1_path), "tier2_splicing": str(tier2_path)},
    )
    assert result["results"] is None
    assert len(result["folds"]) == 2  # logged the 3 missing configured tiers, still built folds for what's present


def test_run_leave_one_out_orchestrates_with_fake_callables(tmp_path):
    config_path = _write_leave_one_out_config(tmp_path)
    tier1_path = _write_tier_manifest(tmp_path, "tier1_field_tamper", 4)
    tier2_path = _write_tier_manifest(tmp_path, "tier2_splicing", 4)
    tier4_path = _write_tier_manifest(tmp_path, "tier4_full_synthetic", 4)
    out_path = tmp_path / "loo_results.json"

    trained_on = []

    def fake_train_fn(train_tiers):
        trained_on.append(sorted(train_tiers))
        return "fake-model"

    def fake_eval_fn(model_handle, held_out_examples):
        assert model_handle == "fake-model"
        return [1.0] * len(held_out_examples)  # perfect "generalization" for this fake model

    result = run_leave_one_out(
        str(config_path),
        {"tier1_field_tamper": str(tier1_path), "tier2_splicing": str(tier2_path),
         "tier4_full_synthetic": str(tier4_path)},
        train_fn=fake_train_fn, eval_fn=fake_eval_fn, out_path=str(out_path),
    )

    assert len(trained_on) == 3  # one training call per fold
    assert result["results"]["overall"]["mean"] == 1.0
    for tier_result in result["results"]["per_tier"].values():
        assert tier_result["mean"] == 1.0
    assert out_path.is_file()


# ---------------------------------------------------------------------------
# Phase 6: adversarial_rounds.py
# ---------------------------------------------------------------------------

def test_build_eval_set_caps_genuine_but_not_tampered(tmp_path):
    genuine_manifest = {"entries": [
        {"path": f"g{i}.jpg", "split": "test", "document_code": "x",
         "ground_truth": {"name": f"N{i}", "dob": "01.01.2000", "id_number": f"X{i}",
                           "address": None, "expiry": "01.01.2030"}}
        for i in range(5)
    ]}
    genuine_path = tmp_path / "genuine.json"
    genuine_path.write_text(json.dumps(genuine_manifest), encoding="utf-8")

    tier2_manifest = {"entries": [
        {"success": True, "source_image": "g0.jpg", "forged_image": "g0_tier2.jpg", "bbox_xyxy": [1, 2, 3, 4]},
        {"success": True, "source_image": "g1.jpg", "forged_image": "g1_tier2.jpg", "bbox_xyxy": [5, 6, 7, 8]},
    ]}
    tier2_path = tmp_path / "tier2.json"
    tier2_path.write_text(json.dumps(tier2_manifest), encoding="utf-8")

    eval_set = build_eval_set(str(genuine_path), {"tier2_splicing": str(tier2_path)}, n_genuine=2, split="test")
    tiers = [e["tier"] for e in eval_set]
    assert tiers.count("genuine") == 2  # capped
    assert tiers.count("tier2_splicing") == 2  # both tampered examples kept, uncapped
    assert all(e["target"]["tamper_verdict"] in ("genuine", "tampered") for e in eval_set)


def test_mine_failures_caps_and_filters_incorrect():
    eval_results = [
        {"image_path": "a.jpg", "correct": True},
        {"image_path": "b.jpg", "correct": False},
        {"image_path": "c.jpg", "correct": False},
        {"image_path": "d.jpg", "correct": False},
    ]
    failures = mine_failures(eval_results, cap=2)
    assert [f["image_path"] for f in failures] == ["b.jpg", "c.jpg"]  # first 2 failures, order preserved


def test_build_accuracy_curve_tags_rounds_in_order():
    curve = build_accuracy_curve([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], n_resamples=100)
    assert [point["round"] for point in curve] == [0, 1, 2]
    assert [point["mean"] for point in curve] == [0.0, 0.5, 1.0]


def _write_adversarial_config(tmp_path, num_rounds=3, cap=200) -> Path:
    config = {"adversarial_rounds": {"num_rounds": num_rounds, "failure_sample_cap": cap, "confidence_level": 0.95}}
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


def test_run_adversarial_rounds_stops_without_callables(tmp_path):
    config_path = _write_adversarial_config(tmp_path)
    result = run_adversarial_rounds(str(config_path), eval_set=[{"image_path": "a.jpg"}])
    assert result["rounds"] is None


def test_run_adversarial_rounds_orchestrates_and_improves(tmp_path):
    config_path = _write_adversarial_config(tmp_path, num_rounds=3, cap=5)
    eval_set = [{"image_path": f"{i}.jpg"} for i in range(10)]
    out_path = tmp_path / "adv_results.json"

    # Fake model that "improves" each round: round r gets the first r*3 examples right.
    def fake_train_fn(failures, round_num):
        return round_num

    def fake_eval_fn(round_num, eval_set):
        return [{"image_path": ex["image_path"], "correct": i < round_num * 3}
                for i, ex in enumerate(eval_set)]

    result = run_adversarial_rounds(str(config_path), eval_set, train_fn=fake_train_fn, eval_fn=fake_eval_fn,
                                     out_path=str(out_path))

    assert len(result["rounds"]) == 3
    accuracies = [r["accuracy"]["mean"] for r in result["rounds"]]
    assert accuracies == sorted(accuracies)  # monotonically improving, matching the fake model
    assert len(result["accuracy_curve"]) == 3
    assert out_path.is_file()
    # Round 0 (round_num=0 -> 0 correct) mines min(cap, 10) = 5 failures
    assert result["rounds"][0]["num_failures_mined"] == 5


# ---------------------------------------------------------------------------
# Phase 9: quantization_bench.py
# ---------------------------------------------------------------------------

def _write_quantization_config(tmp_path, precisions=("fp16", "int8", "int4")) -> Path:
    config = {"quantization_bench": {"precisions": list(precisions), "eval_sample_size": 200}}
    path = tmp_path / "training_config.yaml"
    path.write_text(yaml.dump(config), encoding="utf-8")
    return path


def test_cost_per_million_verifications_scales_with_latency():
    cheap = cost_per_million_verifications(0.1, gpu_cost_per_hour=3.6)
    expensive = cost_per_million_verifications(0.2, gpu_cost_per_hour=3.6)
    assert expensive == pytest.approx(cheap * 2)
    assert cheap == pytest.approx(100)  # 0.1s * (3.6/3600 $/s) * 1e6


def test_run_quantization_bench_stops_without_callables(tmp_path):
    training_config_path = _write_quantization_config(tmp_path)
    model_config_path = tmp_path / "model_config.yaml"
    model_config_path.write_text(yaml.dump({"model": {"name": "x"}}), encoding="utf-8")

    result = run_quantization_bench(str(model_config_path), str(training_config_path), eval_examples=[{}])
    assert result["results"] is None
    assert result["precisions"] == ["fp16", "int8", "int4"]


def test_run_quantization_bench_orchestrates_with_fake_callables(tmp_path):
    training_config_path = _write_quantization_config(tmp_path, precisions=("fp16", "int4"))
    model_config_path = tmp_path / "model_config.yaml"
    model_config_path.write_text(yaml.dump({"model": {"name": "x"}}), encoding="utf-8")
    out_path = tmp_path / "quant_results.json"

    # Fake model: int4 is faster but less accurate than fp16 (the realistic tradeoff).
    latency_by_precision = {"fp16": 0.5, "int4": 0.1}
    accuracy_by_precision = {"fp16": 1.0, "int4": 0.5}

    def fake_load_fn(precision, model_config):
        return precision

    def fake_eval_fn(precision, eval_examples):
        return [{"correct": i < len(eval_examples) * accuracy_by_precision[precision],
                  "latency_seconds": latency_by_precision[precision]} for i in range(len(eval_examples))]

    result = run_quantization_bench(
        str(model_config_path), str(training_config_path), eval_examples=[{}] * 10,
        load_fn=fake_load_fn, eval_fn=fake_eval_fn, out_path=str(out_path),
    )

    assert result["results"]["fp16"]["avg_latency_seconds"] == 0.5
    assert result["results"]["int4"]["avg_latency_seconds"] == 0.1
    assert result["results"]["fp16"]["accuracy"]["mean"] == 1.0
    assert result["results"]["int4"]["accuracy"]["mean"] == 0.5
    # int4's estimated cost should be lower, tracking its lower latency
    assert (result["results"]["int4"]["estimated_cost_per_million_verifications_usd"] <
            result["results"]["fp16"]["estimated_cost_per_million_verifications_usd"])
    assert out_path.is_file()


# ---------------------------------------------------------------------------
# Phase 7: risk_tiering.py / cost_simulator.py
# ---------------------------------------------------------------------------

_COST_CONFIG = {
    "costs": {
        "false_accept": {"value": 500},
        "false_reject": {"value": 50},
        "manual_review": {"value": 5},
    },
    "thresholds": {"auto_approve_min_confidence": 0.9, "auto_reject_max_confidence": 0.1},
    "threshold_sweep": {
        "auto_approve_candidates": [0.95, 0.9, 0.8],
        "auto_reject_candidates": [0.2, 0.1, 0.05],
    },
}


def test_route_boundaries():
    assert route(0.95, _COST_CONFIG) == "auto_approve"
    assert route(0.9, _COST_CONFIG) == "auto_approve"  # boundary is inclusive
    assert route(0.05, _COST_CONFIG) == "auto_reject"
    assert route(0.1, _COST_CONFIG) == "auto_reject"  # boundary is inclusive
    assert route(0.5, _COST_CONFIG) == "human_review"


def test_route_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        route(1.5, _COST_CONFIG)


def test_compute_cost_counts_false_accept_false_reject_and_review():
    eval_results = [
        {"confidence": 0.95, "true_label": "tampered"},   # auto_approve, WRONG -> false accept ($500)
        {"confidence": 0.05, "true_label": "genuine"},     # auto_reject, WRONG -> false reject ($50)
        {"confidence": 0.5, "true_label": "genuine"},      # human_review ($5)
        {"confidence": 0.95, "true_label": "genuine"},     # auto_approve, correct -> $0
    ]
    result = compute_cost(eval_results, _COST_CONFIG["thresholds"], _COST_CONFIG)
    assert result["counts"]["false_accept"] == 1
    assert result["counts"]["false_reject"] == 1
    assert result["counts"]["human_review"] == 1
    assert result["total_cost"] == 500 + 50 + 5
    assert result["avg_cost_per_doc"] == pytest.approx((500 + 50 + 5) / 4)


def test_sweep_thresholds_finds_best_and_skips_invalid_pairs():
    eval_results = [{"confidence": c, "true_label": label} for c, label in [
        (0.99, "genuine"), (0.02, "tampered"), (0.5, "genuine"), (0.5, "tampered"),
    ]]
    swept = sweep_thresholds(eval_results, _COST_CONFIG)
    # 3x3 grid, minus pairs where reject >= approve (none here since candidates are disjoint ranges)
    assert len(swept["curve"]) == 9
    assert swept["best"]["avg_cost_per_doc"] == min(c["avg_cost_per_doc"] for c in swept["curve"])
    for combo in swept["curve"]:
        assert combo["thresholds"]["auto_reject_max_confidence"] < combo["thresholds"]["auto_approve_min_confidence"]


def test_generation_confidence_to_p_genuine():
    assert generation_confidence_to_p_genuine(0.9, "genuine") == 0.9
    assert generation_confidence_to_p_genuine(0.9, "tampered") == pytest.approx(0.1)
    assert generation_confidence_to_p_genuine(0.7, None) == 0.5  # unparseable verdict -> maximally uncertain


def test_explain_decision_auto_approve():
    document_result = {
        "image_path": "doc.jpg",
        "parsed": {"tamper_verdict": "genuine", "explanation": "All fields consistent, no visual tampering."},
        "confidence": 0.97,
    }
    text = explain_decision(document_result, _COST_CONFIG)
    assert "Decision: auto approve" in text
    assert "0.97" in text
    assert "All fields consistent" in text
    assert "500" in text  # cites the false-accept dollar rationale for this tier


def test_explain_decision_human_review_with_similar_cases():
    document_result = {
        "image_path": "doc.jpg",
        "parsed": {"tamper_verdict": "tampered", "explanation": "Expiry date font looks mismatched."},
        "confidence": 0.5,
    }
    similar_cases = [
        {"case_id": "case_12", "tamper_verdict": "tampered", "similarity": 0.88},
        {"case_id": "case_07", "tamper_verdict": "genuine", "similarity": 0.61},
    ]
    text = explain_decision(document_result, _COST_CONFIG, similar_cases=similar_cases)
    assert "Decision: human review" in text
    assert "case_12" in text and "case_07" in text
    assert "5" in text  # cites the manual-review dollar rationale for this tier


def test_explain_decision_rejects_out_of_range_confidence():
    with pytest.raises(ValueError):
        explain_decision({"parsed": {}, "confidence": 1.2}, _COST_CONFIG)


# ---------------------------------------------------------------------------
# Phase 6/8: case_index.py (pure logic + save/load roundtrip only — build_index
# and query need a real sentence-transformers model download, excluded from
# this fast suite per the same reasoning as field_tamper.py's EasyOCR path;
# verified manually instead, see results/tables/phase6_7_groundwork_summary.md)
# ---------------------------------------------------------------------------

def test_case_text_combines_available_fields():
    case = {"document_code": "lva_passport", "tamper_verdict": "tampered", "tier": "tier1_field_tamper",
            "explanation": "DOB field font mismatch"}
    text = case_text(case)
    assert "lva_passport" in text
    assert "tampered" in text
    assert "tier1_field_tamper" in text
    assert "DOB field font mismatch" in text


def test_case_text_falls_back_to_case_id_when_no_other_fields():
    assert case_text({"case_id": "case-42"}) == "case-42"


def test_save_and_load_index_roundtrip(tmp_path):
    # Fabricated embeddings (no real model call) — keeps this test fast; the
    # embedding step itself is verified manually (module docstring).
    index = {
        "model_name": "all-MiniLM-L6-v2",
        "cases": [{"case_id": "c0", "explanation": "font mismatch"}, {"case_id": "c1", "explanation": "splice"}],
        "embeddings": np.array([[1.0, 0.0], [0.0, 1.0]], dtype="float32"),
    }
    out_path = tmp_path / "index"
    save_index(index, str(out_path))
    assert out_path.with_suffix(".npz").is_file()
    assert out_path.with_suffix(".json").is_file()

    loaded = load_index(str(out_path))
    assert loaded["model_name"] == "all-MiniLM-L6-v2"
    assert loaded["cases"] == index["cases"]
    assert np.allclose(loaded["embeddings"], index["embeddings"])
