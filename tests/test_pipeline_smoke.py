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

import pytest
import yaml
from PIL import Image

from src.data_generation.acquire_dataset import _load_annotations, _map_ground_truth, build_genuine_manifest
from src.data_generation.degrade import DEGRADATION_KINDS, degrade_image
from src.data_generation.field_tamper import _mutate_digits
from src.eval.clean_eval import _extract_json, build_eval_sample
from src.eval.metrics import bootstrap_ci, field_exact_match, field_similarity
from src.training.checkpoint_utils import latest_checkpoint
from src.training.sft_train import _apply_tier1_field_overrides, build_conversation, build_sft_examples
from src.utils.image_utils import unique_stem
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

    train_examples = build_sft_examples(str(genuine_path), str(tier1_path), str(tier2_path), split="train")
    # g1.jpg is split=test, so only g0's genuine + its tier1 + tier2 derivatives land in train
    assert len(train_examples) == 3
    tiers = {e["tier"] for e in train_examples}
    assert tiers == {"genuine", "tier1_field_tamper", "tier2_splicing"}

    tier1_example = next(e for e in train_examples if e["tier"] == "tier1_field_tamper")
    assert tier1_example["target"]["id_number"] == "X9"  # overridden by the tamper
    assert tier1_example["target"]["tamper_verdict"] == "tampered"
    assert tier1_example["target"]["tamper_regions"] == [[1, 2, 3, 4]]

    test_examples = build_sft_examples(str(genuine_path), str(tier1_path), str(tier2_path), split="test")
    assert len(test_examples) == 1
    assert test_examples[0]["tier"] == "genuine"


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
