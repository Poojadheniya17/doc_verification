"""Sanity checks: does the pipeline run end to end on a handful of samples.

Phase 1: only checks scaffolding integrity (configs parse, folder tree exists).
Phase 2: exercises degrade.py and field_tamper.py's pure logic against tiny
in-memory fixtures. Deliberately does NOT call tamper_fields/splice_document
end-to-end here — those need a real downloaded MIDV-2020 sample and EasyOCR's
model weights (~20s/image on CPU), which isn't guaranteed present in a fresh
clone or CI. That path is verified manually (see README data section) instead;
this file stays fast and dependency-free so it can run on every commit.
"""

import random
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.data_generation.acquire_dataset import build_genuine_manifest
from src.data_generation.degrade import DEGRADATION_KINDS, degrade_image
from src.data_generation.field_tamper import _mutate_digits
from src.utils.image_utils import unique_stem

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
