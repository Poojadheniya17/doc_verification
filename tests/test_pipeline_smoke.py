"""Sanity checks: does the pipeline run end to end on a handful of samples.

Phase 1: only checks scaffolding integrity (configs parse, folder tree exists).
Grows in each later phase to smoke-test that phase's script on a tiny sample —
this file is the running record that "config-driven, not notebook-driven" holds.
"""

from pathlib import Path

import yaml

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
