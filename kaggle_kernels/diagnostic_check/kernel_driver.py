"""Cheap, no-GPU, no-pip-install diagnostic: does the mounted dataset actually
have all 1000 raw MIDV-2020 images and all 5 tier manifests? Written to get a
direct, unambiguous answer from Kaggle's own runtime after two Kaggle jobs
disagreed with a local pre-push verification (local diff showed the staging
directory byte-identical to the full local repo before the push completed,
but a real Kaggle run still couldn't find a specific file afterward) — rather
than guessing whether this is a real re-sync failure or a mount/propagation
lag, checking directly costs seconds, not hours of GPU time.
"""

from pathlib import Path

candidates = [
    Path("/kaggle/input/doc-verification-data"),
    Path("/kaggle/input/datasets/poojadheniya/doc-verification-data"),
]
INPUT_ROOT = next((c for c in candidates if (c / "src").is_dir()), None)
if INPUT_ROOT is None:
    raise RuntimeError(f"Could not find the mounted dataset's src/ dir under any of {candidates}")
print(f"=== Resolved INPUT_ROOT = {INPUT_ROOT} ===", flush=True)

DOC_CODES = ["alb_id", "aze_passport", "esp_id", "est_id", "fin_id", "grc_passport",
             "lva_passport", "rus_internalpassport", "srb_passport", "svk_id"]

print("=== Raw image counts per document code ===", flush=True)
total = 0
for code in DOC_CODES:
    d = INPUT_ROOT / "data" / "raw" / "midv2020_templates" / "images" / code
    n = len(list(d.glob("*.jpg"))) if d.is_dir() else -1
    total += max(n, 0)
    print(f"{code}: {n}", flush=True)
print(f"=== Total raw images: {total} (expect 1000) ===", flush=True)

specific_check = INPUT_ROOT / "data" / "raw" / "midv2020_templates" / "images" / "alb_id" / "11.jpg"
print(f"=== alb_id/11.jpg exists: {specific_check.is_file()} ===", flush=True)

print("=== Tier manifest files ===", flush=True)
ALL_TIERS = ["tier1_field_tamper", "tier2_splicing", "tier3_inpainting", "tier4_full_synthetic", "tier5_recapture"]
for tier in ALL_TIERS:
    tier_dir = INPUT_ROOT / "data" / "synthetic_forgeries" / tier
    short = tier.split("_")[0]
    correct_path = tier_dir / f"{short}_manifest.json"
    wrong_path = tier_dir / f"{tier}_manifest.json"
    print(f"{tier}: dir_exists={tier_dir.is_dir()}, "
          f"correct_name({correct_path.name})_exists={correct_path.is_file()}, "
          f"full_name({wrong_path.name})_exists={wrong_path.is_file()}", flush=True)

checkpoint_path = INPUT_ROOT / "checkpoints" / "sft_v24_final" / "adapter_model.safetensors"
print(f"=== checkpoint exists: {checkpoint_path.is_file()} ===", flush=True)

print("=== Manifest path-separator check (backslash = broken on Linux) ===", flush=True)
import json  # noqa: E402

manifest_files = [
    INPUT_ROOT / "data" / "processed" / "genuine_manifest_templates.json",
] + [
    INPUT_ROOT / "data" / "synthetic_forgeries" / tier / f"{tier.split('_')[0]}_manifest.json"
    for tier in ALL_TIERS
]
path_keys = ("path", "source_image", "forged_image", "donor_face_image", "donor_image")
total_backslash = 0
for mf in manifest_files:
    if not mf.is_file():
        print(f"{mf}: MISSING", flush=True)
        continue
    d = json.loads(mf.read_text(encoding="utf-8"))
    bad = sum(1 for e in d.get("entries", []) for k in path_keys if e.get(k) and "\\" in e[k])
    total_backslash += bad
    print(f"{mf.name}: {len(d.get('entries', []))} entries, backslash paths={bad}", flush=True)
print(f"=== Total backslash path occurrences across all manifests: {total_backslash} (expect 0) ===", flush=True)

print("Done.", flush=True)
