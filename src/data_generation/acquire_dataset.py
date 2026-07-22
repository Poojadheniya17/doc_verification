"""Acquires the MIDV-2020 public identity document dataset.

Not in the original file-tree sketch (added during Phase 2) — the checklist called
for "public dataset acquisition + preprocessing" without naming a file; this is that
script.

Source: ftp://smartengines.com/midv-2020/ (official; also mirrored at
http://l3i-share.univ-lr.fr per the paper, https://arxiv.org/abs/2107.00396).

Default archive is `templates` (863MB) rather than scan_upright/photo/clips
(1.1GB-53GB): templates.tar holds the clean, front-facing, ideal-quality image for
each of the 1000 mock documents (10 doc types x 100 samples). We generate our own
forgeries and degradations from these clean images (that's the whole point of
Tiers 1-5 and degrade.py) rather than needing MIDV-2020's own captured-condition
variants (photo/scan/clips) — so the smallest archive is also the right one.

`--max-bytes` supports constrained/local environments (this project's CPU-only
local dev tier): it downloads a byte-range prefix of the tar and extracts whatever
complete members land inside that prefix, rather than the whole archive. This is
a real (not synthetic placeholder) subset of MIDV-2020, just partial coverage —
logged as such in the manifest. Kaggle / a full local run should omit --max-bytes.

Note: templates.tar's tar member order is all /images/<code>/*.jpg for every one
of the 10 document codes, THEN all /annotations/<code>.json files at the very end.
So a partial download below ~863MB will contain images but no ground-truth field
annotations. field_tamper.py is therefore written to locate text fields via OCR
(pytesseract/easyocr) rather than depend on the ground-truth JSON — which also
happens to mirror how a real inference-time pipeline works (no oracle annotations
at inference either).
"""

import argparse
import hashlib
import json
import random
import subprocess
import tarfile
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger("acquire_dataset")

FTP_BASE = "ftp://smartengines.com/midv-2020"

ARCHIVES = {
    "templates": {"file": "templates.tar", "size_bytes": 863_293_440},
    "scan_upright": {"file": "scan_upright.tar", "size_bytes": 1_149_143_040},
    "scan_rotated": {"file": "scan_rotated.tar", "size_bytes": 1_122_682_880},
    "photo": {"file": "photo.tar", "size_bytes": 4_002_467_840},
    "clips": {"file": "clips.tar", "size_bytes": 10_303_651_840},
}


def _download_full(url: str, dest: Path, retries: int = 3) -> None:
    """Full download with resume (curl -C -), retried on transient failures."""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["curl", "-sS", "-C", "-", "-o", str(dest), url],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        logger.info(f"download attempt {attempt}/{retries} failed: {result.stderr.strip()}")
    raise RuntimeError(f"Failed to download {url} after {retries} attempts")


def _download_prefix(url: str, dest: Path, max_bytes: int, retries: int = 3) -> None:
    """Single-shot byte-range download of the first `max_bytes` of `url`.

    Not resumable across retries (a partial range read that fails partway
    restarts from 0) — acceptable because max_bytes is meant to stay small
    enough for a single attempt to complete in a few minutes.
    """
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            ["curl", "-sS", "-r", f"0-{max_bytes - 1}", "-o", str(dest), url],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            return
        logger.info(f"prefix download attempt {attempt}/{retries} failed: {result.stderr.strip()}")
    raise RuntimeError(f"Failed to download prefix of {url} after {retries} attempts")


def _extract_best_effort(tar_path: Path, out_dir: Path) -> list[str]:
    """Streams a (possibly truncated) tar and extracts every member it can read
    cleanly, stopping without raising at the first corrupt/incomplete member.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with tarfile.open(tar_path, mode="r|") as tar:
        while True:
            try:
                member = tar.next()
            except Exception as e:
                logger.info(f"stopped extraction at truncation/error: {e}")
                break
            if member is None:
                break
            try:
                tar.extract(member, path=out_dir, filter="data")
                if member.isfile():
                    extracted.append(member.name)
            except Exception as e:
                logger.info(f"stopped extraction at member '{member.name}': {e}")
                break
    return extracted


def acquire(
    archive: str,
    out_root: Path,
    max_bytes: int | None = None,
    verify_checksum: bool = False,
) -> dict:
    if archive not in ARCHIVES:
        raise ValueError(f"Unknown archive '{archive}', expected one of {list(ARCHIVES)}")

    spec = ARCHIVES[archive]
    url = f"{FTP_BASE}/dataset/{spec['file']}"
    out_root.mkdir(parents=True, exist_ok=True)
    tar_path = out_root / spec["file"]
    extract_dir = out_root / f"midv2020_{archive}"

    is_partial = max_bytes is not None and max_bytes < spec["size_bytes"]
    if is_partial:
        logger.info(f"Downloading first {max_bytes:,} of {spec['size_bytes']:,} bytes from {url}")
        _download_prefix(url, tar_path, max_bytes)
    else:
        logger.info(f"Downloading full archive ({spec['size_bytes']:,} bytes) from {url}")
        _download_full(url, tar_path)

    if verify_checksum and not is_partial:
        md5 = hashlib.md5(tar_path.read_bytes()).hexdigest()
        logger.info(f"md5: {md5} (compare against ftp://smartengines.com/midv-2020/md5.txt manually)")

    extracted_members = _extract_best_effort(tar_path, extract_dir)
    codes_seen = sorted({Path(m).parts[1] for m in extracted_members if Path(m).parts[0] == "images"})

    manifest = {
        "archive": archive,
        "source_url": url,
        "partial": is_partial,
        "bytes_downloaded": tar_path.stat().st_size,
        "archive_full_size_bytes": spec["size_bytes"],
        "num_files_extracted": len(extracted_members),
        "document_codes_covered": codes_seen,
        "extract_dir": str(extract_dir),
    }
    manifest_path = out_root / f"midv2020_{archive}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote manifest to {manifest_path}: {manifest['num_files_extracted']} files, "
                f"codes={codes_seen}, partial={is_partial}")
    return manifest


def _load_annotations(annotations_dir: Path, code: str) -> dict[str, dict[str, str]]:
    """Loads MIDV-2020's VIA-format annotation file for one document code and
    returns {filename: {field_name: value}}.

    Must open with explicit encoding="utf-8" — this dataset spans 10 countries'
    scripts (Cyrillic for rus_internalpassport, Greek for grc_passport, Azerbaijani
    for aze_passport, etc.), and Windows' platform-default encoding (cp1252) fails
    to decode 6 of the 10 annotation files outright. Found by hitting
    UnicodeDecodeError while inspecting these files with a bare open() call.
    """
    annotation_path = annotations_dir / f"{code}.json"
    if not annotation_path.exists():
        return {}
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    by_filename = {}
    for entry in data.get("_via_img_metadata", {}).values():
        fields = {r["region_attributes"]["field_name"]: r["region_attributes"]["value"]
                  for r in entry["regions"] if r["region_attributes"].get("value")}
        by_filename[entry["filename"]] = fields
    return by_filename


def _map_ground_truth(fields: dict[str, str]) -> dict[str, str | None]:
    """Maps MIDV-2020's per-document-type field_name schema (which varies: e.g.
    'number' is the passport/ID number on every one of the 10 codes, but only
    some codes also have a separate 'id_number' personal code, and 'residence_line0/1'
    (address) only appears on srb_passport — the rest have no address field at
    all, which is realistic: most ID documents in this dataset simply don't
    print one) onto this project's target extraction schema.
    """
    name_parts = [fields.get(p) for p in ("name", "surname") if fields.get(p)]
    address_parts = [fields.get(f"residence_line{i}") for i in (0, 1) if fields.get(f"residence_line{i}")]
    return {
        "name": " ".join(name_parts) if name_parts else None,
        "dob": fields.get("birth_date"),
        "id_number": fields.get("number"),
        "address": " ".join(address_parts) if address_parts else None,
        "expiry": fields.get("expiry_date"),
    }


def build_genuine_manifest(
    extract_dir: Path,
    out_path: Path,
    split_ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> dict:
    """Assigns each genuine base image to train/val/test, stratified per document
    code so every split sees every doc type in roughly the same proportion —
    important with only 2-10 codes present, where a plain random split could
    starve val/test of an entire document type by chance. Also attaches
    ground-truth extraction fields from MIDV-2020's annotations/ when present
    (see _load_annotations) — needed for real extraction-accuracy scoring in
    clean_eval.py, not just tamper-detection accuracy.
    """
    images_dir = extract_dir / "images"
    annotations_dir = extract_dir / "annotations"
    by_code: dict[str, list[str]] = {}
    for code_dir in sorted(images_dir.iterdir()) if images_dir.exists() else []:
        if code_dir.is_dir():
            by_code[code_dir.name] = sorted(str(p) for p in code_dir.glob("*.jpg"))

    rng = random.Random(seed)
    entries = []
    for code, paths in by_code.items():
        annotations = _load_annotations(annotations_dir, code)
        shuffled = paths[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * split_ratios[0])
        n_val = int(n * split_ratios[1])
        for i, path in enumerate(shuffled):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            raw_fields = annotations.get(Path(path).name, {})
            entries.append({
                "path": path,
                "document_code": code,
                "split": split,
                "ground_truth": _map_ground_truth(raw_fields) if raw_fields else None,
            })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_extract_dir": str(extract_dir),
        "split_ratios": {"train": split_ratios[0], "val": split_ratios[1], "test": split_ratios[2]},
        "seed": seed,
        "num_images": len(entries),
        "num_document_codes": len(by_code),
        "entries": entries,
    }
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info(f"Wrote genuine-image manifest to {out_path}: {len(entries)} images "
                f"across {len(by_code)} document codes")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="templates", choices=list(ARCHIVES))
    parser.add_argument("--out-dir", default="data/raw")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--max-bytes", type=int, default=None,
                         help="Cap download to first N bytes (local/constrained dev). Omit for a full run.")
    parser.add_argument("--verify-checksum", action="store_true")
    parser.add_argument("--skip-download", action="store_true",
                         help="Rebuild the processed manifest from an already-extracted dir without re-downloading.")
    args = parser.parse_args()

    extract_dir = Path(args.out_dir) / f"midv2020_{args.archive}"
    if not args.skip_download:
        acquire(
            archive=args.archive,
            out_root=Path(args.out_dir),
            max_bytes=args.max_bytes,
            verify_checksum=args.verify_checksum,
        )

    build_genuine_manifest(
        extract_dir=extract_dir,
        out_path=Path(args.processed_dir) / f"genuine_manifest_{args.archive}.json",
    )


if __name__ == "__main__":
    # Run as `python -m src.data_generation.acquire_dataset` from the repo root
    # so the `src.*` absolute imports above resolve correctly.
    main()
