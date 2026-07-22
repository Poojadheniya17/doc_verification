"""Builds a self-contained package (code + only the data files actually
referenced by a given set of manifests) for upload as a private Kaggle
Dataset, so a Kaggle kernel can mount it at /kaggle/input/<slug> and resolve
every path training_config.yaml's kaggle.data_root expects.

Necessary because our manifests store local Windows paths (this machine's
data/ is gitignored and never leaves it) — Kaggle's Linux filesystem can't see
this machine at all, so the actual image bytes + path-rewritten manifests have
to travel together in one uploaded bundle.
"""

import json
import shutil
from pathlib import Path

from src.utils.logging_utils import get_logger

logger = get_logger("kaggle_package")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _to_package_relpath(local_path: str) -> str:
    """Local paths are normally already repo-root-relative (just sometimes
    Windows-separated, e.g. 'data\\raw\\midv2020_templates\\images\\x\\00.jpg').
    But if a caller passes an absolute path instead (e.g. Path(...).resolve()),
    make it relative to REPO_ROOT rather than silently mis-joining later —
    Path(out_root) / <absolute path> discards out_root entirely per pathlib's
    join semantics, which put a staged manifest back into the *source* repo
    instead of the staging directory the first time this was tested.
    """
    p = Path(local_path)
    if p.is_absolute():
        p = p.relative_to(REPO_ROOT)
    return str(p.as_posix())


def collect_referenced_paths(*example_lists: list[dict], path_keys: tuple[str, ...] = ("image_path", "path")) -> set[str]:
    """Pulls every real image path referenced across one or more example lists
    (from build_sft_examples / build_eval_sample / manifest "entries" lists —
    anything that's a list of dicts with an image-path-like key).
    """
    paths = set()
    for examples in example_lists:
        for entry in examples:
            for key in path_keys:
                if key in entry and entry[key]:
                    paths.add(entry[key])
    return paths


def stage_package(image_paths: set[str], manifest_paths: dict[str, str], out_dir: str,
                   code_dirs: tuple[str, ...] = ("src", "config")) -> dict:
    """Copies the given image files (preserving their relative data/ layout),
    copies each manifest with its path fields rewritten to match, and copies
    code_dirs verbatim. Returns a summary (file counts, total size).
    """
    out_root = Path(out_dir)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    for local_path in image_paths:
        rel = _to_package_relpath(local_path)
        src_path = REPO_ROOT / rel
        dest_path = out_root / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dest_path)

    for manifest_key, manifest_path in manifest_paths.items():
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        _rewrite_paths_in_place(manifest)
        rel = _to_package_relpath(manifest_path)
        dest_path = out_root / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Staged rewritten manifest: {manifest_key} -> {dest_path}")

    for code_dir in code_dirs:
        shutil.copytree(REPO_ROOT / code_dir, out_root / code_dir,
                         ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    total_bytes = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file())
    n_files = sum(1 for f in out_root.rglob("*") if f.is_file())
    logger.info(f"Staged package at {out_root}: {n_files} files, {total_bytes / 1e6:.1f} MB")
    return {"out_dir": str(out_root), "n_files": n_files, "total_mb": round(total_bytes / 1e6, 1)}


def _rewrite_paths_in_place(obj) -> None:
    """Walks a manifest's nested dict/list structure and normalizes every
    string value that looks like one of our local data/ paths to a forward-slash
    relative path — the same normalization collect_referenced_paths' consumers
    already expect, just applied inside the manifest file itself.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and ("data\\" in value or value.startswith("data/")):
                obj[key] = _to_package_relpath(value)
            else:
                _rewrite_paths_in_place(value)
    elif isinstance(obj, list):
        for item in obj:
            _rewrite_paths_in_place(item)
