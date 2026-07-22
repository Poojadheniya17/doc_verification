"""Shared config loading + environment (local/Kaggle) path resolution.

Not in the original file tree sketch — added during Phase 1 scaffolding because
every training/eval script needs to resolve `environment: local|kaggle` from
training_config.yaml into concrete paths, and duplicating that logic per-script
would violate the "config-driven, not script-driven" requirement.
"""

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_paths(training_config: dict[str, Any]) -> dict[str, Path]:
    """Resolve data/output/checkpoint roots for the active environment.

    `environment` must be "local" or "kaggle" (set in training_config.yaml, or
    override by mutating the loaded dict before calling this — CLI flags in
    each script should do `config["environment"] = args.environment`).
    """
    env = training_config["environment"]
    if env not in ("local", "kaggle"):
        raise ValueError(f"Unknown environment '{env}', expected 'local' or 'kaggle'")

    raw_paths = training_config["paths"][env]
    return {key: Path(value) for key, value in raw_paths.items()}
