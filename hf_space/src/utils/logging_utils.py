"""Experiment tracking wrapper (W&B) + standard logging setup.

Wraps wandb so training/eval scripts can no-op gracefully when WANDB_API_KEY
isn't set (e.g. local CPU smoke runs) rather than crashing — real runs on
Kaggle set the key via .env / Kaggle secrets.
"""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


class ExperimentTracker:
    """Thin wandb wrapper. Disabled (log-to-console-only) if no API key is present,
    so local smoke runs never require W&B credentials.
    """

    def __init__(self, project: str, run_name: str, config: dict, enabled: bool | None = None):
        self.logger = get_logger(run_name)
        self.enabled = enabled if enabled is not None else bool(os.environ.get("WANDB_API_KEY"))
        self._run = None
        if self.enabled:
            import wandb

            self._run = wandb.init(project=project, name=run_name, config=config)
        else:
            self.logger.info("WANDB_API_KEY not set — tracking disabled, logging to console only")

    def log(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            self._run.log(metrics, step=step)
        else:
            self.logger.info(f"step={step} {metrics}")

    def finish(self) -> None:
        if self.enabled:
            self._run.finish()
