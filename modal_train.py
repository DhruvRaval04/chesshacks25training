"""
Modal entrypoint to run `train.py` on a managed GPU worker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import modal

APP_NAME = "chesshacks25-train"
ARTIFACT_VOLUME_NAME = "chesshacks25-artifacts"
ARTIFACT_MOUNT_PATH = Path("/artifacts")
PROJECT_ROOT = Path("/root/project")


def build_image() -> modal.Image:
    """
    Base image with Stockfish and project requirements pre-installed.
    """

    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("stockfish")
        .pip_install_from_requirements("requirements.txt")
    )


app = modal.App(APP_NAME)
image = build_image()
artifacts = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME, create_if_missing=True
)


@app.function(
    image=image,
    gpu=modal.gpu.A10G(),
    timeout=60 * 60 * 12,
    volumes={str(ARTIFACT_MOUNT_PATH): artifacts},
)
def train_remote(
    total_steps: int = 100_000,
    rollout_length: int = 512,
    mini_batch_size: int = 256,
    update_epochs: int = 4,
    learning_rate: float = 3e-4,
    agent_color_schedule: str = "random",
    checkpoint_subdir: str = "checkpoints",
    metrics_filename: str = "metrics.csv",
    hf_save_subdir: Optional[str] = "hf_exports",
    hf_repo_id: Optional[str] = None,
    hf_push_to_hub: bool = False,
    hf_private: bool = False,
    hf_commit_message: str = "Add chess policy checkpoint",
    hf_token_env_var: Optional[str] = None,
):
    """
    Launch the PPO training loop on Modal.

    Use `modal run modal_train.py::train_remote --total-steps 200000 ...`
    to customize hyperparameters.
    """
    import os

    os.chdir(PROJECT_ROOT)
    from train import PPOConfig, run_training  # pylint: disable=import-error

    artifact_root = ARTIFACT_MOUNT_PATH
    checkpoint_dir = artifact_root / checkpoint_subdir
    metrics_path = artifact_root / metrics_filename
    hf_save_dir = (
        artifact_root / hf_save_subdir if hf_save_subdir is not None else None
    )

    cfg = PPOConfig(
        total_steps=total_steps,
        rollout_length=rollout_length,
        mini_batch_size=mini_batch_size,
        update_epochs=update_epochs,
        learning_rate=learning_rate,
        checkpoint_dir=checkpoint_dir,
        metrics_path=metrics_path,
        agent_color_schedule=agent_color_schedule,
        hf_save_dir=hf_save_dir,
        hf_repo_id=hf_repo_id,
        hf_push_to_hub=hf_push_to_hub,
        hf_private=hf_private,
        hf_commit_message=hf_commit_message,
    )

    if hf_token_env_var:
        cfg.hf_token = os.environ.get(hf_token_env_var) or cfg.hf_token

    run_training(cfg)

