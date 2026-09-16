# utils/logger.py

import os
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, cfg):
        self.use_wandb  = cfg.logging.use_wandb
        self.log_dir    = cfg.logging.log_dir
        self.run        = None
        self.writer     = None

        os.makedirs(self.log_dir, exist_ok=True)

        # ── TensorBoard ────────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=self.log_dir)
        print(f"TensorBoard logs → {self.log_dir}")

        # ── Weights & Biases ───────────────────────────────────────────
        if self.use_wandb:
            try:
                import wandb
                self.run = wandb.init(
                    project = cfg.logging.get("wandb_project", "av-physics-mae"),
                    entity  = cfg.logging.get("wandb_entity",  None),
                    config  = dict(cfg),
                    dir     = self.log_dir,
                )
                print(f"W&B run → {self.run.url}")
            except ImportError:
                print("wandb not installed — skipping W&B logging.")
                self.use_wandb = False

    # ------------------------------------------------------------------
    def log(self, metrics: dict, step: int):
        """
        Log a dictionary of scalar metrics at a given step.

        Args:
            metrics : e.g. {"loss": 0.42, "loss_video": 0.21, "loss_audio": 0.21}
            step    : epoch or global step number
        """
        # TensorBoard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, global_step=step)

        # W&B
        if self.use_wandb and self.run is not None:
            self.run.log(metrics, step=step)

    # ------------------------------------------------------------------
    def log_image(self, tag: str, image, step: int):
        """
        Log a single image (numpy array or PIL Image).
        Used by visualization.py to log masked vs reconstructed patches.

        Args:
            tag   : label shown in dashboard e.g. "reconstruction/video"
            image : (H, W, 3) numpy array, values in [0, 1]
            step  : epoch or global step number
        """
        if self.writer is not None:
            import torch
            if not isinstance(image, torch.Tensor):
                import numpy as np
                image = torch.from_numpy(np.array(image))
            # TensorBoard expects (C, H, W)
            if image.ndim == 3 and image.shape[-1] in (1, 3, 4):
                image = image.permute(2, 0, 1)
            self.writer.add_image(tag, image, global_step=step)

        if self.use_wandb and self.run is not None:
            import wandb
            import numpy as np
            if hasattr(image, "numpy"):
                image = image.numpy()
            self.run.log({tag: wandb.Image(image)}, step=step)

    # ------------------------------------------------------------------
    def close(self):
        """
        Call at the end of training to flush and close all writers.
        """
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

        if self.use_wandb and self.run is not None:
            self.run.finish()