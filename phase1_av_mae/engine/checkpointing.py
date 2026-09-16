# engine/checkpointing.py

import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf


def save_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    cfg,
    ckpt_dir: str,
    is_latest: bool = False,
):
    """
    Save model weights, optimizer state, scheduler state,
    and config snapshot into one .pt file.

    Saves two files:
        epoch_XXX.pt  — permanent checkpoint for this epoch
        latest.pt     — always overwritten, used for resuming
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    state = {
        "epoch"     : epoch,
        "model"     : model.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "scheduler" : scheduler.state_dict(),
        "cfg"       : OmegaConf.to_container(cfg, resolve=True),
    }

    # permanent epoch checkpoint
    epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:03d}.pt")
    torch.save(state, epoch_path)
    print(f"Checkpoint saved → {epoch_path}")

    # latest checkpoint (for resuming)
    if is_latest:
        latest_path = os.path.join(ckpt_dir, "latest.pt")
        torch.save(state, latest_path)
        print(f"Latest   saved → {latest_path}")


# ----------------------------------------------------------------------
def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler = None,
) -> int:
    """
    Load a checkpoint back into model (and optionally optimizer
    and scheduler).

    Returns:
        next_epoch : int  — epoch to resume from (saved_epoch + 1)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # always load to CPU first — avoids GPU memory conflicts
    state = torch.load(path, map_location="cpu")

    model.load_state_dict(state["model"])
    print(f"Model weights loaded from {path} (epoch {state['epoch']})")

    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
        print("Optimizer state restored.")

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
        print("Scheduler state restored.")

    return state["epoch"] + 1   # resume from next epoch


# ----------------------------------------------------------------------
def load_encoder_weights(
    path: str,
    model: nn.Module,
    strict: bool = False,
) -> None:
    """
    Load only the encoder weights from a checkpoint into a model.
    Used in Phase 2 when we freeze the Phase 1 backbone and attach
    a new detector head on top.

    strict=False allows loading a partial state dict — keys that
    don't match (e.g. decoder weights) are safely ignored.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(path, map_location="cpu")

    # filter to encoder keys only
    encoder_state = {
        k: v for k, v in state["model"].items()
        if k.startswith("video_encoder") or k.startswith("audio_encoder")
    }

    missing, unexpected = model.load_state_dict(encoder_state, strict=strict)

    print(f"Encoder weights loaded from {path}")
    if missing:
        print(f"  Missing keys  : {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")