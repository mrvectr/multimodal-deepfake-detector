# utils/visualization.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange


# ----------------------------------------------------------------------
def unpatchify_video(
    patches: torch.Tensor,
    patch_size: tuple,
    num_frames: int,
    img_size: int,
) -> torch.Tensor:
    """
    Reconstruct video tensor from flat patch sequence.

    Args:
        patches    : (B, N, pt * ph * pw * 3)
        patch_size : (pt, ph, pw)
        num_frames : total frames T
        img_size   : H = W

    Returns:
        video : (B, 3, T, H, W)  values in original scale
    """
    pt, ph, pw = patch_size
    T = num_frames  // pt
    H = img_size    // ph
    W = img_size    // pw

    video = rearrange(
        patches,
        "b (t h w) (pt ph pw c) -> b c (t pt) (h ph) (w pw)",
        t=T, h=H, w=W, pt=pt, ph=ph, pw=pw, c=3,
    )
    return video


def unpatchify_audio(
    patches: torch.Tensor,
    patch_size: tuple,
    n_mels: int,
    target_length: int,
) -> torch.Tensor:
    """
    Reconstruct mel-spectrogram from flat patch sequence.

    Args:
        patches       : (B, N, pf * pt * 1)
        patch_size    : (pf, pt)
        n_mels        : total mel bins
        target_length : total time frames

    Returns:
        mel : (B, 1, n_mels, target_length)
    """
    pf, pt = patch_size
    F = n_mels       // pf
    T = target_length // pt

    mel = rearrange(
        patches,
        "b (f t) (pf pt c) -> b c (f pf) (t pt)",
        f=F, t=T, pf=pf, pt=pt, c=1,
    )
    return mel


# ----------------------------------------------------------------------
def apply_mask_to_video(
    video: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple,
    num_frames: int,
    img_size: int,
    mask_value: float = 0.5,
) -> torch.Tensor:
    """
    Grey out masked patches in the original video for visualization.

    Args:
        video      : (B, 3, T, H, W)
        mask       : (B, N)  — 1 = masked, 0 = visible
        patch_size : (pt, ph, pw)
        mask_value : pixel value to fill masked patches with

    Returns:
        masked_video : (B, 3, T, H, W)
    """
    from losses.reconstruction import patchify_video
    patches = patchify_video(video, patch_size)           # (B, N, patch_dim)

    # expand mask to patch_dim
    mask_exp = mask.unsqueeze(-1).expand_as(patches)      # (B, N, patch_dim)

    # replace masked patches with mask_value
    patches_masked = patches * (1 - mask_exp) + mask_value * mask_exp

    return unpatchify_video(patches_masked, patch_size, num_frames, img_size)


def apply_mask_to_audio(
    mel: torch.Tensor,
    mask: torch.Tensor,
    patch_size: tuple,
    n_mels: int,
    target_length: int,
    mask_value: float = 0.0,
) -> torch.Tensor:
    """
    Grey out masked patches in the mel-spectrogram for visualization.
    """
    from losses.reconstruction import patchify_audio
    patches = patchify_audio(mel, patch_size)

    mask_exp = mask.unsqueeze(-1).expand_as(patches)
    patches_masked = patches * (1 - mask_exp) + mask_value * mask_exp

    return unpatchify_audio(patches_masked, patch_size, n_mels, target_length)


# ----------------------------------------------------------------------
def visualize_video_reconstruction(
    original: torch.Tensor,    # (B, 3, T, H, W)
    masked: torch.Tensor,      # (B, 3, T, H, W)
    reconstructed: torch.Tensor,  # (B, 3, T, H, W)
    sample_idx: int = 0,
    frame_idx: int = 0,
) -> np.ndarray:
    """
    Create a side-by-side image:
        Original | Masked Input | Reconstructed

    Returns:
        grid : (H, 3*W, 3) numpy array in [0, 1]
    """
    def to_numpy(t):
        # take one sample, one frame, clamp to [0,1]
        frame = t[sample_idx, :, frame_idx]       # (3, H, W)
        frame = frame.detach().cpu().clamp(0, 1)
        return frame.permute(1, 2, 0).numpy()     # (H, W, 3)

    orig  = to_numpy(original)
    mask  = to_numpy(masked)
    recon = to_numpy(reconstructed)

    grid = np.concatenate([orig, mask, recon], axis=1)  # (H, 3W, 3)
    return grid


def visualize_audio_reconstruction(
    original: torch.Tensor,       # (B, 1, n_mels, T)
    masked: torch.Tensor,         # (B, 1, n_mels, T)
    reconstructed: torch.Tensor,  # (B, 1, n_mels, T)
    sample_idx: int = 0,
) -> np.ndarray:
    """
    Create a side-by-side mel-spectrogram image:
        Original | Masked Input | Reconstructed

    Returns:
        grid : (n_mels, 3*T, 1) numpy array
    """
    def to_numpy(t):
        mel = t[sample_idx, 0]                    # (n_mels, T)
        return mel.detach().cpu().numpy()

    orig  = to_numpy(original)
    mask  = to_numpy(masked)
    recon = to_numpy(reconstructed)

    grid = np.concatenate([orig, mask, recon], axis=1)  # (n_mels, 3T)
    return grid


# ----------------------------------------------------------------------
def save_reconstruction_plot(
    original_video: torch.Tensor,
    masked_video: torch.Tensor,
    recon_video: torch.Tensor,
    original_mel: torch.Tensor,
    masked_mel: torch.Tensor,
    recon_mel: torch.Tensor,
    save_path: str,
    epoch: int,
):
    """
    Save a full reconstruction comparison figure to disk.
    Called by the trainer every save_every epochs.

    Layout:
        Row 1 — Video : Original | Masked | Reconstructed
        Row 2 — Audio : Original | Masked | Reconstructed
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"Reconstruction — Epoch {epoch}", fontsize=13)

    # ── Video row ──────────────────────────────────────────────────────
    def show_frame(ax, tensor, title):
        frame = tensor[0, :, 0].detach().cpu().clamp(0, 1)
        ax.imshow(frame.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")

    show_frame(axes[0, 0], original_video,  "Video — Original")
    show_frame(axes[0, 1], masked_video,    "Video — Masked")
    show_frame(axes[0, 2], recon_video,     "Video — Reconstructed")

    # ── Audio row ──────────────────────────────────────────────────────
    def show_mel(ax, tensor, title):
        mel = tensor[0, 0].detach().cpu().numpy()
        ax.imshow(mel, aspect="auto", origin="lower", cmap="magma")
        ax.set_title(title)
        ax.axis("off")

    show_mel(axes[1, 0], original_mel,  "Audio — Original")
    show_mel(axes[1, 1], masked_mel,    "Audio — Masked")
    show_mel(axes[1, 2], recon_mel,     "Audio — Reconstructed")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Reconstruction plot saved → {save_path}")


# ----------------------------------------------------------------------
import os