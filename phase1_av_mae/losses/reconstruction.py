# losses/reconstruction.py

import torch
import torch.nn as nn
from einops import rearrange


def patchify_video(
    video: torch.Tensor,
    patch_size: tuple = (2, 16, 16),
) -> torch.Tensor:
    """
    Break raw video into patches — this is the reconstruction target.

    Args:
        video      : (B, 3, T, H, W)
        patch_size : (pt, ph, pw)

    Returns:
        patches : (B, num_patches, pt * ph * pw * 3)
    """
    pt, ph, pw = patch_size
    patches = rearrange(
        video,
        "b c (t pt) (h ph) (w pw) -> b (t h w) (pt ph pw c)",
        pt=pt, ph=ph, pw=pw,
    )
    return patches


def patchify_audio(
    mel: torch.Tensor,
    patch_size: tuple = (16, 16),
) -> torch.Tensor:
    """
    Break mel-spectrogram into patches — this is the reconstruction target.

    Args:
        mel        : (B, 1, n_mels, T)
        patch_size : (pf, pt)

    Returns:
        patches : (B, num_patches, pf * pt * 1)
    """
    pf, pt = patch_size
    patches = rearrange(
        mel,
        "b c (f pf) (t pt) -> b (f t) (pf pt c)",
        pf=pf, pt=pt,
    )
    return patches


def normalize_patches(patches: torch.Tensor) -> torch.Tensor:
    """
    Normalize each patch independently to zero mean and unit variance.
    This is the norm_pix_loss trick from MAE — makes the loss focus on
    structure rather than brightness/loudness level.

    Args:
        patches : (B, N, patch_dim)

    Returns:
        normalized patches : (B, N, patch_dim)
    """
    mean = patches.mean(dim=-1, keepdim=True)
    var  = patches.var(dim=-1, keepdim=True)
    return (patches - mean) / (var + 1e-6).sqrt()


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    MSE loss computed ONLY on masked patches.
    Visible patches are excluded — the model must not be rewarded
    for copying what it already saw.

    Args:
        pred      : (B, N, patch_dim)  — decoder output
        target    : (B, N, patch_dim)  — patchified original
        mask      : (B, N)             — 1 = masked, 0 = visible
        normalize : if True, normalize target patches before MSE

    Returns:
        scalar loss
    """
    if normalize:
        target = normalize_patches(target)

    # MSE per element, then mean over patch_dim
    loss = (pred - target) ** 2          # (B, N, patch_dim)
    loss = loss.mean(dim=-1)             # (B, N)

    # average only over masked positions
    loss = (loss * mask).sum() / mask.sum()

    return loss