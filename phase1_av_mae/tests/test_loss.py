# tests/test_loss.py

import torch
import pytest
from losses.reconstruction import (
    patchify_video,
    patchify_audio,
    normalize_patches,
    reconstruction_loss,
)


def test_patchify_video_shape():
    B, C, T, H, W = 2, 3, 4, 32, 32
    video   = torch.randn(B, C, T, H, W)
    patches = patchify_video(video, patch_size=(2, 16, 16))

    # (T/pt) * (H/ph) * (W/pw) = 2 * 2 * 2 = 8
    # patch_dim = 2 * 16 * 16 * 3 = 1536
    assert patches.shape == (B, 8, 1536), f"Wrong shape: {patches.shape}"


def test_patchify_audio_shape():
    B = 2
    mel     = torch.randn(B, 1, 32, 32)
    patches = patchify_audio(mel, patch_size=(16, 16))

    # (32/16) * (32/16) = 2 * 2 = 4
    # patch_dim = 16 * 16 * 1 = 256
    assert patches.shape == (B, 4, 256), f"Wrong shape: {patches.shape}"


def test_normalize_patches_zero_mean():
    patches = torch.randn(2, 10, 64)
    normed  = normalize_patches(patches)

    means = normed.mean(dim=-1)
    assert torch.allclose(means, torch.zeros_like(means), atol=1e-5), \
        "Normalized patches should have zero mean"


def test_loss_only_on_masked_patches():
    B, N, D = 2, 10, 64
    pred   = torch.randn(B, N, D)
    target = torch.randn(B, N, D)

    # mask only the first 5 patches
    mask         = torch.zeros(B, N)
    mask[:, :5]  = 1.0

    loss_partial = reconstruction_loss(pred, target, mask,   normalize=False)
    loss_full    = reconstruction_loss(pred, target, torch.ones(B, N), normalize=False)

    assert loss_partial.item() != loss_full.item(), \
        "Partial mask and full mask should give different losses"


def test_perfect_reconstruction_gives_zero_loss():
    B, N, D = 2, 10, 64
    pred   = torch.randn(B, N, D)
    target = pred.clone()          # identical to pred
    mask   = torch.ones(B, N)

    loss = reconstruction_loss(pred, target, mask, normalize=False)
    assert loss.item() < 1e-6, \
        f"Perfect reconstruction should give ~0 loss, got {loss.item()}"


def test_loss_is_scalar():
    pred   = torch.randn(2, 20, 128)
    target = torch.randn(2, 20, 128)
    mask   = torch.ones(2, 20)

    loss = reconstruction_loss(pred, target, mask)
    assert loss.shape == torch.Size([]), "Loss must be a scalar"