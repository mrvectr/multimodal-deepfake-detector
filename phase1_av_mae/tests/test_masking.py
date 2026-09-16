# tests/test_masking.py

import torch
import pytest
from data.masking import random_masking


def test_output_shapes():
    B, N, D = 2, 196, 384
    x = torch.randn(B, N, D)
    x_vis, mask, ids_restore = random_masking(x, mask_ratio=0.4)

    N_keep = int(N * 0.6)
    assert x_vis.shape      == (B, N_keep, D),  f"x_vis shape wrong: {x_vis.shape}"
    assert mask.shape        == (B, N),          f"mask shape wrong: {mask.shape}"
    assert ids_restore.shape == (B, N),          f"ids_restore shape wrong: {ids_restore.shape}"


def test_mask_ratio():
    B, N, D = 4, 200, 128
    x = torch.randn(B, N, D)
    _, mask, _ = random_masking(x, mask_ratio=0.4)

    # each sample should have exactly 40% masked
    masked_count = mask.sum(dim=1)   # (B,)
    expected     = int(N * 0.4)
    assert (masked_count == expected).all(), \
        f"Expected {expected} masked per sample, got {masked_count}"


def test_mask_binary():
    x = torch.randn(2, 100, 64)
    _, mask, _ = random_masking(x, mask_ratio=0.5)
    unique = mask.unique()
    assert set(unique.tolist()).issubset({0.0, 1.0}), \
        f"Mask contains non-binary values: {unique}"


def test_ids_restore_is_permutation():
    B, N, D = 2, 50, 32
    x = torch.randn(B, N, D)
    _, _, ids_restore = random_masking(x, mask_ratio=0.4)

    for b in range(B):
        assert ids_restore[b].sort().values.tolist() == list(range(N)), \
            "ids_restore is not a valid permutation"


def test_different_samples_get_different_masks():
    x = torch.randn(4, 100, 64)
    _, mask, _ = random_masking(x, mask_ratio=0.4)
    # not all rows should be identical
    assert not (mask[0] == mask[1]).all(), \
        "All samples got identical masks — randomness broken"