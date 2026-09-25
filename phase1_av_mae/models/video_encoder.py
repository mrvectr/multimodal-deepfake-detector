# models/video_encoder.py

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from models.positional_encoding import get_1d_sincos_pos_embed_from_grid


class VideoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: tuple = (2, 16, 16),
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()

        self.patch_size  = patch_size
        self.embed_dim   = embed_dim
        self.t_patches   = num_frames // patch_size[0]
        self.h_patches   = img_size   // patch_size[1]
        self.w_patches   = img_size   // patch_size[2]
        self.num_patches = self.t_patches * self.h_patches * self.w_patches

        self.patch_embed = nn.Conv3d(
            in_channels  = 3,
            out_channels = embed_dim,
            kernel_size  = patch_size,
            stride       = patch_size,
        )

        self.register_buffer("pos_embed", self._build_pos_embed())

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    def _build_pos_embed(self) -> torch.Tensor:
        t = torch.arange(self.t_patches, dtype=torch.float32)
        h = torch.arange(self.h_patches, dtype=torch.float32)
        w = torch.arange(self.w_patches, dtype=torch.float32)

        # use full embed_dim for each axis then ADD them together
        pe_t = get_1d_sincos_pos_embed_from_grid(self.embed_dim, t)  # (T, D)
        pe_h = get_1d_sincos_pos_embed_from_grid(self.embed_dim, h)  # (H, D)
        pe_w = get_1d_sincos_pos_embed_from_grid(self.embed_dim, w)  # (W, D)

        pe = (
            pe_t[:, None, None, :] +
            pe_h[None, :, None, :] +
            pe_w[None, None, :, :]
        )  # (T, H, W, D)

        pe = pe.reshape(-1, self.embed_dim)  # (T*H*W, D)
        return pe.unsqueeze(0)               # (1, N, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)                              # (B, D, t, h, w)
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, T*H*W, D)  # (B, N, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)