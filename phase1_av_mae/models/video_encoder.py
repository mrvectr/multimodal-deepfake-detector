# models/video_encoder.py

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from models.positional_encoding import get_1d_sincos_pos_embed_from_grid


class VideoEncoder(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        num_frames: int = 16,
        patch_size: tuple = (2, 16, 16),   # (temporal, H, W)
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # number of patches along each axis
        self.t_patches = num_frames  // patch_size[0]
        self.h_patches = img_size   // patch_size[1]
        self.w_patches = img_size   // patch_size[2]
        self.num_patches = self.t_patches * self.h_patches * self.w_patches

        # ── 1. Tube tokenizer ──────────────────────────────────────────
        # Conv3d with kernel = stride = patch_size turns every
        # (pt × ph × pw) tube into one embed_dim-dimensional token.
        self.patch_embed = nn.Conv3d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ── 2. Positional embedding (frozen sinusoidal) ────────────────
        self.register_buffer(
            "pos_embed",
            self._build_pos_embed(),   # (1, num_patches, embed_dim)
        )

        # ── 3. Transformer encoder blocks ─────────────────────────────
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def _build_pos_embed(self) -> torch.Tensor:
        """
        Factorised sinusoidal pos encoding:
        each patch gets  pos_t + pos_h + pos_w  added together.
        Shape returned: (1, num_patches, embed_dim)
        """
        t = torch.arange(self.t_patches, dtype=torch.float32)
        h = torch.arange(self.h_patches, dtype=torch.float32)
        w = torch.arange(self.w_patches, dtype=torch.float32)

        # each axis gets embed_dim // 3 dims  (we trim/pad to embed_dim)
        d = self.embed_dim // 3

        pe_t = get_1d_sincos_pos_embed_from_grid(d, t)   # (T, d)
        pe_h = get_1d_sincos_pos_embed_from_grid(d, h)   # (H, d)
        pe_w = get_1d_sincos_pos_embed_from_grid(d, w)   # (W, d)

        # broadcast and add over a (T, H, W, d) grid
        pe = (
            pe_t[:, None, None, :] +   # (T, 1, 1, d)
            pe_h[None, :, None, :] +   # (1, H, 1, d)
            pe_w[None, None, :, :]     # (1, 1, W, d)
        )                               # (T, H, W, d)

        pe = pe.reshape(-1, d)          # (T*H*W, d)

        # pad to embed_dim if embed_dim % 3 != 0
        if d * 3 < self.embed_dim:
            pad = torch.zeros(pe.shape[0], self.embed_dim - d * 3)
            pe = torch.cat([pe, pad], dim=1)

        return pe.unsqueeze(0)          # (1, num_patches, embed_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : (B, 3, T, H, W)  — raw video frames

        Returns:
            tokens : (B, num_patches, embed_dim)  — full encoded sequence
                     (no masking here; masking is applied outside)
        """
        # tokenize → (B, embed_dim, t, h, w)
        x = self.patch_embed(x)

        # flatten spatial/temporal dims → (B, num_patches, embed_dim)
        x = rearrange(x, "b d t h w -> b (t h w) d")

        # add positional encoding
        x = x + self.pos_embed

        # pass through transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x