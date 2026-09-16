# models/decoder.py

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from models.positional_encoding import get_1d_sincos_pos_embed_from_grid


class CrossModalDecoder(nn.Module):
    def __init__(
        self,
        # patch dimensions (needed to compute reconstruction target size)
        video_patch_size: tuple = (2, 16, 16),   # (t, h, w)
        audio_patch_size: tuple = (16, 16),       # (f, t)
        num_video_patches: int = 392,             # T/pt * H/ph * W/pw
        num_audio_patches: int = 64,              # F/pf * T/pt

        # encoder output dim → decoder input dim
        encoder_embed_dim: int = 384,
        decoder_embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ):
        super().__init__()

        self.num_video_patches = num_video_patches
        self.num_audio_patches = num_audio_patches
        self.decoder_embed_dim = decoder_embed_dim

        # patch pixel/mel counts — what the decoder must reconstruct
        self.video_patch_dim = video_patch_size[0] * video_patch_size[1] * video_patch_size[2] * 3
        self.audio_patch_dim = audio_patch_size[0] * audio_patch_size[1] * 1

        # ── 1. Project encoder dim → decoder dim ──────────────────────
        self.video_proj = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        self.audio_proj = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # ── 2. Learnable mask tokens ───────────────────────────────────
        # one vector that stands in for every masked patch position
        self.mask_token_video = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_audio = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        # ── 3. Decoder positional embeddings (full sequence) ──────────
        self.register_buffer(
            "video_pos_embed",
            self._build_pos_embed(num_video_patches, decoder_embed_dim),
        )
        self.register_buffer(
            "audio_pos_embed",
            self._build_pos_embed(num_audio_patches, decoder_embed_dim),
        )

        # ── 4. Transformer decoder blocks ─────────────────────────────
        self.blocks = nn.ModuleList([
            Block(dim=decoder_embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        # ── 5. Reconstruction heads ────────────────────────────────────
        # projects decoder output back to raw patch pixel / mel values
        self.video_head = nn.Linear(decoder_embed_dim, self.video_patch_dim)
        self.audio_head = nn.Linear(decoder_embed_dim, self.audio_patch_dim)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        # initialise mask tokens close to zero
        nn.init.normal_(self.mask_token_video, std=0.02)
        nn.init.normal_(self.mask_token_audio, std=0.02)

    # ------------------------------------------------------------------
    def _build_pos_embed(self, num_patches: int, dim: int) -> torch.Tensor:
        pos = torch.arange(num_patches, dtype=torch.float32)
        pe  = get_1d_sincos_pos_embed_from_grid(dim, pos)   # (N, dim)
        return pe.unsqueeze(0)                               # (1, N, dim)

    # ------------------------------------------------------------------
    def _restore_sequence(
        self,
        visible_tokens: torch.Tensor,   # (B, N_visible, D)
        mask_token: nn.Parameter,       # (1, 1, D)
        ids_restore: torch.Tensor,      # (B, N_total)
        pos_embed: torch.Tensor,        # (1, N_total, D)
    ) -> torch.Tensor:
        """
        Insert mask tokens at masked positions and unshuffle
        back to the original patch order.

        Returns: (B, N_total, D)
        """
        B = visible_tokens.shape[0]
        N_total = ids_restore.shape[1]
        N_visible = visible_tokens.shape[1]
        N_masked = N_total - N_visible

        # expand mask token to fill all masked positions
        mask_tokens = mask_token.expand(B, N_masked, -1)  # (B, N_masked, D)

        # concatenate: visible first, then mask tokens
        full = torch.cat([visible_tokens, mask_tokens], dim=1)  # (B, N_total, D)

        # unshuffle to original patch order using ids_restore
        ids = ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1])
        full = torch.gather(full, dim=1, index=ids)             # (B, N_total, D)

        # add full positional embedding (every position, not just visible)
        full = full + pos_embed

        return full

    # ------------------------------------------------------------------
    def forward(
        self,
        video_visible: torch.Tensor,   # (B, N_v_visible, encoder_dim)
        audio_visible: torch.Tensor,   # (B, N_a_visible, encoder_dim)
        ids_restore_v: torch.Tensor,   # (B, N_v_total)
        ids_restore_a: torch.Tensor,   # (B, N_a_total)
    ):
        """
        Returns:
            pred_video : (B, num_video_patches, video_patch_dim)
            pred_audio : (B, num_audio_patches, audio_patch_dim)
        """
        # project both modalities to decoder dim
        video_tokens = self.video_proj(video_visible)   # (B, N_v_vis, D_dec)
        audio_tokens = self.audio_proj(audio_visible)   # (B, N_a_vis, D_dec)

        # restore full sequence by inserting mask tokens
        video_full = self._restore_sequence(
            video_tokens, self.mask_token_video,
            ids_restore_v, self.video_pos_embed,
        )   # (B, N_v_total, D_dec)

        audio_full = self._restore_sequence(
            audio_tokens, self.mask_token_audio,
            ids_restore_a, self.audio_pos_embed,
        )   # (B, N_a_total, D_dec)

        # concatenate both modalities into one sequence
        # cross-modal context: video can attend to audio tokens and vice versa
        x = torch.cat([video_full, audio_full], dim=1)  # (B, N_v+N_a, D_dec)

        # transformer decoder blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # split back into video and audio
        pred_video = x[:, :self.num_video_patches, :]   # (B, N_v, D_dec)
        pred_audio = x[:, self.num_video_patches:, :]   # (B, N_a, D_dec)

        # project to patch pixel / mel values
        pred_video = self.video_head(pred_video)   # (B, N_v, video_patch_dim)
        pred_audio = self.audio_head(pred_audio)   # (B, N_a, audio_patch_dim)

        return pred_video, pred_audio