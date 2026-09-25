# models/audio_encoder.py

import torch
import torch.nn as nn
import torchaudio.transforms as AT
from timm.models.vision_transformer import Block
from models.positional_encoding import get_1d_sincos_pos_embed_from_grid


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        target_length: int = 128,
        patch_size: tuple = (16, 16),
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 160,
    ):
        super().__init__()

        self.patch_size    = patch_size
        self.embed_dim     = embed_dim
        self.n_mels        = n_mels
        self.target_length = target_length
        self.sample_rate   = sample_rate
        self.n_fft         = n_fft
        self.hop_length    = hop_length
        self.f_patches     = n_mels        // patch_size[0]
        self.t_patches     = target_length // patch_size[1]
        self.num_patches   = self.f_patches * self.t_patches

        self.patch_embed = nn.Conv2d(
            in_channels  = 1,
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
        f = torch.arange(self.f_patches, dtype=torch.float32)
        t = torch.arange(self.t_patches, dtype=torch.float32)

        # use full embed_dim for each axis then ADD them together
        pe_f = get_1d_sincos_pos_embed_from_grid(self.embed_dim, f)  # (F, D)
        pe_t = get_1d_sincos_pos_embed_from_grid(self.embed_dim, t)  # (T, D)

        pe = (
            pe_f[:, None, :] +
            pe_t[None, :, :]
        )  # (F, T, D)

        pe = pe.reshape(-1, self.embed_dim)  # (F*T, D)
        return pe.unsqueeze(0)               # (1, N, D)

    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_transform = AT.MelSpectrogram(
            sample_rate = self.sample_rate,
            n_fft       = self.n_fft,
            hop_length  = self.hop_length,
            n_mels      = self.n_mels,
        ).to(waveform.device)

        mel = mel_transform(waveform)       # (B, n_mels, T)
        mel = torch.log(mel + 1e-6)

        T = mel.shape[-1]
        if T < self.target_length:
            mel = torch.nn.functional.pad(mel, (0, self.target_length - T))
        else:
            mel = mel[:, :, :self.target_length]

        return mel.unsqueeze(1)             # (B, 1, n_mels, target_length)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = self._extract_mel(waveform)     # (B, 1, n_mels, target_length)
        x = self.patch_embed(x)             # (B, D, F, T)
        B, D, F, T = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, F*T, D)  # (B, N, D)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        return self.norm(x)