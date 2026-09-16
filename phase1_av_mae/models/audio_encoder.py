# models/audio_encoder.py

import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import Block
from models.positional_encoding import get_1d_sincos_pos_embed_from_grid


class AudioEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 128,
        target_length: int = 128,        # time frames in mel-spectrogram
        patch_size: tuple = (16, 16),    # (mel_bins, time)
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_mels = n_mels
        self.target_length = target_length

        # number of patches along each axis
        self.f_patches = n_mels        // patch_size[0]   # frequency axis
        self.t_patches = target_length // patch_size[1]   # time axis
        self.num_patches = self.f_patches * self.t_patches

        # ── 1. Mel-spectrogram extractor ───────────────────────────────
        self.mel = torch.nn.Sequential(
            torchaudio_melspec(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
            )
        )

        # ── 2. Patch tokenizer ─────────────────────────────────────────
        # Conv2d with kernel = stride = patch_size treats the
        # mel-spectrogram as an image: (1, n_mels, time) → tokens
        self.patch_embed = nn.Conv2d(
            in_channels=1,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # ── 3. Positional embedding (frozen sinusoidal) ────────────────
        self.register_buffer(
            "pos_embed",
            self._build_pos_embed(),   # (1, num_patches, embed_dim)
        )

        # ── 4. Transformer encoder blocks ─────────────────────────────
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

    # ------------------------------------------------------------------
    def _build_pos_embed(self) -> torch.Tensor:
        """
        Factorised sinusoidal pos encoding over frequency and time axes.
        Shape returned: (1, num_patches, embed_dim)
        """
        f = torch.arange(self.f_patches, dtype=torch.float32)
        t = torch.arange(self.t_patches, dtype=torch.float32)

        d = self.embed_dim // 2

        pe_f = get_1d_sincos_pos_embed_from_grid(d, f)   # (F, d)
        pe_t = get_1d_sincos_pos_embed_from_grid(d, t)   # (T, d)

        # broadcast and add over (F, T, d) grid
        pe = (
            pe_f[:, None, :] +   # (F, 1, d)
            pe_t[None, :, :]     # (1, T, d)
        )                         # (F, T, d)

        pe = pe.reshape(-1, d)    # (F*T, d)

        # pad to embed_dim if embed_dim is odd
        if d * 2 < self.embed_dim:
            pad = torch.zeros(pe.shape[0], self.embed_dim - d * 2)
            pe = torch.cat([pe, pad], dim=1)

        return pe.unsqueeze(0)    # (1, num_patches, embed_dim)

    # ------------------------------------------------------------------
    def _extract_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform to log mel-spectrogram and pad/crop
        to a fixed time length.

        Args:
            waveform : (B, samples)

        Returns:
            mel : (B, 1, n_mels, target_length)
        """
        import torchaudio.transforms as AT
        mel_transform = AT.MelSpectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            n_mels=self.n_mels,
        ).to(waveform.device)

        mel = mel_transform(waveform)              # (B, n_mels, T)
        mel = torch.log(mel + 1e-6)               # log compression

        # pad or crop time axis to target_length
        T = mel.shape[-1]
        if T < self.target_length:
            mel = torch.nn.functional.pad(mel, (0, self.target_length - T))
        else:
            mel = mel[:, :, :self.target_length]

        return mel.unsqueeze(1)                    # (B, 1, n_mels, target_length)

    # ------------------------------------------------------------------
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform : (B, samples)  — raw audio at 16kHz

        Returns:
            tokens : (B, num_patches, embed_dim)
        """
        # mel-spectrogram → (B, 1, n_mels, target_length)
        x = self._extract_mel(waveform)

        # tokenize → (B, embed_dim, f, t)
        x = self.patch_embed(x)

        # flatten → (B, num_patches, embed_dim)
        x = rearrange(x, "b d f t -> b (f t) d")

        # add positional encoding
        x = x + self.pos_embed

        # transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x