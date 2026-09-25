# models/crossmodal_mae.py

import torch
import torch.nn as nn

from models.video_encoder import VideoEncoder
from models.audio_encoder import AudioEncoder
from models.decoder import CrossModalDecoder
from data.masking import random_masking
from losses.reconstruction import patchify_video, patchify_audio, reconstruction_loss


class CrossModalMAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.mask_ratio_video = cfg.model.mask_ratio_video
        self.mask_ratio_audio = cfg.model.mask_ratio_audio
        self.norm_pix_loss    = cfg.model.norm_pix_loss

        video_patch_size = tuple(cfg.model.video_patch_size)
        audio_patch_size = tuple(cfg.model.audio_patch_size)
        encoder_dim      = cfg.model.encoder_embed_dim
        target_len       = getattr(cfg.data, "target_length", 128)

        self.video_patch_size = video_patch_size
        self.audio_patch_size = audio_patch_size

        t_p, h_p, w_p = video_patch_size
        f_p, a_p      = audio_patch_size

        self.num_video_patches = (cfg.data.num_frames // t_p) * (cfg.data.img_size // h_p) * (cfg.data.img_size // w_p)
        self.num_audio_patches = (cfg.data.n_mels // f_p) * (target_len // a_p)

        self.video_encoder = VideoEncoder(
            img_size   = cfg.data.img_size,
            num_frames = cfg.data.num_frames,
            patch_size = video_patch_size,
            embed_dim  = encoder_dim,
            depth      = cfg.model.encoder_depth,
            num_heads  = cfg.model.encoder_num_heads,
        )

        self.audio_encoder = AudioEncoder(
            n_mels        = cfg.data.n_mels,
            target_length = target_len,
            patch_size    = audio_patch_size,
            embed_dim     = encoder_dim,
            depth         = cfg.model.encoder_depth,
            num_heads     = cfg.model.encoder_num_heads,
            sample_rate   = cfg.data.sample_rate,
            n_fft         = cfg.data.n_fft,
            hop_length    = cfg.data.hop_length,
        )

        self.decoder = CrossModalDecoder(
            video_patch_size  = video_patch_size,
            audio_patch_size  = audio_patch_size,
            num_video_patches = self.num_video_patches,
            num_audio_patches = self.num_audio_patches,
            encoder_embed_dim = encoder_dim,
            decoder_embed_dim = cfg.model.decoder_embed_dim,
            depth             = cfg.model.decoder_depth,
            num_heads         = cfg.model.decoder_num_heads,
        )

    def forward(self, video: torch.Tensor, waveform: torch.Tensor):
        # ── Tokenize ──────────────────────────────────────────────────
        v = self.video_encoder.patch_embed(video)        # (B, D, t, h, w)
        B, D, t, h, w = v.shape
        video_tokens = v.permute(0, 2, 3, 4, 1).reshape(B, t*h*w, D)
        video_tokens = video_tokens + self.video_encoder.pos_embed

        mel = self.audio_encoder._extract_mel(waveform)  # (B, 1, n_mels, T)
        a = self.audio_encoder.patch_embed(mel)           # (B, D, F, T)
        B, D, F, T = a.shape
        audio_tokens = a.permute(0, 2, 3, 1).reshape(B, F*T, D)
        audio_tokens = audio_tokens + self.audio_encoder.pos_embed

        # ── Mask ──────────────────────────────────────────────────────
        video_visible, mask_v, ids_restore_v = random_masking(video_tokens, self.mask_ratio_video)
        audio_visible, mask_a, ids_restore_a = random_masking(audio_tokens, self.mask_ratio_audio)

        # ── Encode visible tokens only ────────────────────────────────
        for block in self.video_encoder.blocks:
            video_visible = block(video_visible)
        video_visible = self.video_encoder.norm(video_visible)

        for block in self.audio_encoder.blocks:
            audio_visible = block(audio_visible)
        audio_visible = self.audio_encoder.norm(audio_visible)

        # ── Decode cross-modally ──────────────────────────────────────
        pred_video, pred_audio = self.decoder(
            video_visible, audio_visible,
            ids_restore_v, ids_restore_a,
        )

        # ── Loss on masked patches only ───────────────────────────────
        target_video = patchify_video(video, self.video_patch_size)
        target_audio = patchify_audio(mel,   self.audio_patch_size)

        loss_video = reconstruction_loss(pred_video, target_video, mask_v, normalize=self.norm_pix_loss)
        loss_audio = reconstruction_loss(pred_audio, target_audio, mask_a, normalize=self.norm_pix_loss)

        return loss_video + loss_audio, loss_video, loss_audio