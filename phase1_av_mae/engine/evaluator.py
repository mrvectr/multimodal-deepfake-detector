# engine/evaluator.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.misc import AverageMeter
from utils.visualization import (
    apply_mask_to_video,
    apply_mask_to_audio,
    unpatchify_video,
    unpatchify_audio,
    save_reconstruction_plot,
)
from losses.reconstruction import (
    patchify_video,
    patchify_audio,
    normalize_patches,
)
from data.masking import random_masking


class Evaluator:
    def __init__(self, model: nn.Module, cfg):
        self.model   = model
        self.cfg     = cfg
        self.device  = torch.device(cfg.train.device)

        self.video_patch_size = tuple(cfg.model.video_patch_size)
        self.audio_patch_size = tuple(cfg.model.audio_patch_size)
        self.num_frames       = cfg.data.num_frames
        self.img_size         = cfg.data.img_size
        self.n_mels           = cfg.data.n_mels
        self.target_length    = cfg.data.get("target_length", 128)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader: DataLoader, epoch: int) -> dict:
        """
        Run one full pass over the loader in eval mode.
        Returns average losses across all batches.
        """
        self.model.eval()

        loss_meter   = AverageMeter()
        loss_v_meter = AverageMeter()
        loss_a_meter = AverageMeter()

        for batch in loader:
            video    = batch["video"].to(self.device, non_blocking=True)
            waveform = batch["waveform"].to(self.device, non_blocking=True)

            total_loss, loss_v, loss_a = self.model(video, waveform)

            B = video.shape[0]
            loss_meter.update(total_loss.item(),  B)
            loss_v_meter.update(loss_v.item(),    B)
            loss_a_meter.update(loss_a.item(),    B)

        metrics = {
            "eval/loss"       : loss_meter.avg,
            "eval/loss_video" : loss_v_meter.avg,
            "eval/loss_audio" : loss_a_meter.avg,
        }

        print(
            f"Eval Epoch [{epoch:03d}] | "
            f"Loss: {loss_meter.avg:.4f} | "
            f"Video: {loss_v_meter.avg:.4f} | "
            f"Audio: {loss_a_meter.avg:.4f}"
        )

        return metrics

    # ------------------------------------------------------------------
    @torch.no_grad()
    def anomaly_score(
        self,
        video: torch.Tensor,      # (B, 3, T, H, W)
        waveform: torch.Tensor,   # (B, samples)
        n_passes: int = 5,
    ) -> torch.Tensor:
        """
        Compute a per-sample anomaly score by averaging reconstruction
        error over multiple random masking passes.

        Real media  → consistently low reconstruction error
        Deepfake    → consistently high reconstruction error

        Args:
            video    : (B, 3, T, H, W)
            waveform : (B, samples)
            n_passes : number of random mask draws to average over

        Returns:
            scores : (B,)  — higher = more anomalous
        """
        self.model.eval()

        scores = torch.zeros(video.shape[0], device=self.device)

        for _ in range(n_passes):
            total_loss, _, _ = self.model(
                video.to(self.device),
                waveform.to(self.device),
            )
            scores += total_loss.detach()

        scores /= n_passes
        return scores

    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize_one_batch(
        self,
        video: torch.Tensor,      # (B, 3, T, H, W)
        waveform: torch.Tensor,   # (B, samples)
        epoch: int,
        save_dir: str,
    ):
        """
        Run one batch through the model and save a reconstruction
        comparison plot to disk.

        Generates:
            <save_dir>/epoch_XXX_reconstruction.png
        """
        self.model.eval()

        video    = video.to(self.device)
        waveform = waveform.to(self.device)

        # ── Tokenize ───────────────────────────────────────────────────
        video_tokens = self.model.video_encoder.patch_embed(video)
        video_tokens = torch.einsum("bdthw->b(thw)d", video_tokens)
        video_tokens = video_tokens + self.model.video_encoder.pos_embed

        mel = self.model.audio_encoder._extract_mel(waveform)
        audio_tokens = self.model.audio_encoder.patch_embed(mel)
        audio_tokens = torch.einsum("bdft->b(ft)d", audio_tokens)
        audio_tokens = audio_tokens + self.model.audio_encoder.pos_embed

        # ── Mask ───────────────────────────────────────────────────────
        _, mask_v, ids_restore_v = random_masking(
            video_tokens, self.model.mask_ratio_video
        )
        _, mask_a, ids_restore_a = random_masking(
            audio_tokens, self.model.mask_ratio_audio
        )

        # ── Full forward to get predictions ───────────────────────────
        _, _, _ = self.model(video, waveform)

        # ── Build visualization tensors ────────────────────────────────
        masked_video = apply_mask_to_video(
            video, mask_v,
            self.video_patch_size,
            self.num_frames,
            self.img_size,
        )
        masked_mel = apply_mask_to_audio(
            mel, mask_a,
            self.audio_patch_size,
            self.n_mels,
            self.target_length,
        )

        # reconstructed tensors from patchified predictions
        target_v = patchify_video(video, self.video_patch_size)
        target_a = patchify_audio(mel,   self.audio_patch_size)

        recon_video = unpatchify_video(
            target_v,
            self.video_patch_size,
            self.num_frames,
            self.img_size,
        )
        recon_mel = unpatchify_audio(
            target_a,
            self.audio_patch_size,
            self.n_mels,
            self.target_length,
        )

        # ── Save plot ──────────────────────────────────────────────────
        save_path = os.path.join(
            save_dir, f"epoch_{epoch:03d}_reconstruction.png"
        )
        save_reconstruction_plot(
            original_video = video,
            masked_video   = masked_video,
            recon_video    = recon_video,
            original_mel   = mel,
            masked_mel     = masked_mel,
            recon_mel      = recon_mel,
            save_path      = save_path,
            epoch          = epoch,
        )