# engine/trainer.py

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.misc import AverageMeter, set_seed, load_config
from utils.logger import Logger
from engine.checkpointing import save_checkpoint, load_checkpoint


def build_optimizer(model: nn.Module, cfg) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.weight_decay,
    )


def build_scheduler(optimizer, cfg) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Linear warmup followed by cosine annealing.
    """
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 1e-6 / cfg.train.lr,
        end_factor   = 1.0,
        total_iters  = cfg.train.warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max  = cfg.train.epochs - cfg.train.warmup_epochs,
        eta_min = cfg.train.min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers  = [warmup, cosine],
        milestones  = [cfg.train.warmup_epochs],
    )


# ----------------------------------------------------------------------
class Trainer:
    def __init__(self, model: nn.Module, cfg, train_loader: DataLoader):

        self.cfg          = cfg
        self.device       = torch.device(cfg.train.device)
        self.model        = model.to(self.device)
        self.train_loader = train_loader

        self.optimizer  = build_optimizer(model, cfg)
        self.scheduler  = build_scheduler(self.optimizer, cfg)
        self.logger     = Logger(cfg)
        self.start_epoch = 0

        # ── AMP scaler (only active when amp: true in config) ──────────
        self.use_amp = cfg.train.amp and self.device.type == "cuda"
        self.scaler  = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # ── Resume from checkpoint if available ────────────────────────
        ckpt_dir = cfg.logging.checkpoint_dir
        latest   = os.path.join(ckpt_dir, "latest.pt")
        if os.path.exists(latest):
            self.start_epoch = load_checkpoint(
                latest, self.model, self.optimizer, self.scheduler
            )
            print(f"Resumed from epoch {self.start_epoch}")

    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> dict:

        self.model.train()

        loss_meter  = AverageMeter()
        loss_v_meter = AverageMeter()
        loss_a_meter = AverageMeter()
        time_meter  = AverageMeter()

        end = time.time()

        for step, batch in enumerate(self.train_loader):

            video    = batch["video"].to(self.device, non_blocking=True)
            waveform = batch["waveform"].to(self.device, non_blocking=True)

            # ── Forward ────────────────────────────────────────────────
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                total_loss, loss_v, loss_a = self.model(video, waveform)

            # ── Backward ───────────────────────────────────────────────
            self.optimizer.zero_grad()
            self.scaler.scale(total_loss).backward()

            # gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.grad_clip
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # ── Meters ─────────────────────────────────────────────────
            B = video.shape[0]
            loss_meter.update(total_loss.item(),   B)
            loss_v_meter.update(loss_v.item(),     B)
            loss_a_meter.update(loss_a.item(),     B)
            time_meter.update(time.time() - end)
            end = time.time()

            # ── Step-level logging ─────────────────────────────────────
            if step % self.cfg.logging.log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch [{epoch:03d}] "
                    f"Step [{step:04d}/{len(self.train_loader):04d}] | "
                    f"Loss: {loss_meter.avg:.4f} | "
                    f"Video: {loss_v_meter.avg:.4f} | "
                    f"Audio: {loss_a_meter.avg:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Time: {time_meter.avg:.2f}s"
                )

        return {
            "loss"       : loss_meter.avg,
            "loss_video" : loss_v_meter.avg,
            "loss_audio" : loss_a_meter.avg,
        }

    # ------------------------------------------------------------------
    def train(self):

        print(f"Starting training from epoch {self.start_epoch}")
        print(f"Total epochs : {self.cfg.train.epochs}")
        print("-" * 60)

        for epoch in range(self.start_epoch, self.cfg.train.epochs):

            metrics = self._train_one_epoch(epoch)

            self.scheduler.step()

            # ── Epoch-level logging ────────────────────────────────────
            self.logger.log(metrics, step=epoch)

            print(
                f"Epoch [{epoch:03d}] Complete | "
                f"Avg Loss: {metrics['loss']:.4f} | "
                f"Video: {metrics['loss_video']:.4f} | "
                f"Audio: {metrics['loss_audio']:.4f}"
            )
            print("-" * 60)

            # ── Checkpointing ──────────────────────────────────────────
            if epoch % self.cfg.logging.save_every == 0 or epoch == self.cfg.train.epochs - 1:
                save_checkpoint(
                    epoch      = epoch,
                    model      = self.model,
                    optimizer  = self.optimizer,
                    scheduler  = self.scheduler,
                    cfg        = self.cfg,
                    ckpt_dir   = self.cfg.logging.checkpoint_dir,
                    is_latest  = True,
                )

        print("Training complete.")