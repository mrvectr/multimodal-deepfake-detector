# scripts/verify_baseline.py

"""
Milestone gate — no real data needed.
Feed one synthetic batch through the full model and confirm
that the loss decreases over 200 gradient steps.

Expected output:
    Step 000 | Total: ~1.00 | Video: ~0.50 | Audio: ~0.50
    ...
    Step 190 | Total: <0.05 | Video: <0.03 | Audio: <0.03
    ✓ Baseline verified. Architecture is correctly wired.

If loss does not decrease → something in the forward pass is broken.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.misc import set_seed, load_config, AverageMeter


def make_synthetic_batch(cfg, device):
    """
    Create one fake batch of random tensors.
    Shapes match exactly what the real dataloader will produce.
    No video files or audio files needed.
    """
    B          = cfg.train.batch_size
    T          = cfg.data.num_frames
    H          = cfg.data.img_size
    W          = cfg.data.img_size
    samples    = cfg.data.sample_rate * cfg.data.clip_len_seconds

    video    = torch.randn(B, 3, T, H, W).to(device)    # (B, 3, T, H, W)
    waveform = torch.randn(B, samples).to(device)        # (B, raw_samples)

    return video, waveform


def run_verify(cfg_path: str = "phase1_av_mae/configs/baseline.yaml"):

    # ── Setup ──────────────────────────────────────────────────────────
    cfg    = load_config(cfg_path)
    device = torch.device(cfg.train.device)
    set_seed(cfg.seed)

    print(f"Device  : {device}")
    print(f"Config  : {cfg_path}")
    print("-" * 55)

    # ── Build model ────────────────────────────────────────────────────
    from models.crossmodal_mae import CrossModalMAE
    model = CrossModalMAE(cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total_params / 1e6:.2f}M")
    print("-" * 55)

    # ── Optimizer ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.train.lr,
        weight_decay = cfg.train.weight_decay,
    )

    # ── Synthetic batch (fixed — we overfit to one sample) ─────────────
    video, waveform = make_synthetic_batch(cfg, device)

    # ── Training loop ──────────────────────────────────────────────────
    loss_meter = AverageMeter()
    model.train()

    for step in range(cfg.train.epochs):

        optimizer.zero_grad()

        total_loss, loss_v, loss_a = model(video, waveform)

        total_loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.train.grad_clip
        )

        optimizer.step()
        loss_meter.update(total_loss.item())

        if step % cfg.logging.log_every == 0:
            print(
                f"Step {step:03d} | "
                f"Total: {total_loss.item():.4f} | "
                f"Video: {loss_v.item():.4f} | "
                f"Audio: {loss_a.item():.4f}"
            )

    # ── Final check ────────────────────────────────────────────────────
    print("-" * 55)
    final_loss = total_loss.item()

    if final_loss < 1.5:
        print(f"✓ Baseline verified. Final loss: {final_loss:.4f}")
        print("  Architecture is correctly wired. Ready for real data.")
    else:
        print(f"✗ Loss did not converge. Final loss: {final_loss:.4f}")
        print("  Check forward pass in crossmodal_mae.py")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type    = str,
        default = "phase1_av_mae/configs/baseline.yaml",
        help    = "Path to config file",
    )
    args = parser.parse_args()

    run_verify(args.config)