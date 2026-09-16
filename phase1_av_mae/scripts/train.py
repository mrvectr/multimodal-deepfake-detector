# scripts/train.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch

from utils.misc import set_seed, load_config
from models.crossmodal_mae import CrossModalMAE
from data.dataset import build_dataloader
from engine.trainer import Trainer
from engine.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type    = str,
        default = "phase1_av_mae/configs/full_train.yaml",
        help    = "Path to config file",
    )
    parser.add_argument(
        "--train_list",
        type    = str,
        required = True,
        help    = "Path to train file list .txt",
    )
    parser.add_argument(
        "--val_list",
        type    = str,
        default = None,
        help    = "Path to val file list .txt (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg.seed)

    device = torch.device(cfg.train.device)
    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print("-" * 60)

    # ── Model ──────────────────────────────────────────────────────────
    model = CrossModalMAE(cfg).to(device)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters : {total / 1e6:.2f}M")
    print("-" * 60)

    # ── Data ───────────────────────────────────────────────────────────
    train_loader = build_dataloader(args.train_list, cfg, split="train")

    val_loader = None
    if args.val_list:
        val_loader = build_dataloader(args.val_list, cfg, split="val")

    # ── Trainer ────────────────────────────────────────────────────────
    trainer   = Trainer(model, cfg, train_loader)
    evaluator = Evaluator(model, cfg)

    # ── Training loop ──────────────────────────────────────────────────
    for epoch in range(trainer.start_epoch, cfg.train.epochs):

        # train one epoch
        metrics = trainer._train_one_epoch(epoch)
        trainer.scheduler.step()
        trainer.logger.log(metrics, step=epoch)

        # optional validation
        if val_loader is not None and epoch % cfg.logging.save_every == 0:
            val_metrics = evaluator.evaluate(val_loader, epoch)
            trainer.logger.log(val_metrics, step=epoch)

        # save checkpoint
        if epoch % cfg.logging.save_every == 0 or epoch == cfg.train.epochs - 1:
            from engine.checkpointing import save_checkpoint
            save_checkpoint(
                epoch     = epoch,
                model     = model,
                optimizer = trainer.optimizer,
                scheduler = trainer.scheduler,
                cfg       = cfg,
                ckpt_dir  = cfg.logging.checkpoint_dir,
                is_latest = True,
            )

            # save reconstruction plot
            sample = next(iter(train_loader))
            evaluator.visualize_one_batch(
                video    = sample["video"],
                waveform = sample["waveform"],
                epoch    = epoch,
                save_dir = os.path.join(cfg.logging.log_dir, "reconstructions"),
            )

    trainer.logger.close()
    print("Training complete.")


if __name__ == "__main__":
    main()