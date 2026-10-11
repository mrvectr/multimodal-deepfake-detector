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
    parser = argparse.ArgumentParser(
        description="Train Cross-Modal Audio-Visual Masked Autoencoder"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="phase1_av_mae/configs/full_train.yaml",
        help="Path to config file",
    )

    # --------------------------------------------------------------
    # Training data
    # --------------------------------------------------------------
    parser.add_argument(
        "--train_video_dir",
        type=str,
        required=True,
        help="Directory containing training video files",
    )

    parser.add_argument(
        "--train_audio_dir",
        type=str,
        required=True,
        help="Directory containing training audio files",
    )

    # --------------------------------------------------------------
    # Optional validation data
    # --------------------------------------------------------------
    parser.add_argument(
        "--val_video_dir",
        type=str,
        default=None,
        help="Directory containing validation video files",
    )

    parser.add_argument(
        "--val_audio_dir",
        type=str,
        default=None,
        help="Directory containing validation audio files",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --------------------------------------------------------------
    # Configuration
    # --------------------------------------------------------------
    cfg = load_config(args.config)

    set_seed(cfg.seed)

    device = torch.device(cfg.train.device)

    print(f"Device : {device}")
    print(f"Config : {args.config}")
    print("-" * 60)

    # --------------------------------------------------------------
    # Model
    # --------------------------------------------------------------
    model = CrossModalMAE(cfg).to(device)

    total_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"Parameters : {total_params / 1e6:.2f}M")
    print("-" * 60)

    # --------------------------------------------------------------
    # Training Data
    # --------------------------------------------------------------
    train_loader = build_dataloader(
        video_dir=args.train_video_dir,
        audio_dir=args.train_audio_dir,
        cfg=cfg,
        split="train",
    )

    # --------------------------------------------------------------
    # Validation Data
    # --------------------------------------------------------------
    val_loader = None

    if args.val_video_dir and args.val_audio_dir:

        val_loader = build_dataloader(
            video_dir=args.val_video_dir,
            audio_dir=args.val_audio_dir,
            cfg=cfg,
            split="val",
        )

    # --------------------------------------------------------------
    # Trainer + Evaluator
    # --------------------------------------------------------------
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_loader=train_loader,
    )

    evaluator = Evaluator(
        model=model,
        cfg=cfg,
    )

    # --------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------
    for epoch in range(
        trainer.start_epoch,
        cfg.train.epochs,
    ):

        # Train
        metrics = trainer._train_one_epoch(epoch)

        # Update scheduler
        trainer.scheduler.step()

        # Logging
        trainer.logger.log(
            metrics,
            step=epoch,
        )

        print(
            f"Epoch [{epoch:03d}] Complete | "
            f"Avg Loss: {metrics['loss']:.4f} | "
            f"Video: {metrics['loss_video']:.4f} | "
            f"Audio: {metrics['loss_audio']:.4f}"
        )

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------
        if (
            val_loader is not None
            and epoch % cfg.logging.save_every == 0
        ):

            val_metrics = evaluator.evaluate(
                val_loader,
                epoch,
            )

            trainer.logger.log(
                val_metrics,
                step=epoch,
            )

        # ----------------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------------
        if (
            epoch % cfg.logging.save_every == 0
            or epoch == cfg.train.epochs - 1
        ):

            from engine.checkpointing import save_checkpoint

            save_checkpoint(
                epoch=epoch,
                model=model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                cfg=cfg,
                ckpt_dir=cfg.logging.checkpoint_dir,
                is_latest=True,
            )

            # ------------------------------------------------------
            # Reconstruction visualization
            # ------------------------------------------------------
            try:

                sample = next(iter(train_loader))

                evaluator.visualize_one_batch(
                    video=sample["video"],
                    waveform=sample["waveform"],
                    epoch=epoch,
                    save_dir=os.path.join(
                        cfg.logging.log_dir,
                        "reconstructions",
                    ),
                )

            except Exception as e:

                print(
                    f"Warning: reconstruction visualization "
                    f"failed: {e}"
                )

    # --------------------------------------------------------------
    # Finish
    # --------------------------------------------------------------
    trainer.logger.close()

    print("-" * 60)
    print("Training complete.")


if __name__ == "__main__":
    main()