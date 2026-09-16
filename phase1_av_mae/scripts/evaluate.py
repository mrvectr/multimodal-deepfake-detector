# scripts/evaluate.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import torch

from utils.misc import set_seed, load_config
from models.crossmodal_mae import CrossModalMAE
from data.dataset import build_dataloader
from engine.evaluator import Evaluator
from engine.checkpointing import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--file_list",  type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)

    # ── Model ──────────────────────────────────────────────────────────
    model = CrossModalMAE(cfg).to(device)
    load_checkpoint(args.checkpoint, model)
    model.eval()

    # ── Data ───────────────────────────────────────────────────────────
    loader = build_dataloader(args.file_list, cfg, split="val")

    # ── Evaluate ───────────────────────────────────────────────────────
    evaluator = Evaluator(model, cfg)
    metrics   = evaluator.evaluate(loader, epoch=0)

    print("\nEvaluation Results:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:<25} : {v:.4f}")


if __name__ == "__main__":
    main()