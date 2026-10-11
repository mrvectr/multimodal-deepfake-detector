# scripts/run_on_data.py

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
from utils.misc import set_seed, load_config
from models.crossmodal_mae import CrossModalMAE
from data.dataset import build_dataloader


def main():
    cfg = load_config("configs/baseline.yaml")
    set_seed(cfg.seed)
    device = torch.device(cfg.train.device)

    VIDEO_DIR  = r"C:\Users\Swayam Shah\Documents\internship\data\RAW\video"
    AUDIO_DIR  = r"C:\Users\Swayam Shah\Documents\internship\data\RAW\audio"
    OUTPUT_DIR = r"C:\Users\Swayam Shah\Documents\internship\data\output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # build dataloader
    loader = build_dataloader(VIDEO_DIR, AUDIO_DIR, cfg, split="val")

    # build model
    model = CrossModalMAE(cfg).to(device)
    model.eval()

    print(f"\nRunning on {len(loader.dataset)} samples...\n")
    print(f"{'File':<40} {'Total Loss':>12} {'Video Loss':>12} {'Audio Loss':>12}")
    print("-" * 80)

    results = []

    with torch.no_grad():
        for batch in loader:
            video    = batch["video"].to(device)
            waveform = batch["waveform"].to(device)
            paths    = batch["path"]

            total_loss, loss_v, loss_a = model(video, waveform)

            for i, path in enumerate(paths):
                fname = os.path.basename(path)
                print(f"{fname:<40} {total_loss.item():>12.4f} {loss_v.item():>12.4f} {loss_a.item():>12.4f}")
                results.append({
                    "file"       : fname,
                    "total_loss" : total_loss.item(),
                    "loss_video" : loss_v.item(),
                    "loss_audio" : loss_a.item(),
                })

    # save results to output folder
    import json
    out_path = os.path.join(OUTPUT_DIR, "reconstruction_errors.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved → {out_path}")
    print("\nHow to read the output:")
    print("  Low  loss (~1.0-1.5) → model sees this as physically coherent (real)")
    print("  High loss (>2.0)     → model struggled to reconstruct → anomaly signal")


if __name__ == "__main__":
    main()