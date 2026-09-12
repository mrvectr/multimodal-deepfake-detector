# multimodal-deepfake-detector

A physics-grounded multimodal system to detect AI-generated and manipulated media.

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | The "Reality" Foundation (Data Strategy) | In Progress |
| Phase 2 | Multimodal Invariant Extraction | ⬜Pending |
| Phase 3 | The Fusion & Reasoning - Core Architecture | ⬜Pending |
| Phase 4 | For Generalization - Training Protocol | ⬜Pending |
| Phase 5 | Output Reasoning | ⬜Pending |

---

## Repository Structure

```
multimodal-deepfake-detector/
│
├── phase1_av_mae/               ← Reality Foundation: Cross-Modal Masked Autoencoder
├── phase2_feature_extraction/   ← Multimodal Invariant Extraction
├── phase3_fusion/               ← Fusion & Reasoning Core Architecture
├── phase4_training/             ← Generalization Training Protocol
├── phase5_output/               ← Output Reasoning
│
├── shared/                      ← Utilities shared across all phases
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Phase 1 — The "Reality" Foundation `phase1_av_mae/`

Before building the detector, we establish a baseline of what real human audio and video looks like using Self-Supervised Learning on authentic media only. We train a Cross-Modal Masked Autoencoder — mask 40% of the video patches and 40% of the audio mel-spectrograms, then force the model to reconstruct the missing video using only the audio and vice versa. Training data: LRS3, VoxCeleb2, and AV-Deepfake1M++ real subsets. No fake data is used.

```
phase1_av_mae/
│
├── configs/
│   ├── baseline.yaml            ← Small config for testing without real data
│   └── full_train.yaml          ← Full config for LRS3 + VoxCeleb2 training
│
├── data/
│   ├── dataset.py               ← Loads video and audio clips
│   ├── masking.py               ← Randomly hides 40% of patches
│   ├── transforms.py            ← Video and audio augmentations
│   └── samplers.py              ← Clip sampling from long videos
│
├── models/
│   ├── positional_encoding.py   ← Spatial and temporal position encoding
│   ├── video_encoder.py         ← 3D tube tokenizer + ViT encoder
│   ├── audio_encoder.py         ← Mel-spectrogram patchifier + ViT encoder
│   ├── decoder.py               ← Cross-modal reconstruction decoder
│   └── crossmodal_mae.py        ← Master model
│
├── losses/
│   ├── reconstruction.py        ← MSE on masked patches only
│   └── contrastive.py           ← Cross-modal contrastive loss
│
├── engine/
│   ├── trainer.py               ← Training loop
│   ├── evaluator.py             ← Reconstruction error on new clips
│   └── checkpointing.py         ← Save and load weights
│
├── utils/
│   ├── misc.py                  ← Seeds, device setup, helpers
│   ├── logger.py                ← W&B and TensorBoard logging
│   └── visualization.py         ← Masked vs reconstructed patch plots
│
├── scripts/
│   ├── verify_baseline.py       ← Sanity check: overfit on one synthetic sample
│   ├── train.py                 ← Main training entry point
│   └── evaluate.py              ← Run model on a clip, return reconstruction error
│
└── tests/
    ├── test_masking.py
    ├── test_forward_pass.py
    └── test_loss.py
```

### Tech Stack

| Tool | Role |
|---|---|
| PyTorch | Core deep learning framework |
| torchaudio | Audio loading and mel-spectrogram extraction |
| timm | ViT transformer blocks |
| einops | Tensor reshaping for patch operations |
| PyAV | Video decoding |
| OmegaConf | YAML config management |
| Weights & Biases | Experiment tracking |
