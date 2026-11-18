# multimodal-deepfake-detector

A physics-grounded multimodal system to detect AI-generated and manipulated media.

---

## Project Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | The "Reality" Foundation (Data Strategy) | Complete✅ |
| Phase 2 | Multimodal Invariant Extraction | ⏳In progress |
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
