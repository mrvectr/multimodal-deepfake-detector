# data/transforms.py

import torch
import torchvision.transforms as T
import torchaudio.transforms as AT


def get_video_transform(img_size: int = 224, split: str = "train"):
    """
    Video frame augmentations.
    Train: random horizontal flip + color jitter + normalize
    Val  : normalize only
    """
    if split == "train":
        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return T.Compose([
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225],
            ),
        ])


def get_audio_transform(sample_rate: int = 16000, split: str = "train"):
    """
    Audio augmentations applied to raw waveform.
    Train: random time masking + frequency masking (SpecAugment)
    Val  : no augmentation
    """
    if split == "train":
        return torch.nn.Sequential(
            AT.FrequencyMasking(freq_mask_param=30),
            AT.TimeMasking(time_mask_param=100),
        )
    else:
        return torch.nn.Identity()