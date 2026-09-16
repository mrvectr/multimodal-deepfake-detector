# data/dataset.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset

import av
import torchaudio


class AVDataset(Dataset):
    """
    Base Audio-Visual dataset.

    Expects a plain text file where each line is an absolute path
    to a video file (.mp4, .avi, .mkv). Audio is extracted directly
    from the video container — no separate audio files needed.

    File list format (one path per line):
        /data/lrs3/trainval/5536038-00_06_06-00_06_11.mp4
        /data/voxceleb2/dev/mp4/id00012/21Uxsk56VDQ/00001.mp4
        ...
    """

    def __init__(
        self,
        file_list: str,        # path to .txt file containing video paths
        cfg,
        split: str = "train",  # "train" or "val"
    ):
        self.cfg          = cfg
        self.split        = split
        self.num_frames   = cfg.data.num_frames
        self.img_size     = cfg.data.img_size
        self.sample_rate  = cfg.data.sample_rate
        self.clip_seconds = cfg.data.clip_len_seconds
        self.n_samples    = self.sample_rate * self.clip_seconds

        # load file list
        with open(file_list, "r") as f:
            self.videos = [line.strip() for line in f if line.strip()]

        print(f"[{split}] {len(self.videos)} video clips loaded.")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.videos)

    # ------------------------------------------------------------------
    def _load_video_frames(self, path: str) -> torch.Tensor:
        """
        Decode exactly self.num_frames evenly spaced frames from a video.

        Returns:
            frames : (3, T, H, W)  float32 in [0, 1]
        """
        container = av.open(path)
        stream    = container.streams.video[0]

        total_frames = stream.frames
        if total_frames == 0:
            # some containers don't store frame count — decode and count
            total_frames = sum(1 for _ in container.decode(video=0))
            container.seek(0)

        # evenly spaced frame indices
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        indices = set(indices.tolist())

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                img = frame.to_image().convert("RGB")
                img = img.resize((self.img_size, self.img_size))
                arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
                frames.append(arr)
            if len(frames) == self.num_frames:
                break

        container.close()

        # pad with zeros if not enough frames were decoded
        while len(frames) < self.num_frames:
            frames.append(np.zeros(
                (self.img_size, self.img_size, 3), dtype=np.float32
            ))

        frames = np.stack(frames, axis=0)             # (T, H, W, 3)
        frames = torch.from_numpy(frames)
        frames = frames.permute(3, 0, 1, 2)           # (3, T, H, W)
        return frames

    # ------------------------------------------------------------------
    def _load_audio(self, path: str) -> torch.Tensor:
        """
        Extract audio from the video container and resample to
        self.sample_rate. Pad or crop to exactly self.n_samples.

        Returns:
            waveform : (n_samples,)  float32 mono
        """
        container = av.open(path)

        # check if audio stream exists
        if not container.streams.audio:
            container.close()
            return torch.zeros(self.n_samples)

        waveform_chunks = []
        for frame in container.decode(audio=0):
            chunk = torch.from_numpy(
                frame.to_ndarray().mean(axis=0)   # mix to mono
            ).float()
            waveform_chunks.append(chunk)

        container.close()

        if not waveform_chunks:
            return torch.zeros(self.n_samples)

        waveform = torch.cat(waveform_chunks, dim=0)   # (total_samples,)

        # resample if needed
        src_rate = frame.sample_rate
        if src_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=src_rate, new_freq=self.sample_rate
            )

        # pad or crop to fixed length
        if waveform.shape[0] < self.n_samples:
            waveform = torch.nn.functional.pad(
                waveform, (0, self.n_samples - waveform.shape[0])
            )
        else:
            waveform = waveform[:self.n_samples]

        return waveform   # (n_samples,)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        path = self.videos[idx]

        try:
            video    = self._load_video_frames(path)   # (3, T, H, W)
            waveform = self._load_audio(path)           # (n_samples,)
        except Exception as e:
            # if a file is corrupted, return a zero sample and move on
            print(f"Warning: failed to load {path} — {e}")
            video    = torch.zeros(3, self.num_frames, self.img_size, self.img_size)
            waveform = torch.zeros(self.n_samples)

        return {
            "video"    : video,      # (3, T, H, W)  float32 [0, 1]
            "waveform" : waveform,   # (n_samples,)  float32
            "path"     : path,
        }


# ----------------------------------------------------------------------
def build_file_list(root_dir: str, out_path: str, extensions=(".mp4", ".avi", ".mkv")):
    """
    Walk a dataset root directory and write all video paths to a .txt file.

    Usage:
        build_file_list("/data/lrs3", "file_lists/lrs3_train.txt")

    Then pass "file_lists/lrs3_train.txt" as file_list to AVDataset.
    """
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                paths.append(os.path.join(dirpath, fname))

    paths.sort()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"File list saved → {out_path} ({len(paths)} videos)")
    return paths


# ----------------------------------------------------------------------
def build_dataloader(file_list: str, cfg, split: str = "train"):
    """
    Convenience function — build dataset + DataLoader in one call.

    Usage:
        loader = build_dataloader("file_lists/lrs3_train.txt", cfg)
    """
    from torch.utils.data import DataLoader

    dataset = AVDataset(file_list=file_list, cfg=cfg, split=split)

    loader = DataLoader(
        dataset,
        batch_size  = cfg.train.batch_size,
        shuffle     = (split == "train"),
        num_workers = cfg.train.num_workers,
        pin_memory  = cfg.train.pin_memory,
        drop_last   = (split == "train"),
    )

    return loader