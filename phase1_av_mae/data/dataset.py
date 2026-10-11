# data/dataset.py

import os
from typing import List, Tuple

import av
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from torch.utils.data import Dataset, DataLoader


class AVDataset(Dataset):
    """
    Handles separate video and audio files.

    Expected structure:

        Test_data/
        ├── video/
        │   ├── sample001.mov
        │   ├── sample002.mov
        │   └── ...
        │
        └── audio/
            ├── sample001.wav
            ├── sample002.wav
            └── ...

    Video and audio are paired by matching filename stem.

    Example:
        s2_l_bbim3a.mov <-> s2_l_bbim3a.wav
    """

    VIDEO_EXTS = (".mov", ".mp4", ".avi", ".mkv")
    AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")

    def __init__(
        self,
        video_dir: str,
        audio_dir: str,
        cfg,
        split: str = "train",
    ):
        self.cfg = cfg
        self.split = split

        # --------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------
        self.num_frames = cfg.data.num_frames
        self.img_size = cfg.data.img_size
        self.sample_rate = cfg.data.sample_rate
        self.clip_seconds = cfg.data.clip_len_seconds

        self.n_samples = (
            self.sample_rate * self.clip_seconds
        )

        self.video_dir = video_dir
        self.audio_dir = audio_dir

        # --------------------------------------------------------------
        # Check directories
        # --------------------------------------------------------------
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(
                f"Video directory does not exist:\n{video_dir}"
            )

        if not os.path.isdir(audio_dir):
            raise FileNotFoundError(
                f"Audio directory does not exist:\n{audio_dir}"
            )

        # --------------------------------------------------------------
        # Collect files
        # --------------------------------------------------------------
        video_files = self._collect_files(
            video_dir,
            self.VIDEO_EXTS,
        )

        audio_files = self._collect_files(
            audio_dir,
            self.AUDIO_EXTS,
        )

        # --------------------------------------------------------------
        # Create filename-stem maps
        # --------------------------------------------------------------
        video_map = {
            os.path.splitext(
                os.path.basename(path)
            )[0]: path
            for path in video_files
        }

        audio_map = {
            os.path.splitext(
                os.path.basename(path)
            )[0]: path
            for path in audio_files
        }

        # --------------------------------------------------------------
        # Match video + audio pairs
        # --------------------------------------------------------------
        common_stems = sorted(
            set(video_map.keys())
            & set(audio_map.keys())
        )

        self.pairs: List[Tuple[str, str]] = [
            (
                video_map[stem],
                audio_map[stem],
            )
            for stem in common_stems
        ]

        # --------------------------------------------------------------
        # Print dataset information
        # --------------------------------------------------------------
        print("-" * 60)
        print(f"[{split}] Audio-Visual Dataset")
        print(f"Video directory : {video_dir}")
        print(f"Audio directory : {audio_dir}")
        print(f"Video files     : {len(video_files)}")
        print(f"Audio files     : {len(audio_files)}")
        print(f"Matched pairs   : {len(self.pairs)}")
        print("-" * 60)

        # --------------------------------------------------------------
        # Stop if nothing matched
        # --------------------------------------------------------------
        if len(self.pairs) == 0:
            raise ValueError(
                "No matched video/audio pairs found.\n\n"
                f"Video directory : {video_dir}\n"
                f"Audio directory : {audio_dir}\n\n"
                "Make sure filenames match, for example:\n"
                "  video/sample001.mov\n"
                "  audio/sample001.wav"
            )

        # --------------------------------------------------------------
        # Report unmatched files
        # --------------------------------------------------------------
        unmatched_videos = sorted(
            set(video_map.keys())
            - set(audio_map.keys())
        )

        unmatched_audio = sorted(
            set(audio_map.keys())
            - set(video_map.keys())
        )

        if unmatched_videos:
            print(
                f"Warning: {len(unmatched_videos)} "
                "video files have no matching audio."
            )

        if unmatched_audio:
            print(
                f"Warning: {len(unmatched_audio)} "
                "audio files have no matching video."
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _collect_files(
        directory: str,
        extensions: tuple,
    ) -> List[str]:
        """
        Collect supported files from a directory.
        """

        files = []

        for filename in os.listdir(directory):

            full_path = os.path.join(
                directory,
                filename,
            )

            if not os.path.isfile(full_path):
                continue

            if filename.lower().endswith(
                extensions
            ):
                files.append(full_path)

        return sorted(files)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.pairs)

    # ------------------------------------------------------------------
    def _load_video_frames(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Decode exactly self.num_frames frames.

        Returns:
            Tensor:
                (3, T, H, W)

            dtype:
                float32

            range:
                [0, 1]
        """

        container = av.open(path)

        try:

            # ----------------------------------------------------------
            # Check video stream
            # ----------------------------------------------------------
            if not container.streams.video:
                raise RuntimeError(
                    f"No video stream found in: {path}"
                )

            stream = container.streams.video[0]

            total_frames = stream.frames

            # ----------------------------------------------------------
            # Some containers don't expose frame count
            # ----------------------------------------------------------
            if total_frames is None or total_frames <= 0:

                decoded_frames = list(
                    container.decode(video=0)
                )

                total_frames = len(decoded_frames)

                if total_frames == 0:
                    raise RuntimeError(
                        f"Video contains no frames: {path}"
                    )

                frames_to_use = decoded_frames

            else:

                frames_to_use = None

            # ----------------------------------------------------------
            # Select evenly spaced frames
            # ----------------------------------------------------------
            indices = np.linspace(
                0,
                total_frames - 1,
                self.num_frames,
                dtype=np.int64,
            )

            indices = indices.tolist()

            frames = []

            # ----------------------------------------------------------
            # Already decoded
            # ----------------------------------------------------------
            if frames_to_use is not None:

                for idx in indices:

                    frame = frames_to_use[idx]

                    img = frame.to_image().convert(
                        "RGB"
                    )

                    img = img.resize(
                        (
                            self.img_size,
                            self.img_size,
                        )
                    )

                    arr = np.asarray(
                        img,
                        dtype=np.float32,
                    ) / 255.0

                    frames.append(arr)

            # ----------------------------------------------------------
            # Normal decoding
            # ----------------------------------------------------------
            else:

                wanted = set(indices)

                for i, frame in enumerate(
                    container.decode(video=0)
                ):

                    if i in wanted:

                        img = frame.to_image().convert(
                            "RGB"
                        )

                        img = img.resize(
                            (
                                self.img_size,
                                self.img_size,
                            )
                        )

                        arr = np.asarray(
                            img,
                            dtype=np.float32,
                        ) / 255.0

                        frames.append(arr)

                    if len(frames) == self.num_frames:
                        break

        finally:
            container.close()

        # --------------------------------------------------------------
        # Safety padding
        # --------------------------------------------------------------
        while len(frames) < self.num_frames:

            frames.append(
                np.zeros(
                    (
                        self.img_size,
                        self.img_size,
                        3,
                    ),
                    dtype=np.float32,
                )
            )

        # --------------------------------------------------------------
        # Convert:
        #
        # (T, H, W, 3)
        #       ->
        # (3, T, H, W)
        # --------------------------------------------------------------
        frames = np.stack(
            frames[:self.num_frames],
            axis=0,
        )

        frames = torch.from_numpy(
            frames
        )

        frames = frames.permute(
            3,
            0,
            1,
            2,
        ).contiguous()

        return frames

    # ------------------------------------------------------------------
    def _load_audio(
        self,
        path: str,
    ) -> torch.Tensor:
        """
        Load a separate audio file.

        Uses SoundFile instead of torchaudio.load()
        to avoid TorchCodec DLL issues on Windows.

        Steps:
            1. Load audio
            2. Convert to mono
            3. Resample if necessary
            4. Pad/crop to fixed duration

        Returns:
            waveform:
                (n_samples,)
        """

        # --------------------------------------------------------------
        # Load audio using SoundFile
        # --------------------------------------------------------------
        audio, sr = sf.read(
            path,
            dtype="float32",
            always_2d=True,
        )

        # --------------------------------------------------------------
        # Convert channels -> mono
        #
        # SoundFile returns:
        #   (samples, channels)
        # --------------------------------------------------------------
        waveform = torch.from_numpy(
            audio.mean(axis=1)
        ).float()

        # --------------------------------------------------------------
        # Resample
        # --------------------------------------------------------------
        if sr != self.sample_rate:

            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sr,
                new_freq=self.sample_rate,
            )

        # --------------------------------------------------------------
        # Pad / crop
        # --------------------------------------------------------------
        if waveform.shape[0] < self.n_samples:

            waveform = F.pad(
                waveform,
                (
                    0,
                    self.n_samples
                    - waveform.shape[0],
                ),
            )

        else:

            waveform = waveform[
                :self.n_samples
            ]

        return waveform.contiguous()

    # ------------------------------------------------------------------
    def __getitem__(
        self,
        idx: int,
    ) -> dict:

        video_path, audio_path = self.pairs[idx]

        try:

            video = self._load_video_frames(
                video_path
            )

            waveform = self._load_audio(
                audio_path
            )

        except Exception as e:

            print(
                "\nWARNING: Failed to load AV pair"
            )
            print(
                f"Video : {video_path}"
            )
            print(
                f"Audio : {audio_path}"
            )
            print(
                f"Error : {e}\n"
            )

            # ----------------------------------------------------------
            # Fallback tensors
            # ----------------------------------------------------------
            video = torch.zeros(
                3,
                self.num_frames,
                self.img_size,
                self.img_size,
                dtype=torch.float32,
            )

            waveform = torch.zeros(
                self.n_samples,
                dtype=torch.float32,
            )

        return {
            "video": video,
            "waveform": waveform,
            "path": video_path,
            "video_path": video_path,
            "audio_path": audio_path,
        }


# ----------------------------------------------------------------------
def build_dataloader(
    video_dir: str,
    audio_dir: str,
    cfg,
    split: str = "train",
):
    """
    Build Audio-Visual Dataset + DataLoader.
    """

    dataset = AVDataset(
        video_dir=video_dir,
        audio_dir=audio_dir,
        cfg=cfg,
        split=split,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.train.num_workers,
        pin_memory=cfg.train.pin_memory,
        drop_last=False,
    )

    return loader