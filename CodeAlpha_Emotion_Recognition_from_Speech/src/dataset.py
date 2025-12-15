import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Dict
from .preprocess import load_audio, mfcc_features

class EmotionDataset(Dataset):
    def __init__(
        self,
        filepaths: List[str],
        labels: List[str],
        label2id: Dict[str, int],
        sr: int,
        duration_sec: float,
        n_mfcc: int,
        n_fft: int,
        hop_length: int,
    ):
        self.filepaths = filepaths
        self.labels = labels
        self.label2id = label2id
        self.sr = sr
        self.duration_sec = duration_sec
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = self.filepaths[idx]
        y = load_audio(path, sr=self.sr, duration_sec=self.duration_sec)
        mfcc = mfcc_features(
            y, sr=self.sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft, hop_length=self.hop_length
        )
        # CNN expects (C, H, W). Use C=1, H=n_mfcc, W=T
        x = torch.from_numpy(mfcc).unsqueeze(0)  # (1, n_mfcc, T)
        label = self.label2id[self.labels[idx]]
        y_t = torch.tensor(label, dtype=torch.long)
        return x, y_t
