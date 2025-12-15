import os
import numpy as np
import librosa
from typing import Tuple, List

def list_wavs_by_label(data_dir: str) -> Tuple[List[str], List[str]]:
    """
    Expects structure:
      data_dir/
        angry/*.wav
        happy/*.wav
        ...
    Returns filepaths and string labels.
    """
    filepaths, labels = [], []
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fn in sorted(os.listdir(label_dir)):
            if fn.lower().endswith(".wav"):
                filepaths.append(os.path.join(label_dir, fn))
                labels.append(label)
    return filepaths, labels

def load_audio(path: str, sr: int, duration_sec: float) -> np.ndarray:
    y, _ = librosa.load(path, sr=sr, mono=True)
    target_len = int(sr * duration_sec)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]
    return y.astype(np.float32)

def mfcc_features(
    y: np.ndarray,
    sr: int,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    # Standardize per-sample (helps training stability)
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-8)
    return mfcc.astype(np.float32)  # shape: (n_mfcc, T)
