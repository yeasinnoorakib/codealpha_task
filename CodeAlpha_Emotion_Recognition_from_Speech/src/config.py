from dataclasses import dataclass

@dataclass
class CFG:
    # Data
    data_dir: str = "data"  
    sample_rate: int = 16000
    duration_sec: float = 3.0  
    n_mfcc: int = 40
    n_fft: int = 1024
    hop_length: int = 256

    # Training
    seed: int = 42
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-3
    num_workers: int = 0
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Output
    out_dir: str = "models"
    figures_dir: str = "reports/figures"
