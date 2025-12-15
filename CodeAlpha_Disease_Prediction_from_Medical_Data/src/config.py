from dataclasses import dataclass

@dataclass
class CFG:
    data_path: str = "data/dataset.csv"   
    target_col: str = "target"          
    test_ratio: float = 0.2
    seed: int = 42

    out_dir: str = "models"
    figures_dir: str = "reports/figures"
