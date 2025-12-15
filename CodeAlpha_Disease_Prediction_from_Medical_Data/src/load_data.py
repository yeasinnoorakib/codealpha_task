import pandas as pd
from .config import CFG

def load_dataset(cfg: CFG) -> pd.DataFrame:
    df = pd.read_csv(cfg.data_path)
    if cfg.target_col not in df.columns:
        raise ValueError(f"target_col='{cfg.target_col}' not found in columns: {list(df.columns)}")
    return df
