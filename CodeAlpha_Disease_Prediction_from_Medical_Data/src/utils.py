import os

def ensure_dirs(*paths: str):
    for p in paths:
        os.makedirs(p, exist_ok=True)
