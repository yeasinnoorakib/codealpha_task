import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from .config import CFG
from .utils import ensure_dirs
from .preprocess import list_wavs_by_label
from .dataset import EmotionDataset
from .model import MFCC_CNN

def plot_confusion(cm, classes, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(len(classes)), classes, rotation=45, ha="right")
    plt.yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run():
    cfg = CFG()
    ensure_dirs(cfg.figures_dir)

    ckpt_path = os.path.join(cfg.out_dir, "best.pt")
    if not os.path.exists(ckpt_path):
        raise RuntimeError("No checkpoint found. Train first: python -m src.train")

    filepaths, labels = list_wavs_by_label(cfg.data_dir)

    # rebuild splits exactly the same way (seeded stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.seed, stratify=labels
    )
    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test,
        random_state=cfg.seed, stratify=y_temp
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    test_ds = EmotionDataset(X_test, y_test, label2id, cfg.sample_rate, cfg.duration_sec,
                            cfg.n_mfcc, cfg.n_fft, cfg.hop_length)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MFCC_CNN(num_classes=len(label2id)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    preds, true = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            p = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(p.tolist())
            true.extend(yb.numpy().tolist())

    target_names = [id2label[i] for i in range(len(label2id))]
    print(classification_report(true, preds, target_names=target_names, digits=4))

    cm = confusion_matrix(true, preds)
    out_path = os.path.join(cfg.figures_dir, "confusion_matrix.png")
    plot_confusion(cm, target_names, out_path)
    print(f"Saved confusion matrix to: {out_path}")

if __name__ == "__main__":
    run()
