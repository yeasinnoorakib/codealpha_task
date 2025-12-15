import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .config import CFG
from .utils import seed_everything, ensure_dirs
from .preprocess import list_wavs_by_label
from .dataset import EmotionDataset
from .model import MFCC_CNN
def run():
    cfg = CFG()
    seed_everything(cfg.seed)
    ensure_dirs(cfg.out_dir, cfg.figures_dir)

    filepaths, labels = list_wavs_by_label(cfg.data_dir)
    if len(filepaths) == 0:
        raise RuntimeError(
            f"No .wav files found. Put your data like: {cfg.data_dir}/happy/*.wav, {cfg.data_dir}/angry/*.wav"
        )

    unique_labels = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique_labels)}
    num_classes = len(unique_labels)

    X_train, X_temp, y_train, y_temp = train_test_split(
        filepaths, labels, test_size=cfg.val_ratio + cfg.test_ratio,
        random_state=cfg.seed, stratify=labels
    )
    rel_test = cfg.test_ratio / (cfg.val_ratio + cfg.test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test,
        random_state=cfg.seed, stratify=y_temp
    )

    train_ds = EmotionDataset(X_train, y_train, label2id, cfg.sample_rate, cfg.duration_sec,
                             cfg.n_mfcc, cfg.n_fft, cfg.hop_length)
    val_ds = EmotionDataset(X_val, y_val, label2id, cfg.sample_rate, cfg.duration_sec,
                           cfg.n_mfcc, cfg.n_fft, cfg.hop_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MFCC_CNN(num_classes=num_classes).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_path = os.path.join(cfg.out_dir, "best.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs} [train]"):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        model.eval()
        val_preds, val_true = [], []
        val_losses = []
        with torch.no_grad():
            for xb, yb in tqdm(val_loader, desc=f"Epoch {epoch}/{cfg.epochs} [val]"):
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_true.extend(yb.cpu().numpy().tolist())

        val_acc = accuracy_score(val_true, val_preds)
        val_f1 = f1_score(val_true, val_preds, average="macro")
        print(
            f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f} "
            f"val_loss={np.mean(val_losses):.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {"model_state": model.state_dict(), "label2id": label2id, "cfg": cfg.__dict__},
                best_path
            )
            print(f"  Saved best checkpoint to {best_path}")

    print("Training done.")
    print("Next: python -m src.evaluate")

if __name__ == "__main__":
    run()