import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

from .config import CFG
from .utils import ensure_dirs
from .load_data import load_dataset

def plot_confusion(cm, out_path):
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def run():
    cfg = CFG()
    ensure_dirs(cfg.figures_dir)

    model_path = os.path.join(cfg.out_dir, "best_model.joblib")
    if not os.path.exists(model_path):
        raise RuntimeError("No trained model found. Train first: python -m src.train")

    pack = joblib.load(model_path)
    pipe = pack["pipeline"]
    model_name = pack["model_name"]

    df = load_dataset(cfg)
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]

    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(f"Model: {model_name}")
    print(classification_report(y, pred, digits=4))

    cm = confusion_matrix(y, pred)
    cm_path = os.path.join(cfg.figures_dir, "confusion_matrix.png")
    plot_confusion(cm, cm_path)
    print(f"Saved confusion matrix to: {cm_path}")

    roc_path = os.path.join(cfg.figures_dir, "roc_curve.png")
    plot_roc(y, proba, roc_path)
    print(f"Saved ROC curve to: {roc_path}")

if __name__ == "__main__":
    run()
