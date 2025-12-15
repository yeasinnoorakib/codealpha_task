import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from .config import CFG
from .utils import ensure_dirs

def run():
    cfg = CFG()
    ensure_dirs(cfg.figures_dir)

    model_path = os.path.join(cfg.out_dir, "best_model.joblib")
    pack = joblib.load(model_path)
    pipe = pack["pipeline"]

    model = pipe.named_steps["model"]
    preprocess = pipe.named_steps["preprocess"]

    # Get transformed feature names (works for OneHotEncoder)
    try:
        feature_names = preprocess.get_feature_names_out()
    except Exception:
        feature_names = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        idx = np.argsort(importances)[::-1][:15]
        top_imp = importances[idx]
        if feature_names is not None:
            top_names = [feature_names[i] for i in idx]
        else:
            top_names = [f"f{i}" for i in idx]

        plt.figure()
        plt.bar(range(len(top_imp)), top_imp)
        plt.title("Top Feature Importances")
        plt.xticks(range(len(top_imp)), top_names, rotation=45, ha="right")
        plt.tight_layout()
        out_path = os.path.join(cfg.figures_dir, "feature_importance.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved feature importance to: {out_path}")
    else:
        print("Best model has no feature_importances_. If LogisticRegression, you can use coef_ instead.")

if __name__ == "__main__":
    run()
