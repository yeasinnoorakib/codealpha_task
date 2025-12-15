import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from .config import CFG
from .utils import ensure_dirs
from .load_data import load_dataset

def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop"
    )

def run():
    cfg = CFG()
    ensure_dirs(cfg.out_dir, cfg.figures_dir)

    df = load_dataset(cfg)
    X = df.drop(columns=[cfg.target_col])
    y = df[cfg.target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_ratio, random_state=cfg.seed, stratify=y
    )

    preprocess = build_preprocess(X_train)

    models = {
        "logreg": LogisticRegression(max_iter=2000),
        "svm_rbf": SVC(probability=True),
        "rf": RandomForestClassifier(n_estimators=300, random_state=cfg.seed),
    }

    best_name, best_auc, best_pipe = None, -1.0, None
    results = []

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        auc = roc_auc_score(y_test, proba)
        f1 = f1_score(y_test, pred)
        acc = accuracy_score(y_test, pred)

        results.append((name, acc, f1, auc))
        print(f"{name}: acc={acc:.4f} f1={f1:.4f} roc_auc={auc:.4f}")

        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    out_path = os.path.join(cfg.out_dir, "best_model.joblib")
    joblib.dump({"model_name": best_name, "pipeline": best_pipe, "cfg": cfg.__dict__}, out_path)
    print(f"Saved best model: {best_name} (roc_auc={best_auc:.4f}) to {out_path}")

    # Save comparison CSV
    comp = pd.DataFrame(results, columns=["model", "accuracy", "f1", "roc_auc"])
    comp.to_csv(os.path.join(cfg.figures_dir, "model_comparison.csv"), index=False)
    print(f"Saved model comparison to {os.path.join(cfg.figures_dir, 'model_comparison.csv')}")

if __name__ == "__main__":
    run()
