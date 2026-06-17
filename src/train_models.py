"""
train_models.py
---------------
Trains four classifiers on the preprocessed churn data:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost

Each model is:
  - Trained with cross-validation to report stable CV accuracy
  - Saved to models/<model_name>.pkl using joblib
  - Returned in a dict for downstream evaluation

Hyperparameters are kept simple but realistic.
"""

import os
import time
import joblib
import numpy as np

from sklearn.linear_model      import LogisticRegression
from sklearn.tree              import DecisionTreeClassifier
from sklearn.ensemble          import RandomForestClassifier
from sklearn.model_selection   import StratifiedKFold, cross_val_score
from xgboost                   import XGBClassifier

MODELS_DIR = "models"
SEED       = 42
CV_FOLDS   = 5


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────

def build_models() -> dict:
    """
    Returns a dictionary of {name: unfitted_estimator}.

    Logistic Regression
      - Baseline linear model
      - max_iter=1000 ensures convergence on scaled data

    Decision Tree
      - Interpretable; prone to overfitting → max_depth limits this

    Random Forest
      - Ensemble of trees; robust, lower variance than single tree
      - n_estimators=200 for stable predictions

    XGBoost
      - Gradient boosting; typically best performer on tabular data
      - scale_pos_weight handles any residual class imbalance
      - eval_metric='logloss' suppresses deprecation warning
    """
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=SEED,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=SEED,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        ),
    }
    return models


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_all(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: int = CV_FOLDS,
) -> dict:
    """
    Trains every model, runs stratified k-fold CV, saves to disk.

    Returns
    -------
    trained_models : dict  {name: fitted_estimator}
    cv_results     : dict  {name: {"mean": float, "std": float}}
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models         = build_models()
    trained_models = {}
    cv_results     = {}

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SEED)

    print("=" * 60)
    print(f"TRAINING  ({cv}-Fold Stratified CV)")
    print("=" * 60)

    for name, model in models.items():
        start = time.time()

        # ── Cross-validation ────────────────────────────────────────────────
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=skf, scoring="f1", n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std  = cv_scores.std()

        # ── Fit on full training set ────────────────────────────────────────
        model.fit(X_train, y_train)
        elapsed = time.time() - start

        # ── Persist to disk ─────────────────────────────────────────────────
        model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
        joblib.dump(model, model_path)

        trained_models[name] = model
        cv_results[name]     = {"mean": cv_mean, "std": cv_std}

        print(f"\n  {name}")
        print(f"    CV F1 : {cv_mean:.4f}  (±{cv_std:.4f})")
        print(f"    Time  : {elapsed:.2f}s")
        print(f"    Saved : {model_path}")

    print("\n[train] All models trained and saved.\n")
    return trained_models, cv_results


def load_model(name: str):
    """Load a saved model from models/<name>.pkl."""
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved model at {path}. Run training first.")
    return joblib.load(path)


if __name__ == "__main__":
    import pandas as pd
    from feature_engineering import prepare_data

    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, feat_names, arts = prepare_data(df)
    trained, cv_res = train_all(X_train, y_train)
    print("CV Results Summary:")
    for name, res in cv_res.items():
        print(f"  {name:<22} F1: {res['mean']:.4f} ± {res['std']:.4f}")
