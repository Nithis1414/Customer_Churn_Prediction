"""
feature_engineering.py
-----------------------
Handles:
  1. Preprocessing  — encode categoricals, scale numerics
  2. Class balancing — SMOTE oversampling on the training set
  3. Feature selection — SelectKBest (chi-squared / f_classif) + feature importance

Returns ready-to-train (X_train, X_test, y_train, y_test) arrays and
a feature-importance plot saved to plots/feature_importance.png.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling    import SMOTE

PLOTS_DIR  = "plots"
SEED       = 42
TEST_SIZE  = 0.20
TOP_K      = 8          # number of best features to keep


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Encode & clean
# ─────────────────────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop CustomerID (not a signal)
    - Label-encode binary / low-cardinality categoricals
    """
    df = df.copy()
    df.drop(columns=["CustomerID"], inplace=True, errors="ignore")

    # Encode Gender: Male → 1, Female → 0
    le = LabelEncoder()
    df["Gender"] = le.fit_transform(df["Gender"])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"[feature_eng] Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")
    print(f"[feature_eng] Train churn rate : {y_train.mean():.2%}")
    print(f"[feature_eng] Test  churn rate : {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Scale numeric features
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """StandardScaler fitted ONLY on train — applied to both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — SMOTE: balance the training set
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train: pd.Series):
    """
    SMOTE (Synthetic Minority Over-sampling Technique):
      - Creates synthetic minority-class (churned) examples.
      - Applied ONLY on the training split to prevent data leakage.
    """
    smote = SMOTE(random_state=SEED)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    before = pd.Series(y_train).value_counts().to_dict()
    after  = pd.Series(y_resampled).value_counts().to_dict()

    print(f"\n[feature_eng] SMOTE applied:")
    print(f"  Before → {before}")
    print(f"  After  → {after}\n")

    return X_resampled, y_resampled


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Feature selection (SelectKBest with ANOVA F-test)
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    k: int = TOP_K,
):
    """
    Ranks features by ANOVA F-score and keeps the top-k.
    Returns reduced arrays and the selected feature names.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    scores        = selector.scores_
    support       = selector.get_support()
    selected_names = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    print(f"[feature_eng] Top-{k} features selected:")
    for name in selected_names:
        print(f"  • {name}")

    # ── Plot feature scores ──────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    score_df = pd.DataFrame({"Feature": feature_names, "F-Score": scores}) \
                 .sort_values("F-Score", ascending=True)

    colors = ["#DD8452" if name in selected_names else "#AEC6E8"
              for name in score_df["Feature"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(score_df["Feature"], score_df["F-Score"], color=colors, edgecolor="white")
    ax.set_xlabel("ANOVA F-Score", fontweight="bold")
    ax.set_title(f"Feature Selection — Top {k} highlighted in orange",
                 fontweight="bold")

    # Annotate bars
    for bar, val in zip(bars, score_df["F-Score"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[feature_eng] Feature importance plot saved → {path}\n")

    return X_train_sel, X_test_sel, selected_names, selector


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline function
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, use_smote: bool = True, use_selection: bool = True):
    """
    Full preprocessing pipeline:
      encode → split → scale → (SMOTE) → (feature selection)

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, artifacts
    artifacts = dict with scaler, selector (may be None)
    """
    df_enc = encode_features(df)
    X_train, X_test, y_train, y_test = split_data(df_enc)

    feature_names = X_train.columns.tolist()

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    if use_smote:
        X_train_sc, y_train = apply_smote(X_train_sc, y_train)

    selector = None
    if use_selection:
        X_train_sc, X_test_sc, feature_names, selector = select_features(
            X_train_sc, X_test_sc, y_train, feature_names
        )

    artifacts = {"scaler": scaler, "selector": selector, "feature_names": feature_names}
    return X_train_sc, X_test_sc, y_train, y_test, feature_names, artifacts


if __name__ == "__main__":
    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, feat_names, arts = prepare_data(df)
    print(f"Final X_train shape : {X_train.shape}")
    print(f"Final X_test  shape : {X_test.shape}")
