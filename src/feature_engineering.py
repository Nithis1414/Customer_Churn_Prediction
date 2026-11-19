"""
feature_engineering.py
-----------------------
Fully dataset-independent preprocessing pipeline.

Works with ANY CSV that has:
  - A binary target column (default name: "Churn", configurable)
  - Any mix of numeric and categorical columns

Steps
─────
1. Auto-detect and drop ID columns
2. Auto-detect and encode ALL categorical columns (LabelEncoder)
3. Train / test split (stratified)
4. StandardScaler on numeric features
5. SMOTE balancing (training only)
6. SelectKBest feature selection (top-k, auto-adjusted)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection   import train_test_split
from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling    import SMOTE

PLOTS_DIR = "plots"
SEED      = 42
TEST_SIZE = 0.20
TOP_K     = 8       # max features to keep; auto-reduced if dataset has fewer


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Auto-detect and drop ID-like columns
# ─────────────────────────────────────────────────────────────────────────────

def drop_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that look like identifiers — high-cardinality strings
    where almost every value is unique (>90% unique values).
    Also drops any column whose name contains 'id' or 'ID' (case-insensitive).
    """
    df = df.copy()
    cols_to_drop = []

    for col in df.columns:
        # Name-based check
        if "id" in col.lower() and col.lower() != "churn":
            cols_to_drop.append(col)
            continue
        # Cardinality check — string column where >90% values are unique
        if df[col].dtype == object:
            uniqueness = df[col].nunique() / len(df)
            if uniqueness > 0.90:
                cols_to_drop.append(col)

    if cols_to_drop:
        print(f"[feature_eng] Dropping ID/high-cardinality columns: {cols_to_drop}")
        df.drop(columns=cols_to_drop, inplace=True)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Auto-encode ALL categorical columns
# ─────────────────────────────────────────────────────────────────────────────

def encode_all_categoricals(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Finds every non-numeric column (except the target) and label-encodes it.
    Also handles Yes/No → 1/0 and True/False → 1/0 automatically.
    """
    df = df.copy()
    encoded_cols = []

    for col in df.columns:
        if col == target_col:
            continue

        if df[col].dtype == object or str(df[col].dtype) == "category":
            # Try Yes/No first
            unique_vals = df[col].str.strip().str.lower().unique() if df[col].dtype == object else df[col].unique()
            yes_no = {"yes", "no", "true", "false", "y", "n"}

            if set(str(v).lower() for v in unique_vals if pd.notna(v)) <= yes_no:
                df[col] = df[col].str.strip().str.lower().map(
                    {"yes": 1, "no": 0, "true": 1, "false": 0, "y": 1, "n": 0}
                ).fillna(0).astype(int)
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

            encoded_cols.append(col)

    # Also encode target if it is Yes/No text
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.lower().map(
            {"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0, "churn": 1}
        ).fillna(0).astype(int)

    if encoded_cols:
        print(f"[feature_eng] Encoded categorical columns: {encoded_cols}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Coerce remaining columns to numeric, drop unparseable
# ─────────────────────────────────────────────────────────────────────────────

def coerce_numeric(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """Force all feature columns to numeric. Drop rows with NaN."""
    df = df.copy()
    for col in df.columns:
        if col != target_col:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    before = len(df)
    df.dropna(inplace=True)
    dropped = before - len(df)
    if dropped:
        print(f"[feature_eng] Dropped {dropped} rows with unparseable values.")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Train / test split
# ─────────────────────────────────────────────────────────────────────────────

def split_data(df: pd.DataFrame, target_col: str = "Churn"):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    print(f"[feature_eng] Train size : {X_train.shape[0]:,}  |  Test size : {X_test.shape[0]:,}")
    print(f"[feature_eng] Train churn rate : {y_train.mean():.2%}")
    print(f"[feature_eng] Test  churn rate : {y_test.mean():.2%}")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Scale
# ─────────────────────────────────────────────────────────────────────────────

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — SMOTE
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train: np.ndarray, y_train):
    smote = SMOTE(random_state=SEED)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    before = pd.Series(y_train).value_counts().to_dict()
    after  = pd.Series(y_res).value_counts().to_dict()
    print(f"\n[feature_eng] SMOTE applied:")
    print(f"  Before → {before}")
    print(f"  After  → {after}\n")

    return X_res, y_res


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Feature selection
# ─────────────────────────────────────────────────────────────────────────────

def select_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    feature_names: list,
    k: int = TOP_K,
):
    # Auto-adjust k if dataset has fewer features
    k = min(k, X_train.shape[1])

    selector    = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    scores         = selector.scores_
    support        = selector.get_support()
    selected_names = [feature_names[i] for i in range(len(feature_names)) if support[i]]

    print(f"[feature_eng] Top-{k} features selected:")
    for name in selected_names:
        print(f"  • {name}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(PLOTS_DIR, exist_ok=True)
    score_df = pd.DataFrame({"Feature": feature_names, "F-Score": scores}) \
                 .sort_values("F-Score", ascending=True)

    colors = ["#DD8452" if n in selected_names else "#AEC6E8"
              for n in score_df["Feature"]]

    fig, ax = plt.subplots(figsize=(max(6, len(feature_names) * 0.5), 6))
    bars = ax.barh(score_df["Feature"], score_df["F-Score"],
                   color=colors, edgecolor="white")
    ax.set_xlabel("ANOVA F-Score", fontweight="bold")
    ax.set_title(f"Feature Selection — Top {k} highlighted in orange", fontweight="bold")
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
# Public pipeline — works with ANY dataset
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(
    df: pd.DataFrame,
    target_col: str = "Churn",
    use_smote: bool = True,
    use_selection: bool = True,
):
    """
    Universal preprocessing pipeline.

    Parameters
    ----------
    df           : any pandas DataFrame with a binary target column
    target_col   : name of the target column (default "Churn")
    use_smote    : apply SMOTE balancing on training set
    use_selection: apply SelectKBest feature selection

    Returns
    -------
    X_train, X_test, y_train, y_test, feature_names, artifacts
    artifacts = {scaler, selector, feature_names, target_col}
    """
    # Detect target column automatically if not found
    if target_col not in df.columns:
        # Try common alternatives
        candidates = [c for c in df.columns
                      if c.lower() in ("churn", "churned", "target",
                                       "label", "exited", "attrition")]
        if candidates:
            target_col = candidates[0]
            print(f"[feature_eng] Auto-detected target column: '{target_col}'")
        else:
            raise ValueError(
                f"Target column '{target_col}' not found. "
                f"Available columns: {df.columns.tolist()}"
            )

    print(f"\n[feature_eng] Dataset shape    : {df.shape}")
    print(f"[feature_eng] Target column    : '{target_col}'")
    print(f"[feature_eng] All columns      : {df.columns.tolist()}\n")

    df = drop_id_columns(df)
    df = encode_all_categoricals(df, target_col=target_col)
    df = coerce_numeric(df, target_col=target_col)

    X_train, X_test, y_train, y_test = split_data(df, target_col=target_col)
    feature_names = X_train.columns.tolist()

    X_train_sc, X_test_sc, scaler = scale_features(X_train, X_test)

    if use_smote:
        try:
            X_train_sc, y_train = apply_smote(X_train_sc, y_train)
        except Exception as e:
            print(f"[feature_eng] SMOTE skipped: {e}")

    selector = None
    if use_selection:
        X_train_sc, X_test_sc, feature_names, selector = select_features(
            X_train_sc, X_test_sc, y_train, feature_names
        )

    artifacts = {
        "scaler"       : scaler,
        "selector"     : selector,
        "feature_names": feature_names,
        "target_col"   : target_col,
    }

    return X_train_sc, X_test_sc, y_train, y_test, feature_names, artifacts


if __name__ == "__main__":
    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, feat_names, arts = prepare_data(df)
    print(f"Final X_train shape : {X_train.shape}")
    print(f"Final X_test  shape : {X_test.shape}")
    print(f"Features used       : {feat_names}")
