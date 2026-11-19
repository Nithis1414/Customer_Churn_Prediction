"""
main.py
-------
End-to-end Customer Churn Prediction Pipeline

Steps
─────
1. Generate synthetic dataset  (src/generate_data.py)
2. Exploratory Data Analysis   (src/eda.py)
3. Feature engineering:
     • Label encoding
     • Train/test split (80/20, stratified)
     • Standard scaling
     • SMOTE oversampling on training set
     • SelectKBest feature selection
4. Train models with 5-fold CV:
     • Logistic Regression
     • Decision Tree
     • Random Forest
     • XGBoost
5. Evaluate on held-out test set:
     • Confusion Matrix, Precision, Recall, F1, ROC-AUC
     • Save all plots to plots/

Usage
─────
    # From the project root:
    python main.py

    # Skip data generation (use existing data/churn.csv):
    python main.py --skip-gen

    # Skip EDA (faster iteration):
    python main.py --skip-eda
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

# ── Make src/ importable when running from project root ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_data       import generate_churn_data, save_data
from eda                 import run_eda
from feature_engineering import prepare_data
from train_models        import train_all
from evaluate            import evaluate_all


# ─────────────────────────────────────────────────────────────────────────────
# CLI args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction Pipeline")
    parser.add_argument("--skip-gen",  action="store_true", help="Skip dataset generation")
    parser.add_argument("--skip-eda",  action="store_true", help="Skip EDA plots")
    parser.add_argument("--no-smote",  action="store_true", help="Disable SMOTE balancing")
    parser.add_argument("--no-select", action="store_true", help="Disable feature selection")
    parser.add_argument("--data",      type=str, default="data/churn.csv",
                        help="Path to any CSV dataset (default: data/churn.csv)")
    parser.add_argument("--target",    type=str, default="Churn",
                        help="Name of the target column (default: Churn)")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  CUSTOMER CHURN PREDICTION  —  ML PIPELINE".center(58) + "║")
    print("╚" + "═" * 58 + "╝\n")

    # ── Step 1: Data Generation ───────────────────────────────────────────────
    data_path = args.data
    if not args.skip_gen and data_path == "data/churn.csv":
        print("━━━  STEP 1: DATA GENERATION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        df = generate_churn_data(n_samples=5_000, churn_rate=0.20)
        save_data(df, path=data_path)
    else:
        if not args.skip_gen:
            print(f"[main] Using custom dataset: {data_path}\n")
        else:
            print(f"[main] Skipping data generation — loading: {data_path}\n")
        df = pd.read_csv(data_path)

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    if not args.skip_eda:
        print("━━━  STEP 2: EXPLORATORY DATA ANALYSIS  ━━━━━━━━━━━━━━━━━━\n")
        run_eda(data_path=data_path)
    else:
        print("[main] Skipping EDA.\n")

    # ── Step 3: Feature Engineering ───────────────────────────────────────────
    print("━━━  STEP 3: FEATURE ENGINEERING  ━━━━━━━━━━━━━━━━━━━━━━━━\n")
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test, feat_names, artifacts = prepare_data(
        df,
        target_col   = args.target,
        use_smote    = not args.no_smote,
        use_selection= not args.no_select,
    )
    print(f"[main] Training features : {feat_names}")
    print(f"[main] X_train : {X_train.shape}  |  X_test : {X_test.shape}\n")

    # ── Step 4: Model Training ────────────────────────────────────────────────
    print("━━━  STEP 4: MODEL TRAINING  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    trained_models, cv_results = train_all(X_train, y_train)

    # ── Step 5: Evaluation ────────────────────────────────────────────────────
    print("━━━  STEP 5: EVALUATION  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    results_df = evaluate_all(trained_models, X_test, np.array(y_test))

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + "  PIPELINE COMPLETE".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print("\nArtifacts generated:")
    print("  data/   → churn.csv")
    print("  models/ → LogisticRegression.pkl, DecisionTree.pkl,")
    print("            RandomForest.pkl, XGBoost.pkl")
    print("  plots/  → eda_*.png, feature_importance.png,")
    print("            eval_confusion_matrix_*.png,")
    print("            eval_roc_curves.png,")
    print("            eval_model_comparison.png\n")


if __name__ == "__main__":
    main()
