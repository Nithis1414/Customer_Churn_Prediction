"""
evaluate.py
-----------
Evaluates all trained models on the held-out test set and produces:

Console output
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC per model
  - Full classification report

Plots saved to plots/
  - eval_confusion_matrix_<model>.png   — one per model
  - eval_roc_curves.png                 — all models on one chart
  - eval_model_comparison.png           — bar chart of key metrics

Key ML concepts demonstrated
  - Confusion Matrix   (TP, FP, TN, FN)
  - Precision          (of all predicted positives, how many are correct?)
  - Recall / Sensitivity (of all actual positives, how many did we catch?)
  - F1-Score           (harmonic mean of precision and recall)
  - ROC-AUC            (area under the receiver operating characteristic curve)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

PLOTS_DIR = "plots"
sns.set_theme(style="whitegrid")


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy" : accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall"   : recall_score(y_true, y_pred, zero_division=0),
        "F1-Score" : f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_true, y_prob),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """
    Annotated heatmap showing:
      TN (top-left)  FP (top-right)
      FN (bot-left)  TP (bot-right)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Pred: Retained", "Pred: Churned"],
        yticklabels=["Actual: Retained", "Actual: Churned"],
        linewidths=1, linecolor="white",
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )

    ax.set_title(f"Confusion Matrix — {model_name}", fontweight="bold", fontsize=13)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")

    # Explanation text below plot
    info = (
        f"TN={tn}  FP={fp}  FN={fn}  TP={tp}\n"
        f"Precision={tp/(tp+fp+1e-9):.3f}  "
        f"Recall={tp/(tp+fn+1e-9):.3f}"
    )
    fig.text(0.5, -0.04, info, ha="center", fontsize=9, color="gray")

    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, f"eval_confusion_matrix_{model_name}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Confusion matrix saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# ROC curves (all models on one chart)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(roc_data: dict) -> None:
    """
    roc_data = {model_name: (fpr, tpr, auc_score)}
    """
    plt.figure(figsize=(8, 6))

    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
        plt.plot(fpr, tpr, color=color, lw=2, label=f"{name}  (AUC={auc_val:.3f})")

    plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    plt.xlabel("False Positive Rate", fontweight="bold")
    plt.ylabel("True Positive Rate (Recall)", fontweight="bold")
    plt.title("ROC Curves — All Models", fontweight="bold", fontsize=13)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(True, alpha=0.3)

    path = os.path.join(PLOTS_DIR, "eval_roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] ROC curve plot saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Model comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_model_comparison(results_df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

    x      = np.arange(len(metrics))
    width  = 0.18
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (_, row) in enumerate(results_df.iterrows()):
        offset = (i - len(results_df) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, [row[m] for m in metrics],
                        width=width, label=row["Model"],
                        color=colors[i % len(colors)], edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight="bold")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontweight="bold")
    ax.set_title("Model Performance Comparison", fontweight="bold", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, alpha=0.6)

    path = os.path.join(PLOTS_DIR, "eval_model_comparison.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate] Model comparison chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Runs evaluation for every trained model.

    Returns
    -------
    results_df : DataFrame with one row per model and columns per metric
    """
    os.makedirs(PLOTS_DIR, exist_ok=True)

    all_metrics = []
    roc_data    = {}

    print("=" * 60)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]   # probability of churn

        metrics = compute_metrics(y_test, y_pred, y_prob)
        all_metrics.append({"Model": name, **metrics})

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr, metrics["ROC-AUC"])

        # ── Console output ────────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        for metric, val in metrics.items():
            print(f"  {metric:<12}: {val:.4f}")
        print(f"\n  Classification Report:\n")
        print(classification_report(
            y_test, y_pred,
            target_names=["Retained (0)", "Churned (1)"],
            digits=4,
        ))

        # ── Individual confusion matrix ───────────────────────────────────────
        plot_confusion_matrix(y_test, y_pred, name)

    # ── Combined plots ────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)
    plot_roc_curves(roc_data)
    plot_model_comparison(results_df)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(results_df.set_index("Model").round(4).to_string())

    best = results_df.loc[results_df["F1-Score"].idxmax(), "Model"]
    print(f"\n  Best model by F1-Score: ★ {best}\n")

    return results_df


if __name__ == "__main__":
    import pandas as pd
    from feature_engineering import prepare_data
    from train_models import train_all

    df = pd.read_csv("data/churn.csv")
    X_train, X_test, y_train, y_test, feat_names, arts = prepare_data(df)
    trained, _ = train_all(X_train, y_train)
    evaluate_all(trained, X_test, np.array(y_test))
