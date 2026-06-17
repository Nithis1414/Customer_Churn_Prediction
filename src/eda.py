"""
eda.py
------
Exploratory Data Analysis for the customer churn dataset.

Outputs:
  - Console summary (shape, dtypes, missing values, class balance)
  - plots/eda_class_balance.png
  - plots/eda_numeric_distributions.png
  - plots/eda_correlation_heatmap.png
  - plots/eda_churn_vs_features.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for scripts
import seaborn as sns

PLOTS_DIR = "plots"
DATA_PATH = "data/churn.csv"

sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_fig(filename: str) -> None:
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[eda] Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# EDA steps
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    """Print basic dataset statistics to console."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape}")
    print(f"Memory usage   : {df.memory_usage(deep=True).sum() / 1024:.1f} KB\n")

    print("── Data Types ──────────────────────────────────────────────")
    print(df.dtypes.to_string())

    print("\n── Missing Values ──────────────────────────────────────────")
    missing = df.isnull().sum()
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "  None found ✓")

    print("\n── Numeric Summary ─────────────────────────────────────────")
    print(df.describe(include=[np.number]).T.to_string())

    print("\n── Target Class Balance ────────────────────────────────────")
    vc = df["Churn"].value_counts()
    vcp = df["Churn"].value_counts(normalize=True) * 100
    print(f"  0 (Retained) : {vc[0]:>5}  ({vcp[0]:.1f}%)")
    print(f"  1 (Churned)  : {vc[1]:>5}  ({vcp[1]:.1f}%)")
    print(f"  Imbalance ratio: {vc[0]/vc[1]:.2f}:1\n")


def plot_class_balance(df: pd.DataFrame) -> None:
    """Bar chart showing churn vs retained counts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    counts = df["Churn"].value_counts()
    labels = ["Retained (0)", "Churned (1)"]
    colors = ["#4C72B0", "#DD8452"]

    # Count bar
    axes[0].bar(labels, counts.values, color=colors, edgecolor="white", width=0.5)
    axes[0].set_title("Class Distribution — Counts", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 30, str(v), ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(
        counts.values, labels=labels, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    axes[1].set_title("Class Distribution — Proportion", fontweight="bold")

    fig.suptitle("Target Variable: Churn Distribution", fontsize=14, fontweight="bold", y=1.02)
    save_fig("eda_class_balance.png")


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """Histogram + KDE for every numeric feature, coloured by churn label."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Churn"]

    n_cols = 3
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
            subset = df[df["Churn"] == label][col]
            axes[i].hist(subset, bins=30, alpha=0.5, color=color,
                         label=f"{'Retained' if label==0 else 'Churned'}", density=True)
        axes[i].set_title(col, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")
        axes[i].legend(fontsize=8)

    # Hide any unused subplots
    for j in range(len(numeric_cols), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Feature Distributions by Churn Label",
                 fontsize=14, fontweight="bold", y=1.01)
    save_fig("eda_numeric_distributions.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations for numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))   # show lower triangle only
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, square=True,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Pearson Correlation Heatmap", fontsize=14, fontweight="bold")
    save_fig("eda_correlation_heatmap.png")


def plot_churn_vs_features(df: pd.DataFrame) -> None:
    """Box plots: churn group vs key continuous features."""
    key_features = ["Tenure", "MonthlyCharges", "SatisfactionScore", "ComplaintCount"]
    fig, axes = plt.subplots(1, len(key_features), figsize=(16, 5))

    for ax, feat in zip(axes, key_features):
        data_by_label = [
            df[df["Churn"] == 0][feat].values,
            df[df["Churn"] == 1][feat].values,
        ]
        bp = ax.boxplot(data_by_label, patch_artist=True,
                        tick_labels=["Retained", "Churned"],
                        medianprops={"color": "black", "linewidth": 2})
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontweight="bold")
        ax.set_ylabel(feat)

    fig.suptitle("Key Features vs Churn", fontsize=14, fontweight="bold")
    save_fig("eda_churn_vs_features.png")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_eda(data_path: str = DATA_PATH) -> pd.DataFrame:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    df = pd.read_csv(data_path)

    print_summary(df)
    plot_class_balance(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    plot_churn_vs_features(df)

    print("[eda] EDA complete.\n")
    return df


if __name__ == "__main__":
    run_eda()
