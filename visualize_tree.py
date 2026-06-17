"""
visualize_tree.py
-----------------
Loads the trained Decision Tree and renders it as:
  1. A text flowchart (console)
  2. plots/decision_tree_visual.png  (graphical)

Run from project root:
    python visualize_tree.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

# ── Load artifacts ────────────────────────────────────────────────────────────
model = joblib.load("models/DecisionTree.pkl")

# Feature names saved by the pipeline (top-8 selected)
feature_names = [
    "Age", "Tenure", "MonthlyCharges", "TotalCharges",
    "HasCreditCard", "IsActiveMember", "ComplaintCount", "SatisfactionScore"
]
class_names = ["Retained", "Churned"]

# ── Text representation ───────────────────────────────────────────────────────
print("=" * 60)
print("DECISION TREE — TEXT REPRESENTATION (depth ≤ 3)")
print("=" * 60)
text_tree = export_text(model, feature_names=feature_names, max_depth=3)
print(text_tree)

# ── Graphical representation ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(28, 10))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=class_names,
    filled=True,           # colour nodes by majority class
    rounded=True,
    max_depth=3,           # show top 3 levels only (readable)
    fontsize=9,
    ax=ax,
    impurity=True,
    proportion=False,
)
ax.set_title("Decision Tree — Customer Churn (top 3 levels)", fontsize=14, fontweight="bold")
plt.tight_layout()

os.makedirs("plots", exist_ok=True)
out = "plots/decision_tree_visual.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Tree plot saved → {out}")
