"""
dashboard.py
------------
Interactive Streamlit dashboard for the Customer Churn Prediction project.

Sections
────────
1. Overview     — dataset stats and class balance
2. EDA          — live interactive charts (filter by churn label)
3. Model Results— metrics table, ROC curves, confusion matrices
4. Predict      — enter a single customer's details → get churn probability

Run:
    streamlit run dashboard.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    confusion_matrix, roc_curve, roc_auc_score,
    accuracy_score, precision_score, recall_score, f1_score,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH     = "data/churn.csv"
MODELS_DIR    = "models"
FEATURE_NAMES = [
    "Age", "Tenure", "MonthlyCharges", "TotalCharges",
    "HasCreditCard", "IsActiveMember", "ComplaintCount", "SatisfactionScore",
]
MODEL_NAMES   = ["LogisticRegression", "DecisionTree", "RandomForest", "XGBoost"]
COLORS        = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_models():
    models = {}
    for name in MODEL_NAMES:
        path = os.path.join(MODELS_DIR, f"{name}.pkl")
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

@st.cache_resource
def load_pipeline_artifacts():
    """Re-run preprocessing to get X_test / y_test for evaluation."""
    from feature_engineering import prepare_data
    df = load_data()
    X_train, X_test, y_train, y_test, feat_names, arts = prepare_data(
        df, use_smote=True, use_selection=True
    )
    return X_test, np.array(y_test), arts["scaler"], arts["selector"]

# ═════════════════════════════════════════════════════════════════════════════
# Sidebar
# ═════════════════════════════════════════════════════════════════════════════
st.sidebar.image(
    "https://img.icons8.com/fluency/96/combo-chart.png", width=60
)
st.sidebar.title("📉 Churn Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "🔍 EDA", "📊 Model Results", "🎯 Predict Customer", "📂 Custom Dataset"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Built with Scikit-learn · XGBoost · Streamlit")

# ═════════════════════════════════════════════════════════════════════════════
# Page 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.title("Customer Churn Prediction Dashboard")
    st.markdown(
        "Predict which customers are likely to **leave** the company using ML. "
        "Use the sidebar to explore the data, compare models, or predict a single customer."
    )

    df = load_data()

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churned", f"{df['Churn'].sum():,}", delta=f"{df['Churn'].mean():.1%} rate", delta_color="inverse")
    col3.metric("Retained", f"{(df['Churn']==0).sum():,}")
    col4.metric("Features", str(df.shape[1] - 2))  # minus CustomerID and Churn

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # Class balance pie
    with col_l:
        st.subheader("Class Balance")
        counts = df["Churn"].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie(
            counts.values,
            labels=["Retained", "Churned"],
            colors=["#4C72B0", "#DD8452"],
            autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        ax.set_title("Churn Distribution", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    # Raw data preview
    with col_r:
        st.subheader("Raw Data Sample")
        st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Descriptive Statistics")
    st.dataframe(df.describe(include=[np.number]).T.round(2), use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# Page 2 — EDA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 EDA":
    st.title("Exploratory Data Analysis")
    df = load_data()

    # Filter by churn
    st.sidebar.subheader("EDA Filters")
    show_group = st.sidebar.radio("Show customers", ["All", "Retained only", "Churned only"])
    if show_group == "Retained only":
        df_view = df[df["Churn"] == 0]
    elif show_group == "Churned only":
        df_view = df[df["Churn"] == 1]
    else:
        df_view = df

    numeric_cols = ["Age", "Tenure", "MonthlyCharges", "TotalCharges",
                    "ComplaintCount", "SatisfactionScore"]

    # Feature distribution
    st.subheader("Feature Distribution")
    selected_feat = st.selectbox("Select feature", numeric_cols)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Histogram split by churn
    for label, color, lname in [(0, "#4C72B0", "Retained"), (1, "#DD8452", "Churned")]:
        axes[0].hist(df[df["Churn"] == label][selected_feat],
                     bins=30, alpha=0.6, color=color, label=lname, density=True)
    axes[0].set_title(f"{selected_feat} — by Churn Label", fontweight="bold")
    axes[0].legend()
    axes[0].set_xlabel(selected_feat)
    axes[0].set_ylabel("Density")

    # Box plot
    data_groups = [df[df["Churn"] == 0][selected_feat].values,
                   df[df["Churn"] == 1][selected_feat].values]
    bp = axes[1].boxplot(data_groups, patch_artist=True,
                         tick_labels=["Retained", "Churned"],
                         medianprops={"color": "black", "linewidth": 2})
    for patch, color in zip(bp["boxes"], ["#4C72B0", "#DD8452"]):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    axes[1].set_title(f"{selected_feat} — Box Plot", fontweight="bold")
    axes[1].set_ylabel(selected_feat)

    st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 7))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, linewidths=0.5, ax=ax2, cbar_kws={"shrink": 0.8})
    ax2.set_title("Pearson Correlation", fontweight="bold")
    st.pyplot(fig2); plt.close()

    # Scatter plot
    st.subheader("Scatter Plot")
    c1, c2 = st.columns(2)
    x_feat = c1.selectbox("X axis", numeric_cols, index=1)  # Tenure
    y_feat = c2.selectbox("Y axis", numeric_cols, index=2)  # MonthlyCharges

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    for label, color, lname in [(0, "#4C72B0", "Retained"), (1, "#DD8452", "Churned")]:
        subset = df_view[df_view["Churn"] == label] if show_group == "All" else df_view
        ax3.scatter(df[df["Churn"] == label][x_feat],
                    df[df["Churn"] == label][y_feat],
                    alpha=0.3, s=15, color=color, label=lname)
    ax3.set_xlabel(x_feat, fontweight="bold")
    ax3.set_ylabel(y_feat, fontweight="bold")
    ax3.set_title(f"{x_feat} vs {y_feat}", fontweight="bold")
    ax3.legend()
    st.pyplot(fig3); plt.close()

# ═════════════════════════════════════════════════════════════════════════════
# Page 3 — Model Results
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Results":
    st.title("Model Evaluation Results")

    models = load_models()
    if not models:
        st.error("No trained models found. Run `python main.py` first.")
        st.stop()

    X_test, y_test, scaler, selector = load_pipeline_artifacts()

    # Compute metrics for all models
    rows, roc_data = [], {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model"    : name,
            "Accuracy" : round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1-Score" : round(f1_score(y_test, y_pred, zero_division=0), 4),
            "ROC-AUC"  : round(roc_auc_score(y_test, y_prob), 4),
        })
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)

    results_df = pd.DataFrame(rows).set_index("Model")

    # Highlight best per column
    st.subheader("Metrics Table")
    st.dataframe(
        results_df.style.highlight_max(axis=0, color="#d4edda"),
        use_container_width=True,
    )

    st.markdown("---")
    col_l, col_r = st.columns(2)

    # ROC curves
    with col_l:
        st.subheader("ROC Curves")
        fig, ax = plt.subplots(figsize=(6, 5))
        for (name, (fpr, tpr)), color in zip(roc_data.items(), COLORS):
            auc = results_df.loc[name, "ROC-AUC"]
            ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves — All Models", fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        st.pyplot(fig); plt.close()

    # Model comparison bar
    with col_r:
        st.subheader("Model Comparison")
        metrics   = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        x         = np.arange(len(metrics))
        width     = 0.18
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        for i, (model_name, row) in enumerate(results_df.iterrows()):
            offset = (i - len(results_df) / 2 + 0.5) * width
            ax2.bar(x + offset, [row[m] for m in metrics], width=width,
                    label=model_name, color=COLORS[i], edgecolor="white")
        ax2.set_xticks(x); ax2.set_xticklabels(metrics, fontweight="bold")
        ax2.set_ylim(0, 1.1); ax2.set_ylabel("Score")
        ax2.set_title("Performance Comparison", fontweight="bold")
        ax2.legend(fontsize=8); ax2.axhline(0.5, ls="--", color="gray", alpha=0.5)
        st.pyplot(fig2); plt.close()

    # Confusion matrix — selectable
    st.markdown("---")
    st.subheader("Confusion Matrix")
    selected_model = st.selectbox("Select model", list(models.keys()))
    model = models[selected_model]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                xticklabels=["Pred: Retained", "Pred: Churned"],
                yticklabels=["Actual: Retained", "Actual: Churned"],
                linewidths=1, annot_kws={"size": 14, "weight": "bold"})
    ax3.set_title(f"Confusion Matrix — {selected_model}", fontweight="bold")
    st.pyplot(fig3); plt.close()

    # Metric explainer below
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("True Negatives (TN)",  f"{tn}", help="Correctly predicted: Retained")
    c2.metric("False Positives (FP)", f"{fp}", help="Predicted Churned but actually Retained")
    c3.metric("False Negatives (FN)", f"{fn}", help="Missed churners — most costly!")
    c4.metric("True Positives (TP)",  f"{tp}", help="Correctly predicted: Churned")

# ═════════════════════════════════════════════════════════════════════════════
# Page 4 — Predict single customer
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Predict Customer":
    st.title("Predict Churn for a Single Customer")
    st.markdown("Adjust the sliders and dropdowns below, then click **Predict**.")

    models = load_models()
    if not models:
        st.error("No trained models found. Run `python main.py` first.")
        st.stop()

    _, _, scaler, selector = load_pipeline_artifacts()

    # ── Input form ────────────────────────────────────────────────────────────
    with st.form("predict_form"):
        st.subheader("Customer Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            age              = st.slider("Age", 18, 70, 35)
            gender           = st.selectbox("Gender", ["Male", "Female"])
            tenure           = st.slider("Tenure (months)", 1, 72, 24)
            num_products     = st.slider("Num Products", 1, 4, 2)

        with col2:
            monthly_charges  = st.slider("Monthly Charges ($)", 20.0, 120.0, 65.0, step=0.5)
            total_charges    = st.slider("Total Charges ($)", 0.0, 8500.0,
                                         float(monthly_charges * tenure), step=10.0)
            has_credit_card  = st.selectbox("Has Credit Card", [1, 0],
                                             format_func=lambda x: "Yes" if x else "No")

        with col3:
            is_active        = st.selectbox("Is Active Member", [1, 0],
                                             format_func=lambda x: "Yes" if x else "No")
            complaints       = st.slider("Complaint Count", 0, 6, 1)
            satisfaction     = st.slider("Satisfaction Score (1–10)", 1, 10, 6)

        selected_model_name = st.selectbox("Model to use", list(models.keys()))
        submitted = st.form_submit_button("🔮 Predict", use_container_width=True)

    if submitted:
        # Build raw feature vector — must match the 10 columns the scaler was
        # fitted on (post-encoding, pre-selection):
        # Age, Gender, Tenure, MonthlyCharges, TotalCharges, NumProducts,
        # HasCreditCard, IsActiveMember, ComplaintCount, SatisfactionScore
        gender_enc = 1 if gender == "Male" else 0
        raw = np.array([[age, gender_enc, tenure, monthly_charges, total_charges,
                         num_products, has_credit_card, is_active,
                         complaints, satisfaction]],
                       dtype=float)

        # Scale
        raw_scaled = scaler.transform(raw)

        # Feature selection (if selector was fitted)
        if selector is not None:
            raw_final = selector.transform(raw_scaled)
        else:
            raw_final = raw_scaled

        # Predict
        model     = models[selected_model_name]
        pred      = model.predict(raw_final)[0]
        prob      = model.predict_proba(raw_final)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        # Big result card
        if pred == 1:
            st.error(f"⚠️  **This customer is likely to CHURN**")
        else:
            st.success(f"✅  **This customer is likely to STAY**")

        # Gauge-style probability bar
        col_a, col_b = st.columns([1, 2])
        col_a.metric("Churn Probability", f"{prob:.1%}")
        col_b.progress(float(prob), text=f"Risk level: {prob:.1%}")

        # Probability breakdown
        st.markdown("---")
        st.subheader("All Models — Churn Probability")
        prob_rows = []
        for mname, mobj in models.items():
            p = mobj.predict_proba(raw_final)[0][1]
            prob_rows.append({"Model": mname, "Churn Probability": round(p, 4),
                              "Prediction": "Churn" if p >= 0.5 else "Stay"})

        prob_df = pd.DataFrame(prob_rows).set_index("Model")
        st.dataframe(
            prob_df.style
                   .background_gradient(subset=["Churn Probability"], cmap="RdYlGn_r")
                   .format({"Churn Probability": "{:.1%}"}),
            use_container_width=True,
        )

        # Visual bar chart of probabilities
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.barh(prob_df.index, prob_df["Churn Probability"],
                       color=["#DD8452" if p >= 0.5 else "#4C72B0"
                               for p in prob_df["Churn Probability"]],
                       edgecolor="white")
        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Decision threshold (0.5)")
        ax.set_xlim(0, 1)
        ax.set_xlabel("Churn Probability")
        ax.set_title("Churn Probability by Model", fontweight="bold")
        ax.legend(fontsize=9)
        for bar, val in zip(bars, prob_df["Churn Probability"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=10)
        st.pyplot(fig); plt.close()

        # Feature summary
        st.markdown("---")
        st.subheader("Input Summary")
        input_summary = {
            "Age": age, "Gender": gender, "Tenure (months)": tenure,
            "Monthly Charges": f"${monthly_charges}", "Total Charges": f"${total_charges}",
            "Num Products": num_products, "Has Credit Card": "Yes" if has_credit_card else "No",
            "Active Member": "Yes" if is_active else "No",
            "Complaint Count": complaints, "Satisfaction Score": f"{satisfaction}/10",
        }
        st.table(pd.Series(input_summary, name="Value"))

# ═════════════════════════════════════════════════════════════════════════════
# Page 5 — Custom Dataset Upload  (no mapping — fully automatic)
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📂 Custom Dataset":
    from sklearn.model_selection   import train_test_split
    from sklearn.preprocessing     import LabelEncoder, StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from imblearn.over_sampling    import SMOTE
    from sklearn.linear_model      import LogisticRegression as LR
    from sklearn.tree              import DecisionTreeClassifier
    from sklearn.ensemble          import RandomForestClassifier
    from xgboost                   import XGBClassifier

    st.title("Upload Your Own Dataset")
    st.markdown(
        "Upload any CSV file. Select the target (churn) column. "
        "The pipeline uses **all remaining columns automatically** — "
        "no mapping required."
    )

    # ── Step 1: Upload ────────────────────────────────────────────────────────
    st.subheader("Step 1 — Upload CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("👆 Upload a CSV file to get started.")
        st.markdown("---")
        st.subheader("Popular public datasets you can try")
        st.markdown(
            "- [IBM Telco Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)\n"
            "- [Bank Customer Churn](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling)\n"
            "- [E-commerce Churn](https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction)"
        )
        st.stop()

    try:
        custom_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    st.success(f"File uploaded — {custom_df.shape[0]:,} rows, {custom_df.shape[1]} columns")
    st.dataframe(custom_df.head(5), use_container_width=True)

    # ── Step 2: Pick target column only ──────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 2 — Select Target Column")
    st.markdown("Which column contains the **Churn label** (0 = retained, 1 = churned)?")

    all_cols = custom_df.columns.tolist()

    # Auto-detect likely target column
    auto_target = all_cols[0]
    for c in all_cols:
        if c.lower() in ("churn", "churned", "target", "label", "exited", "attrition"):
            auto_target = c
            break

    target_col = st.selectbox(
        "Target column",
        all_cols,
        index=all_cols.index(auto_target),
        help="This is the column the model will learn to predict."
    )

    # Show unique values of target
    st.write(f"Unique values in `{target_col}`: {sorted(custom_df[target_col].dropna().unique().tolist())}")

    # ── Step 3: Options + Train ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 3 — Train & Evaluate")

    col_opt1, col_opt2 = st.columns(2)
    use_smote_c = col_opt1.checkbox("Apply SMOTE balancing", value=True)
    use_sel_c   = col_opt2.checkbox("Apply feature selection", value=True)

    if st.button("🚀 Train Models on This Dataset", use_container_width=True):

        # ── Auto-preprocess ───────────────────────────────────────────────────
        df_work = custom_df.copy()

        # Drop ID-like columns (high cardinality strings, not target)
        drop_cols = []
        for c in df_work.columns:
            if c == target_col:
                continue
            if "id" in c.lower():
                drop_cols.append(c)
                continue
            if df_work[c].dtype == object and df_work[c].nunique() / len(df_work) > 0.9:
                drop_cols.append(c)
        if drop_cols:
            st.write(f"Auto-dropped ID/high-cardinality columns: {drop_cols}")
            df_work.drop(columns=drop_cols, inplace=True)

        # Auto-encode target
        if df_work[target_col].dtype == object:
            mapping_vals = {"yes": 1, "no": 0, "true": 1, "false": 0,
                            "churn": 1, "no churn": 0, "1": 1, "0": 0}
            df_work[target_col] = (
                df_work[target_col].astype(str).str.strip().str.lower()
                .map(mapping_vals)
            )

        # Auto-encode all other categorical columns
        encoded_info = []
        for c in df_work.columns:
            if c == target_col:
                continue
            if df_work[c].dtype == object or str(df_work[c].dtype) == "category":
                uniq = set(str(v).strip().lower() for v in df_work[c].dropna().unique())
                if uniq <= {"yes", "no", "true", "false", "y", "n"}:
                    df_work[c] = df_work[c].astype(str).str.strip().str.lower() \
                                  .map({"yes": 1, "no": 0, "true": 1, "false": 0,
                                        "y": 1, "n": 0}).fillna(0).astype(int)
                else:
                    le = LabelEncoder()
                    df_work[c] = le.fit_transform(df_work[c].astype(str))
                encoded_info.append(c)

        # Coerce remaining to numeric, drop NaN rows
        for c in df_work.columns:
            df_work[c] = pd.to_numeric(df_work[c], errors="coerce")
        df_work.dropna(inplace=True)

        if encoded_info:
            st.write(f"Auto-encoded text columns: {encoded_info}")
        st.write(f"Clean dataset: **{len(df_work):,} rows** ready for training")

        if len(df_work) < 50:
            st.error("Not enough valid rows after cleaning (need at least 50).")
            st.stop()

        if df_work[target_col].nunique() < 2:
            st.error(f"Target column `{target_col}` must have both 0 and 1 values.")
            st.stop()

        # ── Split ─────────────────────────────────────────────────────────────
        feat_cols_c = [c for c in df_work.columns if c != target_col]
        X = df_work[feat_cols_c].values
        y = df_work[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── Scale ─────────────────────────────────────────────────────────────
        scaler_c = StandardScaler()
        X_train  = scaler_c.fit_transform(X_train)
        X_test   = scaler_c.transform(X_test)

        # ── SMOTE ─────────────────────────────────────────────────────────────
        if use_smote_c:
            try:
                sm = SMOTE(random_state=42)
                X_train, y_train = sm.fit_resample(X_train, y_train)
                st.write(f"After SMOTE — training samples: **{len(y_train):,}**")
            except Exception as e:
                st.warning(f"SMOTE skipped: {e}")

        # ── Feature selection ─────────────────────────────────────────────────
        selector_c = None
        final_feat_names = feat_cols_c
        if use_sel_c:
            k = min(8, X_train.shape[1])
            selector_c  = SelectKBest(score_func=f_classif, k=k)
            X_train     = selector_c.fit_transform(X_train, y_train)
            X_test      = selector_c.transform(X_test)
            final_feat_names = [feat_cols_c[i] for i in range(len(feat_cols_c))
                                if selector_c.get_support()[i]]
            st.write(f"Top-{k} features selected: **{final_feat_names}**")

        # ── Train 4 models ────────────────────────────────────────────────────
        custom_models = {
            "LogisticRegression": LR(max_iter=1000, random_state=42),
            "DecisionTree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
            "RandomForest"      : RandomForestClassifier(n_estimators=100,
                                                         random_state=42, n_jobs=-1),
            "XGBoost"           : XGBClassifier(n_estimators=100, eval_metric="logloss",
                                                random_state=42, n_jobs=-1),
        }

        pb = st.progress(0, text="Training models...")
        rows_c, roc_data_c = [], {}

        for i, (mname, mdl) in enumerate(custom_models.items()):
            mdl.fit(X_train, y_train)
            yp  = mdl.predict(X_test)
            ypr = mdl.predict_proba(X_test)[:, 1]
            rows_c.append({
                "Model"    : mname,
                "Accuracy" : round(accuracy_score(y_test, yp), 4),
                "Precision": round(precision_score(y_test, yp, zero_division=0), 4),
                "Recall"   : round(recall_score(y_test, yp, zero_division=0), 4),
                "F1-Score" : round(f1_score(y_test, yp, zero_division=0), 4),
                "ROC-AUC"  : round(roc_auc_score(y_test, ypr), 4),
            })
            fpr, tpr, _ = roc_curve(y_test, ypr)
            roc_data_c[mname] = (fpr, tpr)
            pb.progress((i + 1) / 4, text=f"Trained {mname}")

        pb.empty()
        st.success("All 4 models trained successfully!")

        # ── Metrics table ─────────────────────────────────────────────────────
        results_c = pd.DataFrame(rows_c).set_index("Model")
        st.markdown("---")
        st.subheader("Model Metrics")
        st.dataframe(
            results_c.style.highlight_max(axis=0, color="#d4edda"),
            use_container_width=True,
        )
        best = results_c["F1-Score"].idxmax()
        st.info(f"Best model by F1-Score: **{best}** ({results_c.loc[best,'F1-Score']:.4f})")

        # ── Charts ────────────────────────────────────────────────────────────
        cl, cr = st.columns(2)
        with cl:
            st.subheader("ROC Curves")
            fig, ax = plt.subplots(figsize=(6, 5))
            for (mn, (fpr, tpr)), color in zip(roc_data_c.items(), COLORS):
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{mn} ({results_c.loc[mn,'ROC-AUC']})")
            ax.plot([0,1],[0,1],"k--",lw=1)
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curves", fontweight="bold")
            ax.legend(fontsize=8); ax.grid(alpha=0.3)
            st.pyplot(fig); plt.close()

        with cr:
            st.subheader("Model Comparison")
            ml = ["Accuracy","Precision","Recall","F1-Score","ROC-AUC"]
            x  = np.arange(len(ml)); w = 0.18
            fig2, ax2 = plt.subplots(figsize=(8,5))
            for i, (mn, row) in enumerate(results_c.iterrows()):
                off = (i - len(results_c)/2 + 0.5) * w
                ax2.bar(x+off, [row[m] for m in ml], width=w,
                        label=mn, color=COLORS[i], edgecolor="white")
            ax2.set_xticks(x); ax2.set_xticklabels(ml, fontweight="bold")
            ax2.set_ylim(0,1.1); ax2.legend(fontsize=8)
            ax2.set_title("Performance Comparison", fontweight="bold")
            st.pyplot(fig2); plt.close()

        # ── Confusion matrices ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Confusion Matrices")
        cm_cols = st.columns(4)
        for i, (mn, mdl) in enumerate(custom_models.items()):
            yp  = mdl.predict(X_test)
            cm  = confusion_matrix(y_test, yp)
            fig3, ax3 = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax3,
                        xticklabels=["Stay","Churn"],
                        yticklabels=["Stay","Churn"],
                        annot_kws={"size":12,"weight":"bold"}, linewidths=1)
            ax3.set_title(mn, fontweight="bold", fontsize=10)
            ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
            cm_cols[i].pyplot(fig3); plt.close()

        # ── KPI cards ─────────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Dataset Overview")
        k1,k2,k3,k4 = st.columns(4)
        churn_rate = df_work[target_col].mean()
        k1.metric("Total Rows",       f"{len(df_work):,}")
        k2.metric("Features Used",    str(len(feat_cols_c)))
        k3.metric("Churn Rate",       f"{churn_rate:.1%}")
        k4.metric("Features in Model",str(len(final_feat_names)))

        # ── Auto-generated Predict form ───────────────────────────────────────
        st.markdown("---")
        st.subheader("Predict a Single Customer")
        st.markdown(
            "Enter values for each feature below. "
            "The form is **automatically generated** from your dataset's columns."
        )

        with st.form("custom_predict_form"):
            input_vals = {}
            form_cols  = st.columns(3)

            for idx, feat in enumerate(final_feat_names):
                col_data = df_work[feat]
                col_min  = float(col_data.min())
                col_max  = float(col_data.max())
                col_mean = float(col_data.mean())
                col_ui   = form_cols[idx % 3]

                # Binary column → selectbox
                if sorted(col_data.unique().tolist()) == [0, 1] or \
                   sorted(col_data.unique().tolist()) == [0.0, 1.0]:
                    input_vals[feat] = col_ui.selectbox(
                        feat, [0, 1],
                        format_func=lambda x: "Yes (1)" if x else "No (0)",
                        key=f"cpred_{feat}"
                    )
                else:
                    # Numeric slider
                    step = 1.0 if col_max - col_min > 10 else 0.1
                    input_vals[feat] = col_ui.slider(
                        feat,
                        min_value=col_min,
                        max_value=col_max,
                        value=round(col_mean, 2),
                        step=step,
                        key=f"cpred_{feat}"
                    )

            pred_model_name = st.selectbox(
                "Model to use", list(custom_models.keys()), key="cpred_model"
            )
            pred_submitted = st.form_submit_button(
                "🔮 Predict Churn", use_container_width=True
            )

        if pred_submitted:
            raw_input  = np.array([[input_vals[f] for f in final_feat_names]], dtype=float)

            # We need to scale using the full feature set then select
            # Re-build full feature vector (fill unselected with 0)
            full_input = np.zeros((1, len(feat_cols_c)))
            for fi, fname in enumerate(feat_cols_c):
                if fname in input_vals:
                    full_input[0, fi] = input_vals[fname]

            full_scaled = scaler_c.transform(full_input)
            if selector_c is not None:
                final_input = selector_c.transform(full_scaled)
            else:
                final_input = full_scaled

            pred_model = custom_models[pred_model_name]
            pred_label = pred_model.predict(final_input)[0]
            pred_prob  = pred_model.predict_proba(final_input)[0][1]

            st.markdown("---")
            if pred_label == 1:
                st.error(f"⚠️  **This customer is likely to CHURN** ({pred_prob:.1%} probability)")
            else:
                st.success(f"✅  **This customer is likely to STAY** ({pred_prob:.1%} probability)")

            st.progress(float(pred_prob), text=f"Churn risk: {pred_prob:.1%}")

            # All models
            st.subheader("All Models — Churn Probability")
            all_probs = []
            for mn, mdl in custom_models.items():
                p = mdl.predict_proba(final_input)[0][1]
                all_probs.append({"Model": mn, "Churn Probability": f"{p:.1%}",
                                  "Prediction": "🔴 Churn" if p >= 0.5 else "🟢 Stay"})
            st.dataframe(pd.DataFrame(all_probs).set_index("Model"),
                         use_container_width=True)
