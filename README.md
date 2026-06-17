# Customer Churn Prediction

A complete, interview-ready Machine Learning project that predicts whether a customer will leave a company (churn).

---

## Project Structure

```
customer_churn_prediction/
│
├── data/
│   └── churn.csv                  ← generated synthetic dataset (5 000 rows)
│
├── models/
│   ├── LogisticRegression.pkl
│   ├── DecisionTree.pkl
│   ├── RandomForest.pkl
│   └── XGBoost.pkl
│
├── plots/
│   ├── eda_class_balance.png
│   ├── eda_numeric_distributions.png
│   ├── eda_correlation_heatmap.png
│   ├── eda_churn_vs_features.png
│   ├── feature_importance.png
│   ├── eval_confusion_matrix_<model>.png  (4 files)
│   ├── eval_roc_curves.png
│   └── eval_model_comparison.png
│
├── src/
│   ├── generate_data.py           ← synthetic dataset generator
│   ├── eda.py                     ← exploratory data analysis + plots
│   ├── feature_engineering.py     ← encoding, scaling, SMOTE, feature selection
│   ├── train_models.py            ← model definitions, CV training, persistence
│   └── evaluate.py                ← metrics, confusion matrix, ROC curves
│
├── main.py                        ← single entry-point for the full pipeline
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python main.py

# 3. Flags
python main.py --skip-gen    # use existing data/churn.csv
python main.py --skip-eda    # skip EDA plots (faster)
python main.py --no-smote    # disable SMOTE balancing
python main.py --no-select   # use all features (no SelectKBest)
```

---

## Dataset Features

| Feature           | Type        | Description                              |
|-------------------|-------------|------------------------------------------|
| CustomerID        | string      | Unique identifier (dropped before model) |
| Age               | int         | Customer age in years                    |
| Gender            | categorical | Male / Female                            |
| Tenure            | int         | Months with the company (1–72)           |
| MonthlyCharges    | float       | Monthly bill amount ($20–$120)           |
| TotalCharges      | float       | Cumulative charges                       |
| NumProducts       | int         | Number of products subscribed (1–4)      |
| HasCreditCard     | binary      | 1 = yes, 0 = no                          |
| IsActiveMember    | binary      | 1 = active, 0 = inactive                 |
| ComplaintCount    | int         | Support complaints filed                 |
| SatisfactionScore | int         | Survey score 1–10                        |
| **Churn**         | **binary**  | **1 = churned, 0 = retained (target)**   |

Class distribution: ~80 % retained / ~20 % churned (realistic imbalance).

---

## ML Concepts Covered

### Classification
Binary classification (Churn: yes / no). Four algorithms are compared.

### Data Balancing — SMOTE
The dataset is imbalanced (~80/20). SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic churned-customer samples in the training set to prevent the model from ignoring the minority class.

> **Important:** SMOTE is applied **only to training data** to prevent data leakage.

### Feature Selection — SelectKBest
ANOVA F-test scores each feature's statistical relationship with the target. The top-8 features are selected, reducing dimensionality and noise.

### Evaluation Metrics

```
Confusion Matrix
─────────────────────────────────────
                Predicted 0  Predicted 1
Actual 0 (TN)       TN           FP
Actual 1 (TP)       FN           TP
─────────────────────────────────────

Precision = TP / (TP + FP)   — of all predicted churns, how many were real?
Recall    = TP / (TP + FN)   — of all actual churns, how many did we catch?
F1-Score  = 2 × (P × R) / (P + R)   — harmonic mean; good for imbalanced data
ROC-AUC   — area under the ROC curve; 1.0 = perfect, 0.5 = random
```

For churn prediction, **Recall** is often the most business-critical metric — missing a churner (FN) is usually more costly than a false alarm (FP).

---

## Algorithms

### 1. Logistic Regression
- Linear baseline; fast and interpretable.
- Outputs probabilities via the sigmoid function.
- Good when features have a linear relationship with the log-odds of churn.

### 2. Decision Tree
- Splits data on the most informative features recursively.
- Highly interpretable (can be visualised as a flowchart).
- `max_depth=6` prevents overfitting.

### 3. Random Forest
- Ensemble of 200 decision trees trained on random feature subsets.
- Reduces variance compared to a single tree.
- Built-in feature importance via mean decrease in impurity.

### 4. XGBoost
- Gradient boosting: each tree corrects errors of the previous.
- Typically the best performer on structured/tabular data.
- Regularisation (L1/L2) controls overfitting.

---

## Interview Talking Points

1. **Why SMOTE over class_weight?**  
   SMOTE synthesises new samples; `class_weight` only reweights loss. SMOTE gives the model more diverse minority examples to learn from.

2. **Why F1 over Accuracy?**  
   With 80/20 imbalance a model predicting "never churn" scores 80 % accuracy. F1 penalises both FP and FN, giving a fairer picture.

3. **Why StandardScaler only on train?**  
   Fitting the scaler on test data would leak test statistics into training — a form of data leakage that inflates performance estimates.

4. **When would you prefer Recall over Precision?**  
   When the cost of missing a churner (FN) is higher than wrongly flagging a loyal customer (FP) — which is typical in retention campaigns.

5. **Random Forest vs XGBoost?**  
   Random Forest builds trees in parallel (bagging); XGBoost builds them sequentially (boosting). XGBoost usually wins on accuracy; Random Forest is faster to tune.
