"""
generate_data.py
----------------
Generates a realistic synthetic customer churn dataset and saves it to data/churn.csv.

Features generated:
  - CustomerID       : unique identifier
  - Age              : customer age
  - Gender           : Male / Female
  - Tenure           : months with the company
  - MonthlyCharges   : monthly bill amount
  - TotalCharges     : total amount billed
  - NumProducts      : number of products subscribed
  - HasCreditCard    : 0 / 1
  - IsActiveMember   : 0 / 1
  - ComplaintCount   : number of support complaints
  - SatisfactionScore: survey score 1-10
  - Churn            : target — 1 = churned, 0 = retained
"""

import numpy as np
import pandas as pd
import os

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

N_SAMPLES   = 5_000   # total customers
CHURN_RATE  = 0.20    # ~20 % churn — realistic and imbalanced


def generate_churn_data(n_samples: int = N_SAMPLES, churn_rate: float = CHURN_RATE) -> pd.DataFrame:
    """
    Build a synthetic churn dataset with plausible correlations:
      - Short tenure   → higher churn probability
      - High complaints→ higher churn probability
      - Low satisfaction → higher churn probability
      - Inactive member → higher churn probability
    """

    # ── basic demographics ────────────────────────────────────────────────────
    age     = np.random.randint(18, 70, size=n_samples)
    gender  = np.random.choice(["Male", "Female"], size=n_samples)
    tenure  = np.random.randint(1, 72, size=n_samples)          # months (1–72)

    # ── product / billing ────────────────────────────────────────────────────
    num_products      = np.random.randint(1, 5, size=n_samples)
    has_credit_card   = np.random.randint(0, 2, size=n_samples)
    is_active_member  = np.random.randint(0, 2, size=n_samples)
    monthly_charges   = np.round(np.random.uniform(20, 120, size=n_samples), 2)
    total_charges     = np.round(monthly_charges * tenure + np.random.normal(0, 50, n_samples), 2)
    total_charges     = np.clip(total_charges, 0, None)  # no negatives

    # ── service quality ───────────────────────────────────────────────────────
    complaint_count    = np.random.poisson(lam=1.0, size=n_samples)   # 0-5 typical
    satisfaction_score = np.random.randint(1, 11, size=n_samples)     # 1-10

    # ── churn probability (rule-based, not random) ────────────────────────────
    churn_score = (
        0.30 * (1 - tenure / 72)          # short tenure → risky
        + 0.25 * (complaint_count / 5)    # more complaints → risky
        + 0.20 * (1 - satisfaction_score / 10)  # low satisfaction → risky
        + 0.15 * (1 - is_active_member)   # inactive → risky
        + 0.10 * (monthly_charges / 120)  # high bill → mildly risky
    )

    # Normalise to [0,1] and add noise
    churn_score = churn_score / churn_score.max()
    churn_score += np.random.normal(0, 0.05, n_samples)
    churn_score = np.clip(churn_score, 0, 1)

    # Convert to binary labels while respecting the target churn_rate
    threshold = np.percentile(churn_score, (1 - churn_rate) * 100)
    churn     = (churn_score >= threshold).astype(int)

    # ── assemble DataFrame ────────────────────────────────────────────────────
    df = pd.DataFrame({
        "CustomerID"       : [f"CUST{str(i).zfill(5)}" for i in range(1, n_samples + 1)],
        "Age"              : age,
        "Gender"           : gender,
        "Tenure"           : tenure,
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges,
        "NumProducts"      : num_products,
        "HasCreditCard"    : has_credit_card,
        "IsActiveMember"   : is_active_member,
        "ComplaintCount"   : complaint_count,
        "SatisfactionScore": satisfaction_score,
        "Churn"            : churn,
    })

    return df


def save_data(df: pd.DataFrame, path: str = "data/churn.csv") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[generate_data] Saved {len(df):,} rows → {path}")
    print(f"[generate_data] Churn distribution:\n{df['Churn'].value_counts(normalize=True).round(3)}\n")


if __name__ == "__main__":
    df = generate_churn_data()
    save_data(df, path="data/churn.csv")
