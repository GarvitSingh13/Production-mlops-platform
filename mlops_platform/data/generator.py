"""
Synthetic Enterprise Dataset Generator
======================================
Generates realistic, production-grade tabular customer data with realistic
feature interactions, business rules, and intentional drift modes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from mlops_platform.config.settings import settings


def generate_synthetic_churn_data(
    n_samples: int = 5000,
    random_seed: int = 42,
    drift_mode: bool = False,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Generate synthetic customer records with realistic business logic.

    Args:
        n_samples: Number of records to generate.
        random_seed: Random state for deterministic reproduction.
        drift_mode: If True, shifts feature distributions to simulate production drift.
        output_path: Optional path to save the generated dataset as CSV.

    Returns:
        Pandas DataFrame conforming to CustomerDataRecord contract.
    """
    rng = np.random.default_rng(random_seed)

    # 1. Customer Identifiers
    customer_ids = [f"CUST-{100000 + i}" for i in range(n_samples)]

    # 2. Features (with conditional distribution shift if drift_mode=True)
    if not drift_mode:
        # Baseline Distributions
        credit_score = rng.normal(loc=650, scale=80, size=n_samples).clip(300, 850).astype(int)
        age = rng.normal(loc=42, scale=12, size=n_samples).clip(18, 90).astype(int)
        tenure_months = rng.integers(low=1, high=72, size=n_samples)
        monthly_charges = rng.normal(loc=65.0, scale=25.0, size=n_samples).clip(18.0, 180.0).round(2)
        num_support_tickets = rng.poisson(lam=1.5, size=n_samples).clip(0, 15)
        contract_types = rng.choice(
            ["month-to-month", "one-year", "two-year"],
            size=n_samples,
            p=[0.55, 0.25, 0.20],
        )
        payment_methods = rng.choice(
            ["credit_card", "bank_transfer", "electronic_check"],
            size=n_samples,
            p=[0.35, 0.35, 0.30],
        )
        paperless = rng.choice([0, 1], size=n_samples, p=[0.40, 0.60])
    else:
        # DRIFTED Distributions (Simulating inflation, product instability, and dissatisfaction)
        # - Credit scores shift downward
        # - Monthly charges shift upward
        # - Support tickets jump significantly (product bugs)
        # - More month-to-month contracts
        credit_score = rng.normal(loc=590, scale=95, size=n_samples).clip(300, 850).astype(int)
        age = rng.normal(loc=38, scale=14, size=n_samples).clip(18, 90).astype(int)
        tenure_months = rng.integers(low=1, high=48, size=n_samples)  # Shorter tenure
        monthly_charges = rng.normal(loc=88.0, scale=28.0, size=n_samples).clip(25.0, 220.0).round(2)  # Higher price
        num_support_tickets = rng.poisson(lam=4.2, size=n_samples).clip(0, 25)  # Spike in complaints!
        contract_types = rng.choice(
            ["month-to-month", "one-year", "two-year"],
            size=n_samples,
            p=[0.75, 0.15, 0.10],
        )
        payment_methods = rng.choice(
            ["credit_card", "bank_transfer", "electronic_check"],
            size=n_samples,
            p=[0.20, 0.30, 0.50],
        )
        paperless = rng.choice([0, 1], size=n_samples, p=[0.25, 0.75])

    # Realistic Total Charges with slight noise
    noise = rng.normal(loc=0.0, scale=15.0, size=n_samples)
    total_charges = np.maximum(
        monthly_charges,
        (tenure_months * monthly_charges + noise).round(2)
    )

    # 3. Ground Truth Churn Logic (Log-Odds with non-linear feature interactions)
    # Higher churn risk: month-to-month, high support tickets, high monthly charges, low credit score
    # Lower churn risk: two-year contract, long tenure, high credit score
    contract_risk = np.where(contract_types == "month-to-month", 1.2, np.where(contract_types == "one-year", -0.4, -1.5))
    payment_risk = np.where(payment_methods == "electronic_check", 0.6, -0.2)

    log_odds = (
        -1.8
        + 0.55 * num_support_tickets
        + 0.015 * (monthly_charges - 65.0)
        - 0.04 * tenure_months
        - 0.003 * (credit_score - 650)
        + contract_risk
        + payment_risk
        + 0.3 * paperless
        + rng.normal(0, 0.3, size=n_samples)  # Unobserved stochastic factor
    )

    churn_prob = 1.0 / (1.0 + np.exp(-log_odds))
    churn = (rng.uniform(0, 1, size=n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "credit_score": credit_score,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "num_support_tickets": num_support_tickets,
        "contract_type": contract_types,
        "payment_method": payment_methods,
        "has_paperless_billing": paperless,
        "churn": churn,
    })

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


if __name__ == "__main__":
    # Generate baseline training dataset
    df_baseline = generate_synthetic_churn_data(
        n_samples=6000,
        random_seed=settings.RANDOM_SEED,
        drift_mode=False,
        output_path=settings.BASELINE_DATA_PATH
    )
    print(f"[OK] Generated baseline dataset: {df_baseline.shape} -> {settings.BASELINE_DATA_PATH}")

    # Generate drifted inference dataset for Stage 8 monitoring
    df_drifted = generate_synthetic_churn_data(
        n_samples=1500,
        random_seed=settings.RANDOM_SEED + 99,
        drift_mode=True,
        output_path=settings.DRIFTED_DATA_PATH
    )
    print(f"[OK] Generated drifted dataset: {df_drifted.shape} -> {settings.DRIFTED_DATA_PATH}")
