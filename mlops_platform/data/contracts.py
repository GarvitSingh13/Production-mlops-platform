"""
Data Contracts & Schema Validation
===================================
Enforces strict schema constraints, data types, value ranges, and allowable
categories on training data and live inference payloads.
"""

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

# Allowable Categorical Values
ContractType = Literal["month-to-month", "one-year", "two-year"]
PaymentMethod = Literal["credit_card", "bank_transfer", "electronic_check"]


class ContractValidationError(Exception):
    """Raised when incoming dataset or payload violates the data quality contract."""

    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        message = f"Data Contract Violation: Found {len(errors)} validation failure(s):\n"
        for err in errors[:10]:  # Show top 10 violations
            message += f" - [{err.get('field', 'general')}]: {err.get('message')}\n"
        if len(errors) > 10:
            message += f" ... and {len(errors) - 10} more errors.\n"
        super().__init__(message)


class InferencePayload(BaseModel):
    """Schema for online inference requests (without target)."""

    customer_id: str = Field(..., description="Unique customer identifier", min_length=3)
    credit_score: int = Field(..., ge=300, le=850, description="Credit score between 300 and 850")
    age: int = Field(..., ge=18, le=100, description="Customer age between 18 and 100")
    tenure_months: int = Field(..., ge=0, le=120, description="Tenure with service in months")
    monthly_charges: float = Field(..., gt=0.0, le=1000.0, description="Monthly recurring charge")
    total_charges: float = Field(..., ge=0.0, description="Cumulative lifetime charges")
    num_support_tickets: int = Field(..., ge=0, le=50, description="Support tickets opened in past year")
    contract_type: ContractType = Field(..., description="Type of customer contract")
    payment_method: PaymentMethod = Field(..., description="Payment method used")
    has_paperless_billing: int = Field(..., ge=0, le=1, description="Binary flag for paperless billing (0 or 1)")

    @model_validator(mode="after")
    def validate_total_charges_consistency(self) -> "InferencePayload":
        """Validate that total charges is roughly consistent with tenure and monthly charges."""
        if self.tenure_months == 0 and self.total_charges > self.monthly_charges:
            raise ValueError("Customer with 0 tenure months cannot have total_charges greater than monthly_charges.")
        return self


class CustomerDataRecord(InferencePayload):
    """Schema for historical training records (includes target column)."""

    churn: int = Field(..., ge=0, le=1, description="Target label: 1 if customer churned, 0 otherwise")


def validate_dataframe_contract(
    df: pd.DataFrame,
    is_training: bool = True,
    raise_on_error: bool = True
) -> list[dict[str, Any]]:
    """
    Vectorized high-performance batch validation of an entire DataFrame.
    Checks column presence, missing values, numerical bounds, and category sets.

    Args:
        df: Input pandas DataFrame to validate.
        is_training: If True, requires the target column 'churn'.
        raise_on_error: If True, raises ContractValidationError if any violation is found.

    Returns:
        List of error dictionaries detailing any contract violations.
    """
    errors: list[dict[str, Any]] = []

    # 1. Required Columns Check
    expected_cols = [
        "customer_id",
        "credit_score",
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
        "contract_type",
        "payment_method",
        "has_paperless_billing",
    ]
    if is_training:
        expected_cols.append("churn")

    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        errors.append({
            "field": "columns",
            "message": f"Missing required columns in dataset: {missing_cols}"
        })
        if raise_on_error:
            raise ContractValidationError(errors)

    # 2. Null Value Check
    null_counts = df[expected_cols].isnull().sum()
    cols_with_nulls = null_counts[null_counts > 0]
    for col, count in cols_with_nulls.items():
        errors.append({
            "field": col,
            "message": f"Column '{col}' contains {count} null/NaN values (nulls are not permitted)."
        })

    # 3. Numerical Bounds Validation
    bounds = {
        "credit_score": (300, 850),
        "age": (18, 100),
        "tenure_months": (0, 120),
        "monthly_charges": (0.01, 1000.0),
        "total_charges": (0.0, 100000.0),
        "num_support_tickets": (0, 50),
        "has_paperless_billing": (0, 1),
    }
    if is_training:
        bounds["churn"] = (0, 1)

    for col, (min_val, max_val) in bounds.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            violating_min = (df[col] < min_val).sum()
            violating_max = (df[col] > max_val).sum()
            if violating_min > 0:
                errors.append({
                    "field": col,
                    "message": f"Column '{col}' has {violating_min} value(s) below minimum allowed ({min_val})."
                })
            if violating_max > 0:
                errors.append({
                    "field": col,
                    "message": f"Column '{col}' has {violating_max} value(s) above maximum allowed ({max_val})."
                })

    # 4. Categorical Domain Validation
    valid_categories = {
        "contract_type": {"month-to-month", "one-year", "two-year"},
        "payment_method": {"credit_card", "bank_transfer", "electronic_check"},
    }

    for col, allowed_set in valid_categories.items():
        if col in df.columns:
            actual_categories = set(df[col].dropna().unique())
            invalid_cats = actual_categories - allowed_set
            if invalid_cats:
                errors.append({
                    "field": col,
                    "message": f"Column '{col}' contains invalid categories: {invalid_cats}. Allowed: {allowed_set}"
                })

    if errors and raise_on_error:
        raise ContractValidationError(errors)

    return errors
