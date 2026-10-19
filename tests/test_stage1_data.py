"""
Unit & Integration Tests for Stage 1: Data Contracts & Pipeline
===============================================================
Verifies data generation, contract bounds, invalid input rejection,
leak-free preprocessor fitting, and stratified split parity.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from mlops_platform.config.settings import settings
from mlops_platform.data.contracts import (
    ContractValidationError,
    InferencePayload,
    validate_dataframe_contract,
)
from mlops_platform.data.generator import generate_synthetic_churn_data
from mlops_platform.data.pipeline import DataPipeline


class TestDataGenerator:
    """Tests for synthetic dataset generator."""

    def test_generator_shape_and_columns(self):
        df = generate_synthetic_churn_data(n_samples=200, random_seed=42)
        assert len(df) == 200
        assert settings.TARGET_COLUMN in df.columns
        for num_col in settings.NUMERICAL_FEATURES:
            assert num_col in df.columns
        for cat_col in settings.CATEGORICAL_FEATURES:
            assert cat_col in df.columns

    def test_generator_churn_rate_sanity(self):
        df = generate_synthetic_churn_data(n_samples=1000, random_seed=42)
        churn_rate = df[settings.TARGET_COLUMN].mean()
        # Churn rate should be realistically between 15% and 45%
        assert 0.15 <= churn_rate <= 0.45

    def test_generator_drifted_data_differs(self):
        df_base = generate_synthetic_churn_data(n_samples=500, random_seed=42, drift_mode=False)
        df_drift = generate_synthetic_churn_data(n_samples=500, random_seed=42, drift_mode=True)
        # In drifted dataset, support tickets should be significantly higher
        assert df_drift["num_support_tickets"].mean() > df_base["num_support_tickets"].mean()


class TestDataContracts:
    """Tests for Pydantic contracts and batch DataFrame validation."""

    def test_valid_dataframe_passes(self):
        df = generate_synthetic_churn_data(n_samples=100, random_seed=42)
        errors = validate_dataframe_contract(df, is_training=True, raise_on_error=True)
        assert len(errors) == 0

    def test_missing_column_raises_error(self):
        df = generate_synthetic_churn_data(n_samples=50, random_seed=42)
        df_dropped = df.drop(columns=["monthly_charges"])
        with pytest.raises(ContractValidationError) as exc_info:
            validate_dataframe_contract(df_dropped, is_training=True, raise_on_error=True)
        assert "monthly_charges" in str(exc_info.value)

    def test_negative_age_raises_error(self):
        df = generate_synthetic_churn_data(n_samples=50, random_seed=42)
        df.loc[0, "age"] = -10  # Invalid age
        with pytest.raises(ContractValidationError) as exc_info:
            validate_dataframe_contract(df, is_training=True, raise_on_error=True)
        assert "age" in str(exc_info.value)

    def test_out_of_bounds_credit_score_raises_error(self):
        df = generate_synthetic_churn_data(n_samples=50, random_seed=42)
        df.loc[0, "credit_score"] = 999  # Max valid is 850
        with pytest.raises(ContractValidationError) as exc_info:
            validate_dataframe_contract(df, is_training=True, raise_on_error=True)
        assert "credit_score" in str(exc_info.value)

    def test_invalid_category_raises_error(self):
        df = generate_synthetic_churn_data(n_samples=50, random_seed=42)
        df.loc[0, "contract_type"] = "decade-to-decade"  # Unknown contract
        with pytest.raises(ContractValidationError) as exc_info:
            validate_dataframe_contract(df, is_training=True, raise_on_error=True)
        assert "contract_type" in str(exc_info.value)

    def test_null_values_raise_error(self):
        df = generate_synthetic_churn_data(n_samples=50, random_seed=42)
        df.loc[2, "tenure_months"] = np.nan
        with pytest.raises(ContractValidationError) as exc_info:
            validate_dataframe_contract(df, is_training=True, raise_on_error=True)
        assert "tenure_months" in str(exc_info.value)

    def test_pydantic_inference_payload_valid(self):
        payload = InferencePayload(
            customer_id="CUST-9999",
            credit_score=720,
            age=35,
            tenure_months=24,
            monthly_charges=75.50,
            total_charges=1812.0,
            num_support_tickets=1,
            contract_type="one-year",
            payment_method="credit_card",
            has_paperless_billing=1,
        )
        assert payload.customer_id == "CUST-9999"

    def test_pydantic_inference_payload_invalid(self):
        with pytest.raises(ValidationError):
            InferencePayload(
                customer_id="CUST-9999",
                credit_score=200,  # Below 300
                age=15,           # Below 18
                tenure_months=12,
                monthly_charges=50.0,
                total_charges=600.0,
                num_support_tickets=0,
                contract_type="invalid-contract",
                payment_method="credit_card",
                has_paperless_billing=1,
            )


class TestDataPipeline:
    """Tests for complete data pipeline execution and artifact generation."""

    def test_pipeline_execution_and_stratification(self, tmp_path):
        raw_csv = tmp_path / "raw_test.csv"
        generate_synthetic_churn_data(n_samples=600, random_seed=42, output_path=raw_csv)

        pipeline = DataPipeline(random_seed=42)
        summary = pipeline.run(raw_data_path=raw_csv, save_artifacts=False)

        # Proportions: ~70% train (420), ~15% val (90), ~15% test (90)
        assert summary["train_shape"][0] == 420
        assert summary["val_shape"][0] == 90
        assert summary["test_shape"][0] == 90

        # Verify stratified balance across splits: churn rates within 4% of each other
        train_churn = summary["train_churn_rate"]
        val_churn = summary["val_churn_rate"]
        test_churn = summary["test_churn_rate"]

        assert abs(train_churn - val_churn) < 0.05
        assert abs(train_churn - test_churn) < 0.05

        # Check transformed features (no NaNs)
        assert not summary["train_df"].isnull().values.any()
        assert not summary["val_df"].isnull().values.any()
        assert not summary["test_df"].isnull().values.any()
