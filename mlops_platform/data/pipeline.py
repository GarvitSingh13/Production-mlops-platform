"""
Data Processing & Feature Pipeline
==================================
Handles raw data ingestion, contract validation, stratified splitting,
leak-free feature transformations (StandardScaler + OneHotEncoder),
and Parquet serialization.
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_platform.config.settings import settings
from mlops_platform.data.contracts import validate_dataframe_contract
from mlops_platform.data.generator import generate_synthetic_churn_data


class DataPipeline:
    """Production-grade tabular data processing and feature engineering pipeline."""

    def __init__(self, random_seed: int = settings.RANDOM_SEED):
        self.random_seed = random_seed
        self.preprocessor: ColumnTransformer = None
        self.feature_names_: list = []

    def build_preprocessor(self) -> ColumnTransformer:
        """Create a Scikit-Learn ColumnTransformer for numerical and categorical features."""
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, settings.NUMERICAL_FEATURES),
                ("cat", categorical_pipeline, settings.CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        return preprocessor

    def run(
        self,
        raw_data_path: Path = settings.BASELINE_DATA_PATH,
        save_artifacts: bool = True
    ) -> dict[str, Any]:
        """
        Execute full data pipeline: Ingestion -> Contract Validation -> Split -> Transform -> Save.

        Returns:
            Dict containing metadata, shapes, class balances, and transformed DataFrames.
        """
        # 1. Ingest Data (or generate if missing)
        if not raw_data_path.exists():
            print(f"[INFO] Raw data not found at {raw_data_path}. Generating baseline dataset...")
            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
            df_raw = generate_synthetic_churn_data(
                n_samples=6000,
                random_seed=self.random_seed,
                drift_mode=False,
                output_path=raw_data_path,
            )
        else:
            df_raw = pd.read_csv(raw_data_path)

        # 2. Validate Data Contract
        validate_dataframe_contract(df_raw, is_training=True, raise_on_error=True)

        # 3. Stratified Train / Val / Test Split
        # First split into Train (70%) and Temp (30%)
        train_df, temp_df = train_test_split(
            df_raw,
            test_size=(settings.VAL_RATIO + settings.TEST_RATIO),
            random_state=self.random_seed,
            stratify=df_raw[settings.TARGET_COLUMN],
        )

        # Split Temp into Val (15% overall) and Test (15% overall)
        relative_val_size = settings.VAL_RATIO / (settings.VAL_RATIO + settings.TEST_RATIO)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1.0 - relative_val_size),
            random_state=self.random_seed,
            stratify=temp_df[settings.TARGET_COLUMN],
        )

        # 4. Feature Extraction & Pipeline Fitting (Train Set ONLY)
        self.preprocessor = self.build_preprocessor()

        X_train_raw = train_df[settings.NUMERICAL_FEATURES + settings.CATEGORICAL_FEATURES]
        y_train = train_df[settings.TARGET_COLUMN].values

        X_val_raw = val_df[settings.NUMERICAL_FEATURES + settings.CATEGORICAL_FEATURES]
        y_val = val_df[settings.TARGET_COLUMN].values

        X_test_raw = test_df[settings.NUMERICAL_FEATURES + settings.CATEGORICAL_FEATURES]
        y_test = test_df[settings.TARGET_COLUMN].values

        # Fit strictly on train; transform all splits
        X_train_transformed = self.preprocessor.fit_transform(X_train_raw)
        X_val_transformed = self.preprocessor.transform(X_val_raw)
        X_test_transformed = self.preprocessor.transform(X_test_raw)

        # Extract transformed feature names
        num_features = settings.NUMERICAL_FEATURES
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
        cat_features = cat_encoder.get_feature_names_out(settings.CATEGORICAL_FEATURES).tolist()
        self.feature_names_ = num_features + cat_features

        # Assemble clean transformed DataFrames
        train_processed = pd.DataFrame(X_train_transformed, columns=self.feature_names_)
        train_processed[settings.TARGET_COLUMN] = y_train

        val_processed = pd.DataFrame(X_val_transformed, columns=self.feature_names_)
        val_processed[settings.TARGET_COLUMN] = y_val

        test_processed = pd.DataFrame(X_test_transformed, columns=self.feature_names_)
        test_processed[settings.TARGET_COLUMN] = y_test

        # 5. Persist Processed Artifacts
        if save_artifacts:
            settings.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

            train_processed.to_parquet(settings.TRAIN_DATA_PATH, index=False)
            val_processed.to_parquet(settings.VAL_DATA_PATH, index=False)
            test_processed.to_parquet(settings.TEST_DATA_PATH, index=False)
            joblib.dump(self.preprocessor, settings.PREPROCESSOR_PATH)

        summary = {
            "total_records": len(df_raw),
            "train_shape": train_processed.shape,
            "val_shape": val_processed.shape,
            "test_shape": test_processed.shape,
            "num_features": len(self.feature_names_),
            "train_churn_rate": float(train_processed[settings.TARGET_COLUMN].mean()),
            "val_churn_rate": float(val_processed[settings.TARGET_COLUMN].mean()),
            "test_churn_rate": float(test_processed[settings.TARGET_COLUMN].mean()),
            "feature_names": self.feature_names_,
            "train_df": train_processed,
            "val_df": val_processed,
            "test_df": test_processed,
            "preprocessor": self.preprocessor,
        }
        return summary


def run_data_pipeline() -> dict[str, Any]:
    """Helper entrypoint to execute the pipeline with default settings."""
    pipeline = DataPipeline()
    return pipeline.run()


if __name__ == "__main__":
    result = run_data_pipeline()
    print(f"[OK] Pipeline finished. Transformed {result['num_features']} features.")
    print(f"     Train: {result['train_shape']} | Val: {result['val_shape']} | Test: {result['test_shape']}")
