from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the MLOps Platform."""

    model_config = SettingsConfigDict(arbitrary_types_allowed=True)

    # Project Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = PROJECT_ROOT / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = PROJECT_ROOT / "models"
    METRICS_DIR: Path = PROJECT_ROOT / "metrics"

    # Dataset Files
    BASELINE_DATA_PATH: Path = RAW_DATA_DIR / "customer_churn_baseline.csv"
    DRIFTED_DATA_PATH: Path = RAW_DATA_DIR / "customer_churn_drifted.csv"
    TRAIN_DATA_PATH: Path = PROCESSED_DATA_DIR / "train.parquet"
    VAL_DATA_PATH: Path = PROCESSED_DATA_DIR / "val.parquet"
    TEST_DATA_PATH: Path = PROCESSED_DATA_DIR / "test.parquet"
    PREPROCESSOR_PATH: Path = PROCESSED_DATA_DIR / "preprocessor.joblib"

    # Feature Schema Definitions
    ID_COLUMN: str = "customer_id"
    TARGET_COLUMN: str = "churn"

    NUMERICAL_FEATURES: list[str] = [
        "credit_score",
        "age",
        "tenure_months",
        "monthly_charges",
        "total_charges",
        "num_support_tickets",
    ]

    CATEGORICAL_FEATURES: list[str] = [
        "contract_type",
        "payment_method",
        "has_paperless_billing",
    ]

    # Data Splitting
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15
    TEST_RATIO: float = 0.15
    RANDOM_SEED: int = 42

    # MLflow Configuration
    MLFLOW_TRACKING_URI: str = "sqlite:///mlruns.db"
    MLFLOW_EXPERIMENT_NAME: str = "enterprise_customer_churn"
    MLFLOW_REGISTERED_MODEL_NAME: str = "CustomerChurnChampion"

    # API Serving
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_TITLE: str = "Enterprise MLOps Inference Service"
    API_VERSION: str = "1.0.0"


# Global Settings Singleton
settings = Settings()
