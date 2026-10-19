"""
Live Model Explainability Engine (SHAP)
=======================================
Computes real-time Shapley Additive exPlanations (SHAP) at serving time,
returning the top N driving factors for every individual prediction.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import shap
from mlops_platform.serving.schemas import FeatureExplanation


class SHAPExplainer:
    """Manages real-time SHAP feature attributions for inference payloads."""

    def __init__(
        self,
        model: Any,
        feature_names: List[str],
        background_data: Optional[pd.DataFrame] = None,
    ):
        self.model = model
        self.feature_names = feature_names

        # Use 50 samples as background reference dataset for fast computation
        if background_data is not None and len(background_data) > 0:
            bg_sample = background_data.head(50).values
        else:
            bg_sample = np.zeros((1, len(feature_names)))

        # Initialize appropriate SHAP Explainer
        try:
            self.explainer = shap.LinearExplainer(self.model, bg_sample)
        except Exception:
            self.explainer = shap.Explainer(self.model, bg_sample)

    def explain_instance(
        self,
        transformed_row: np.ndarray,
        top_n: int = 3
    ) -> List[FeatureExplanation]:
        """
        Compute top N feature contributions for a single transformed instance.

        Args:
            transformed_row: 1D numpy array of transformed features (shape [n_features]).
            top_n: Number of top driving features to return.

        Returns:
            List of FeatureExplanation models sorted by relative impact.
        """
        row_2d = transformed_row.reshape(1, -1)
        raw_shap_values = self.explainer.shap_values(row_2d)

        # Handle different SHAP output formats (binary classification returns 1D or 2D array)
        if isinstance(raw_shap_values, list):
            # List of arrays [class_0, class_1]
            values = np.array(raw_shap_values[1]).flatten()
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 3:
            # Shape [1, n_features, 2]
            values = raw_shap_values[0, :, 1]
        elif isinstance(raw_shap_values, np.ndarray) and raw_shap_values.ndim == 2:
            # Shape [1, n_features]
            values = raw_shap_values[0]
        else:
            values = np.array(raw_shap_values).flatten()

        abs_values = np.abs(values)
        total_abs = np.sum(abs_values)
        if total_abs == 0:
            total_abs = 1e-6

        # Get indices of top N features sorted by absolute contribution
        top_indices = np.argsort(abs_values)[::-1][:top_n]

        explanations: List[FeatureExplanation] = []
        for idx in top_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"feat_{idx}"
            raw_val = float(values[idx])
            percentage = round((abs_values[idx] / total_abs) * 100.0, 2)
            direction = "increases_risk" if raw_val > 0 else "decreases_risk"

            explanations.append(
                FeatureExplanation(
                    feature=feat_name,
                    shap_value=round(raw_val, 4),
                    impact_percentage=percentage,
                    direction=direction,
                )
            )

        return explanations
