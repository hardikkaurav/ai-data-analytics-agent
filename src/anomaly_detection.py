# src/anomaly_detection.py
import pandas as pd
import numpy as np
from scipy import stats

def anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Detects anomalies using Z-score on the first numeric column found.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        return {
            "zscore_anomalies_count": 0,
            "isolation_forest_anomalies_count": 0,
            "top_zscore_anomalies": []
        }

    # Use the first numeric column for simplicity (e.g., Sales)
    target_col = numeric_cols[0]
    
    # Calculate Z-Scores
    z_scores = np.abs(stats.zscore(df[target_col].dropna()))
    anomalies = df.loc[df[target_col].index[z_scores > 3]] # Threshold: 3 std devs

    return {
        "zscore_anomalies_count": len(anomalies),
        "isolation_forest_anomalies_count": "N/A (Not Implemented)",
        "top_zscore_anomalies": anomalies[[target_col]].head(5).to_dict()
    }