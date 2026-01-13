# src/eda.py

import pandas as pd

def basic_eda(df: pd.DataFrame) -> dict:
    """
    Perform automated exploratory data analysis
    """
    eda_results = {}

    # Dataset shape
    eda_results["rows"] = df.shape[0]
    eda_results["columns"] = df.shape[1]

    # Missing values
    missing = df.isnull().sum()
    eda_results["missing_values"] = missing[missing > 0].to_dict()

    # Numeric summary
    numeric_cols = df.select_dtypes(include="number")
    eda_results["numeric_summary"] = numeric_cols.describe().to_dict()

    # Categorical summary (top values)
    categorical_cols = df.select_dtypes(include="object")
    cat_summary = {}

    for col in categorical_cols.columns:
        cat_summary[col] = df[col].value_counts().head(5).to_dict()

    eda_results["categorical_summary"] = cat_summary

    return eda_results
