# src/data_loader.py

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV data and perform basic cleaning
    """
    df = pd.read_csv(file_path)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Convert date columns if present
    date_cols = ["order_date", "ship_date"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df
