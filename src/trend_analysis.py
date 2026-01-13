import pandas as pd

def sales_trend_analysis(
    df: pd.DataFrame,
    date_col: str = "order_date",
    value_col: str = "sales",
    freq: str = "M"
) -> dict:

    results = {}

    if date_col not in df.columns or value_col not in df.columns:
        return {"error": "Required columns not found"}

    # Copy to avoid modifying original DataFrame
    ts_df = df[[date_col, value_col]].dropna().copy()

    # 🔥 FORCE datetime conversion (THIS FIXES THE ERROR)
    ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
    ts_df = ts_df.dropna(subset=[date_col])

    # Set datetime index
    ts_df = ts_df.set_index(date_col)

    # Resample
    ts_df = ts_df.resample(freq).sum().reset_index()

    # Growth rate
    ts_df["growth_rate"] = ts_df[value_col].pct_change()

    avg_growth = ts_df["growth_rate"].mean()

    if avg_growth > 0.02:
        trend = "increasing"
    elif avg_growth < -0.02:
        trend = "decreasing"
    else:
        trend = "stable"

    results["frequency"] = freq
    results["trend"] = trend
    results["average_growth_rate"] = round(avg_growth, 4)

    # Handle edge cases (single row)
    if len(ts_df) > 1:
        results["best_period"] = ts_df.loc[ts_df["growth_rate"].idxmax()].to_dict()
        results["worst_period"] = ts_df.loc[ts_df["growth_rate"].idxmin()].to_dict()
    else:
        results["best_period"] = {}
        results["worst_period"] = {}

    results["time_series"] = ts_df.to_dict(orient="records")

    return results
