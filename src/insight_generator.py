# src/insight_generator.py
from typing import Dict


def build_insight_prompt(
    eda: Dict,
    trends: Dict,
    anomalies: Dict
) -> str:
    """
    Build a structured prompt for the LLM using analytics outputs
    """

    prompt = f"""
You are a senior data analyst.

Your task:
- Analyze the provided analytics results
- Generate clear business insights
- Do NOT invent numbers
- Do NOT assume causes without evidence
- If unsure, say so clearly

### Dataset Overview
Rows: {eda.get("rows")}
Columns: {eda.get("columns")}

### Missing Values
{eda.get("missing_values")}

### Trend Analysis
Overall trend: {trends.get("trend")}
Average growth rate: {trends.get("average_growth_rate")}
Best period: {trends.get("best_period")}
Worst period: {trends.get("worst_period")}

### Anomaly Detection
Z-score anomalies: {anomalies.get("zscore_anomalies_count")}
Isolation Forest anomalies: {anomalies.get("isolation_forest_anomalies_count")}
Top anomalies (Z-score): {anomalies.get("top_zscore_anomalies")}
Top anomalies (Isolation Forest): {anomalies.get("top_iforest_anomalies")}

### Output Requirements
1. Executive summary (3–4 sentences)
2. Key insights (bullet points)
3. Potential business implications
4. Limitations of the analysis
"""

    return prompt
